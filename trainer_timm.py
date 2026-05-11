import logging
import os
import tempfile
import sys
import copy
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.utils.parametrize as P
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import argparse
import tqdm
import subprocess
import json
from pathlib import Path

import timm
from timm.data import Mixup, create_transform, create_loader, resolve_model_data_config
from timm.data.mixup import mixup_target, cutmix_bbox_and_lam
from timm.loss import SoftTargetCrossEntropy, LabelSmoothingCrossEntropy
from timm.optim import create_optimizer_v2
from timm.scheduler import CosineLRScheduler, MultiStepLRScheduler
from timm.data.random_erasing import RandomErasing as TimmRandomErasing

from pc_model import PCNet
from data_utils import ToPackedRGGB, RawImgDataset, load_and_register_buffer, get_parametrized_weight_mods, PackedRGGBToRGB
from scangen.data import NoiseCIFARDataset, MyNoiseCIFARDataset
from distillation import CRDLoss, CRDOptions, MGDLoss
from distillation import SimKD, SRRLLoss, TeacherFeatureExtractor


from trainer import TrainerCiFar, _normalize_dataset_name, _CIFAR_STATS

class TrainerCiFarTimmStyle(TrainerCiFar):
    """
    Single timm-style trainer for CIFAR.

    Supports:
      - custom nn.Module models
      - timm-created models
      - native CIFAR input, e.g. 32x32
      - resized CIFAR input, e.g. 224x224
      - training from scratch
      - ImageNet-pretrained fine-tuning

    Intentionally does NOT support CRD.
    """

    def __init__(
        self,
        *args,

        # -------------------------
        # Optional timm model creation
        # -------------------------
        timm_model_name=None,
        pretrained=False,
        in_chans=3,

        # -------------------------
        # Data config
        # -------------------------
        # If True, use timm model's pretrained data config:
        # input_size, mean, std, interpolation.
        # This is recommended for ImageNet-pretrained fine-tuning.
        use_model_data_config=False,

        # Manual overrides. These have highest priority.
        # Examples:
        #   timm_input_size=(3, 32, 32)
        #   timm_input_size=(3, 224, 224)
        timm_input_size=None,
        timm_mean=None,
        timm_std=None,
        interpolation=None,

        # -------------------------
        # Optimizer
        # -------------------------
        timm_opt="adamw",
        opt_eps=None,
        opt_betas=None,
        momentum=0.9,

        # -------------------------
        # Scheduler / warmup
        # -------------------------
        timm_sched="cosine",
        min_lr=1e-6,
        warmup_lr=1e-6,
        decay_rate=0.1,

        # -------------------------
        # RGB augmentation
        # -------------------------
        timm_aug=True,

        # Good CIFAR default.
        # For 224x224 ImageNet-style fine-tuning, you may prefer (0.08, 1.0).
        timm_train_scale=(0.75, 1.0),
        timm_train_ratio=(1.0, 1.0),

        hflip=0.5,
        vflip=0.0,
        color_jitter=0.1,
        auto_augment="rand-m9-mstd0.5-inc1",

        # Random erasing.
        re_prob=0.25,
        re_mode="pixel",
        re_count=1,

        # -------------------------
        # Mixup / CutMix / smoothing
        # -------------------------
        label_smoothing=0.1,
        mixup_alpha=0.2,
        cutmix_alpha=1.0,
        cutmix_minmax=None,
        mixup_prob=1.0,
        switch_prob=0.5,
        mixup_mode="batch",

        # -------------------------
        # For rggb-like data (4,16,16)
        # -------------------------
        convert_non_rgb_to_rgb=False,
        non_rgb_spatial_aug=True,
        non_rgb_crop_padding=2,
        non_rgb_affine_degrees=0,
        non_rgb_affine_translate=None,
        non_rgb_affine_shear=None,

        # -------------------------
        # Loader
        # -------------------------
        num_workers=2,
        pin_memory=True,
        persistent_workers=False,

        is_timm_model=True,

        **kwargs,
    ):
        self.is_timm_model = is_timm_model
        # ------------------------------------------------------------
        # Disable CRD in this trainer.
        # ------------------------------------------------------------
        distill_method = kwargs.get("distill_method", "none")
        distill_method = (distill_method or "none").lower()
        if "crd" in distill_method:
            raise ValueError(
                "TrainerCiFarTimmStyle does not support CRD. "
                "Use distill_method='none' or 'kd'."
            )

        # ------------------------------------------------------------
        # Optional timm model construction.
        # If caller already passes model=..., this is not needed.
        # ------------------------------------------------------------
        dataset_name = _normalize_dataset_name(kwargs.get("dataset_name", "cifar10"))
        num_classes = 100 if dataset_name == "cifar100" else 10

        if timm_model_name is not None:
            kwargs["model"] = timm.create_model(
                timm_model_name,
                pretrained=pretrained,
                num_classes=num_classes,
                in_chans=in_chans,
            )

        # ------------------------------------------------------------
        # Store config before parent __init__.
        # Parent __init__ calls self._get_optimizer() and self._prepare_cifar().
        # ------------------------------------------------------------
        self.timm_model_name = timm_model_name
        self.pretrained = pretrained
        self.in_chans = in_chans

        self.use_model_data_config = use_model_data_config
        self.timm_input_size = timm_input_size
        self.timm_mean = timm_mean
        self.timm_std = timm_std
        self.interpolation = interpolation

        self.timm_opt = timm_opt
        self.opt_eps = opt_eps
        self.opt_betas = opt_betas
        self.momentum = momentum

        self.timm_sched = timm_sched
        self.min_lr = min_lr
        self.warmup_lr = warmup_lr
        self.decay_rate = decay_rate

        self.timm_aug = timm_aug
        self.timm_train_scale = timm_train_scale
        self.timm_train_ratio = timm_train_ratio
        self.hflip = hflip
        self.vflip = vflip
        self.color_jitter = color_jitter
        self.auto_augment = auto_augment
        self.re_prob = re_prob
        self.re_mode = re_mode
        self.re_count = re_count

        self.label_smoothing = label_smoothing
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.cutmix_minmax = cutmix_minmax
        self.mixup_prob = mixup_prob
        self.switch_prob = switch_prob
        self.mixup_mode = mixup_mode

        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers

        # Parent receives lr_reduce_on but does not store it.
        self.lr_reduce_on = kwargs.get("lr_reduce_on", "80,122,150,225,262")
        self.T_0 = kwargs.get("T0", None)

        self.mixup_fn = None
        self.train_loss_fn = None

        # For rggb-like data (4,16,16)
        self.convert_non_rgb_to_rgb = convert_non_rgb_to_rgb

        self.non_rgb_spatial_aug = non_rgb_spatial_aug
        self.non_rgb_crop_padding = non_rgb_crop_padding
        self.non_rgb_affine_degrees = non_rgb_affine_degrees
        self.non_rgb_affine_translate = non_rgb_affine_translate
        self.non_rgb_affine_shear = non_rgb_affine_shear

        # evaluate() uses hard labels, so keep eval loss as hard-label CE.
        if "loss_fn" not in kwargs:
            kwargs["loss_fn"] = nn.CrossEntropyLoss()

        super().__init__(*args, **kwargs)

        # Replace parent's torch scheduler/warmup with timm scheduler.
        self.warmup_scheduler = None
        self._build_timm_scheduler()
        self._build_timm_loss_and_mixup()

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------
    def _get_optimizer(self, optim_type, lr, weight_decay):
        """
        Called inside TrainerCiFar.__init__.

        Keep same signature as parent, but use timm optimizer factory.
        """
        opt_kwargs = dict(
            opt=self.timm_opt,
            lr=lr,
            weight_decay=weight_decay,
            momentum=self.momentum,
        )

        if self.opt_eps is not None:
            opt_kwargs["eps"] = self.opt_eps

        if self.opt_betas is not None:
            opt_kwargs["betas"] = self.opt_betas

        return create_optimizer_v2(self.model, **opt_kwargs)

    # ------------------------------------------------------------------
    # Scheduler
    # ------------------------------------------------------------------
    def _build_timm_scheduler(self):
        """
        Timm epoch-level scheduler.

        train() calls:
            self.scheduler.step(epoch + 1)
        """
        if self.timm_sched == "cosine":
            if self.T_0 is None:
                logging.warning("Using cosine LR with NO restart.")
                self.scheduler = CosineLRScheduler(
                    self.optimizer,
                    t_initial=self.num_epochs,
                    lr_min=self.min_lr,
                    warmup_t=self.warmup_epoch,
                    warmup_lr_init=self.warmup_lr,
                    warmup_prefix=True,
                )
            else:
                logging.warning("Using cosine LR with restart. T0: {}".format(self.T_0))
                self.scheduler = CosineLRScheduler(
                    self.optimizer,
                    t_initial=self.T_0,
                    cycle_mul=2,
                    cycle_limit=999,
                    lr_min=self.min_lr,
                    warmup_t=self.warmup_epoch,
                    warmup_lr_init=self.warmup_lr,
                    warmup_prefix=True,
                )

        elif self.timm_sched == "multistep":
            decay_t = list(map(int, self.lr_reduce_on.split(",")))

            self.scheduler = MultiStepLRScheduler(
                self.optimizer,
                decay_t=decay_t,
                decay_rate=self.decay_rate,
                warmup_t=self.warmup_epoch,
                warmup_lr_init=self.warmup_lr,
            )

        else:
            raise ValueError(f"Unknown timm_sched: {self.timm_sched}")

    # ------------------------------------------------------------------
    # Data config / transforms
    # ------------------------------------------------------------------
    def _get_timm_data_config(self, dataset_name):
        """
        Resolve input_size / mean / std / interpolation.

        Priority:
          1. explicit manual arguments:
                timm_input_size, timm_mean, timm_std, interpolation
          2. if use_model_data_config=True:
                timm model's pretrained data config
          3. default:
                CIFAR stats and 32x32 input
        """
        cifar_mean, cifar_std = _CIFAR_STATS[dataset_name]

        model_cfg = {}
        if self.use_model_data_config:
            try:
                model_cfg = resolve_model_data_config(self.model)
            except Exception:
                model_cfg = {}

        input_size = (
            self.timm_input_size
            if self.timm_input_size is not None
            else model_cfg.get("input_size", (3, 32, 32))
        )

        mean = (
            self.timm_mean
            if self.timm_mean is not None
            else model_cfg.get("mean", cifar_mean)
        )

        std = (
            self.timm_std
            if self.timm_std is not None
            else model_cfg.get("std", cifar_std)
        )

        interpolation = (
            self.interpolation
            if self.interpolation is not None
            else model_cfg.get("interpolation", "bicubic")
        )

        return input_size, mean, std, interpolation

    def _infer_timm_input_size(self, img_type):
        if img_type == "rgb":
            input_size, _, _, _ = self._get_timm_data_config(self.dataset_name)
            return input_size

        # For non-RGB branches, keep the original custom convention.
        # This trainer is mainly intended for RGB/timm models.
        if self.timm_input_size is not None:
            return self.timm_input_size

        if img_type in {"rggb", "scanGFI"}:
            return (4, 16, 16)

        return (3, 16, 16)

    def _set_transform_recursive(self, dataset, transform, set_teacher=False):
        """
        Recursively set transform on wrapped datasets.

        This handles wrappers like DatasetWithIndex or paired datasets.
        By default, do not touch teacher_dataset.
        """
        if hasattr(dataset, "transform"):
            dataset.transform = transform

        if hasattr(dataset, "dataset"):
            self._set_transform_recursive(
                dataset.dataset,
                transform,
                set_teacher=set_teacher,
            )

        if hasattr(dataset, "student_dataset"):
            self._set_transform_recursive(
                dataset.student_dataset,
                transform,
                set_teacher=set_teacher,
            )

        if set_teacher and hasattr(dataset, "teacher_dataset"):
            self._set_transform_recursive(
                dataset.teacher_dataset,
                transform,
                set_teacher=set_teacher,
            )

    def _build_rgb_timm_transforms(self, dataset_name):
        input_size, mean, std, interpolation = self._get_timm_data_config(dataset_name)

        transform_train = create_transform(
            input_size=input_size,
            is_training=True,
            use_prefetcher=False,

            # Crop / flip.
            scale=self.timm_train_scale,
            ratio=self.timm_train_ratio,
            hflip=self.hflip,
            vflip=self.vflip,

            # Color / policy augmentation.
            color_jitter=self.color_jitter,
            auto_augment=self.auto_augment,
            interpolation=interpolation,

            # Normalization.
            mean=mean,
            std=std,

            # Random erasing.
            re_prob=self.re_prob,
            re_mode=self.re_mode,
            re_count=self.re_count,
        )

        test_kwargs = dict(
            input_size=input_size,
            is_training=False,
            use_prefetcher=False,
            interpolation=interpolation,
            mean=mean,
            std=std,
        )

        if input_size[-2:] == (32, 32):
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        else:
            transform_test = create_transform(**test_kwargs)

        return transform_train, transform_test

    def _build_scanGFI_to_rgb_timm_transforms(self, dataset_name):
        # mean, std = _CIFAR_STATS[dataset_name]
        logging.warning("Building transforms for converted to rgb images.")
        input_size = self.timm_input_size or (3, 32, 32)
        mean, std = (0.0,) * input_size[0], (1.0,) * input_size[0]
        interpolation = self.interpolation or "bicubic"

        transform_train = transforms.Compose([
            PackedRGGBToRGB(),
            transforms.ToPILImage(),
            create_transform(
                input_size=input_size,
                is_training=True,
                use_prefetcher=False,
                scale=self.timm_train_scale,
                ratio=self.timm_train_ratio,
                hflip=self.hflip,
                vflip=self.vflip,
                color_jitter=self.color_jitter,
                auto_augment=self.auto_augment,
                interpolation=interpolation,
                mean=mean,
                std=std,
                re_prob=self.re_prob,
                re_mode=self.re_mode,
                re_count=self.re_count,
            ),
        ])

        test_kwargs = dict(
            input_size=input_size,
            is_training=False,
            use_prefetcher=False,
            interpolation=interpolation,
            mean=mean,
            std=std,
        )

        if input_size[-2:] == (32, 32):
            transform_test = transforms.Compose([
                PackedRGGBToRGB(),
            ])
        else:
            transform_test = transforms.Compose([
                PackedRGGBToRGB(),
                transforms.ToPILImage(),
                create_transform(**test_kwargs),
            ])

        return transform_train, transform_test

    def _build_non_rgb_timm_transforms(self, img_type):
        logging.warning("Building transforms for non rgb images.")
        spatial_size = self.timm_input_size[-1] if self.timm_input_size is not None else 16

        train_steps = []

        if self.non_rgb_spatial_aug:
            train_steps.append(
                transforms.RandomCrop(
                    spatial_size,
                    padding=self.non_rgb_crop_padding,
                )
            )

            if self.hflip > 0.0:
                train_steps.append(transforms.RandomHorizontalFlip(p=self.hflip))

            if self.vflip > 0.0:
                train_steps.append(transforms.RandomVerticalFlip(p=self.vflip))

            if (
                    self.non_rgb_affine_degrees
                    or self.non_rgb_affine_translate is not None
                    or self.non_rgb_affine_shear is not None
            ):
                train_steps.append(
                    transforms.RandomAffine(
                        degrees=self.non_rgb_affine_degrees,
                        translate=self.non_rgb_affine_translate,
                        shear=self.non_rgb_affine_shear,
                    )
                )

        if self.re_prob > 0.0:
            train_steps.append(
                TimmRandomErasing(
                    probability=self.re_prob,
                    mode=self.re_mode,
                    max_count=self.re_count,
                    device="cpu",
                )
            )

        transform_train = transforms.Compose(train_steps) if train_steps else None
        transform_test = None

        return transform_train, transform_test

    def _prepare_cifar(self, img_type, dataset_name):
        """
        Parent builds datasets/wrappers first.

        Then this subclass:
          - replaces RGB transforms with timm create_transform
          - replaces dataloaders with timm create_loader

        For rggb/scanGFI/raw, original custom transforms are preserved.
        The original _prepare_cifar has special branches for rgb/rggb/scanGFI/raw,
        so this subclass only overrides the RGB transform path.
        """
        super()._prepare_cifar(img_type, dataset_name)

        dataset_name = _normalize_dataset_name(dataset_name)
        input_size = self._infer_timm_input_size(img_type)

        if img_type == "rgb" and self.timm_aug:
            transform_train, transform_test = self._build_rgb_timm_transforms(dataset_name)

            self._set_transform_recursive(
                self.train_set,
                transform_train,
                set_teacher=False,
            )
            self._set_transform_recursive(
                self.val_set,
                transform_test,
                set_teacher=False,
            )
        elif img_type == "scanGFI" and self.timm_aug and self.convert_non_rgb_to_rgb:
            transform_train, transform_test = self._build_scanGFI_to_rgb_timm_transforms(dataset_name)
            self._set_transform_recursive(self.train_set, transform_train, set_teacher=False)
            self._set_transform_recursive(self.val_set, transform_test, set_teacher=False)
        elif img_type == "scanGFI" and self.timm_aug:
            transform_train, transform_test = self._build_non_rgb_timm_transforms(img_type)
            if transform_train is not None:
                self._set_transform_recursive(self.train_set, transform_train, set_teacher=False)

        elif img_type != "rgb":
            logging.warning(
                "timm RGB transforms are not applied for img_type=%s. "
                "Keeping original custom transforms.",
                img_type,
            )

        self.train_dataloader = torch.utils.data.DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
            drop_last=True,
        )

        self.val_dataloader = torch.utils.data.DataLoader(
            self.val_set,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )
        ##########################################################################################
        # Note: We are not using create_loader of timm because it will overwrite the transform
        # of the dataset. Since we already set the transforms before, we don't need to set
        # the transform via create_loader.
        ##########################################################################################
        # self.train_dataloader = create_loader(
        #     self.train_set,
        #     input_size=input_size,
        #     batch_size=self.batch_size,
        #     is_training=True,
        #     use_prefetcher=False,
        #     no_aug=True,
        #     re_prob=0.0,
        #     num_workers=self.num_workers,
        #     pin_memory=self.pin_memory,
        #     persistent_workers=self.persistent_workers,
        # )
        #
        # self.val_dataloader = create_loader(
        #     self.val_set,
        #     input_size=input_size,
        #     batch_size=self.test_batch_size,
        #     is_training=False,
        #     use_prefetcher=False,
        #     num_workers=0,
        #     pin_memory=self.pin_memory,
        #     persistent_workers=self.persistent_workers,
        # )

        if self.teacher_eval_loader is None:
            self.teacher_eval_loader = self.val_dataloader

    # ------------------------------------------------------------------
    # Loss / mixup
    # ------------------------------------------------------------------
    def _build_timm_loss_and_mixup(self):
        num_classes = 100 if self.dataset_name == "cifar100" else 10

        mixup_active = (
            self.mixup_alpha > 0.0
            or self.cutmix_alpha > 0.0
            or self.cutmix_minmax is not None
        )

        # KD + mixup is okay only if teacher sees the same mixed input.
        # If orig_t_inp=True, teacher_inputs are separate, so disable for safety.
        if mixup_active and self._kd_enabled and self.orig_t_inp:
            logging.warning(
                "Disabling mixup/cutmix because KD with orig_t_inp=True uses "
                "separate teacher inputs. To enable it correctly, teacher_inputs "
                "must be mixed with the same lambda/permutation."
            )
            mixup_active = False

        if mixup_active:
            self.mixup_fn = Mixup(
                mixup_alpha=self.mixup_alpha,
                cutmix_alpha=self.cutmix_alpha,
                cutmix_minmax=self.cutmix_minmax,
                prob=self.mixup_prob,
                switch_prob=self.switch_prob,
                mode=self.mixup_mode,
                label_smoothing=self.label_smoothing,
                num_classes=num_classes,
            )
            self.train_loss_fn = SoftTargetCrossEntropy()

        elif self.label_smoothing > 0.0:
            self.mixup_fn = None
            self.train_loss_fn = LabelSmoothingCrossEntropy(
                smoothing=self.label_smoothing
            )

        else:
            self.mixup_fn = None
            self.train_loss_fn = self.loss_fn

    # ------------------------------------------------------------------
    # Train loop
    # ------------------------------------------------------------------
    def train(self):
        """
        Same structure as parent train(), but timm scheduler uses:
            self.scheduler.step(epoch + 1)

        Parent train() calls scheduler.step() with no epoch argument. :contentReference[oaicite:2]{index=2}
        """
        train_loss_list, val_acc_list = [], []
        best_acc, val_acc, best_epoch = 0.0, 0.0, 0
        best_top5 = None
        val_top5 = None
        best_model_path = None

        for epoch in range(self.num_epochs):
            print("Training epoch {} / {}".format(epoch, self.num_epochs))

            train_loss = self.train_one_epoch(epoch)

            if (epoch + 1) % self.eval_every == 0 and epoch >= self.skip_eval_epochs:
                train_acc, train_top5, _, _ = self.evaluate(self.train_dataloader)
                val_acc, val_top5, _, _ = self.evaluate(self.val_dataloader)

                train_loss_list.append(train_loss)
                val_acc_list.append(val_acc)

                if self.dataset_name == "cifar100":
                    print(
                        "Validation top1: {}, top5: {}; Train top1: {}, top5: {}".format(
                            val_acc,
                            val_top5,
                            train_acc,
                            train_top5,
                        )
                    )
                else:
                    print("Validation acc: {}, Train acc: {}".format(val_acc, train_acc))

                if val_acc > best_acc:
                    best_acc = val_acc
                    best_epoch = epoch + 1
                    best_top5 = val_top5
                    best_model_path = self._save_model_ckpt(
                        val_acc,
                        epoch + 1,
                        "_best_ckpt.pth",
                    )

            self.scheduler.step(epoch + 1)

        _ = self._save_model_ckpt(val_acc, self.num_epochs, "_last_ckpt.pth")

        print("----- Train finished, Model Name: {} -----".format(self.model_name))
        print(
            "----- Total number of parameters: {} M -----".format(
                sum(p.numel() for p in self.model.parameters()) / 1e6
            )
        )

        if self.dataset_name == "cifar100":
            print(
                "----- Best top1: {}, Best top5: {}, Best epoch: {} -----".format(
                    best_acc,
                    best_top5,
                    best_epoch,
                )
            )
        else:
            print("----- Best acc: {}, Best epoch: {} -----".format(best_acc, best_epoch))

        print("----- Model path: {} -----".format(best_model_path))
        print("--------------------------------------------------------------------------")

        return train_loss_list, val_acc_list

    def train_one_epoch(self, epoch):
        """
        Same basic logic as parent train_one_epoch(), but:
          - optional timm Mixup/CutMix
          - timm train_loss_fn
          - no CRD
          - no parent torch warmup_scheduler step
        """
        self.model.train()
        running_loss, n_samples = 0.0, 0

        progress_bar = tqdm.tqdm(
            enumerate(self.train_dataloader),
            total=len(self.train_dataloader),
            desc="Training",
        )

        for _i, _data in progress_bar:
            teacher_inputs = None

            # Keep compatibility with the original batch formats.
            if self.orig_t_inp and isinstance(_data, (list, tuple)):
                if len(_data) == 3:
                    inputs, teacher_inputs, labels = _data
                elif len(_data) == 4:
                    inputs, teacher_inputs, labels, _ = _data
                elif len(_data) == 5:
                    inputs, teacher_inputs, labels, _, _ = _data
                else:
                    raise RuntimeError(
                        f"Unexpected batch format with orig_t_inp=True: len={len(_data)}"
                    )

            elif isinstance(_data, (list, tuple)) and len(_data) == 4:
                inputs, labels, _, _ = _data

            elif isinstance(_data, (list, tuple)) and len(_data) == 3:
                inputs, labels, _ = _data

            else:
                inputs, labels = _data

            n_samples += inputs.size(0)

            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            if teacher_inputs is not None:
                teacher_inputs = teacher_inputs.to(self.device)

            labels_for_ce = labels

            if self.mixup_fn is not None:
                inputs, labels_for_ce = self.mixup_fn(inputs, labels)

            self.optimizer.zero_grad()

            outputs, student_feat = self._student_forward(inputs)

            teacher_logits = None
            if self.teacher_model is not None and self._kd_enabled:
                teacher_logits, _ = self._teacher_forward(
                    teacher_inputs if teacher_inputs is not None else inputs
                )

            ce_loss = self.train_loss_fn(outputs, labels_for_ce)

            kd_loss = None
            if self._kd_enabled:
                if teacher_logits is None:
                    raise RuntimeError("Teacher logits not available for KD.")

                kd_loss = F.kl_div(
                    F.log_softmax(outputs / self.distill_temperature, dim=1),
                    F.softmax(teacher_logits / self.distill_temperature, dim=1),
                    reduction="batchmean",
                ) * (self.distill_temperature ** 2)

            if self._kd_enabled:
                loss = (1.0 - self.distill_alpha) * ce_loss + self.distill_alpha * kd_loss
            else:
                loss = ce_loss

            loss.backward()

            if self.max_norm is not None:
                nn.utils.clip_grad_norm_(
                    list(self.model.parameters()),
                    max_norm=self.max_norm,
                )

            self.optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            avg_loss = running_loss / n_samples

            postfix = {
                "Iter": f"{_i + 1}/{len(self.train_dataloader)}",
                "Loss": f"{avg_loss:.4f}",
                "LR": self.optimizer.param_groups[0]["lr"],
            }

            if self.noisy_model is not None:
                postfix["Noise"] = f"{self.noisy_model.current_noise_level:.3f}"

            if self.mixup_fn is not None:
                postfix["Mix"] = "on"

            progress_bar.set_postfix(postfix)

        running_loss /= n_samples
        return running_loss

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------
    def _save_model_ckpt(self, acc, epoch, suffix=""):
        """
        Generic checkpoint saving.

        This works for timm models because it does not require model.init_args.
        The original checkpoint path assumes init_args and reconstructs the
        model class, which is not generally available for timm models. :contentReference[oaicite:3]{index=3}
        """
        if not self.is_timm_model:
            return super()._save_model_ckpt(acc, epoch, suffix=suffix)
        save_to = os.path.join(self.save_path, self.model_name)
        os.makedirs(save_to, exist_ok=True)
        save_pth_path = os.path.join(str(save_to), self.model_name + suffix)

        state = {
            "net": self.model.state_dict(),
            "net_type": self.model.__class__.__name__,
            "acc": acc,
            "epoch": epoch,
            "dataset_name": self.dataset_name,
            "img_type": self.img_type,
            "timm_model_name": self.timm_model_name,
            "pretrained": self.pretrained,
            "use_model_data_config": self.use_model_data_config,
            "timm_input_size": self.timm_input_size,
            "timm_mean": self.timm_mean,
            "timm_std": self.timm_std,
            "interpolation": self.interpolation,
        }

        if hasattr(self.model, "init_args"):
            state["init_args"] = self.model.init_args

        torch.save(state, save_pth_path)
        return save_pth_path


class TrainerCiFarTimmStyleFeatureKD(TrainerCiFarTimmStyle):
    """
    Generic timm trainer for feature-based distillation.

    Designed for methods like:
        SRRL
        MGD
        ReviewKD

    Student inference architecture is unchanged:
        outputs = self.model(inputs)

    Feature-KD modules are training-only.
    """

    feature_kd_name = None  # e.g. "srrl", "mgd", "reviewkd"

    def __init__(
        self,
        *args,
        feature_kd_beta=1.0,
        **kwargs,
    ):
        raw_distill_method = (kwargs.get("distill_method", "none") or "none").lower()
        raw_distill_method = raw_distill_method.replace("+", "_")

        if "crd" in raw_distill_method:
            raise ValueError(
                f"{self.__class__.__name__} does not support CRD. "
                "Use the original trainer for CRD."
            )

        if self.feature_kd_name is None:
            raise ValueError("feature_kd_name must be set in subclass.")

        self._feature_kd_requested = self.feature_kd_name in raw_distill_method

        base_tokens = [
            token for token in raw_distill_method.split("_")
            if token not in {"", self.feature_kd_name}
        ]
        kwargs["distill_method"] = "_".join(base_tokens) if base_tokens else "none"

        self.feature_kd_beta = float(feature_kd_beta)

        # To support mixup/cutmix with orig_t_inp=True when training with distillation.
        self._paired_mixup_active = False
        self._paired_mixup_fn = None

        super().__init__(*args, **kwargs)

        self._feature_kd_loss = None
        self._teacher_extractor = None

        if self._feature_kd_requested:
            if self.teacher_model is None:
                raise ValueError(f"{self.feature_kd_name.upper()} requires teacher_model.")

            self._teacher_extractor = self._make_teacher_extractor()
            self._teacher_extractor = self._teacher_extractor.to(self.device)

            self._build_feature_kd_from_one_batch()
            self._add_feature_kd_to_optimizer()

            # Rebuild timm scheduler after adding auxiliary module params.
            self._build_timm_scheduler()

    # ------------------------------------------------------------------
    # Methods intended to be overridden by subclasses
    # ------------------------------------------------------------------
    def _make_teacher_extractor(self):
        """
        Override if a method needs a different teacher feature extractor.

        For SRRL/MGD, final 4D feature is enough.
        For ReviewKD, this should return multi-stage teacher features.
        """
        return TeacherFeatureExtractor(self.teacher_model)

    def _make_feature_kd_loss(self, student_features, teacher_features):
        """
        Build the method-specific auxiliary loss module.

        Args:
            student_features: usually a list of student feature maps
            teacher_features: usually a list of teacher feature maps

        Returns:
            nn.Module with:
                trainable_parameters()
                forward(...)
        """
        raise NotImplementedError

    def _compute_feature_kd_loss(
        self,
        student_features,
        teacher_features,
        teacher_logits,
    ):
        """
        Compute method-specific feature KD loss.

        Returns:
            loss, log_dict
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Shared utilities
    # ------------------------------------------------------------------
    def _build_timm_loss_and_mixup(self):
        # Parent TrainerCiFarTimmStyle disables Mixup/CutMix when
        # _kd_enabled and orig_t_inp=True. For feature-KD, we want to
        # keep Mixup/CutMix and apply paired mixing ourselves.
        old_kd_enabled = getattr(self, "_kd_enabled", False)

        if self._feature_kd_requested and self.orig_t_inp:
            self._kd_enabled = False

        super()._build_timm_loss_and_mixup()
        self._kd_enabled = old_kd_enabled

        self._paired_mixup_active = False
        self._paired_mixup_fn = None

        if self._feature_kd_requested and self.orig_t_inp and self.mixup_fn is not None:
            assert self.mixup_mode == "batch" and self.cutmix_minmax is None

            logging.warning(
                "Using paired timm-style Mixup/CutMix for %s with orig_t_inp=True. ",
                self.feature_kd_name.upper(),
            )

            # Keep timm's Mixup object for its params / label smoothing config.
            # Do not call it directly because it only mixes one input tensor.
            self._paired_mixup_active = True
            self._paired_mixup_fn = self.mixup_fn

            # Disable normal one-input timm Mixup path.
            # train_loss_fn remains SoftTargetCrossEntropy from super().
            self.mixup_fn = None

    def _scale_bbox_to_tensor(self, bbox, src_shape, dst_shape):
        """
        Scale a CutMix bbox from src H/W to dst H/W.

        bbox is timm-style: (yl, yh, xl, xh).
        """
        yl, yh, xl, xh = bbox

        src_h, src_w = src_shape[-2:]
        dst_h, dst_w = dst_shape[-2:]

        yl2 = int(round(float(yl) / float(src_h) * dst_h))
        yh2 = int(round(float(yh) / float(src_h) * dst_h))
        xl2 = int(round(float(xl) / float(src_w) * dst_w))
        xh2 = int(round(float(xh) / float(src_w) * dst_w))

        yl2 = max(0, min(dst_h, yl2))
        yh2 = max(0, min(dst_h, yh2))
        xl2 = max(0, min(dst_w, xl2))
        xh2 = max(0, min(dst_w, xh2))

        return yl2, yh2, xl2, xh2

    def _paired_timm_mixup_cutmix(self, inputs, teacher_inputs, labels):
        """
        Timm-aligned paired Mixup/CutMix.

        Aligns with timm batch mode:
          - uses Mixup._params_per_batch()
          - uses x.flip(0) as the paired source
          - uses cutmix_bbox_and_lam()
          - uses mixup_target()
        """
        if self._paired_mixup_fn is None:
            raise RuntimeError("_paired_mixup_fn is None.")

        assert len(inputs) % 2 == 0, "Batch size should be even when using timm Mixup."

        lam, use_cutmix = self._paired_mixup_fn._params_per_batch()

        mixed_inputs = inputs.clone()
        mixed_teacher_inputs = teacher_inputs.clone()

        if lam != 1.0:
            if use_cutmix:
                # Generate bbox on student input resolution, exactly like timm batch mode.
                bbox_s, lam = cutmix_bbox_and_lam(
                    mixed_inputs.shape,
                    lam,
                    ratio_minmax=self._paired_mixup_fn.cutmix_minmax,
                    correct_lam=self._paired_mixup_fn.correct_lam,
                )
                syl, syh, sxl, sxh = bbox_s

                # Apply corresponding normalized bbox to teacher input resolution.
                tyl, tyh, txl, txh = self._scale_bbox_to_tensor(
                    bbox_s,
                    src_shape=mixed_inputs.shape,
                    dst_shape=mixed_teacher_inputs.shape,
                )

                mixed_inputs[:, :, syl:syh, sxl:sxh] = inputs.flip(0)[:, :, syl:syh, sxl:sxh]
                mixed_teacher_inputs[:, :, tyl:tyh, txl:txh] = teacher_inputs.flip(0)[:, :, tyl:tyh, txl:txh]

            else:
                mixed_inputs = mixed_inputs.mul(lam).add_(inputs.flip(0).mul(1.0 - lam))
                mixed_teacher_inputs = mixed_teacher_inputs.mul(lam).add_(
                    teacher_inputs.flip(0).mul(1.0 - lam)
                )

        labels_for_ce = mixup_target(
            labels,
            self._paired_mixup_fn.num_classes,
            lam,
            self._paired_mixup_fn.label_smoothing,
        )

        return mixed_inputs, mixed_teacher_inputs, labels_for_ce

    def _unpack_train_batch(self, _data):
        teacher_inputs = None

        if self.orig_t_inp and isinstance(_data, (list, tuple)):
            if len(_data) == 3:
                inputs, teacher_inputs, labels = _data
            elif len(_data) == 4:
                inputs, teacher_inputs, labels, _ = _data
            elif len(_data) == 5:
                inputs, teacher_inputs, labels, _, _ = _data
            else:
                raise RuntimeError(
                    f"Unexpected batch format with orig_t_inp=True: len={len(_data)}"
                )

        elif isinstance(_data, (list, tuple)) and len(_data) == 4:
            inputs, labels, _, _ = _data

        elif isinstance(_data, (list, tuple)) and len(_data) == 3:
            inputs, labels, _ = _data

        else:
            inputs, labels = _data

        return inputs, teacher_inputs, labels

    def _student_forward_feature_kd(self, inputs):
        """
        Default: use model(inputs, is_feat=True), and return all features.

        PCNet returns:
            [feat], out

        so this works directly for SRRL/MGD.

        For ReviewKD, modify PCNet to return multiple features:
            [feat1, feat2, feat3, feat4], out
        """
        if self.noisy_model is not None and self.model.training:
            noisy_params = self.noisy_model.gen_noisy_params()
            result = torch.func.functional_call(
                self.model,
                noisy_params,
                (inputs,),
                {"is_feat": True},
            )
        else:
            result = self.model(inputs, is_feat=True)

        if not isinstance(result, (tuple, list)) or len(result) != 2:
            raise RuntimeError(
                "Expected student model to return (features, logits) when is_feat=True."
            )

        features, outputs = result

        if not isinstance(features, (tuple, list)):
            features = [features]

        if len(features) == 0:
            raise RuntimeError("Student returned an empty feature list.")

        return outputs, list(features)

    def _prepare_feature_kd_teacher_inputs(self, inputs):
        if self.img_type == "scanGFI":
            return self._prepare_teacher_inputs(inputs)
        return inputs

    def _teacher_forward_feature_kd(self, inputs):
        """
        Default: final teacher feature only.

        Return:
            teacher_logits
            [teacher_feat]

        ReviewKD can override this to return multiple teacher features.
        """
        if self._teacher_extractor is None:
            raise RuntimeError("Teacher extractor was not initialized.")

        inputs = self._prepare_feature_kd_teacher_inputs(inputs)

        with torch.no_grad():
            teacher_feat = self._teacher_extractor.forward_features(inputs)
            teacher_logits = self._teacher_extractor.forward_logits_from_feature(
                teacher_feat
            )

        return teacher_logits, [teacher_feat]

    def _build_feature_kd_from_one_batch(self):
        was_training = self.model.training

        self.model.eval()
        self.teacher_model.eval()

        _data = next(iter(self.train_dataloader))
        inputs, teacher_inputs, _ = self._unpack_train_batch(_data)

        inputs = inputs.to(self.device)
        if teacher_inputs is not None:
            teacher_inputs = teacher_inputs.to(self.device)

        with torch.no_grad():
            _, student_features = self._student_forward_feature_kd(inputs)
            _, teacher_features = self._teacher_forward_feature_kd(
                teacher_inputs if teacher_inputs is not None else inputs
            )

        self._feature_kd_loss = self._make_feature_kd_loss(
            student_features,
            teacher_features,
        ).to(self.device)

        if was_training:
            self.model.train()
        else:
            self.model.eval()

    def _add_feature_kd_to_optimizer(self):
        if self._feature_kd_loss is None:
            return

        params = list(self._feature_kd_loss.trainable_parameters())
        if not params:
            return

        base_group = self.optimizer.param_groups[0]

        new_group = {
            "params": params,
            "lr": base_group["lr"],
            "weight_decay": base_group.get("weight_decay", 0.0),
        }

        for key in ("betas", "eps", "momentum"):
            if key in base_group:
                new_group[key] = base_group[key]

        self.optimizer.add_param_group(new_group)

    def _kd_loss(self, student_logits, teacher_logits):
        return F.kl_div(
            F.log_softmax(student_logits / self.distill_temperature, dim=1),
            F.softmax(teacher_logits / self.distill_temperature, dim=1),
            reduction="batchmean",
        ) * (self.distill_temperature ** 2)

    # ------------------------------------------------------------------
    # Generic feature-KD train loop
    # ------------------------------------------------------------------
    def train_one_epoch(self, epoch):
        if not self._feature_kd_requested:
            return super().train_one_epoch(epoch)

        self.model.train()
        self.teacher_model.eval()

        if self._teacher_extractor is not None:
            self._teacher_extractor.eval()

        if self._feature_kd_loss is not None:
            self._feature_kd_loss.train()

        running_loss, n_samples = 0.0, 0

        progress_bar = tqdm.tqdm(
            enumerate(self.train_dataloader),
            total=len(self.train_dataloader),
            desc="Training",
        )

        for _i, _data in progress_bar:
            inputs, teacher_inputs, labels = self._unpack_train_batch(_data)

            n_samples += inputs.size(0)

            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            if teacher_inputs is not None:
                teacher_inputs = teacher_inputs.to(self.device)

            labels_for_ce = labels

            if self._paired_mixup_active:
                if teacher_inputs is None:
                    raise RuntimeError(
                        "Paired Mixup/CutMix requires teacher_inputs, but teacher_inputs is None."
                    )

                inputs, teacher_inputs, labels_for_ce = self._paired_timm_mixup_cutmix(
                    inputs,
                    teacher_inputs,
                    labels,
                )
            elif self.mixup_fn is not None:
                # teacher_inputs is available when orig_t_inp=True.
                # In this case, we should use self._paired_timm_mixup_cutmix
                assert teacher_inputs is None
                inputs, labels_for_ce = self.mixup_fn(inputs, labels)

            self.optimizer.zero_grad()

            outputs, student_features = self._student_forward_feature_kd(inputs)

            teacher_logits, teacher_features = self._teacher_forward_feature_kd(
                teacher_inputs if teacher_inputs is not None else inputs
            )

            ce_loss = self.train_loss_fn(outputs, labels_for_ce)

            kd_loss = None
            if self._kd_enabled:
                kd_loss = self._kd_loss(outputs, teacher_logits)
                base_loss = (
                    (1.0 - self.distill_alpha) * ce_loss
                    + self.distill_alpha * kd_loss
                )
            else:
                base_loss = ce_loss

            feature_kd_loss, feature_logs = self._compute_feature_kd_loss(
                student_features=student_features,
                teacher_features=teacher_features,
                teacher_logits=teacher_logits,
            )

            loss = base_loss + self.feature_kd_beta * feature_kd_loss

            loss.backward()

            if self.max_norm is not None:
                grad_params = list(self.model.parameters())
                if self._feature_kd_loss is not None:
                    grad_params += list(self._feature_kd_loss.trainable_parameters())

                nn.utils.clip_grad_norm_(grad_params, max_norm=self.max_norm)

            self.optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            avg_loss = running_loss / n_samples

            postfix = {
                "Iter": f"{_i + 1}/{len(self.train_dataloader)}",
                "Loss": f"{avg_loss:.4f}",
                "CE": f"{ce_loss.item():.4f}",
                self.feature_kd_name.upper(): f"{feature_kd_loss.item():.4f}",
                "LR": self.optimizer.param_groups[0]["lr"],
            }

            if kd_loss is not None:
                postfix["KD"] = f"{kd_loss.item():.4f}"

            for k, v in feature_logs.items():
                if torch.is_tensor(v):
                    postfix[k] = f"{v.item():.4f}"

            if self.noisy_model is not None:
                postfix["Noise"] = f"{self.noisy_model.current_noise_level:.3f}"

            if self.mixup_fn is not None or self._paired_mixup_active:
                postfix["Mix"] = "on"

            progress_bar.set_postfix(postfix)

        running_loss /= n_samples
        return running_loss

    def _save_model_ckpt(self, acc, epoch, suffix=""):
        save_to = os.path.join(self.save_path, self.model_name)
        default_path = os.path.join(str(save_to), self.model_name + suffix)

        save_pth_path = super()._save_model_ckpt(acc, epoch, suffix=suffix)

        if not (self._feature_kd_requested and self._feature_kd_loss is not None):
            return save_pth_path

        feature_kd_state = {
            "name": self.feature_kd_name,
            "beta": self.feature_kd_beta,
            "module_class": self._feature_kd_loss.__class__.__name__,
            "state_dict": self._feature_kd_loss.state_dict(),
        }

        paths_to_update = []
        for path in [default_path, save_pth_path]:
            if path not in paths_to_update and os.path.exists(path):
                paths_to_update.append(path)

        for path in paths_to_update:
            state = torch.load(path, map_location="cpu", weights_only=False)
            state["feature_kd"] = feature_kd_state
            torch.save(state, path)

        return save_pth_path

    def load_feature_kd_from_ckpt(self, ckpt_path, strict=True):
        if self._feature_kd_loss is None:
            raise RuntimeError("Feature-KD module has not been initialized.")

        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)

        if "feature_kd" not in ckpt:
            logging.warning("No feature_kd state found. Training with newly initialized aux module.")
            return

        saved_name = ckpt["feature_kd"].get("name", None)
        if saved_name != self.feature_kd_name:
            raise ValueError(
                f"Feature-KD method mismatch: checkpoint has {saved_name}, "
                f"current trainer has {self.feature_kd_name}."
            )

        self._feature_kd_loss.load_state_dict(
            ckpt["feature_kd"]["state_dict"],
            strict=strict,
        )


class TrainerCiFarTimmStyleSRRL(TrainerCiFarTimmStyleFeatureKD):
    feature_kd_name = "srrl"

    def __init__(
        self,
        *args,
        srrl_beta=1.0,
        srrl_stat_weight=1.0,
        srrl_pred_weight=1.0,
        **kwargs,
    ):
        self.srrl_stat_weight = float(srrl_stat_weight)
        self.srrl_pred_weight = float(srrl_pred_weight)

        super().__init__(
            *args,
            feature_kd_beta=srrl_beta,
            **kwargs,
        )

    def _make_teacher_extractor(self):
        return TeacherFeatureExtractor(self.teacher_model)

    def _make_feature_kd_loss(self, student_features, teacher_features):
        student_feat = student_features[-1]
        teacher_feat = teacher_features[-1]

        if student_feat.dim() != 4:
            raise RuntimeError(
                f"SRRL requires 4D student feature [B,C,H,W], got {tuple(student_feat.shape)}."
            )

        if teacher_feat.dim() != 4:
            raise RuntimeError(
                f"SRRL requires 4D teacher feature [B,C,H,W], got {tuple(teacher_feat.shape)}."
            )

        return SRRLLoss(
            in_channels=student_feat.size(1),
            out_channels=teacher_feat.size(1),
            stat_weight=self.srrl_stat_weight,
            pred_weight=self.srrl_pred_weight,
        )

    def _compute_feature_kd_loss(
        self,
        student_features,
        teacher_features,
        teacher_logits,
    ):
        loss, logs = self._feature_kd_loss(
            feat_student=student_features[-1],
            feat_teacher=teacher_features[-1],
            teacher_logits=teacher_logits,
            teacher_head=self._teacher_extractor.classify_feature,
            return_dict=True,
        )

        return loss, {
            "Stat": logs["srrl_stat_loss"],
            "Pred": logs["srrl_pred_loss"],
        }


class TrainerCiFarTimmStyleMGD(TrainerCiFarTimmStyleFeatureKD):
    feature_kd_name = "mgd"

    def __init__(
        self,
        *args,
        mgd_alpha=7e-5,
        mgd_lambda=0.5,
        mgd_mask_mode="channel",
        **kwargs,
    ):
        self.mgd_alpha = float(mgd_alpha)
        self.mgd_lambda = float(mgd_lambda)
        self.mgd_mask_mode = mgd_mask_mode

        # MGDLoss already applies alpha_mgd internally, following the official code.
        # So keep the generic outer feature_kd_beta at 1.0.
        super().__init__(
            *args,
            feature_kd_beta=1.0,
            **kwargs,
        )

    def _make_feature_kd_loss(self, student_features, teacher_features):
        student_feat = student_features[-1]
        teacher_feat = teacher_features[-1]

        if student_feat.dim() != 4:
            raise RuntimeError(
                f"MGD requires 4D student feature [B,C,H,W], got {tuple(student_feat.shape)}."
            )

        if teacher_feat.dim() != 4:
            raise RuntimeError(
                f"MGD requires 4D teacher feature [B,C,H,W], got {tuple(teacher_feat.shape)}."
            )

        return MGDLoss(
            student_channels=student_feat.size(1),
            teacher_channels=teacher_feat.size(1),
            alpha_mgd=self.mgd_alpha,
            lambda_mgd=self.mgd_lambda,
            mask_mode=self.mgd_mask_mode,
        )

    def _compute_feature_kd_loss(
        self,
        student_features,
        teacher_features,
        teacher_logits,
    ):
        loss, logs = self._feature_kd_loss(
            feat_student=student_features[-1],
            feat_teacher=teacher_features[-1],
            return_dict=True,
        )

        return loss, {
            "MGDraw": logs["mgd_raw_loss"],
        }


class TrainerCiFarTimmStyleReviewKD(TrainerCiFarTimmStyleFeatureKD):
    pass

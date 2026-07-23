import tqdm
import inspect
import logging
import os
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
import timm

from timm.data import Mixup, create_transform, resolve_model_data_config
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.loss import SoftTargetCrossEntropy, LabelSmoothingCrossEntropy

from trainer import TrainerCiFar, WrappedNoisyModel
from distillation import CRDOptions

from trainer_timm import (
    TrainerCiFarTimmStyle,
    TrainerCiFarTimmStyleFeatureKD,
    TrainerCiFarTimmStyleSRRL,
    TrainerCiFarTimmStyleMGD,
)


class TrainerImageNetTimmStyle(TrainerCiFarTimmStyle):
    """
    ImageNet-1K / ILSVRC2012 timm-style trainer.

    Expected dataset layout:

        imagenet_root/
            train/
                n01440764/*.JPEG
                n01443537/*.JPEG
                ...
            val/
                n01440764/*.JPEG
                n01443537/*.JPEG
                ...

    This class intentionally does NOT call TrainerCiFarTimmStyle.__init__(),
    because that constructor is CIFAR-specific.

    Reused from TrainerCiFarTimmStyle:
      - _get_optimizer()
      - _build_timm_scheduler()
      - _build_rgb_timm_transforms()
      - _student_forward()
      - _teacher_forward()
      - train_one_epoch()
    """

    def __init__(
        self,
        *args,

        # ImageNet-specific
        imagenet_root=None,
        data_root=None,
        num_classes=1000,

        # ImageNet-style defaults different from CIFAR timm trainer
        use_model_data_config=True,
        timm_train_scale=(0.08, 1.0),
        timm_train_ratio=(3.0 / 4.0, 4.0 / 3.0),
        color_jitter=0.4,
        num_workers=8,
        persistent_workers=True,

        # AMP / gradient accumulation
        amp_enabled=False,
        amp_dtype="bf16",
        grad_accum_steps=1,

        save_every=10,

        **kwargs,
    ):
        if imagenet_root is None:
            imagenet_root = data_root
        if imagenet_root is None:
            raise ValueError("Please provide imagenet_root or data_root.")

        # ------------------------------------------------------------
        # Pop timm-style args that TrainerCiFar.__init__ does not know.
        # Defaults mostly match TrainerCiFarTimmStyle, except ImageNet defaults.
        # ------------------------------------------------------------
        self.timm_model_name = kwargs.pop("timm_model_name", None)
        self.pretrained = kwargs.pop("pretrained", False)
        self.in_chans = kwargs.pop("in_chans", 3)
        self.is_timm_model = kwargs.pop("is_timm_model", True)

        self.use_model_data_config = use_model_data_config
        self.timm_input_size = kwargs.pop("timm_input_size", None)
        self.timm_mean = kwargs.pop("timm_mean", None)
        self.timm_std = kwargs.pop("timm_std", None)
        self.interpolation = kwargs.pop("interpolation", None)

        self.timm_opt = kwargs.pop("timm_opt", "adamw")
        self.opt_eps = kwargs.pop("opt_eps", None)
        self.opt_betas = kwargs.pop("opt_betas", None)
        self.momentum = kwargs.pop("momentum", 0.9)

        self.timm_sched = kwargs.pop("timm_sched", "cosine")
        self.min_lr = kwargs.pop("min_lr", 1e-6)
        self.warmup_lr = kwargs.pop("warmup_lr", 1e-6)
        self.decay_rate = kwargs.pop("decay_rate", 0.1)

        self.timm_aug = kwargs.pop("timm_aug", True)
        self.timm_train_scale = timm_train_scale
        self.timm_train_ratio = timm_train_ratio
        self.hflip = kwargs.pop("hflip", 0.5)
        self.vflip = kwargs.pop("vflip", 0.0)
        self.color_jitter = color_jitter
        self.auto_augment = kwargs.pop("auto_augment", "rand-m9-mstd0.5-inc1")

        self.re_prob = kwargs.pop("re_prob", 0.25)
        self.re_mode = kwargs.pop("re_mode", "pixel")
        self.re_count = kwargs.pop("re_count", 1)

        self.label_smoothing = kwargs.pop("label_smoothing", 0.1)
        self.mixup_alpha = kwargs.pop("mixup_alpha", 0.2)
        self.cutmix_alpha = kwargs.pop("cutmix_alpha", 1.0)
        self.cutmix_minmax = kwargs.pop("cutmix_minmax", None)
        self.mixup_prob = kwargs.pop("mixup_prob", 1.0)
        self.switch_prob = kwargs.pop("switch_prob", 0.5)
        self.mixup_mode = kwargs.pop("mixup_mode", "batch")

        self.num_workers = num_workers
        self.pin_memory = kwargs.pop("pin_memory", True)
        self.persistent_workers = persistent_workers

        self.amp_enabled = bool(amp_enabled)
        self.grad_accum_steps = max(1, int(grad_accum_steps))

        self.save_every = save_every

        amp_dtype = str(amp_dtype).lower()
        if amp_dtype in {"bf16", "bfloat16"}:
            self.amp_dtype = torch.bfloat16
        elif amp_dtype in {"fp16", "float16"}:
            self.amp_dtype = torch.float16
        else:
            raise ValueError(f"Unknown amp_dtype: {amp_dtype}. Use 'bf16' or 'fp16'.")

        self.grad_scaler = None

        # Compatibility attrs used by inherited helper methods.
        # ImageNet trainer only supports rgb, but these attributes prevent
        # inherited methods from failing if they inspect them.
        self.convert_non_rgb_to_rgb = kwargs.pop("convert_non_rgb_to_rgb", False)
        self.non_rgb_spatial_aug = kwargs.pop("non_rgb_spatial_aug", False)
        self.non_rgb_crop_padding = kwargs.pop("non_rgb_crop_padding", 0)
        self.non_rgb_affine_degrees = kwargs.pop("non_rgb_affine_degrees", 0)
        self.non_rgb_affine_translate = kwargs.pop("non_rgb_affine_translate", None)
        self.non_rgb_affine_shear = kwargs.pop("non_rgb_affine_shear", None)

        self.imagenet_root = Path(imagenet_root).expanduser()
        self.num_classes = int(num_classes)

        # ------------------------------------------------------------
        # Bind remaining args using TrainerCiFar.__init__ signature,
        # but do NOT call TrainerCiFar.__init__.
        # ------------------------------------------------------------
        kwargs.setdefault("dataset_name", "imagenet")
        kwargs.setdefault("img_type", "rgb")
        kwargs.setdefault("distill_method", "none")

        sig = inspect.signature(TrainerCiFar.__init__)
        bound = sig.bind_partial(None, *args, **kwargs)
        bound.apply_defaults()
        cfg = {k: v for k, v in bound.arguments.items() if k != "self"}

        dataset_name_norm = str(cfg["dataset_name"]).lower().replace("-", "").replace("_", "")
        if dataset_name_norm not in {
            "imagenet",
            "imagenet1k",
            "ilsvrc",
            "ilsvrc2012",
            "ilsvrc12",
        }:
            raise ValueError(
                f"TrainerImageNetTimmStyle only supports ImageNet, got {cfg['dataset_name']}."
            )

        if cfg["img_type"] != "rgb":
            raise ValueError("TrainerImageNetTimmStyle currently supports only img_type='rgb'.")

        distill_method = (cfg["distill_method"] or "none").lower().replace("+", "_")
        if "crd" in distill_method:
            raise ValueError(
                "TrainerImageNetTimmStyle does not support CRD. "
                "Use distill_method='none' or 'kd'. "
                "For feature KD, use TrainerImageNetTimmStyleSRRL or "
                "TrainerImageNetTimmStyleMGD."
            )
        cfg["distill_method"] = distill_method

        if self.timm_model_name is not None:
            cfg["model"] = timm.create_model(
                self.timm_model_name,
                pretrained=self.pretrained,
                num_classes=self.num_classes,
                in_chans=self.in_chans,
            )

        if cfg.get("model", None) is None:
            raise ValueError("Either pass model=... or timm_model_name=...")

        if cfg.get("model_name", None) is None:
            raise ValueError("model_name must be provided.")

        if cfg.get("save_path", None) is None:
            raise ValueError("save_path must be provided.")

        # ------------------------------------------------------------
        # Same state setup as TrainerCiFar, but with real ImageNet dataset.
        # ------------------------------------------------------------
        self.skip_eval_epochs = cfg["skip_eval_epochs"]
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("----- Using {} device -----".format(self.device))

        if self.device.type != "cuda":
            self.amp_enabled = False
        self.grad_scaler = torch.amp.GradScaler(
            "cuda",
            enabled=(self.amp_enabled and self.amp_dtype is torch.float16),
        )

        self.model = cfg["model"].to(self.device)
        self.model_name = cfg["model_name"]
        self.save_path = cfg["save_path"]

        self.loss_fn = cfg["loss_fn"]
        self.batch_size = cfg["batch_size"]
        self.num_epochs = cfg["num_epochs"]
        self.warmup_epoch = cfg["warmup_epoch"]
        self.test_batch_size = cfg["test_bs"]
        self.max_norm = cfg["max_norm"]
        self.aug = cfg["aug"]
        self.eval_every = cfg["eval_every"]
        self.img_type = "rgb"
        self.dataset_name = "imagenet"

        self.lr_reduce_on = cfg["lr_reduce_on"]
        self.T_0 = cfg["T0"]

        self.distill_alpha = cfg["distill_alpha"]
        self.distill_temperature = cfg["distill_temperature"]
        self.distill_method = cfg["distill_method"]

        self.scangen_noise_config = None
        self.scangen_noise_root = None
        self.teacher_eval_loader = None
        self.teacher_model = None
        self.teacher_input_size = cfg["teacher_input_size"]
        self.teacher_center_crop = cfg["teacher_center_crop"]

        if cfg["teacher_model"] is not None:
            self.teacher_model = cfg["teacher_model"].to(self.device)
            self.teacher_model.eval()
            for param in self.teacher_model.parameters():
                param.requires_grad_(False)

        self._kd_enabled = (
            self.teacher_model is not None
            and self.distill_alpha > 0.0
            and "kd" in self.distill_method
        )

        # CRD is intentionally disabled in this ImageNet timm trainer.
        self._crd_enabled = False
        self._crd_loss = None
        self.crd_beta = cfg["crd_beta"]
        self.neg_sample = cfg["neg_sample"]
        self.orig_t_inp = cfg["orig_t_inp"]

        # Kept for compatibility with inherited methods / state names.
        self._crd_options = CRDOptions(
            contrast_method=cfg["contrast_method"],
            feat_dim=cfg["crd_feat_dim"],
            nce_k=cfg["crd_k"],
            nce_t=cfg["crd_temperature"],
            nce_m=cfg["crd_momentum"],
        )

        self._student_feature_module = self._find_last_linear(self.model)
        self._teacher_feature_module = (
            self._find_last_linear(self.teacher_model)
            if self.teacher_model is not None
            else None
        )

        self._train_sample_count = 0
        self._crd_initialized = False

        self.mixup_fn = None
        self.train_loss_fn = None

        # ------------------------------------------------------------
        # Optimizer
        # Uses inherited TrainerCiFarTimmStyle._get_optimizer().
        # ------------------------------------------------------------
        self.optimizer = self._get_optimizer(
            cfg["optim_type"],
            lr=cfg["learning_rate"],
            weight_decay=cfg["weight_decay"],
        )

        # ------------------------------------------------------------
        # Mismatch-aware training, same logic as TrainerCiFar.
        # ------------------------------------------------------------
        self.noisy_model = None
        self.noise_schedule = None
        effective_noise_type = cfg["noise_type"] or "mul"

        if cfg["mismatch_levels"] is not None or cfg["noise_level"] is not None:
            levels = []

            if cfg["mismatch_levels"]:
                levels.extend(float(lvl) for lvl in cfg["mismatch_levels"])

            if cfg["noise_level"] is not None:
                levels.append(float(cfg["noise_level"]))

            levels = [lvl for lvl in levels if lvl >= 0.0]

            if levels:
                self.noise_schedule = list(dict.fromkeys(levels))
                self.noisy_model = WrappedNoisyModel(
                    model=self.model,
                    noise_levels=self.noise_schedule,
                    noise_type=effective_noise_type,
                )
                logging.warning(
                    "Mismatch-aware training enabled with noise levels %s (%s noise).",
                    self.noise_schedule,
                    effective_noise_type,
                )

        if self.noisy_model is None:
            logging.warning(
                "Mismatch-aware training disabled; proceeding without injected mismatch noise."
            )

        # ------------------------------------------------------------
        # ImageNet data, timm scheduler, timm loss/mixup.
        # ------------------------------------------------------------
        self._prepare_imagenet()

        self.warmup_scheduler = None
        self._build_timm_scheduler()
        self._build_timm_loss_and_mixup()

    # ------------------------------------------------------------------
    # ImageNet data config
    # ------------------------------------------------------------------
    def _get_timm_data_config(self, dataset_name=None):
        """
        Resolve input_size / mean / std / interpolation.

        Priority:
          1. explicit manual args:
                timm_input_size, timm_mean, timm_std, interpolation
          2. if use_model_data_config=True:
                timm model pretrained data config
          3. ImageNet defaults from timm constants
        """
        model_cfg = {}
        if self.use_model_data_config:
            try:
                model_cfg = resolve_model_data_config(self.model)
            except Exception:
                model_cfg = {}

        input_size = (
            self.timm_input_size
            if self.timm_input_size is not None
            else model_cfg.get("input_size", (3, 224, 224))
        )

        mean = (
            self.timm_mean
            if self.timm_mean is not None
            else model_cfg.get("mean", IMAGENET_DEFAULT_MEAN)
        )

        std = (
            self.timm_std
            if self.timm_std is not None
            else model_cfg.get("std", IMAGENET_DEFAULT_STD)
        )

        interpolation = (
            self.interpolation
            if self.interpolation is not None
            else model_cfg.get("interpolation", "bicubic")
        )

        return input_size, mean, std, interpolation

    def _infer_timm_input_size(self, img_type):
        input_size, _, _, _ = self._get_timm_data_config("imagenet")
        return input_size

    # ------------------------------------------------------------------
    # ImageNet dataset
    # ------------------------------------------------------------------
    def _prepare_imagenet(self):
        train_dir = self.imagenet_root / "train"
        val_dir = self.imagenet_root / "val"

        if not train_dir.exists():
            raise FileNotFoundError(f"ImageNet train dir not found: {train_dir}")

        if not val_dir.exists():
            raise FileNotFoundError(f"ImageNet val dir not found: {val_dir}")

        val_class_dirs = [p for p in val_dir.iterdir() if p.is_dir()]
        if len(val_class_dirs) == 0:
            raise RuntimeError(
                f"{val_dir} appears to be flat. Reorganize val into class folders first, "
                "e.g. val/n01440764/*.JPEG."
            )

        if self.timm_aug:
            transform_train, transform_val = self._build_rgb_timm_transforms("imagenet")
        else:
            input_size, mean, std, interpolation = self._get_timm_data_config("imagenet")

            transform_train = create_transform(
                input_size=input_size,
                is_training=True,
                use_prefetcher=False,
                no_aug=True,
                interpolation=interpolation,
                mean=mean,
                std=std,
            )

            transform_val = create_transform(
                input_size=input_size,
                is_training=False,
                use_prefetcher=False,
                interpolation=interpolation,
                mean=mean,
                std=std,
            )

        self.train_set = torchvision.datasets.ImageFolder(
            root=str(train_dir),
            transform=transform_train,
        )

        self.val_set = torchvision.datasets.ImageFolder(
            root=str(val_dir),
            transform=transform_val,
        )

        if len(self.train_set.classes) != self.num_classes:
            logging.warning(
                "Expected %d ImageNet train classes, but found %d in %s.",
                self.num_classes,
                len(self.train_set.classes),
                train_dir,
            )

        if len(self.val_set.classes) != self.num_classes:
            logging.warning(
                "Expected %d ImageNet val classes, but found %d in %s.",
                self.num_classes,
                len(self.val_set.classes),
                val_dir,
            )

        self._train_sample_count = len(self.train_set)

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
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
            drop_last=False,
        )

        if self.teacher_eval_loader is None:
            self.teacher_eval_loader = self.val_dataloader

    # ------------------------------------------------------------------
    # Loss / mixup
    # ------------------------------------------------------------------
    def _build_timm_loss_and_mixup(self):
        """
        Same structure as TrainerCiFarTimmStyle._build_timm_loss_and_mixup(),
        but uses self.num_classes instead of CIFAR hard-coded 10/100.
        """
        mixup_active = (
            self.mixup_alpha > 0.0
            or self.cutmix_alpha > 0.0
            or self.cutmix_minmax is not None
        )

        if mixup_active and self._kd_enabled and self.orig_t_inp:
            logging.warning(
                "Disabling mixup/cutmix because KD with orig_t_inp=True uses "
                "separate teacher inputs. For ImageNet, prefer orig_t_inp=False."
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
                num_classes=self.num_classes,
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

    def _amp_autocast(self):
        return torch.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=(self.amp_enabled and self.device.type == "cuda"),
        )

    def _backward_step_accum(
            self,
            loss,
            grad_params,
            is_update_step,
            accum_divisor=None,
    ):
        if accum_divisor is None:
            accum_divisor = self.grad_accum_steps

        loss_for_backward = loss / accum_divisor

        if self.grad_scaler is not None and self.grad_scaler.is_enabled():
            self.grad_scaler.scale(loss_for_backward).backward()
        else:
            loss_for_backward.backward()

        if not is_update_step:
            return False

        if self.max_norm is not None:
            if self.grad_scaler is not None and self.grad_scaler.is_enabled():
                self.grad_scaler.unscale_(self.optimizer)

            nn.utils.clip_grad_norm_(
                grad_params,
                max_norm=self.max_norm,
            )

        if self.grad_scaler is not None and self.grad_scaler.is_enabled():
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            self.optimizer.step()

        self.optimizer.zero_grad(set_to_none=True)

        return True

    def train_one_epoch(self, epoch):
        self.model.train()
        running_loss, n_samples = 0.0, 0

        num_batches = len(self.train_dataloader)

        progress_bar = tqdm.tqdm(
            enumerate(self.train_dataloader),
            total=num_batches,
            desc="Training",
        )

        self.optimizer.zero_grad(set_to_none=True)

        last_accum_steps = num_batches % self.grad_accum_steps
        if last_accum_steps == 0:
            last_accum_steps = self.grad_accum_steps

        for _i, _data in progress_bar:
            if isinstance(_data, (list, tuple)):
                inputs, labels = _data[0], _data[1]
            else:
                inputs, labels = _data

            n_samples += inputs.size(0)

            inputs = inputs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            labels_for_ce = labels

            if self.mixup_fn is not None:
                inputs, labels_for_ce = self.mixup_fn(inputs, labels)

            with self._amp_autocast():
                outputs, _ = self._student_forward(inputs)
                loss = self.train_loss_fn(outputs, labels_for_ce)

            is_update_step = (
                    ((_i + 1) % self.grad_accum_steps == 0)
                    or ((_i + 1) == num_batches)
            )

            is_last_accum_window = _i >= (num_batches - last_accum_steps)
            accum_divisor = last_accum_steps if is_last_accum_window else self.grad_accum_steps

            did_update = self._backward_step_accum(
                loss=loss,
                grad_params=list(self.model.parameters()),
                is_update_step=is_update_step,
                accum_divisor=accum_divisor,
            )

            running_loss += loss.detach().item() * inputs.size(0)
            avg_loss = running_loss / n_samples

            postfix = {
                "Iter": f"{_i + 1}/{num_batches}",
                "Loss": f"{avg_loss:.4f}",
                "LR": self.optimizer.param_groups[0]["lr"],
            }

            if self.amp_enabled:
                postfix["AMP"] = "bf16" if self.amp_dtype is torch.bfloat16 else "fp16"

            if self.grad_accum_steps > 1:
                postfix["Accum"] = str(self.grad_accum_steps)

            if did_update:
                postfix["Step"] = "yes"

            progress_bar.set_postfix(postfix)

        running_loss /= n_samples
        return running_loss

    def evaluate(self, dataloader):
        correct1 = 0
        correct5 = 0
        total = 0

        pred_list, label_list = [], []

        self.model.eval()
        self.reset_spin_variation_for_inference()
        with torch.no_grad():
            for data in dataloader:
                inputs, labels = data[0], data[1]

                inputs = inputs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                outputs = self.model(inputs)

                max_k = min(5, outputs.size(1))
                _, pred = outputs.topk(max_k, dim=1, largest=True, sorted=True)

                total += labels.size(0)
                correct1 += pred[:, 0].eq(labels).sum().item()
                correct5 += pred.eq(labels.view(-1, 1)).any(dim=1).sum().item()

                pred_list.append(pred[:, 0].detach().cpu())
                label_list.append(labels.detach().cpu())

        top1 = correct1 / total
        top5 = correct5 / total

        return top1, top5, torch.cat(label_list), torch.cat(pred_list)

    def train(self):
        train_loss_list, val_acc_list = [], []
        best_acc, val_acc, best_epoch = 0.0, 0.0, 0
        best_top5 = None
        best_model_path = None

        for epoch in range(self.num_epochs):
            print("Training epoch {} / {}".format(epoch, self.num_epochs))

            train_loss = self.train_one_epoch(epoch)

            if (epoch + 1) % self.eval_every == 0 and epoch >= self.skip_eval_epochs:
                train_acc, train_top5, _, _ = self.evaluate(self.train_dataloader)
                val_acc, val_top5, _, _ = self.evaluate(self.val_dataloader)

                train_loss_list.append(train_loss)
                val_acc_list.append(val_acc)

                print(
                    "Validation top1: {}, top5: {}; Train top1: {}, top5: {}".format(
                        val_acc,
                        val_top5,
                        train_acc,
                        train_top5,
                    )
                )

                if val_acc > best_acc:
                    best_acc = val_acc
                    best_epoch = epoch + 1
                    best_top5 = val_top5
                    best_model_path = self._save_model_ckpt(
                        val_acc,
                        epoch + 1,
                        "_best_ckpt.pth",
                    )

            if (epoch + 1) % self.save_every == 0:
                # note: This is for recovery purpose only. val_acc might still be 0.
                _ = self._save_model_ckpt(val_acc, epoch + 1, "_last_ckpt.pth")

            self.scheduler.step(epoch + 1)

        _ = self._save_model_ckpt(val_acc, self.num_epochs, "_last_ckpt.pth")

        print("----- Train finished, Model Name: {} -----".format(self.model_name))
        print(
            "----- Total number of parameters: {} M -----".format(
                sum(p.numel() for p in self.model.parameters()) / 1e6
            )
        )
        print(
            "----- Best top1: {}, Best top5: {}, Best epoch: {} -----".format(
                best_acc,
                best_top5,
                best_epoch,
            )
        )
        print("----- Model path: {} -----".format(best_model_path))
        print("--------------------------------------------------------------------------")

        return train_loss_list, val_acc_list

# ----------------------------------------------------------------------
# ImageNet feature-KD variants
# ----------------------------------------------------------------------
# These reuse the existing feature-KD logic. The MRO makes
# TrainerCiFarTimmStyleFeatureKD.__init__ call TrainerImageNetTimmStyle.__init__.
# ----------------------------------------------------------------------
class TrainerImageNetTimmStyleFeatureKD(
    TrainerCiFarTimmStyleFeatureKD,
    TrainerImageNetTimmStyle,
):
    pass


class TrainerImageNetTimmStyleSRRL(
    TrainerCiFarTimmStyleSRRL,
    TrainerImageNetTimmStyle,
):
    pass


class TrainerImageNetTimmStyleMGD(
    TrainerCiFarTimmStyleMGD,
    TrainerImageNetTimmStyle,
):
    pass
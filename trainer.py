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

from pc_model import PCNet
from data_utils import ToPackedRGGB, RawImgDataset, load_and_register_buffer, get_parametrized_weight_mods
from scangen.data import NoiseCIFARDataset, MyNoiseCIFARDataset
from distillation import CRDLoss, CRDOptions


class DatasetWithIndex(torch.utils.data.Dataset):
    """Wrap a dataset to also return sample indices."""

    def __init__(self, dataset: torch.utils.data.Dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        if isinstance(data, tuple):
            return (*data, idx)
        return data, idx


class DatasetWithIndexAndContrast(torch.utils.data.Dataset):
    """Wrap a dataset to also return sample indices and contrast indices."""

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        k: int = 4096,
        mode: str = "exact",
        percent: float = 1.0,
    ):
        self.dataset = dataset
        self.k = k
        self.mode = mode

        labels = self._extract_labels(dataset)
        self.labels = torch.as_tensor(labels, dtype=torch.long)

        num_samples = self.labels.numel()
        num_classes = int(self.labels.max().item()) + 1

        self.cls_positive = []
        for c in range(num_classes):
            pos_idx = torch.nonzero(self.labels == c, as_tuple=False).squeeze(1)
            self.cls_positive.append(pos_idx)

        self.cls_negative = []
        for c in range(num_classes):
            neg_idx = torch.nonzero(self.labels != c, as_tuple=False).squeeze(1)
            if 0 < percent < 1:
                n = int(neg_idx.numel() * percent)
                perm = torch.randperm(neg_idx.numel())[:n]
                neg_idx = neg_idx[perm]
            self.cls_negative.append(neg_idx)

    def _extract_labels(self, ds):
        if hasattr(ds, "targets"):
            return ds.targets
        if hasattr(ds, "train_labels"):
            return ds.train_labels
        if hasattr(ds, "labels"):
            return ds.labels

        if hasattr(ds, "dataset"):
            return self._extract_labels(ds.dataset)

        if hasattr(ds, "_clean_file") and hasattr(ds, "indices"):
            labels = torch.as_tensor(ds._clean_file["labels"][:], dtype=torch.long)
            indices = torch.as_tensor(ds.indices, dtype=torch.long)
            return labels.index_select(0, indices)

        raise AttributeError("Could not extract labels from dataset for contrast sampling.")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        if not isinstance(data, tuple) or len(data) < 2:
            raise ValueError("Base dataset must return at least (input, label).")

        if len(data) == 2:
            inp, target = data
        else:
            inp, target = data[:2]

        target_int = int(target)

        if self.mode == "exact":
            pos_idx = torch.tensor([idx], dtype=torch.long)
        elif self.mode == "relax":
            pos_pool = self.cls_positive[target_int]
            rand_i = torch.randint(0, pos_pool.numel(), (1,))
            pos_idx = pos_pool.index_select(0, rand_i)
        else:
            raise NotImplementedError(self.mode)

        neg_pool = self.cls_negative[target_int]
        replace = self.k > neg_pool.numel()

        if replace:
            rand_i = torch.randint(0, neg_pool.numel(), (self.k,))
            neg_idx = neg_pool.index_select(0, rand_i)
        else:
            perm = torch.randperm(neg_pool.numel())[:self.k]
            neg_idx = neg_pool.index_select(0, perm)

        sample_idx = torch.cat((pos_idx, neg_idx), dim=0)

        return inp, target, idx, sample_idx


class StudentTeacherPairDataset(torch.utils.data.Dataset):
    """
    Pair low-res student input with original high-res CIFAR teacher input.
    Assumes the sample orders of the two datasets are aligned.
    """

    def __init__(self, student_dataset: torch.utils.data.Dataset, teacher_dataset: torch.utils.data.Dataset):
        self.student_dataset = student_dataset
        self.teacher_dataset = teacher_dataset

        if len(self.student_dataset) != len(self.teacher_dataset):
            raise ValueError(
                f"Dataset length mismatch: student={len(self.student_dataset)}, teacher={len(self.teacher_dataset)}"
            )

        self.labels = self._extract_labels(self.teacher_dataset)

    def _extract_labels(self, ds):
        if hasattr(ds, "targets"):
            return torch.as_tensor(ds.targets, dtype=torch.long)
        if hasattr(ds, "train_labels"):
            return torch.as_tensor(ds.train_labels, dtype=torch.long)
        if hasattr(ds, "labels"):
            return torch.as_tensor(ds.labels, dtype=torch.long)
        raise AttributeError("Could not extract labels from teacher dataset.")

    def __len__(self):
        return len(self.student_dataset)

    def __getitem__(self, idx):
        student_data = self.student_dataset[idx]
        teacher_data = self.teacher_dataset[idx]

        if not isinstance(student_data, tuple) or len(student_data) < 2:
            raise ValueError("student_dataset must return at least (input, label)")
        if not isinstance(teacher_data, tuple) or len(teacher_data) < 2:
            raise ValueError("teacher_dataset must return at least (input, label)")

        student_input, student_label = student_data[:2]
        teacher_input, teacher_label = teacher_data[:2]

        if int(student_label) != int(teacher_label):
            raise RuntimeError(
                f"Label mismatch at idx={idx}: student={int(student_label)}, teacher={int(teacher_label)}"
            )

        return student_input, teacher_input, student_label


class PairDatasetWithIndex(torch.utils.data.Dataset):
    """Wrap paired dataset to also return index."""

    def __init__(self, dataset: torch.utils.data.Dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        inputs, teacher_inputs, labels = self.dataset[idx]
        return inputs, teacher_inputs, labels, idx


class PairDatasetWithIndexAndContrast(torch.utils.data.Dataset):
    """Wrap paired dataset to also return index and contrast_idx."""

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        k: int = 4096,
        mode: str = "exact",
        percent: float = 1.0,
    ):
        self.dataset = dataset
        self.k = k
        self.mode = mode

        self.labels = torch.as_tensor(dataset.labels, dtype=torch.long)

        num_classes = int(self.labels.max().item()) + 1

        self.cls_positive = []
        for c in range(num_classes):
            pos_idx = torch.nonzero(self.labels == c, as_tuple=False).squeeze(1)
            self.cls_positive.append(pos_idx)

        self.cls_negative = []
        for c in range(num_classes):
            neg_idx = torch.nonzero(self.labels != c, as_tuple=False).squeeze(1)
            if 0 < percent < 1:
                n = int(neg_idx.numel() * percent)
                perm = torch.randperm(neg_idx.numel())[:n]
                neg_idx = neg_idx[perm]
            self.cls_negative.append(neg_idx)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        inputs, teacher_inputs, labels = self.dataset[idx]
        target_int = int(labels)

        if self.mode == "exact":
            pos_idx = torch.tensor([idx], dtype=torch.long)
        elif self.mode == "relax":
            pos_pool = self.cls_positive[target_int]
            rand_i = torch.randint(0, pos_pool.numel(), (1,))
            pos_idx = pos_pool.index_select(0, rand_i)
        else:
            raise NotImplementedError(self.mode)

        neg_pool = self.cls_negative[target_int]
        replace = self.k > neg_pool.numel()

        if replace:
            rand_i = torch.randint(0, neg_pool.numel(), (self.k,))
            neg_idx = neg_pool.index_select(0, rand_i)
        else:
            perm = torch.randperm(neg_pool.numel())[:self.k]
            neg_idx = neg_pool.index_select(0, perm)

        sample_idx = torch.cat((pos_idx, neg_idx), dim=0)

        return inputs, teacher_inputs, labels, idx, sample_idx


_CIFAR_STATS = {
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "cifar100": ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
}


def _normalize_dataset_name(dataset_name=None) -> str:
    name = (dataset_name or "cifar10").lower().replace("-", "").replace("_", "")
    if name in _CIFAR_STATS:
        return name
    raise ValueError(f"Unsupported dataset: {dataset_name}. Use cifar10 or cifar100.")


def _default_cifar_dir(dataset_name: str) -> str:
    name = _normalize_dataset_name(dataset_name)
    return "cifar-10-data"
    # return "cifar-10-data" if name == "cifar10" else "cifar-100-data"


def _resolve_cifar_data_root(img_type: str, input_name: str) -> Path:
    dataset_name = _normalize_dataset_name(input_name)
    hdf5_name = f"{dataset_name}_raw.h5"
    env_root = os.getenv("SCANGEN_DATA_ROOT")
    if env_root:
        env_root_path = Path(env_root).expanduser()
        if env_root_path.is_file():
            if env_root_path.name == hdf5_name:
                return env_root_path.parent
            env_root_path = env_root_path.parent
        candidates = [
            env_root_path,
            env_root_path / img_type,
            env_root_path / _default_cifar_dir(dataset_name) / img_type,
        ]
        for candidate in candidates:
            if (candidate / hdf5_name).exists():
                return candidate
        logging.warning(
            "SCANGEN_DATA_ROOT=%s does not contain %s; falling back to project-relative path.",
            env_root,
            hdf5_name,
        )
    return Path(__file__).resolve().parent.parent / _default_cifar_dir(dataset_name) / img_type


class TrainerCiFar(object):
    def __init__(self, model, model_name, save_path,
                 batch_size=512, optim_type="Adam", weight_decay=1e-3,
                 loss_fn=nn.CrossEntropyLoss(), skip_eval_epochs=0,
                 learning_rate=0.01, num_epochs=300, warmup_epoch=1,
                 lr_reduce_on="80,122,150,225,262", test_bs=512, max_norm=None, aug=False, T0=None,
                 eval_every=1, img_type="rgb", dataset_name="cifar10", noise_level=None, noise_type=None,
                 mismatch_levels=None,
                 contrast_method="memory", neg_sample="label", orig_t_inp=False,
                 distill_alpha=0.0, distill_temperature=1.0, teacher_model=None,
                 distill_method="kd", crd_feat_dim=128, crd_k=16384,
                 crd_temperature=0.07, crd_momentum=0.5, crd_beta=0.8,
                 teacher_input_size=224, teacher_center_crop=True):
        self.skip_eval_epochs = skip_eval_epochs
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('----- Using {} device -----'.format(self.device))

        model = model.to(self.device)
        self.model = model
        self.model_name = model_name
        self.save_path = save_path
        self.optimizer = self._get_optimizer(optim_type, lr=learning_rate, weight_decay=weight_decay)
        # Reuse the LR schedule epoch as before
        if T0 is not None:
            # With restart seems to be better.
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=self.optimizer,
                                                                            T_0=T0, T_mult=2)
            # self.scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer, T_max=num_epochs)
        else:
            self.scheduler = optim.lr_scheduler.MultiStepLR(optimizer=self.optimizer,
                                                            milestones=list(map(int, lr_reduce_on.split(","))))
        self.loss_fn = loss_fn
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.warmup_epoch = warmup_epoch
        self.test_batch_size = test_bs
        self.max_norm = max_norm
        self.aug = aug # use the augmentation in convMixer or not
        self.eval_every = eval_every
        self.img_type = img_type
        self.dataset_name = _normalize_dataset_name(dataset_name)
        self.distill_alpha = distill_alpha
        self.distill_temperature = distill_temperature
        self.distill_method = (distill_method or "none").lower()
        if self.distill_method not in {"none", "kd", "crd", "kd_crd", "kd+crd"}:
            raise ValueError(f"Unsupported distillation method: {self.distill_method}")
        if self.distill_method == "kd+crd":
            self.distill_method = "kd_crd"
        self.scangen_noise_config = None
        self.scangen_noise_root = None
        self.teacher_eval_loader = None
        self.teacher_model = None
        self.teacher_input_size = teacher_input_size
        self.teacher_center_crop = teacher_center_crop
        if teacher_model is not None:
            self.teacher_model = teacher_model.to(self.device)
            self.teacher_model.eval()
            for param in self.teacher_model.parameters():
                param.requires_grad_(False)
        self._kd_enabled = (
            self.teacher_model is not None and self.distill_alpha > 0.0 and "kd" in self.distill_method
        )
        self._crd_enabled = self.teacher_model is not None and "crd" in self.distill_method
        if "crd" in self.distill_method and self.teacher_model is None:
            raise ValueError("CRD distillation requires a teacher model.")
        self.crd_beta = crd_beta
        self.neg_sample = neg_sample
        self.orig_t_inp = orig_t_inp
        self._crd_loss = None
        self._crd_options = CRDOptions(
            contrast_method=contrast_method,
            feat_dim=crd_feat_dim,
            nce_k=crd_k,
            nce_t=crd_temperature,
            nce_m=crd_momentum,
        )
        self._student_feature_module = self._find_last_linear(self.model)
        self._teacher_feature_module = self._find_last_linear(self.teacher_model) if self.teacher_model else None
        if self._crd_enabled:
            if self._student_feature_module is None:
                raise ValueError("CRD requires the student to have a final linear layer.")
            if self._teacher_feature_module is None:
                raise ValueError("CRD requires the teacher to have a final linear layer.")
        self._train_sample_count = 0
        self._crd_initialized = False

        # noise inject training
        self.noisy_model = None
        self.noise_schedule = None
        effective_noise_type = (noise_type or "mul")
        if mismatch_levels is not None or noise_level is not None:
            levels = []
            if mismatch_levels:
                levels.extend(float(lvl) for lvl in mismatch_levels)
            if noise_level is not None:
                levels.append(float(noise_level))
            levels = [lvl for lvl in levels if lvl >= 0.0]
            if levels:
                # Preserve order while removing duplicates
                unique_levels = list(dict.fromkeys(levels))
                self.noise_schedule = unique_levels
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
            logging.warning("Mismatch-aware training disabled; proceeding without injected mismatch noise.")

        self._prepare_cifar(img_type, self.dataset_name)
        # Todo: Change the scheduler to some more flexible one
        if warmup_epoch > 0:
            warmup_iters = len(self.train_dataloader) * self.warmup_epoch
            self.warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer=self.optimizer, start_factor=0.01,
                                                                total_iters=warmup_iters)

    def _find_last_linear(self, model):
        if model is None:
            return None
        last_linear = None
        for module in model.modules():
            if isinstance(module, nn.Linear):
                last_linear = module
        return last_linear

    def _student_forward(self, inputs):
        if self._student_feature_module is None:
            outputs = self.noisy_model(inputs) if self.noisy_model else self.model(inputs)
            return outputs, None

        features = {}

        def hook(_, hook_inputs, __):
            features["feat"] = hook_inputs[0]

        handle = self._student_feature_module.register_forward_hook(hook)
        try:
            outputs = self.noisy_model(inputs) if self.noisy_model else self.model(inputs)
        finally:
            handle.remove()
        student_feat = features.get("feat")
        if student_feat is None and self._crd_enabled:
            raise RuntimeError("Failed to capture student features for CRD.")
        return outputs, student_feat

    def _teacher_forward(self, inputs):
        if self.teacher_model is None:
            return None, None
        if self._teacher_feature_module is None:
            with torch.no_grad():
                outputs = self.teacher_model(inputs)
            return outputs, None

        features = {}

        def hook(_, hook_inputs, __):
            features["feat"] = hook_inputs[0]

        handle = self._teacher_feature_module.register_forward_hook(hook)
        try:
            with torch.no_grad():
                normalized_inputs = inputs
                if self.img_type == "scanGFI":
                    normalized_inputs = self._prepare_teacher_inputs(normalized_inputs)
                outputs = self.teacher_model(normalized_inputs)
        finally:
            handle.remove()
        teacher_feat = features.get("feat")
        if teacher_feat is None and self._crd_enabled:
            raise RuntimeError("Failed to capture teacher features for CRD.")
        if teacher_feat is not None:
            teacher_feat = teacher_feat.detach()
        return outputs, teacher_feat

    def _prepare_teacher_inputs(self, inputs):
        target_size = self.teacher_input_size
        if target_size and inputs.size(-1) != target_size:
            inputs = F.interpolate(inputs, size=(target_size, target_size), mode="bilinear", align_corners=False)
        if self.teacher_center_crop and target_size:
            h, w = inputs.shape[-2:]
            if h >= target_size and w >= target_size:
                top = (h - target_size) // 2
                left = (w - target_size) // 2
                inputs = inputs[..., top:top + target_size, left:left + target_size]
        channels = inputs.size(1)
        mean = torch.full((1, channels, 1, 1), 0.5, device=inputs.device)
        std = torch.full((1, channels, 1, 1), 0.5, device=inputs.device)
        return (inputs - mean) / std

    def _ensure_crd_initialized(self, student_feat, teacher_feat):
        if not self._crd_enabled or self._crd_initialized:
            return
        if student_feat is None or teacher_feat is None:
            raise RuntimeError("Student and teacher features are required to initialize CRD.")
        s_dim = student_feat.size(1)
        t_dim = teacher_feat.size(1)
        options = CRDOptions(
            contrast_method=self._crd_options.contrast_method,
            feat_dim=self._crd_options.feat_dim,
            nce_k=self._crd_options.nce_k,
            nce_t=self._crd_options.nce_t,
            nce_m=self._crd_options.nce_m,
            n_data=self._train_sample_count,
            s_dim=s_dim,
            t_dim=t_dim,
        )
        self._crd_loss = CRDLoss(options).to(self.device)
        self._crd_loss.train()
        self.optimizer.add_param_group({"params": self._crd_loss.parameters()})
        new_lr = self.optimizer.param_groups[-1]["lr"]
        self.optimizer.param_groups[-1]["initial_lr"] = new_lr
        if getattr(self, "scheduler", None) is not None:
            self.scheduler.base_lrs.append(new_lr)
            if hasattr(self.scheduler, "_last_lr"):
                last_lr = self.scheduler._last_lr[-1] if self.scheduler._last_lr else new_lr
                self.scheduler._last_lr.append(last_lr)
        if getattr(self, "warmup_scheduler", None) is not None:
            self.warmup_scheduler.base_lrs.append(new_lr)
            if hasattr(self.warmup_scheduler, "_last_lr"):
                last_lr = self.warmup_scheduler._last_lr[-1] if self.warmup_scheduler._last_lr else new_lr
                self.warmup_scheduler._last_lr.append(last_lr)
        self._crd_initialized = True

    def train(self):
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
                            val_acc, val_top5, train_acc, train_top5
                        )
                    )
                else:
                    print("Validation acc: {}, Train acc: {}".format(val_acc, train_acc))
                if val_acc > best_acc:
                    best_acc = val_acc
                    best_epoch = epoch + 1
                    best_top5 = val_top5
                    best_model_path = self._save_model_ckpt(val_acc, epoch + 1, "_best_ckpt.pth")
            self.scheduler.step()
        _ = self._save_model_ckpt(val_acc, self.num_epochs, "_last_ckpt.pth")
        print("----- Train finished, Model Name: {} -----".format(self.model_name))
        print("----- Total number of parameters: {} M -----".format(sum(p.numel() for p in self.model.parameters()) / 1e6))
        if self.dataset_name == "cifar100":
            print("----- Best top1: {}, Best top5: {}, Best epoch: {} -----".format(best_acc, best_top5, best_epoch))
        else:
            print("----- Best acc: {}, Best epoch: {} -----".format(best_acc, best_epoch))
        print("----- Model path: {} -----".format(best_model_path))
        print("--------------------------------------------------------------------------")
        return train_loss_list, val_acc_list

    def train_one_epoch(self, epoch):
        self.model.train()
        running_loss, n_samples = 0.0, 0
        progress_bar = tqdm.tqdm(enumerate(self.train_dataloader),
                            total=len(self.train_dataloader), desc="Training")
        for _i, _data in progress_bar:
            teacher_inputs = None
            contrast_idx = None
            if self.orig_t_inp and isinstance(_data, (list, tuple)):
                if len(_data) == 5:
                    # self.neg_sample = label, y_i != y_j
                    inputs, teacher_inputs, labels, indices, contrast_idx = _data
                elif len(_data) == 4:
                    # self.neg_sample = index, i != j
                    inputs, teacher_inputs, labels, indices = _data
                elif len(_data) == 3:
                    inputs, teacher_inputs, labels = _data
                    indices = None
                else:
                    raise RuntimeError(f"Unexpected batch format with orig_t_inp=True: len={len(_data)}")
            elif isinstance(_data, (list, tuple)) and len(_data) == 4:
                # self.neg_sample = label, y_i != y_j
                inputs, labels, indices, contrast_idx = _data
            elif isinstance(_data, (list, tuple)) and len(_data) == 3:
                # self.neg_sample = index, i != j
                inputs, labels, indices = _data
            else:
                inputs, labels = _data
                indices = None
            n_samples += inputs.size(0)
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            if teacher_inputs is not None:
                teacher_inputs = teacher_inputs.to(self.device)

            # Zero the parameter gradients
            self.optimizer.zero_grad()

            # Forward + backward + optimize
            outputs, student_feat = self._student_forward(inputs)
            teacher_logits, teacher_feat = None, None
            if self.teacher_model is not None and (self._kd_enabled or self._crd_enabled):
                teacher_logits, teacher_feat = self._teacher_forward(
                    teacher_inputs if teacher_inputs is not None else inputs
                )
            ce_loss = self.loss_fn(outputs, labels)
            kd_loss = None
            crd_loss = None

            if self._kd_enabled:
                if teacher_logits is None:
                    raise RuntimeError("Teacher logits not available for KD.")
                kd_loss = F.kl_div(
                    F.log_softmax(outputs / self.distill_temperature, dim=1),
                    F.softmax(teacher_logits / self.distill_temperature, dim=1),
                    reduction='batchmean',
                ) * (self.distill_temperature ** 2)

            if self._crd_enabled:
                if indices is None:
                    raise RuntimeError("Dataset indices are required for CRD.")
                indices_tensor = indices.to(self.device, dtype=torch.long)
                if teacher_feat is None or student_feat is None:
                    raise RuntimeError("Student and teacher features are required for CRD.")
                self._ensure_crd_initialized(student_feat, teacher_feat)
                crd_loss = self._crd_loss(
                    student_feat,
                    teacher_feat,
                    indices_tensor,
                    None if contrast_idx is None else contrast_idx.to(self.device, dtype=torch.long),
                )
            if self._kd_enabled and kd_loss is None:
                raise RuntimeError("KD loss was not computed despite KD being enabled.")
            if self._crd_enabled and crd_loss is None:
                raise RuntimeError("CRD loss was not computed despite CRD being enabled.")

            if self._kd_enabled and not self._crd_enabled:
                loss = (1.0 - self.distill_alpha) * ce_loss + self.distill_alpha * kd_loss
            elif self._kd_enabled and self._crd_enabled:
                loss = (
                    (1.0 - self.distill_alpha) * ce_loss
                    + self.distill_alpha * kd_loss
                    + self.crd_beta * crd_loss
                )
            elif self._crd_enabled:
                loss = ce_loss + self.crd_beta * crd_loss
            else:
                loss = ce_loss
            loss.backward()
            if self.max_norm is not None:
                grad_params = list(self.model.parameters())
                if self._crd_enabled and self._crd_loss is not None:
                    grad_params += list(self._crd_loss.parameters())
                nn.utils.clip_grad_norm_(grad_params, max_norm=self.max_norm)
            self.optimizer.step()

            # Update running loss and compute average loss
            running_loss += loss.item() * inputs.size(0)
            avg_loss = running_loss / n_samples

            # Update the tqdm progress bar with current iteration and loss
            postfix = {
                "Iter": f"{_i + 1}/{len(self.train_dataloader)}",
                "Loss": f"{avg_loss:.4f}",
                "LR": self.optimizer.param_groups[0]["lr"]
            }
            if self.noisy_model is not None:
                postfix["Noise"] = f"{self.noisy_model.current_noise_level:.3f}"
            progress_bar.set_postfix(postfix)

            if epoch < self.warmup_epoch:
                self.warmup_scheduler.step()

        running_loss /= n_samples
        return running_loss

    def evaluate(self, dataloader):
        correct = 0
        correct_top5 = 0
        total = 0
        running_loss = 0.0
        pred_list, label_list = [], []
        compute_top5 = self.dataset_name == "cifar100"
        with torch.no_grad():
            self.model.eval()
            for data in dataloader:
                if isinstance(data, (list, tuple)):
                    if self.orig_t_inp:
                        if len(data) == 5:
                            inputs, _, labels, _, _ = data
                        elif len(data) == 4:
                            inputs, _, labels, _ = data
                        elif len(data) == 3:
                            inputs, _, labels = data
                        else:
                            inputs, labels = data
                    else:
                        if len(data) == 4:
                            inputs, labels, _, _ = data
                        elif len(data) == 3:
                            inputs, labels, _ = data
                        else:
                            inputs, labels = data
                else:
                    inputs, labels = data

                # move the data to GPU
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # calculate outputs by running inputs through the network
                outputs = self.model(inputs)

                loss = self.loss_fn(outputs, labels)
                running_loss += loss.item()

                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                if compute_top5:
                    max_k = min(5, outputs.size(1))
                    topk = outputs.topk(max_k, dim=1).indices
                    correct_top5 += topk.eq(labels.view(-1, 1)).any(dim=1).sum().item()

                pred_list.append(predicted)
                label_list.append(labels)

        accuracy = correct / total
        top5_acc = (correct_top5 / total) if compute_top5 and total > 0 else None
        return accuracy, top5_acc, torch.cat(label_list), torch.cat(pred_list)

    def _save_then_load(self, save_to):
        model_class = self.model.__class__
        tmp_sd = {
            'net': self.model.state_dict(),
            'init_args': self.model.init_args,
        }
        tmp_sp = os.path.join(str(save_to), "__tmp_model.pth")
        torch.save(tmp_sd, tmp_sp)
        tmp_sd = torch.load(tmp_sp, weights_only=False)
        decoupled_model = model_class(
            **{**tmp_sd['init_args']['model_args'], **tmp_sd['init_args']['kwargs']}).to(self.device)
        p_dict = get_parametrized_weight_mods(self.model)
        _ = load_and_register_buffer(decoupled_model, tmp_sd['net'], self.device, p_dict)
        return decoupled_model

    def _save_model_ckpt(self, acc, epoch, suffix=""):
        save_to = os.path.join(self.save_path, self.model_name)
        os.makedirs(save_to, exist_ok=True)
        save_pth_path = os.path.join(str(save_to), self.model_name + suffix)

        # Need to save then the load the model to totally decouple the parameterization
        flat_model = self._save_then_load(save_to)
        parametrize_flag = False
        for _mod in flat_model.modules():
            if P.is_parametrized(_mod):
                parametrize_flag = True
                P.remove_parametrizations(_mod, "weight", leave_parametrized=True) # Keep the parametrized res
        if parametrize_flag:
            logging.warning("Model Includes parametrized module, saving the non-parametrized model with param baked in.")
            flat_state = {
                'net': flat_model.state_dict(),
                'init_args': self.model.init_args,
                'net_type': self.model.__class__.__name__,
                'acc': acc,
                'epoch': epoch,
            }
            if self._crd_enabled and self._crd_loss is not None:
                flat_state['crd'] = self._crd_loss.state_dict()
            # save the flat model with the same name as before
            torch.save(flat_state, save_pth_path)
            # modify the model name with "full_param" to save the model with full parametrization
            save_pth_path = os.path.join(str(save_to), self.model_name + "_full_param" + suffix)

        state = {
            'net': self.model.state_dict(),
            'init_args': self.model.init_args,
            'net_type': self.model.__class__.__name__,
            'acc': acc,
            'epoch': epoch,
        }
        if self._crd_enabled and self._crd_loss is not None:
            state['crd'] = self._crd_loss.state_dict()
        torch.save(state, save_pth_path)
        return save_pth_path

    def _get_optimizer(self, optim_type, lr, weight_decay):
        if optim_type == "SGD":
            return optim.SGD(self.model.parameters(), momentum=0.9, lr=lr, weight_decay=weight_decay, nesterov=False)
        elif optim_type == "Adam":
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError("Unknown optimizer: {}".format(optim_type))

    def _prepare_cifar(self, img_type, dataset_name):
        """
        Todo: Actually the validation dataset should be split from the train_set.
        After the split, we can change the scheduler into other types depending on the validation result.
        """
        dataset_name = _normalize_dataset_name(dataset_name)
        if img_type in {"rgb", "rggb"}:
            mean, std = _CIFAR_STATS[dataset_name]
            dataset_cls = torchvision.datasets.CIFAR100 if dataset_name == "cifar100" else torchvision.datasets.CIFAR10
            if self.aug:
                if img_type == "rgb":
                    transform_train = transforms.Compose([
                        transforms.RandomResizedCrop(32, scale=(0.75, 1.0), ratio=(1.0, 1.0)),
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.RandAugment(num_ops=1, magnitude=8),
                        transforms.ColorJitter(0.1, 0.1, 0.1),
                        transforms.ToTensor(),
                        transforms.Normalize(mean, std),
                        transforms.RandomErasing(p=0.25),
                    ])
                else:
                    transform_train = transforms.Compose([
                        transforms.ToTensor(),
                        ToPackedRGGB(return_orig=False),
                        transforms.RandomResizedCrop(16, scale=(0.75, 1.0), ratio=(1.0, 1.0)),
                        transforms.RandomHorizontalFlip(p=0.2),
                        transforms.RandomErasing(p=0.1),
                    ])
            else:
                if img_type == "rgb":
                    transform_train = transforms.Compose([
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean, std), ])
                else:
                    # Todo: Normalize rggb data?
                    transform_train = transforms.Compose([
                        transforms.ToTensor(),
                        ToPackedRGGB(return_orig=False),
                        transforms.RandomCrop(16, padding=2),
                        transforms.RandomHorizontalFlip(),
                    ])
            if img_type == "rgb":
                transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std), ])
            else:
                transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    ToPackedRGGB(return_orig=False), ])
            self.train_set = dataset_cls(root='../data', train=True, download=True, transform=transform_train)
            self.val_set = dataset_cls(root='../data', train=False, download=True, transform=transform_test)
        elif img_type == "scanGFI":
            with tempfile.TemporaryDirectory() as tmpdir:
                conf_path = Path(os.path.join(tmpdir, "config.json"))
                subprocess.run("uv run scangen create-config --dataset {} {}".format(dataset_name, str(conf_path)), shell=True)
                config_to_use = conf_path
                created_temp_config = True

                with open(config_to_use) as fp:
                    scangen_config = json.load(fp)
            self.scangen_noise_config = scangen_config["noise"]

            noise_data_root = _resolve_cifar_data_root(img_type, dataset_name)
            self.scangen_noise_root = noise_data_root
            self.train_set = MyNoiseCIFARDataset(
                root=noise_data_root,
                input_name=dataset_name + "_raw",
                train=True,
                noise_config=self.scangen_noise_config,
                device=self.device,
                transform=transforms.Compose([
                    transforms.RandomCrop(16, padding=2),
                    transforms.RandomHorizontalFlip(),
                ])
            )
            if self.orig_t_inp:
                # Use the original cifar data for the teacher model.
                teacher_train_transform_steps = []
                if self.teacher_input_size and self.teacher_input_size > 0:
                    teacher_train_transform_steps.append(transforms.Resize(self.teacher_input_size))
                    if self.teacher_center_crop:
                        teacher_train_transform_steps.append(transforms.CenterCrop(self.teacher_input_size))
                teacher_train_transform_steps.append(transforms.ToTensor())
                teacher_train_transform = transforms.Compose(teacher_train_transform_steps)

                teacher_dataset_cls = torchvision.datasets.CIFAR100 if dataset_name == "cifar100" else torchvision.datasets.CIFAR10
                teacher_train_set = teacher_dataset_cls(
                    root='../data',
                    train=True,
                    download=True,
                    transform=teacher_train_transform,
                )
                self.train_set = StudentTeacherPairDataset(self.train_set, teacher_train_set)

            self.val_set = MyNoiseCIFARDataset(
                root=noise_data_root,
                input_name=dataset_name + "_raw",
                train=False,
                noise_config=self.scangen_noise_config,
                device=self.device,
                seed=0,
            )
            if created_temp_config and config_to_use.exists():
                config_to_use.unlink()

            teacher_transform_steps = []
            if self.orig_t_inp:
                # Evaluate the teacher model with the original cifar dataset.
                if self.teacher_input_size and self.teacher_input_size > 0:
                    teacher_transform_steps.append(transforms.Resize(self.teacher_input_size))
                    if self.teacher_center_crop:
                        teacher_transform_steps.append(transforms.CenterCrop(self.teacher_input_size))
                teacher_transform_steps.append(transforms.ToTensor())
                teacher_transform = transforms.Compose(teacher_transform_steps)

                teacher_dataset_cls = torchvision.datasets.CIFAR100 if dataset_name == "cifar100" else torchvision.datasets.CIFAR10
                teacher_dataset = teacher_dataset_cls(
                    root='../data',
                    train=False,
                    download=True,
                    transform=teacher_transform,
                )
            else:
                # Evaluate the teacher model with the scanGFI dataset.
                teacher_transform_steps = [transforms.ToPILImage()]
                if self.teacher_input_size and self.teacher_input_size > 0:
                    teacher_transform_steps.append(transforms.Resize(self.teacher_input_size))
                    if self.teacher_center_crop:
                        teacher_transform_steps.append(transforms.CenterCrop(self.teacher_input_size))
                teacher_transform_steps.append(transforms.ToTensor())
                teacher_transform = transforms.Compose(teacher_transform_steps)

                teacher_dataset = MyNoiseCIFARDataset(
                    root=noise_data_root,
                    input_name=dataset_name + "_raw",
                    train=False,
                    noise_config=self.scangen_noise_config,
                    device=self.device,
                    transform=teacher_transform,
                    seed=0,
                )

            self.teacher_eval_loader = torch.utils.data.DataLoader(
                teacher_dataset,
                batch_size=self.test_batch_size,
                shuffle=False,
                num_workers=2,
            )
        else:
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomCrop(16, padding=2),
                transforms.RandomHorizontalFlip(),
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
            ])
            raw_root = Path(__file__).resolve().parent.parent / _default_cifar_dir(dataset_name) / img_type
            self.train_set = RawImgDataset(root=str(raw_root), train=True, transform=transform_train)
            self.val_set = RawImgDataset(root=str(raw_root), train=False, transform=transform_test)

        if self._crd_enabled:
            assert self.neg_sample in {"label", "index"}
            if self.orig_t_inp and img_type == "scanGFI":
                if self.neg_sample == "index":
                    # negative samples based on i != j
                    self.train_set = PairDatasetWithIndex(self.train_set)
                elif self.neg_sample == "label":
                    # sampling negative samples with y_i != y_j
                    self.train_set = PairDatasetWithIndexAndContrast(
                        self.train_set,
                        k=self._crd_options.nce_k,
                        mode='exact',
                        percent=1.0,
                    )
                else:
                    raise NotImplementedError
            else:
                if self.neg_sample == "index":
                    # negative samples based on i != j
                    self.train_set = DatasetWithIndex(self.train_set)
                elif self.neg_sample == "label":
                    # sampling negative samples with y_i != y_j
                    self.train_set = DatasetWithIndexAndContrast(
                        self.train_set,
                        k=self._crd_options.nce_k,
                        mode='exact',
                        percent=1.0,
                    )
                else:
                    raise NotImplementedError
        self._train_sample_count = len(self.train_set)

        # Get dataloader
        self.train_dataloader = torch.utils.data.DataLoader(
            self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=2
        )
        self.val_dataloader = torch.utils.data.DataLoader(self.val_set, batch_size=self.test_batch_size, shuffle=False,
                                                          num_workers=2)
        if self.teacher_eval_loader is None:
            self.teacher_eval_loader = self.val_dataloader


class WrappedNoisyModel(nn.Module):
    """
    Injects multiplicative or additive parameter noise during training to emulate weight mismatch.
    """
    def __init__(self, model: nn.Module, noise_levels, noise_type: str = "mul"):
        super().__init__()
        self.model = model
        if isinstance(noise_levels, (int, float)):
            levels = [float(noise_levels)]
        else:
            levels = [float(lvl) for lvl in noise_levels]
        if not levels:
            raise ValueError("noise_levels must contain at least one value.")
        if any(lvl < 0.0 for lvl in levels):
            raise ValueError("noise_levels must be non-negative.")
        # Preserve ordering while removing duplicates
        self.noise_levels = list(dict.fromkeys(levels))
        noise_type_lower = (noise_type or "mul").lower()
        if noise_type_lower not in {"mul", "add"}:
            raise ValueError(f"Unsupported noise_type={noise_type}. Expected 'mul' or 'add'.")
        self.noise_type = noise_type_lower
        # Keeps a list of params free from noise
        self.noise_free_params = {"s_w_Param"}
        self.current_noise_level = 0.0

    def _check_noise_free(self, p_name: str) -> bool:
        return any(_nf_p in p_name for _nf_p in self.noise_free_params)

    def _sample_noise_level(self) -> float:
        level = random.choice(self.noise_levels)
        self.current_noise_level = level
        return level

    def gen_noisy_params(self):
        noise_std = self._sample_noise_level()
        noisy_params = {}
        for _name, _param in self.model.named_parameters():
            if self._check_noise_free(_name) or noise_std == 0.0:
                noisy_params[_name] = _param
                continue
            if self.noise_type == "mul":
                noisy_params[_name] = self._apply_noise_mul(_param, noise_std)
            else:
                noisy_params[_name] = self._apply_noise_add(_param, noise_std)
        return noisy_params

    def _apply_noise_mul(self, p: nn.Parameter, std: float):
        noise_ = torch.randn_like(p, device=p.device, requires_grad=False) * std
        # The noise is multiplicative to emulate parameter mismatch.
        return p.mul(1 + noise_)

    def _apply_noise_add(self, p: nn.Parameter, std: float):
        p_max = p.detach().abs().max()
        scale_val = p_max.item() if p_max is not None else 0.0
        if scale_val == 0.0:
            return p
        noise_ = torch.randn_like(p, device=p.device, requires_grad=False) * std * scale_val
        return p.add(noise_)

    def forward(self, x):
        if not self.model.training:
            return self.model(x)
        noisy_params = self.gen_noisy_params()
        return torch.func.functional_call(self.model, noisy_params, (x,))

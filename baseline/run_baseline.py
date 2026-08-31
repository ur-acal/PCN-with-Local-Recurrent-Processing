#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import logging
import os
import pickle
import random
import sys
from pathlib import Path
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set

import numpy as np
import timm
import torch
import torch.nn as nn
from timm.data import create_transform, resolve_model_data_config, create_loader
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import baseline.cifar_resnet  # registers custom CIFAR models into timm
from baseline.baseline_cifar_configs import TINYIMAGENET_DEFAULTS, get_baseline_config, build_model
from trainer import _CIFAR_STATS
from weight_range_audit import collect_wrn_weight_range_audit_rows, save_audit_csv
from mismatch_utils import ADDITIVE_SCALE_MODES, additive_mismatch_scale
from tinyimagenet_data import (
    TINYIMAGENET_MEAN,
    TINYIMAGENET_STD,
    build_tinyimagenet_datasets,
)


log = logging.getLogger(__name__)


NORM_TYPES = (
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.SyncBatchNorm,
    nn.GroupNorm,
    nn.LayerNorm,
    nn.InstanceNorm1d,
    nn.InstanceNorm2d,
    nn.InstanceNorm3d,
)

BN_TYPES = (
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.SyncBatchNorm,
)


@dataclass
class BNRecalibrationConfig:
    enabled: bool = False
    num_samples: str = "all"
    batch_size: int = 128
    subset_seed: int = 123
    num_workers: int = 4
    diagnostics_dir: str = "./bn_recalibration_diagnostics"


@dataclass
class BNRecalibrationReport:
    num_bn_modules: int
    num_calibration_samples: int
    changed_bn_buffers: List[str]
    conv_linear_classifier_hash: str
    bn_affine_hash: str
    fixed_noisy_param_hash: str
    diagnostics_path: Optional[str] = None


@dataclass
class NoiseApplyRecord:
    name: str
    kind: str
    changed: bool
    all_zero: bool
    numel: int


@dataclass
class NoiseAssertionSummary:
    applied_records: List[NoiseApplyRecord]
    skipped_records: List[NoiseApplyRecord]


class FixedMismatchHelper:
    """
    Add one fixed mismatch realization to a model for one full evaluation trial.

    Policy:
      - By default, apply mismatch to all floating-point parameters.
      - If noise_to_norm=False, skip parameters belonging to normalization modules.
      - If noise_to_norm=True, also perturb normalization parameters and floating buffers
        such as running_mean / running_var.
      - Exact all-zero tensors are allowed to remain unchanged under multiplicative noise.
    """

    def __init__(
        self,
        model: nn.Module,
        noise_sigma: float,
        noise_type: str = "multiplicative",
        noise_to_norm: bool = False,
        include_buffers: bool = True,
        seed: Optional[int] = None,
        exclude_param_names: Optional[Iterable[str]] = None,
        additive_scale_mode: str = "max_abs",
    ):
        self.model = model
        self.noise_sigma = float(noise_sigma)
        self.noise_type = noise_type
        self.noise_to_norm = bool(noise_to_norm)
        self.include_buffers = bool(include_buffers)
        self.seed = seed
        self.exclude_param_names: Set[str] = set(exclude_param_names or [])
        if additive_scale_mode not in ADDITIVE_SCALE_MODES:
            raise ValueError(f"Unsupported additive scale mode: {additive_scale_mode}")
        self.additive_scale_mode = additive_scale_mode

        self._clean_state = None
        self._last_summary = None
        self._param_to_module = self._build_param_to_module_map()
        self._buffer_to_module = self._build_buffer_to_module_map()

    def _build_param_to_module_map(self) -> Dict[str, nn.Module]:
        out = {}
        for module_name, module in self.model.named_modules():
            for local_name, _ in module.named_parameters(recurse=False):
                full_name = f"{module_name}.{local_name}" if module_name else local_name
                out[full_name] = module
        return out

    def _build_buffer_to_module_map(self) -> Dict[str, nn.Module]:
        out = {}
        for module_name, module in self.model.named_modules():
            for local_name, _ in module.named_buffers(recurse=False):
                full_name = f"{module_name}.{local_name}" if module_name else local_name
                out[full_name] = module
        return out

    def snapshot_clean_state(self):
        self._clean_state = {
            k: v.detach().cpu().clone()
            for k, v in self.model.state_dict().items()
        }

    def restore_clean_state(self):
        assert self._clean_state is not None, "Call snapshot_clean_state() before restore_clean_state()."
        state = {k: v.clone() for k, v in self._clean_state.items()}
        self.model.load_state_dict(state, strict=True)

    @staticmethod
    def _is_norm_module(module: Optional[nn.Module]) -> bool:
        return isinstance(module, NORM_TYPES)

    @staticmethod
    def _tensor_all_zero(x: torch.Tensor) -> bool:
        return bool(torch.count_nonzero(x).item() == 0)

    def _apply_noise_(self, x: torch.Tensor):
        if self.noise_sigma <= 0:
            return
        if not x.is_floating_point():
            return

        noise = torch.randn_like(x)
        if self.noise_type == "multiplicative":
            x.mul_(1.0 + self.noise_sigma * noise)
        elif self.noise_type == "additive":
            scale = additive_mismatch_scale(x, self.additive_scale_mode)
            x.add_(self.noise_sigma * noise * scale)
        else:
            raise ValueError(f"Unsupported noise_type: {self.noise_type}")

    def _should_noise_param(self, name: str, p: torch.Tensor) -> bool:
        if not p.is_floating_point():
            return False
        if name in self.exclude_param_names:
            return False

        module = self._param_to_module.get(name, None)
        is_norm = self._is_norm_module(module)

        if is_norm and not self.noise_to_norm:
            return False

        return True

    def _should_noise_buffer(self, name: str, buf: torch.Tensor) -> bool:
        if not self.include_buffers:
            return False
        if not buf.is_floating_point():
            return False

        module = self._buffer_to_module.get(name, None)
        is_norm = self._is_norm_module(module)
        if not is_norm:
            return False

        return self.noise_to_norm

    def add_noise(self) -> NoiseAssertionSummary:
        assert self._clean_state is not None, "Call snapshot_clean_state() before add_noise()."

        if self.seed is not None:
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            np.random.seed(self.seed % (2**32 - 1))
            random.seed(self.seed)

        applied_records = []
        skipped_records = []

        with torch.no_grad():
            for name, p in self.model.named_parameters():
                clean = self._clean_state[name].to(device=p.device, dtype=p.dtype)
                all_zero = self._tensor_all_zero(clean) if p.is_floating_point() else False
                should_apply = self._should_noise_param(name, p)

                if should_apply and self.noise_sigma > 0:
                    self._apply_noise_(p)
                    changed = not torch.equal(p.detach(), clean)
                    if all_zero:
                        changed = False
                    applied_records.append(NoiseApplyRecord(name, "param", changed, all_zero, p.numel()))
                else:
                    skipped_records.append(NoiseApplyRecord(name, "param", False, all_zero, p.numel()))

            for name, buf in self.model.named_buffers():
                clean = self._clean_state[name].to(device=buf.device, dtype=buf.dtype)
                all_zero = self._tensor_all_zero(clean) if buf.is_floating_point() else False
                should_apply = self._should_noise_buffer(name, buf)

                if should_apply and self.noise_sigma > 0:
                    self._apply_noise_(buf)
                    if "running_var" in name:
                        buf.clamp_(min=1e-6)
                    changed = not torch.equal(buf.detach(), clean)
                    if all_zero:
                        changed = False
                    applied_records.append(NoiseApplyRecord(name, "buffer", changed, all_zero, buf.numel()))
                else:
                    skipped_records.append(NoiseApplyRecord(name, "buffer", False, all_zero, buf.numel()))

        summary = NoiseAssertionSummary(applied_records=applied_records, skipped_records=skipped_records)
        self.assert_noise_applied(summary)
        self._last_summary = summary
        return summary

    def assert_noise_applied(self, summary: NoiseAssertionSummary):
        if self.noise_sigma <= 0:
            return

        assert len(summary.applied_records) > 0, "No tensors were selected for mismatch. Check options."

        for rec in summary.applied_records:
            if rec.all_zero:
                log.warning(
                    "Skipping strict assertion for all-zero %s tensor under %s mismatch: %s",
                    rec.kind,
                    self.noise_type,
                    rec.name,
                )
                continue
            assert rec.changed, f"Mismatch was not applied to {rec.kind}: {rec.name}"

        clean_keys = set(self._clean_state.keys())
        state_keys = set(self.model.state_dict().keys())
        assert clean_keys == state_keys, "State dict keys changed after noise application."


def str2bool(v: str) -> bool:
    return v.lower() in ("yes", "true", "t", "1")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def infer_run_name_from_checkpoint_map(checkpoint_map: Dict[str, str]) -> str:
    if not checkpoint_map:
        return ""

    names = []
    for _, ckpt_path in checkpoint_map.items():
        # /.../resize_finetune_resmlp_12_224/resize_finetune_resmlp_12_224_best_ckpt.pth
        # -> resize_finetune_resmlp_12_224
        run_name = os.path.basename(os.path.dirname(ckpt_path.rstrip("/")))
        names.append(run_name)

    return "_".join(names)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate timm baselines under fixed mismatch.")
    parser.add_argument("--model_list", type=str, required=True,
                        help="Comma-separated timm model names, e.g. resnet18,resnet34")
    parser.add_argument("--dataset", type=str, choices=["cifar10", "cifar100", "tinyimagenet", "imagenet"], required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--checkpoint_dir", type=str, default="",
                        help="Optional directory containing <model_name>.pth, <model_name>.pt, or <model_name>.ckpt")
    parser.add_argument("--checkpoint_map", type=str, default="",
                        help="Optional mapping: model_a=/path/a.pth,model_b=/path/b.pth")
    parser.add_argument("--pretrained", type=str2bool, default=False,
                        help="Use timm pretrained weights if no checkpoint is provided.")
    parser.add_argument(
        "--case",
        type=str,
        default="auto",
        choices=[
            "auto",
            "custom_noresize",
            "adapt_noresize_scratch",
            "adapt_noresize_finetune",
            "resize_scratch",
            "resize_finetune",
        ],
    )
    parser.add_argument(
        "--prefer_resize",
        type=str2bool,
        default=False,
        help="For supported timm models, choose resize case instead of adapt_noresize when case=auto.",
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--num_classes", type=int, default=None)
    parser.add_argument("--in_chans", type=int, default=3)
    parser.add_argument("--noise_level_list", type=str, default="0.0,0.01,0.02,0.05")
    parser.add_argument("--noisy_trials", type=int, default=5)
    parser.add_argument("--noise_type", type=str, choices=["multiplicative", "additive"], default="multiplicative")
    parser.add_argument("--additive_scale_mode", choices=ADDITIVE_SCALE_MODES, default="max_abs")
    parser.add_argument("--noise_to_norm", type=str2bool, default=False)
    parser.add_argument(
        "--exclude_conv_bias_from_mismatch",
        type=str2bool,
        default=False,
        help="Exclude only bias parameters owned by convolution modules from mismatch.",
    )
    parser.add_argument("--fold_norm", type=str2bool, default=False)
    parser.add_argument(
        "--fold_norm_mode",
        type=str,
        choices=["sequential", "wrn_preact_no_mismatch_bias"],
        default="sequential",
        help=(
            "Norm folding implementation used when --fold_norm=true. "
            "'wrn_preact_no_mismatch_bias' folds function-preserving WRN Conv/BN pairs "
            "and excludes only the newly induced folded Conv bias parameters from mismatch."
        ),
    )
    parser.add_argument("--results_dir", type=str, default="./results_mismatch_eval")
    parser.add_argument("--pin_memory", type=str2bool, default=False)
    parser.add_argument("--use_amp", type=str2bool, default=False)
    parser.add_argument("--max_eval_batches", type=lambda s: None if s.lower() in {"none", ""} else int(s), default=None,
                        help="Optional smoke-test limit for evaluation batches. Default preserves full evaluation.")
    parser.add_argument("--weight_range_audit_csv", type=str, default="",
                        help="Optional sidecar CSV for clean-checkpoint max_abs/rms/kappa audit of selected tensors.")

    parser.add_argument("--bn_recalibration_enabled", type=str2bool, default=False)
    parser.add_argument("--bn_recalibration_num_samples", type=str, default="all")
    parser.add_argument("--bn_recalibration_batch_size", type=int, default=128)
    parser.add_argument("--bn_recalibration_subset_seed", type=int, default=123)
    parser.add_argument("--bn_recalibration_num_workers", type=int, default=4)
    parser.add_argument("--bn_recalibration_diagnostics_dir", type=str, default="./bn_recalibration_diagnostics")
    return parser.parse_args()


def parse_checkpoint_map(s: str) -> Dict[str, str]:
    out = {}
    if not s:
        return out
    for item in s.split(","):
        item = item.strip()
        if not item:
            continue
        model_name, ckpt_path = item.split("=", 1)
        out[model_name.strip()] = ckpt_path.strip()
    return out


def infer_num_classes(dataset_name: str, explicit_num_classes: Optional[int]) -> int:
    if explicit_num_classes is not None:
        return explicit_num_classes
    if dataset_name == "cifar10":
        return 10
    if dataset_name == "cifar100":
        return 100
    if dataset_name == "tinyimagenet":
        return 200
    if dataset_name == "imagenet":
        return 1000
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def _mark_mismatch_excluded_params(model: nn.Module, names: Iterable[str]):
    current = set(getattr(model, "_mismatch_excluded_param_names", set()))
    current.update(names)
    model._mismatch_excluded_param_names = current


def _replace_child(root: nn.Module, dotted_name: str, new_module: nn.Module):
    parts = dotted_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    leaf = parts[-1]
    if leaf.isdigit():
        parent[int(leaf)] = new_module
    else:
        setattr(parent, leaf, new_module)


def _get_child(root: nn.Module, dotted_name: str) -> nn.Module:
    cur = root
    for part in dotted_name.split("."):
        cur = cur[int(part)] if part.isdigit() else getattr(cur, part)
    return cur


def _fold_conv_bn_pair(model: nn.Module, conv_name: str, bn_name: str) -> Optional[str]:
    conv = _get_child(model, conv_name)
    bn = _get_child(model, bn_name)
    if not isinstance(conv, (nn.Conv1d, nn.Conv2d, nn.Conv3d)) or not isinstance(
        bn, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
    ):
        return None
    fused = torch.nn.utils.fusion.fuse_conv_bn_eval(conv.eval(), bn.eval())
    _replace_child(model, conv_name, fused)
    _replace_child(model, bn_name, nn.Identity())
    return f"{conv_name}.bias" if fused.bias is not None else None


def fold_wrn_preact_norms_no_mismatch_bias(model: nn.Module) -> nn.Module:
    """Fold function-preserving WRN Conv->BN pairs and mark induced biases for mismatch exclusion."""
    model.eval()
    excluded = []

    for layer_name in ("layer1", "layer2", "layer3"):
        layer = getattr(model, layer_name, None)
        if layer is None:
            continue
        for idx, block in enumerate(layer):
            if hasattr(block, "conv1") and hasattr(block, "bn2"):
                bias_name = _fold_conv_bn_pair(model, f"{layer_name}.{idx}.conv1", f"{layer_name}.{idx}.bn2")
                if bias_name:
                    excluded.append(bias_name)

    _mark_mismatch_excluded_params(model, excluded)
    log.warning(
        "WRN preactivation norm folding finished. Folded %d Conv/BN pairs; excluding %d induced bias params from mismatch.",
        len(excluded),
        len(excluded),
    )
    return model


def maybe_fold_norms(model: nn.Module, mode: str = "sequential") -> nn.Module:
    if mode == "wrn_preact_no_mismatch_bias":
        return fold_wrn_preact_norms_no_mismatch_bias(model)
    if mode != "sequential":
        raise ValueError(f"Unsupported fold_norm_mode: {mode}")
    """
    Conservative best-effort Conv/BatchNorm folding.

    This only folds common adjacent Conv + BatchNorm pairs inside nn.Sequential.
    GroupNorm, LayerNorm, InstanceNorm, and non-adjacent BN patterns are left unchanged.
    """
    folded_count = 0

    def _fuse_sequential(seq: nn.Sequential):
        nonlocal folded_count
        i = 0
        while i < len(seq) - 1:
            m1 = seq[i]
            m2 = seq[i + 1]
            if isinstance(m1, (nn.Conv1d, nn.Conv2d, nn.Conv3d)) and isinstance(m2, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                fused = torch.nn.utils.fusion.fuse_conv_bn_eval(m1.eval(), m2.eval())
                seq[i] = fused
                seq[i + 1] = nn.Identity()
                folded_count += 1
                i += 2
            else:
                if isinstance(m1, nn.Sequential):
                    _fuse_sequential(m1)
                i += 1
        if len(seq) > 0 and isinstance(seq[-1], nn.Sequential):
            _fuse_sequential(seq[-1])

    model.eval()
    for _, module in model.named_modules():
        if isinstance(module, nn.Sequential):
            _fuse_sequential(module)

    log.warning("Best-effort norm folding finished. Folded %d Conv/BN pairs.", folded_count)
    return model


def resolve_checkpoint_path(model_name: str, checkpoint_map: Dict[str, str], checkpoint_dir: str) -> Optional[str]:
    if model_name in checkpoint_map:
        return checkpoint_map[model_name]
    if checkpoint_dir:
        candidates = [
            os.path.join(checkpoint_dir, f"{model_name}.pth"),
            os.path.join(checkpoint_dir, f"{model_name}.pt"),
            os.path.join(checkpoint_dir, f"{model_name}.ckpt"),
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
    return None


def load_model_weights(model: nn.Module, ckpt_path: Optional[str], device: torch.device):
    if ckpt_path is None:
        log.warning("No external checkpoint provided. Using current model weights.")
        return

    log.warning("Loading checkpoint: %s", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict):
        if "net" in ckpt:
            state_dict = ckpt["net"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif "model" in ckpt:
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module."):]
        new_state_dict[nk] = v

    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    log.warning("Checkpoint load done. missing=%d unexpected=%d", len(missing), len(unexpected))
    if len(missing) > 0:
        log.warning("Missing keys (first 20): %s", missing[:20])
    if len(unexpected) > 0:
        log.warning("Unexpected keys (first 20): %s", unexpected[:20])


def convolution_bias_parameter_names(model: nn.Module) -> Set[str]:
    conv_types = (
        nn.Conv1d, nn.Conv2d, nn.Conv3d,
        nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d,
    )
    return {
        f"{module_name}.bias" if module_name else "bias"
        for module_name, module in model.named_modules()
        if isinstance(module, conv_types) and module.bias is not None
    }


def _build_eval_transform(model: nn.Module, dataset_name: str, cfg: dict):
    if dataset_name == "tinyimagenet":
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(TINYIMAGENET_MEAN, TINYIMAGENET_STD),
        ])
    if dataset_name in {"cifar10", "cifar100"}:
        mean, std = _CIFAR_STATS[dataset_name]
        input_size = cfg["timm_input_size"]
        test_kwargs = dict(
            input_size=input_size,
            is_training=False,
            use_prefetcher=False,
            interpolation=cfg.get("interpolation", "bicubic"),
            mean=mean,
            std=std,
        )
        if input_size[-2:] == (32, 32):
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        return create_transform(**test_kwargs)

    if dataset_name == "imagenet":
        data_config = resolve_model_data_config(model)
        return create_transform(**data_config, is_training=False)

    raise ValueError(f"Unsupported dataset: {dataset_name}")


def build_test_loader(model: nn.Module, args, cfg: dict) -> DataLoader:
    if args.dataset in {"cifar10", "cifar100"}:
        test_transform = _build_eval_transform(model, args.dataset, cfg)

        if args.dataset == "cifar10":
            test_dataset = datasets.CIFAR10(
                root=args.data_dir,
                train=False,
                download=True,
                transform=test_transform,
            )
        else:
            test_dataset = datasets.CIFAR100(
                root=args.data_dir,
                train=False,
                download=True,
                transform=test_transform,
            )

    elif args.dataset == "tinyimagenet":
        test_transform = _build_eval_transform(model, args.dataset, cfg)
        _, test_dataset = build_tinyimagenet_datasets(
            args.data_dir,
            val_transform=test_transform,
            validate_counts=True,
        )

    elif args.dataset == "imagenet":
        test_transform = _build_eval_transform(model, args.dataset, cfg)
        test_root = os.path.join(args.data_dir, "val")
        test_dataset = datasets.ImageFolder(root=test_root, transform=test_transform)

    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    ##########################################################################################
    # Note: We are not using create_loader of timm because it will overwrite the transform
    # of the dataset. Since we already set the transforms before, we don't need to set
    # the transform via create_loader.
    ##########################################################################################
    # test_loader = create_loader(
    #     test_dataset,
    #     input_size=cfg["timm_input_size"],
    #     batch_size=args.batch_size,
    #     is_training=False,
    #     use_prefetcher=False,
    #     num_workers=args.num_workers,
    #     pin_memory=args.pin_memory,
    #     persistent_workers=False,
    # )
    # x, y = next(iter(test_loader_raw))
    # print(x.shape, x.mean().item(), x.std().item(), y[:10])
    # x, y = next(iter(test_loader))
    # print(x.shape, x.mean().item(), x.std().item(), y[:10])
    test_loader_raw = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=False,
    )
    return test_loader_raw



class UnlabeledDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, _ = self.base_dataset[idx]
        return image


def parse_bn_recalibration_config(args) -> BNRecalibrationConfig:
    return BNRecalibrationConfig(
        enabled=bool(args.bn_recalibration_enabled),
        num_samples=str(args.bn_recalibration_num_samples),
        batch_size=int(args.bn_recalibration_batch_size),
        subset_seed=int(args.bn_recalibration_subset_seed),
        num_workers=int(args.bn_recalibration_num_workers),
        diagnostics_dir=str(args.bn_recalibration_diagnostics_dir),
    )


def _parse_num_samples(raw: str, dataset_len: int) -> int:
    raw = str(raw).strip().lower()
    if raw == "all":
        return dataset_len
    n = int(raw)
    if n <= 0:
        raise ValueError("bn_recalibration_num_samples must be a positive integer or 'all'.")
    return min(n, dataset_len)


def build_bn_calibration_loader(model: nn.Module, args, cfg: dict, recal_cfg: BNRecalibrationConfig) -> DataLoader:
    transform = _build_eval_transform(model, args.dataset, cfg)
    if args.dataset == "tinyimagenet":
        train_dataset, _ = build_tinyimagenet_datasets(
            args.data_dir,
            train_transform=transform,
            validate_counts=True,
        )
    elif args.dataset in {"cifar10", "cifar100"}:
        dataset_cls = datasets.CIFAR10 if args.dataset == "cifar10" else datasets.CIFAR100
        train_dataset = dataset_cls(
            root=args.data_dir,
            train=True,
            download=True,
            transform=transform,
        )
    else:
        raise ValueError("BN recalibration supports CIFAR and Tiny ImageNet train splits only.")

    n = _parse_num_samples(recal_cfg.num_samples, len(train_dataset))
    if n < len(train_dataset):
        g = torch.Generator()
        g.manual_seed(recal_cfg.subset_seed)
        indices = torch.randperm(len(train_dataset), generator=g)[:n].tolist()
        train_dataset = Subset(train_dataset, indices)

    return DataLoader(
        UnlabeledDataset(train_dataset),
        batch_size=recal_cfg.batch_size,
        shuffle=False,
        num_workers=recal_cfg.num_workers,
        pin_memory=args.pin_memory,
        drop_last=False,
    )


def tensor_sha256(x: torch.Tensor) -> str:
    arr = x.detach().cpu().contiguous().numpy()
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode("utf-8"))
    h.update(str(tuple(arr.shape)).encode("utf-8"))
    h.update(arr.tobytes())
    return h.hexdigest()


def combined_hash(named_tensors: Iterable[tuple[str, torch.Tensor]]) -> str:
    h = hashlib.sha256()
    for name, tensor in sorted(named_tensors, key=lambda item: item[0]):
        h.update(name.encode("utf-8"))
        h.update(tensor_sha256(tensor).encode("utf-8"))
    return h.hexdigest()


def _name_to_module_map(model: nn.Module, include_buffers: bool = False) -> Dict[str, nn.Module]:
    out = {}
    for module_name, module in model.named_modules():
        iterator = module.named_buffers(recurse=False) if include_buffers else module.named_parameters(recurse=False)
        for local_name, _ in iterator:
            full_name = f"{module_name}.{local_name}" if module_name else local_name
            out[full_name] = module
    return out


def conv_linear_classifier_tensors(model: nn.Module) -> List[tuple[str, torch.Tensor]]:
    param_to_module = _name_to_module_map(model)
    out = []
    for name, p in model.named_parameters():
        module = param_to_module.get(name)
        name_l = name.lower()
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)) or any(
            token in name_l for token in ("classifier", "fc", "head")
        ):
            out.append((name, p))
    return out


def bn_affine_tensors(model: nn.Module) -> List[tuple[str, torch.Tensor]]:
    param_to_module = _name_to_module_map(model)
    out = []
    for name, p in model.named_parameters():
        if isinstance(param_to_module.get(name), BN_TYPES):
            out.append((name, p))
    return out


def all_parameter_tensors(model: nn.Module) -> List[tuple[str, torch.Tensor]]:
    return list(model.named_parameters())


def _snapshot_hashes(named_tensors: Iterable[tuple[str, torch.Tensor]]) -> Dict[str, str]:
    return {name: tensor_sha256(tensor) for name, tensor in named_tensors}


def _assert_hashes_unchanged(before: Dict[str, str], after: Dict[str, str], label: str):
    if before != after:
        changed = sorted(k for k in before.keys() & after.keys() if before[k] != after[k])
        missing = sorted(before.keys() - after.keys())
        extra = sorted(after.keys() - before.keys())
        raise AssertionError(
            f"{label} changed during BN recalibration. changed={changed[:10]} missing={missing[:10]} extra={extra[:10]}"
        )


def _is_allowed_bn_recal_buffer(name: str, module: Optional[nn.Module]) -> bool:
    local_name = name.rsplit(".", 1)[-1]
    return isinstance(module, BN_TYPES) and local_name in {"running_mean", "running_var", "num_batches_tracked"}


def _assert_only_bn_buffers_changed(model: nn.Module, before_state: Dict[str, torch.Tensor]) -> List[str]:
    buffer_to_module = _name_to_module_map(model, include_buffers=True)
    changed_allowed = []
    with torch.no_grad():
        for name, buf in model.named_buffers():
            before = before_state[name].to(device=buf.device, dtype=buf.dtype if buf.is_floating_point() else buf.dtype)
            changed = not torch.equal(buf.detach(), before)
            if changed and not _is_allowed_bn_recal_buffer(name, buffer_to_module.get(name)):
                raise AssertionError(f"Unexpected buffer changed during BN recalibration: {name}")
            if changed:
                changed_allowed.append(name)
    return changed_allowed


def _assert_non_bn_modules_eval(model: nn.Module):
    bad = []
    for name, module in model.named_modules():
        if name and not isinstance(module, BN_TYPES) and module.training:
            bad.append(name)
    if bad:
        raise AssertionError(f"Non-BN modules entered training mode during BN recalibration: {bad[:10]}")


def _write_bn_recalibration_diagnostics(
    recal_cfg: BNRecalibrationConfig,
    model_arch: str,
    noise_level: float,
    trial: int,
    report: BNRecalibrationReport,
):
    os.makedirs(recal_cfg.diagnostics_dir, exist_ok=True)
    path = os.path.join(recal_cfg.diagnostics_dir, "bn_recalibration.jsonl")
    row = {
        "model": model_arch,
        "noise_level": noise_level,
        "trial": trial,
        "num_bn_modules": report.num_bn_modules,
        "num_calibration_samples": report.num_calibration_samples,
        "changed_bn_buffers": report.changed_bn_buffers,
        "conv_linear_classifier_hash": report.conv_linear_classifier_hash,
        "bn_affine_hash": report.bn_affine_hash,
        "fixed_noisy_param_hash": report.fixed_noisy_param_hash,
    }
    with open(path, "a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
    report.diagnostics_path = path


def recalibrate_batchnorm_statistics(
    model: nn.Module,
    calibration_loader: DataLoader,
    device: torch.device,
    recal_cfg: BNRecalibrationConfig,
    model_arch: str,
    noise_level: float,
    trial: int,
) -> BNRecalibrationReport:
    bn_modules = [(name, module) for name, module in model.named_modules() if isinstance(module, BN_TYPES)]
    if len(bn_modules) == 0:
        log.warning("BN recalibration requested for %s, but the model has no BatchNorm modules; no-op.", model_arch)
        report = BNRecalibrationReport(
            num_bn_modules=0,
            num_calibration_samples=0,
            changed_bn_buffers=[],
            conv_linear_classifier_hash=combined_hash(conv_linear_classifier_tensors(model)),
            bn_affine_hash=combined_hash(bn_affine_tensors(model)),
            fixed_noisy_param_hash=combined_hash(all_parameter_tensors(model)),
        )
        _write_bn_recalibration_diagnostics(recal_cfg, model_arch, noise_level, trial, report)
        model.eval()
        return report

    before_conv_linear_classifier = _snapshot_hashes(conv_linear_classifier_tensors(model))
    before_bn_affine = _snapshot_hashes(bn_affine_tensors(model))
    before_all_params = _snapshot_hashes(all_parameter_tensors(model))
    before_buffers = {name: buf.detach().clone() for name, buf in model.named_buffers()}
    original_requires_grad = {name: p.requires_grad for name, p in model.named_parameters()}
    original_momentum = {name: module.momentum for name, module in bn_modules}

    model.eval()
    for _, p in model.named_parameters():
        p.requires_grad_(False)

    for name, module in bn_modules:
        module.reset_running_stats()
        module.momentum = None
        module.train()

    _assert_non_bn_modules_eval(model)

    sample_count = 0
    with torch.no_grad():
        for inputs in calibration_loader:
            inputs = inputs.to(device, non_blocking=True)
            _ = model(inputs)
            sample_count += inputs.size(0)

    for name, module in bn_modules:
        module.momentum = original_momentum[name]
    for name, p in model.named_parameters():
        p.requires_grad_(original_requires_grad[name])
    model.eval()

    after_conv_linear_classifier = _snapshot_hashes(conv_linear_classifier_tensors(model))
    after_bn_affine = _snapshot_hashes(bn_affine_tensors(model))
    after_all_params = _snapshot_hashes(all_parameter_tensors(model))
    _assert_hashes_unchanged(before_conv_linear_classifier, after_conv_linear_classifier, "Conv/linear/classifier parameters")
    _assert_hashes_unchanged(before_bn_affine, after_bn_affine, "BN affine parameters")
    _assert_hashes_unchanged(before_all_params, after_all_params, "Fixed noisy parameter state")
    changed_bn_buffers = _assert_only_bn_buffers_changed(model, before_buffers)
    _assert_non_bn_modules_eval(model)

    report = BNRecalibrationReport(
        num_bn_modules=len(bn_modules),
        num_calibration_samples=sample_count,
        changed_bn_buffers=changed_bn_buffers,
        conv_linear_classifier_hash=combined_hash(conv_linear_classifier_tensors(model)),
        bn_affine_hash=combined_hash(bn_affine_tensors(model)),
        fixed_noisy_param_hash=combined_hash(all_parameter_tensors(model)),
    )
    _write_bn_recalibration_diagnostics(recal_cfg, model_arch, noise_level, trial, report)
    return report


@torch.no_grad()
def evaluate_once(
    model: nn.Module,
    test_dataloader: DataLoader,
    device: torch.device,
    use_amp: bool = False,
    max_batches: Optional[int] = None,
) -> float:
    model.eval()
    total = 0
    correct = 0

    total_batches = len(test_dataloader) if max_batches is None else min(len(test_dataloader), max_batches)
    pbar = tqdm(enumerate(test_dataloader), total=total_batches, disable=False)
    autocast_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16

    for batch_idx, (inputs, targets) in pbar:
        if max_batches is not None and batch_idx >= max_batches:
            break
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.autocast(
            device_type=device.type,
            dtype=autocast_dtype,
            enabled=use_amp and device.type in ["cuda", "cpu"],
        ):
            outputs = model(inputs)

        preds = outputs.argmax(dim=1)
        correct += preds.eq(targets).sum().item()
        total += targets.size(0)
        accuracy = 100.0 * correct / max(total, 1)
        pbar.set_description(f"Acc {accuracy:.2f}%")

    return 100.0 * correct / max(total, 1)


def save_results_csv(csv_path: str, rows: List[Dict]):
    if len(rows) == 0:
        return
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    log.warning("Using device: %s", device)

    bn_recalibration_cfg = parse_bn_recalibration_config(args)
    if bn_recalibration_cfg.enabled and args.noise_to_norm:
        raise ValueError("BN recalibration is defined for noise_to_norm=False only.")

    model_list = [x.strip() for x in args.model_list.split(",") if x.strip()]
    noise_level_list_ = [float(x.strip()) for x in args.noise_level_list.split(",") if x.strip()]
    checkpoint_map = parse_checkpoint_map(args.checkpoint_map)
    num_classes = infer_num_classes(args.dataset, args.num_classes)

    all_rows = []
    all_noise_acc_spec = {}
    audit_primary_rows = []
    audit_nonprimary_rows = []

    for model_idx, model_arch in enumerate(model_list):
        log.warning("=" * 120)
        log.warning("Evaluating model: %s", model_arch)

        cfg = get_baseline_config(
            model_name=model_arch,
            pretrained=args.pretrained,
            case=args.case,
            prefer_resize=args.prefer_resize,
            extra_overrides=None,
        )
        if args.dataset == "tinyimagenet":
            case_name = cfg["case"]
            model_name = cfg["model_name"]
            cfg.update(TINYIMAGENET_DEFAULTS)
            cfg["case"] = case_name
            cfg["model_name"] = model_name

        model = build_model(model_arch, cfg, num_classes=num_classes).to(device)
        ckpt_path = resolve_checkpoint_path(model_arch, checkpoint_map, args.checkpoint_dir)
        load_model_weights(model, ckpt_path, device)
        if args.fold_norm:
            model = maybe_fold_norms(model, mode=args.fold_norm_mode).to(device)
        test_dataloader = build_test_loader(model, args, cfg)
        calibration_dataloader = None
        if bn_recalibration_cfg.enabled:
            calibration_dataloader = build_bn_calibration_loader(model, args, cfg, bn_recalibration_cfg)

        excluded_param_names = set(getattr(model, "_mismatch_excluded_param_names", set()))
        if args.exclude_conv_bias_from_mismatch:
            excluded_param_names.update(convolution_bias_parameter_names(model))

        mismatch_helper = FixedMismatchHelper(
            model=model,
            noise_sigma=0.0,
            noise_type=args.noise_type,
            noise_to_norm=args.noise_to_norm,
            include_buffers=True,
            exclude_param_names=excluded_param_names,
            additive_scale_mode=args.additive_scale_mode,
        )
        mismatch_helper.snapshot_clean_state()

        if args.weight_range_audit_csv:
            primary_rows, nonprimary_rows = collect_wrn_weight_range_audit_rows(
                model,
                dataset=args.dataset,
                model_name=model_arch,
                should_noise_param=mismatch_helper._should_noise_param,
                param_to_module=mismatch_helper._param_to_module,
            )
            audit_primary_rows.extend(primary_rows)
            audit_nonprimary_rows.extend(nonprimary_rows)

        noise_acc_spec = {}

        for noise_level in noise_level_list_:
            trials = args.noisy_trials if noise_level > 0 else 1
            acc_list = []

            for t in range(trials):
                log.warning("Model=%s | noise_level=%g | trial=%d/%d", model_arch, noise_level, t + 1, trials)
                mismatch_helper.restore_clean_state()
                mismatch_helper.noise_sigma = noise_level
                mismatch_helper.seed = args.seed + 100000 * model_idx + 1000 * int(round(noise_level * 1e6)) + t

                if noise_level > 0:
                    summary = mismatch_helper.add_noise()
                    log.warning("Applied mismatch to %d tensors.", len(summary.applied_records))

                if bn_recalibration_cfg.enabled:
                    assert calibration_dataloader is not None
                    report = recalibrate_batchnorm_statistics(
                        model=model,
                        calibration_loader=calibration_dataloader,
                        device=device,
                        recal_cfg=bn_recalibration_cfg,
                        model_arch=model_arch,
                        noise_level=noise_level,
                        trial=t,
                    )
                    log.warning(
                        "BN recalibration finished: bn_modules=%d samples=%d changed_bn_buffers=%d diagnostics=%s",
                        report.num_bn_modules,
                        report.num_calibration_samples,
                        len(report.changed_bn_buffers),
                        report.diagnostics_path,
                    )

                accuracy = evaluate_once(
                    model,
                    test_dataloader,
                    device=device,
                    use_amp=args.use_amp,
                    max_batches=args.max_eval_batches,
                )
                log.warning("Test Accuracy at noise level %g: %.2f%%", noise_level, accuracy)
                acc_list.append(float(accuracy))

            avg_acc = sum(acc_list) / len(acc_list)
            std_acc = float(np.std(acc_list))
            _nl_key = (model_arch, noise_level)
            noise_acc_spec[_nl_key] = acc_list

            all_rows.append({
                "model": model_arch,
                "noise_level": noise_level,
                "acc": f"{avg_acc:.2f}±{std_acc:.2f}%",
                "noise_type": args.noise_type,
                "additive_scale_mode": args.additive_scale_mode,
                "noise_to_norm": args.noise_to_norm,
                "exclude_conv_bias_from_mismatch": args.exclude_conv_bias_from_mismatch,
            })

        for _nl, _acc in noise_acc_spec.items():
            _model_name, _noise_level = _nl
            log.warning(
                "Model: %s Noise level: %g, Acc:%.2f±%.2f%%",
                _model_name,
                _noise_level,
                sum(_acc) / len(_acc),
                np.std(_acc),
            )

        all_noise_acc_spec.update(noise_acc_spec)

    # This is the model_save_name, like resize_finetune_cifar10_resmlp_12_224
    run_name = infer_run_name_from_checkpoint_map(checkpoint_map)
    if not run_name:
        run_name = "_".join(model_list)

    run_name = run_name.replace("/", "_").replace(":", "_").replace(" ", "_")
    if bn_recalibration_cfg.enabled:
        run_name = f"{run_name}_bn_recal"

    csv_path = os.path.join(args.results_dir, f"summary_{run_name}.csv")
    pkl_path = os.path.join(args.results_dir, f"noise_acc_spec_{run_name}.pkl")

    save_results_csv(csv_path, all_rows)
    with open(pkl_path, "wb") as f:
        pickle.dump(all_noise_acc_spec, f)
    if args.weight_range_audit_csv:
        save_audit_csv(args.weight_range_audit_csv, audit_primary_rows, audit_nonprimary_rows)

    print("\n=== Final Summary ===")
    for row in all_rows:
        print(
            "Model={model:20s} noise={noise_level:<8g} acc={acc} noise_type={noise_type} noise_to_norm={noise_to_norm} exclude_conv_bias={exclude_conv_bias_from_mismatch}".format(**row)
        )
    print(f"\nSaved CSV to: {csv_path}")
    print(f"Saved pickle to: {pkl_path}")
    if args.weight_range_audit_csv:
        print(f"Saved weight-range audit CSV to: {args.weight_range_audit_csv}")


if __name__ == "__main__":
    main()

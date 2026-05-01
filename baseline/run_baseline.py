#!/usr/bin/env python3
import argparse
import csv
import logging
import os
import pickle
import random
import sys
from pathlib import Path
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import timm
import torch
import torch.nn as nn
from timm.data import create_transform, resolve_model_data_config, create_loader
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import baseline.cifar_resnet  # registers custom CIFAR models into timm
from baseline.baseline_cifar_configs import get_baseline_config, build_model
from trainer import _CIFAR_STATS


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
    ):
        self.model = model
        self.noise_sigma = float(noise_sigma)
        self.noise_type = noise_type
        self.noise_to_norm = bool(noise_to_norm)
        self.include_buffers = bool(include_buffers)
        self.seed = seed

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
            x.add_(self.noise_sigma * noise)
        else:
            raise ValueError(f"Unsupported noise_type: {self.noise_type}")

    def _should_noise_param(self, name: str, p: torch.Tensor) -> bool:
        if not p.is_floating_point():
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
                    if all_zero and self.noise_type == "multiplicative":
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
                    changed = not torch.equal(buf.detach(), clean)
                    if all_zero and self.noise_type == "multiplicative":
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
            if rec.all_zero and self.noise_type == "multiplicative":
                log.warning(
                    "Skipping strict assertion for all-zero %s tensor under multiplicative mismatch: %s",
                    rec.kind,
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
    parser.add_argument("--dataset", type=str, choices=["cifar10", "cifar100", "imagenet"], required=True)
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
    parser.add_argument("--noise_to_norm", type=str2bool, default=False)
    parser.add_argument("--fold_norm", type=str2bool, default=False)
    parser.add_argument("--results_dir", type=str, default="./results_mismatch_eval")
    parser.add_argument("--pin_memory", type=str2bool, default=False)
    parser.add_argument("--use_amp", type=str2bool, default=False)
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
    if dataset_name == "imagenet":
        return 1000
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def maybe_fold_norms(model: nn.Module) -> nn.Module:
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


def build_test_loader(model: nn.Module, args, cfg: dict) -> DataLoader:
    if args.dataset in {"cifar10", "cifar100"}:
        mean, std = _CIFAR_STATS[args.dataset]

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
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        else:
            test_transform = create_transform(**test_kwargs)

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

    elif args.dataset == "imagenet":
        data_config = resolve_model_data_config(model)
        test_transform = create_transform(**data_config, is_training=False)
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


@torch.no_grad()
def evaluate_once(model: nn.Module, test_dataloader: DataLoader, device: torch.device, use_amp: bool = False) -> float:
    model.eval()
    total = 0
    correct = 0

    pbar = tqdm(enumerate(test_dataloader), total=len(test_dataloader), disable=False)
    autocast_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16

    for _, (inputs, targets) in pbar:
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

    model_list = [x.strip() for x in args.model_list.split(",") if x.strip()]
    noise_level_list_ = [float(x.strip()) for x in args.noise_level_list.split(",") if x.strip()]
    checkpoint_map = parse_checkpoint_map(args.checkpoint_map)
    num_classes = infer_num_classes(args.dataset, args.num_classes)

    all_rows = []
    all_noise_acc_spec = {}

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

        model = build_model(model_arch, cfg, num_classes=num_classes).to(device)
        ckpt_path = resolve_checkpoint_path(model_arch, checkpoint_map, args.checkpoint_dir)
        load_model_weights(model, ckpt_path, device)
        test_dataloader = build_test_loader(model, args, cfg)

        mismatch_helper = FixedMismatchHelper(
            model=model,
            noise_sigma=0.0,
            noise_type=args.noise_type,
            noise_to_norm=args.noise_to_norm,
            include_buffers=True,
        )
        mismatch_helper.snapshot_clean_state()

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

                accuracy = evaluate_once(model, test_dataloader, device=device, use_amp=args.use_amp)
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
                "noise_to_norm": args.noise_to_norm,
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

    csv_path = os.path.join(args.results_dir, f"summary_{run_name}.csv")
    pkl_path = os.path.join(args.results_dir, f"noise_acc_spec_{run_name}.pkl")

    save_results_csv(csv_path, all_rows)
    with open(pkl_path, "wb") as f:
        pickle.dump(all_noise_acc_spec, f)

    print("\n=== Final Summary ===")
    for row in all_rows:
        print(
            "Model={model:20s} noise={noise_level:<8g} acc={acc} noise_type={noise_type} noise_to_norm={noise_to_norm}".format(**row)
        )
    print(f"\nSaved CSV to: {csv_path}")
    print(f"Saved pickle to: {pkl_path}")


if __name__ == "__main__":
    main()

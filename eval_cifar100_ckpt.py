#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from collections import OrderedDict
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

_CIFAR_STATS = {
    "cifar100": ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a CIFAR-100 checkpoint on the original CIFAR-100 test set."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=str,
        help="path to a .pth checkpoint or state_dict",
    )
    parser.add_argument(
        "--arch",
        default="efficientnet_v2_l",
        choices=("efficientnet_v2_s", "efficientnet_v2_m", "efficientnet_v2_l"),
        help="EfficientNetV2 variant name",
    )
    parser.add_argument(
        "--arch_source",
        default="auto",
        choices=("auto", "torchvision", "hankyul2"),
        help="model implementation to use (auto=detect from checkpoint keys)",
    )
    parser.add_argument("--data_root", default="./data", type=str, help="CIFAR-100 root directory")
    parser.add_argument("--batch_size", default=128, type=int, help="batch size for evaluation")
    parser.add_argument("--num_workers", default=4, type=int, help="dataloader workers")
    parser.add_argument(
        "--resize",
        default=224,
        type=int,
        help="image size for resize+center crop (set 0 to disable)",
    )
    parser.add_argument(
        "--center_crop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="apply center crop after resize",
    )
    parser.add_argument(
        "--device",
        default=None,
        type=str,
        help="device to use (e.g. cuda, cuda:0, cpu); defaults to auto",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="use autocast for faster eval on GPU",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="enforce exact checkpoint match",
    )
    parser.add_argument(
        "--download",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="download CIFAR-100 if missing",
    )
    return parser.parse_args()


def load_checkpoint_state_dict(checkpoint_path: Path) -> OrderedDict[str, torch.Tensor]:
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    if isinstance(checkpoint, nn.Module):
        checkpoint = checkpoint.state_dict()
    if isinstance(checkpoint, dict):
        for key in ("net", "state_dict", "model", "model_state"):
            if key in checkpoint:
                nested = checkpoint[key]
                if isinstance(nested, nn.Module):
                    nested = nested.state_dict()
                checkpoint = nested
                break
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint at {checkpoint_path} does not contain a state dict.")
    state_dict = OrderedDict(checkpoint)
    if any(key.startswith("module.") for key in state_dict):
        state_dict = OrderedDict(
            (key.replace("module.", "", 1), value) for key, value in state_dict.items()
        )
    if any(key.startswith("model.") for key in state_dict):
        state_dict = OrderedDict(
            (key.replace("model.", "", 1), value) for key, value in state_dict.items()
        )
    return state_dict


def infer_checkpoint_source(state_dict: OrderedDict[str, torch.Tensor]) -> str:
    has_stem = any(key.startswith("stem.") for key in state_dict)
    has_blocks = any(key.startswith("blocks.") for key in state_dict)
    if has_stem and has_blocks:
        return "hankyul2"
    if any(key.startswith("features.") for key in state_dict):
        return "torchvision"
    return "unknown"


def resolve_arch_source(
    requested: str,
    state_dict: OrderedDict[str, torch.Tensor] | None,
) -> str:
    if requested != "auto":
        return requested
    if state_dict is None:
        return "torchvision"
    inferred = infer_checkpoint_source(state_dict)
    return inferred if inferred != "unknown" else "torchvision"


def build_torchvision_model(arch: str, num_classes: int) -> nn.Module:
    builder = getattr(torchvision.models, arch, None)
    if builder is None:
        raise ValueError(f"{arch} is not available in this torchvision build.")
    model = builder(weights=None)
    classifier = getattr(model, "classifier", None)
    if isinstance(classifier, nn.Sequential):
        classifier[-1] = nn.Linear(classifier[-1].in_features, num_classes)
    elif isinstance(classifier, nn.Linear):
        model.classifier = nn.Linear(classifier.in_features, num_classes)
    else:
        raise RuntimeError("Unexpected classifier head for EfficientNetV2.")
    return model


def build_hankyul_model(arch: str, num_classes: int) -> nn.Module:
    return torch.hub.load(
        "hankyul2/EfficientNetV2-pytorch",
        arch,
        nclass=num_classes,
        skip_validation=True,
    )


def build_test_loader(
    data_root: str,
    batch_size: int,
    num_workers: int,
    resize: int | None,
    center_crop: bool,
    download: bool,
) -> DataLoader:
    mean, std = _CIFAR_STATS["cifar100"]
    transform_steps = []
    if resize and resize > 0:
        transform_steps.append(transforms.Resize(resize))
        if center_crop:
            transform_steps.append(transforms.CenterCrop(resize))
    transform_steps.extend([transforms.ToTensor(), transforms.Normalize(mean, std)])
    transform = transforms.Compose(transform_steps)
    dataset = torchvision.datasets.CIFAR100(
        root=data_root,
        train=False,
        download=download,
        transform=transform,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, use_amp: bool) -> tuple[float, float]:
    model.eval()
    total = 0
    correct = 0
    correct_top5 = 0
    autocast_ctx = (
        torch.cuda.amp.autocast(enabled=True) if use_amp and device.type == "cuda" else nullcontext()
    )

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            with autocast_ctx:
                outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            max_k = min(5, outputs.size(1))
            topk = outputs.topk(max_k, dim=1).indices
            correct_top5 += topk.eq(targets.view(-1, 1)).any(dim=1).sum().item()

    top1 = 100.0 * correct / max(total, 1)
    top5 = 100.0 * correct_top5 / max(total, 1)
    return top1, top5


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    checkpoint_path = Path(args.checkpoint).expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    state_dict = load_checkpoint_state_dict(checkpoint_path)
    arch_source = resolve_arch_source(args.arch_source, state_dict)
    if args.arch_source == "auto" and arch_source != "torchvision":
        logging.info("Detected %s checkpoint; using %s model implementation.", arch_source, arch_source)

    num_classes = 100
    if arch_source == "hankyul2":
        model = build_hankyul_model(args.arch, num_classes)
    else:
        model = build_torchvision_model(args.arch, num_classes)

    missing, unexpected = model.load_state_dict(state_dict, strict=args.strict)
    if not args.strict and (missing or unexpected):
        logging.warning("State dict load: missing=%s unexpected=%s", missing, unexpected)

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    model.to(device)

    loader = build_test_loader(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        resize=args.resize,
        center_crop=args.center_crop,
        download=args.download,
    )
    top1, top5 = evaluate(model, loader, device, args.amp)
    logging.info("CIFAR-100 Top1: %.2f%% | Top5: %.2f%%", top1, top5)


if __name__ == "__main__":
    main()

'''Train CIFAR teacher with PyTorch.'''
from __future__ import annotations

import argparse
import json
import logging
import os
import random
from collections import OrderedDict
from pathlib import Path
from typing import Sequence, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.models as models

from scangen.data import MyNoiseCIFARDataset
from utils import progress_bar


RAW_MEAN = (0.5, 0.5, 0.5, 0.5)
RAW_STD = (0.5, 0.5, 0.5, 0.5)

def _normalize_dataset_name(dataset_name: str | None) -> str:
    name = (dataset_name or "cifar10").lower().replace("-", "").replace("_", "")
    if name in {"cifar10", "cifar100"}:
        return name
    raise ValueError(f"Unsupported dataset: {dataset_name}. Use cifar10 or cifar100.")


def _default_cifar_dir(dataset_name: str) -> str:
    name = _normalize_dataset_name(dataset_name)
    return "cifar-10-data" if name == "cifar10" else "cifar-100-data"


def _get_efficientnet_v2_builder(arch: str):
    if not arch.startswith("efficientnet_v2_"):
        raise ValueError(f"Unsupported EfficientNetV2 arch: {arch}")
    builder = getattr(models, arch, None)
    if builder is None:
        raise ImportError(f"{arch} is not available in this torchvision build.")
    return builder


def _get_efficientnet_v2_weights(arch: str):
    suffix = arch.split("_")[-1].upper()
    weights_name = f"EfficientNet_V2_{suffix}_Weights"
    weights_enum = getattr(models, weights_name, None)
    if weights_enum is None:
        raise ImportError(
            f"{weights_name} is not available; upgrade torchvision to use EfficientNetV2 weights."
        )
    if hasattr(weights_enum, "IMAGENET1K_V1"):
        return weights_enum.IMAGENET1K_V1
    if hasattr(weights_enum, "DEFAULT"):
        return weights_enum.DEFAULT
    raise RuntimeError(f"No ImageNet weights found for {arch}.")


def _replace_first_conv(model: nn.Module, in_channels: int) -> None:
    first_conv_name, first_conv = _find_first_conv(model)
    if first_conv.in_channels == in_channels:
        return
    new_conv = nn.Conv2d(
        in_channels,
        first_conv.out_channels,
        kernel_size=first_conv.kernel_size,
        stride=first_conv.stride,
        padding=first_conv.padding,
        dilation=first_conv.dilation,
        groups=first_conv.groups,
        bias=first_conv.bias is not None,
        padding_mode=first_conv.padding_mode,
    )
    with torch.no_grad():
        if in_channels >= first_conv.in_channels:
            new_conv.weight[:, :first_conv.in_channels].copy_(first_conv.weight)
            extra = first_conv.weight.mean(dim=1, keepdim=True)
            repeat = in_channels - first_conv.in_channels
            if repeat > 0:
                new_conv.weight[:, first_conv.in_channels:].copy_(extra.repeat(1, repeat, 1, 1))
        else:
            new_conv.weight.copy_(first_conv.weight[:, :in_channels])
        if first_conv.bias is not None:
            new_conv.bias.copy_(first_conv.bias)
    parent = model
    path, attr = first_conv_name.rsplit(".", 1) if "." in first_conv_name else ("", first_conv_name)
    if path:
        for part in path.split("."):
            parent = getattr(parent, part)
    setattr(parent, attr, new_conv)


def _find_first_conv(model: nn.Module) -> tuple[str, nn.Conv2d]:
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            return name, module
    raise RuntimeError("No Conv2d layer found to update input channels.")


def load_efficientnet_v2_4ch(
    num_classes: int,
    arch: str,
    pretrained: bool = True,
    in_channels: int = 4,
) -> nn.Module:
    """Load EfficientNetV2 with ImageNet weights adapted to four input channels."""
    builder = _get_efficientnet_v2_builder(arch)
    weights = _get_efficientnet_v2_weights(arch) if pretrained else None
    model = builder(weights=weights)
    _replace_first_conv(model, in_channels)
    classifier = getattr(model, "classifier", None)
    if isinstance(classifier, nn.Sequential):
        classifier[-1] = nn.Linear(classifier[-1].in_features, num_classes)
    elif isinstance(classifier, nn.Linear):
        model.classifier = nn.Linear(classifier.in_features, num_classes)
    else:
        raise RuntimeError("Unexpected classifier head for EfficientNetV2.")
    return model


def _map_arch_to_hankyul(arch: str) -> str:
    """Map torchvision-style architecture names to hankyul2 naming."""
    # hankyul2 uses different naming conventions
    arch_map = {
        'efficientnet_v2_s': 'efficientnet_v2_s',
        'efficientnet_v2_m': 'efficientnet_v2_m',
        'efficientnet_v2_l': 'efficientnet_v2_l',
        'efficientnet_v2_b4': 'efficientnet_v2_b4',  # May need adjustment based on actual hub naming
    }
    return arch_map.get(arch, arch)


def load_hankyul_efficientnet_v2_4ch(
    num_classes: int,
    arch: str,
    in_channels: int = 4,
) -> nn.Module:
    """Load EfficientNetV2 from hankyul2/EfficientNetV2-pytorch and adapt input channels."""
    hankyul_arch = _map_arch_to_hankyul(arch)
    try:
        model = torch.hub.load(
            "hankyul2/EfficientNetV2-pytorch",
            hankyul_arch,
            nclass=num_classes,
            skip_validation=True,
        )
    except Exception as e:
        # If the mapped name fails, try the original name
        if hankyul_arch != arch:
            logging.warning("Failed to load with mapped arch %s, trying original %s", hankyul_arch, arch)
            model = torch.hub.load(
                "hankyul2/EfficientNetV2-pytorch",
                arch,
                nclass=num_classes,
                skip_validation=True,
            )
        else:
            raise
    _replace_first_conv(model, in_channels)
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train EfficientNetV2 teacher on noisy CIFAR')
    parser.add_argument(
        '--arch',
        default='efficientnet_v2_s',
        choices=('efficientnet_v2_s', 'efficientnet_v2_m', 'efficientnet_v2_l', 'efficientnet_v2_b4'),
        help='EfficientNetV2 variant to train (torchvision naming, or b4 for hankyul2).',
    )
    parser.add_argument(
        '--arch_source',
        default='auto',
        choices=('auto', 'torchvision', 'hankyul2'),
        help='model implementation to use (auto=detect from init/test checkpoint).',
    )
    parser.add_argument(
        '--no_pretrained',
        action='store_true',
        help='disable ImageNet pretrained initialization',
    )
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--gamma', default=0.1, type=float, help='lr decay factor')
    parser.add_argument('--wd', default=1e-6, type=float, help='weight decay')
    parser.add_argument('--ne', default=30, type=int, help='number of epochs')
    parser.add_argument('--nsc', default=10, type=int, help='number of epochs per LR step')
    parser.add_argument('--batch_split', default=1, type=int, help='gradient accumulation factor')
    parser.add_argument('--batch', default=32, type=int, help='effective batch size')
    parser.add_argument('--alpha', default=0.1, type=float, help='mixup interpolation coefficient')
    parser.add_argument('--dataset', default='cifar10', choices=('cifar10', 'cifar100'),
                        help='dataset name (cifar10 or cifar100)')
    parser.add_argument('--train_size', default=160, type=int,
                        help='input resolution for training transforms')
    parser.add_argument('--test_size', default=200, type=int,
                        help='input resolution for test transforms')
    parser.add_argument('--train_transform', default='rrc', choices=('rrc', 'cifar'),
                        help='training augmentation preset (rrc=RandomResizedCrop, cifar=resize+pad+crop)')
    parser.add_argument('--test_center_crop', action='store_true',
                        help='apply center crop for test transforms')
    parser.add_argument('--root', default=None, type=str,
                        help='root directory containing noise data (defaults to SCANGEN_DATA_ROOT)')
    parser.add_argument('--noise_config', default='scangen/config.json', type=str,
                        help='path to scangen noise config JSON')
    parser.add_argument('--checkpoint', default='./checkpoint/ckpt.pth', type=str,
                        help='file path to save the trained teacher checkpoint')
    parser.add_argument('--init_checkpoint', default=None, type=str,
                        help='optional checkpoint to load before training (fine-tuning)')
    parser.add_argument('--mismatch_levels', nargs='*', default=None,
                        help='one or more non-negative mismatch noise levels (e.g. 0.05 0.1)')
    parser.add_argument('--mismatch_type', default='mul', choices=('mul',),
                        help='mismatch noise type (multiplicative only)')
    parser.add_argument('--mismatch_ramp_start', type=float, default=None,
                        help='starting mismatch level for linear ramp (defaults to first mismatch level)')
    parser.add_argument('--mismatch_ramp_epochs', type=int, default=0,
                        help='epochs over which to linearly ramp from start to target mismatch level')
    parser.add_argument('--num_workers', default=1, type=int, help='number of dataloader workers')
    parser.add_argument('--test_only', action='store_true',
                        help='skip training and only evaluate an existing checkpoint')
    return parser.parse_args()


def load_checkpoint_state_dict(checkpoint_path: Path, device: torch.device) -> OrderedDict[str, torch.Tensor]:
    """Load a checkpoint file into an OrderedDict state dict."""
    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    if isinstance(checkpoint, nn.Module):
        checkpoint = checkpoint.state_dict()
    if isinstance(checkpoint, dict):
        for key in ('net', 'state_dict', 'model', 'model_state'):
            if key in checkpoint:
                nested = checkpoint[key]
                if isinstance(nested, nn.Module):
                    nested = nested.state_dict()
                checkpoint = nested
                break
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint at {checkpoint_path} does not contain a state dict")
    state_dict = checkpoint
    if any(key.startswith('module.') for key in state_dict):
        state_dict = OrderedDict(
            (key.replace('module.', '', 1), value) for key, value in state_dict.items()
        )
    else:
        state_dict = OrderedDict(state_dict)
    if any(key.startswith('model.') for key in state_dict):
        state_dict = OrderedDict(
            (key.replace('model.', '', 1), value) for key, value in state_dict.items()
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


def infer_architecture_from_checkpoint(state_dict: OrderedDict[str, torch.Tensor]) -> str | None:
    """Infer EfficientNetV2 architecture from checkpoint by counting blocks."""
    if not any(key.startswith("blocks.") for key in state_dict):
        return None
    
    # Find the maximum block index
    max_block_idx = -1
    for key in state_dict.keys():
        if key.startswith("blocks."):
            # Extract block index from key like "blocks.78.block.linear_bottleneck.0.weight"
            parts = key.split(".")
            if len(parts) >= 2 and parts[0] == "blocks":
                try:
                    block_idx = int(parts[1])
                    max_block_idx = max(max_block_idx, block_idx)
                except ValueError:
                    continue
    
    # EfficientNetV2-L has blocks up to ~78, EfficientNetV2-M up to ~54, EfficientNetV2-S up to ~38
    # EfficientNetV2-B4 (hankyul2) typically has fewer blocks
    if max_block_idx >= 70:
        return "efficientnet_v2_l"
    elif max_block_idx >= 50:
        return "efficientnet_v2_m"
    elif max_block_idx >= 30:
        return "efficientnet_v2_s"
    elif max_block_idx >= 0:
        # Could be B4 or another variant, default to B4 for hankyul2 checkpoints
        return "efficientnet_v2_b4"
    return None


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


def adapt_state_dict_input_channels(
    state_dict: OrderedDict[str, torch.Tensor],
    model: nn.Module,
) -> OrderedDict[str, torch.Tensor]:
    """Adapt checkpoint first-conv weights to match the model input channels."""
    conv_name, conv = _find_first_conv(model)
    weight_key = f"{conv_name}.weight"
    if weight_key not in state_dict:
        return state_dict
    weight = state_dict[weight_key]
    if weight.ndim != 4:
        logging.warning("Skipping input-channel adaptation for %s: expected 4D weight.", weight_key)
        return state_dict
    src_in = weight.shape[1]
    tgt_in = conv.in_channels
    if src_in == tgt_in:
        return state_dict
    if weight.shape[0] != conv.out_channels or weight.shape[2:] != tuple(conv.kernel_size):
        logging.warning(
            "Skipping input-channel adaptation for %s due to shape mismatch (%s vs %s).",
            weight_key,
            tuple(weight.shape),
            (conv.out_channels, tgt_in, *conv.kernel_size),
        )
        return state_dict
    new_weight = weight.new_zeros((weight.shape[0], tgt_in, weight.shape[2], weight.shape[3]))
    if tgt_in > src_in:
        new_weight[:, :src_in].copy_(weight)
        extra = weight.mean(dim=1, keepdim=True)
        repeat = tgt_in - src_in
        new_weight[:, src_in:].copy_(extra.repeat(1, repeat, 1, 1))
    else:
        new_weight.copy_(weight[:, :tgt_in])
    updated = OrderedDict(state_dict)
    updated[weight_key] = new_weight
    logging.info("Adapted %s input channels from %d to %d.", weight_key, src_in, tgt_in)
    return updated


def resolve_noise_data_root(root_argument: str | None, input_name: str) -> Path:
    """Locate the directory that holds the generated HDF5 noise data."""
    dataset_name = _normalize_dataset_name(input_name)
    hdf5_name = f"{dataset_name}_raw.h5"

    if root_argument:
        user_root = Path(root_argument).expanduser()
        if user_root.is_file():
            if user_root.name == hdf5_name:
                return user_root.parent
            msg = f"--root points to {user_root}, but expected {hdf5_name}"
            raise FileNotFoundError(msg)

        candidates = [
            user_root,
            user_root / "scanGFI",
            user_root / _default_cifar_dir(dataset_name) / "scanGFI",
        ]
        for candidate in candidates:
            if (candidate / hdf5_name).exists():
                return candidate
        msg = f"Could not find {hdf5_name} under provided root {root_argument}"
        raise FileNotFoundError(msg)

    env_root = os.getenv("SCANGEN_DATA_ROOT")
    if env_root:
        env_path = Path(env_root).expanduser()
        if env_path.is_file() and env_path.name == hdf5_name:
            return env_path.parent
        env_candidates = [
            env_path,
            env_path / "scanGFI",
            env_path / _default_cifar_dir(dataset_name) / "scanGFI",
        ]
        for candidate in env_candidates:
            if (candidate / hdf5_name).exists():
                return candidate
        logging.warning(
            "SCANGEN_DATA_ROOT=%s does not contain %s; falling back to defaults.",
            env_root,
            hdf5_name,
        )

    default_root = Path(__file__).resolve().parent / _default_cifar_dir(dataset_name) / "scanGFI"
    if (default_root / hdf5_name).exists():
        return default_root

    raise FileNotFoundError(
        f"Could not locate {hdf5_name}. "
        "Provide --root or set SCANGEN_DATA_ROOT to the dataset directory."
    )


def load_noise_config(config_path_str: str) -> dict:
    """Load the noise configuration used for on-the-fly noise generation."""
    candidate_paths = [Path(config_path_str).expanduser()]
    if not candidate_paths[0].is_absolute():
        candidate_paths.append((Path(__file__).resolve().parent / config_path_str).expanduser())

    for path in candidate_paths:
        if path.exists():
            with open(path, 'r', encoding='utf-8') as fp:
                config = json.load(fp)
            return config.get("noise", config)

    raise FileNotFoundError(f"Could not read noise config at {config_path_str}")


def build_transforms(args: argparse.Namespace) -> tuple[transforms.Compose, transforms.Compose]:
    if args.train_transform == "cifar":
        transform_train = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size=args.train_size),
            transforms.Pad(4, padding_mode="reflect"),
            transforms.RandomCrop(size=args.train_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(RAW_MEAN, RAW_STD),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(size=args.train_size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(RAW_MEAN, RAW_STD),
        ])

    transform_test_steps = [
        transforms.ToPILImage(),
        transforms.Resize(size=args.test_size),
    ]
    if args.test_center_crop:
        transform_test_steps.append(transforms.CenterCrop(size=args.test_size))
    transform_test_steps.extend([
        transforms.ToTensor(),
        transforms.Normalize(RAW_MEAN, RAW_STD),
    ])
    transform_test = transforms.Compose(transform_test_steps)
    return transform_train, transform_test


def create_datasets(args: argparse.Namespace) -> tuple[MyNoiseCIFARDataset, MyNoiseCIFARDataset]:
    noise_config = load_noise_config(args.noise_config)
    dataset_name = _normalize_dataset_name(args.dataset)
    noise_root = resolve_noise_data_root(args.root, dataset_name)
    transform_train, transform_test = build_transforms(args)

    train_set = MyNoiseCIFARDataset(
        root=noise_root,
        input_name=dataset_name,
        train=True,
        noise_config=noise_config,
        device=torch.device("cpu"),
        transform=transform_train,
        noisy_inp=True,
    )
    test_set = MyNoiseCIFARDataset(
        root=noise_root,
        input_name=dataset_name,
        train=False,
        noise_config=noise_config,
        device=torch.device("cpu"),
        transform=transform_test,
        noisy_inp=False,
    )
    return train_set, test_set


def mixup_data(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float,
    lam: float,
    count: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Apply mixup augmentation to the batch."""
    if count == 0:
        lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = inputs.size(0)
    index = torch.randperm(batch_size, device=device)
    mixed_inputs = lam * inputs + (1 - lam) * inputs[index, :]
    targets_a, targets_b = targets, targets[index]
    return mixed_inputs, targets_a, targets_b, lam


def mixup_criterion(criterion, pred, targets_a, targets_b, lam):
    return lam * criterion(pred, targets_a) + (1 - lam) * criterion(pred, targets_b)


def sanitize_mismatch_levels(raw_levels: Sequence[float | str] | None) -> list[float]:
    """Parse and validate mismatch noise levels."""
    if not raw_levels:
        return []
    parsed_values: list[float] = []
    for entry in raw_levels:
        if isinstance(entry, str):
            pieces = [piece.strip() for piece in entry.split(',')]
            parsed_values.extend(float(piece) for piece in pieces if piece)
        else:
            parsed_values.append(float(entry))
    if not parsed_values:
        return []
    if any(level < 0.0 for level in parsed_values):
        raise ValueError("Mismatch levels must be non-negative.")
    # Preserve order while removing duplicates
    return list(dict.fromkeys(parsed_values))


class WrappedNoisyModel(nn.Module):
    """Inject multiplicative weight noise during training to emulate device mismatch."""

    def __init__(
        self,
        model: nn.Module,
        noise_levels: Sequence[float],
        noise_type: str = "mul",
    ) -> None:
        super().__init__()
        if not noise_levels:
            raise ValueError("noise_levels must contain at least one value.")
        noise_type_lower = (noise_type or "mul").lower()
        if noise_type_lower != "mul":
            logging.warning(
                "Only multiplicative mismatch is supported; forcing noise_type='mul' (got %s).",
                noise_type,
            )
        self.model = model
        self.noise_levels = list(dict.fromkeys(float(level) for level in noise_levels))
        self.noise_type = "mul"
        self.noise_free_params = {"s_w_Param"}
        self.current_noise_level = 0.0
        self._manual_noise_level: float | None = None

    def _check_noise_free(self, param_name: str) -> bool:
        return any(noise_free in param_name for noise_free in self.noise_free_params)

    def set_manual_noise_level(self, level: float | None) -> None:
        if level is None:
            self._manual_noise_level = None
            return
        if level < 0.0:
            raise ValueError("manual mismatch level must be non-negative.")
        self._manual_noise_level = float(level)

    def _sample_noise_level(self) -> float:
        if self._manual_noise_level is not None:
            level = self._manual_noise_level
        elif len(self.noise_levels) == 1:
            level = self.noise_levels[0]
        else:
            level = random.choice(self.noise_levels)
        self.current_noise_level = level
        return level

    def gen_noisy_params(self) -> dict[str, torch.Tensor]:
        noise_std = self._sample_noise_level()
        noisy_params: dict[str, torch.Tensor] = {}
        for name, param in self.model.named_parameters():
            if self._check_noise_free(name) or noise_std == 0.0:
                noisy_params[name] = param
                continue
            noisy_params[name] = self._apply_noise_mul(param, noise_std)
        return noisy_params

    def _apply_noise_mul(self, param: nn.Parameter, std: float) -> torch.Tensor:
        gaussian = torch.randn_like(param, device=param.device, requires_grad=False)
        noise_multiplier = 1.0 + std * gaussian
        return param * noise_multiplier

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if not self.model.training:
            return self.model(inputs)
        noisy_params = self.gen_noisy_params()
        return torch.func.functional_call(self.model, noisy_params, (inputs,))


def build_linear_ramp_schedule(
    start_level: float,
    target_level: float,
    ramp_epochs: int,
) -> Callable[[int], float]:
    """Create a callable that ramps linearly from start_level to target_level over ramp_epochs."""
    start = float(start_level)
    target = float(target_level)
    if ramp_epochs <= 0:
        return lambda _epoch: target
    if ramp_epochs == 1:
        return lambda _epoch: target
    span = max(ramp_epochs - 1, 1)

    def schedule(epoch: int) -> float:
        if epoch >= ramp_epochs:
            return target
        progress = min(max(epoch / span, 0.0), 1.0)
        return start + (target - start) * progress

    return schedule


def train_one_epoch(
    net: nn.Module,
    trainloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    args: argparse.Namespace,
) -> None:
    print(f'\nEpoch: {epoch}')
    net.train()
    train_loss = 0.0
    correct = 0.0
    total = 0
    count = 0
    lam = 1.0
    optimizer.zero_grad()

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if count == args.batch_split:
            optimizer.step()
            optimizer.zero_grad()
            count = 0

        inputs = inputs.to(device)
        targets = targets.to(device)
        inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, args.alpha, lam, count, device)
        outputs = net(inputs)
        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        loss = loss / args.batch_split
        loss.backward()
        count += 1

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += (
            lam * predicted.eq(targets_a).float().sum().item()
            + (1 - lam) * predicted.eq(targets_b).float().sum().item()
        )

        avg_loss = train_loss / (batch_idx + 1)
        acc = 100.0 * correct / max(total, 1)
        progress_bar(
            batch_idx,
            len(trainloader),
            'Loss: {:.3f} | Acc: {:.3f}% ({:.1f}/{})'.format(avg_loss, acc, correct, total),
        )

    if count > 0:
        optimizer.step()
        optimizer.zero_grad()


def evaluate(
    net: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    dataset_name: str = "cifar10",
) -> tuple[float, float | None]:
    net.eval()
    test_loss = 0.0
    correct = 0
    correct_top5 = 0
    total = 0
    compute_top5 = dataset_name == "cifar100"

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            if compute_top5:
                max_k = min(5, outputs.size(1))
                topk = outputs.topk(max_k, dim=1).indices
                correct_top5 += topk.eq(targets.view(-1, 1)).any(dim=1).sum().item()

            avg_loss = test_loss / (batch_idx + 1)
            acc = 100.0 * correct / max(total, 1)
            top5_acc = 100.0 * correct_top5 / max(total, 1) if compute_top5 else None
            progress_bar(
                batch_idx,
                len(dataloader),
                'Loss: {:.3f} | Acc: {:.3f}%{} ({}/{})'.format(
                    avg_loss,
                    acc,
                    "" if not compute_top5 else " | Top5: {:.3f}%".format(top5_acc),
                    correct,
                    total,
                ),
            )

    acc = 100.0 * correct / max(total, 1)
    top5_acc = 100.0 * correct_top5 / max(total, 1) if compute_top5 else None
    return acc, top5_acc


def main() -> None:
    args = parse_args()
    args.dataset = _normalize_dataset_name(args.dataset)

    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    if args.batch_split < 1:
        raise ValueError("batch_split must be >= 1")
    if args.batch < 1:
        raise ValueError("batch must be >= 1")

    try:
        mismatch_levels = sanitize_mismatch_levels(args.mismatch_levels)
    except ValueError as exc:
        raise SystemExit(f"Invalid mismatch levels: {exc}") from exc
    args.mismatch_levels = mismatch_levels
    mismatch_type = (args.mismatch_type or "mul").lower()
    args.mismatch_type = mismatch_type
    if args.mismatch_ramp_start is not None and args.mismatch_ramp_start < 0.0:
        raise ValueError("mismatch_ramp_start must be >= 0")
    if args.mismatch_ramp_epochs < 0:
        raise ValueError("mismatch_ramp_epochs must be >= 0")
    if args.mismatch_ramp_start is not None and not mismatch_levels:
        logging.warning("mismatch ramp settings ignored because mismatch_levels is empty.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device :", device)

    train_set, test_set = create_datasets(args)
    testloader = DataLoader(
        test_set,
        batch_size=10,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == 'cuda',
    )

    criterion = nn.CrossEntropyLoss()
    num_classes = 100 if args.dataset == "cifar100" else 10
    preload_state: OrderedDict[str, torch.Tensor] | None = None
    init_state: OrderedDict[str, torch.Tensor] | None = None
    init_checkpoint_path: Path | None = None
    checkpoint_path = Path(args.checkpoint).expanduser()
    load_device = torch.device("cpu")

    if args.test_only:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        preload_state = load_checkpoint_state_dict(checkpoint_path, load_device)
    elif args.init_checkpoint:
        init_checkpoint_path = Path(args.init_checkpoint).expanduser()
        if not init_checkpoint_path.is_file():
            raise FileNotFoundError(f"Initial checkpoint not found at {init_checkpoint_path}")
        preload_state = load_checkpoint_state_dict(init_checkpoint_path, load_device)
        init_state = preload_state

    arch_source = resolve_arch_source(args.arch_source, preload_state)
    if args.arch_source == "auto" and preload_state is not None and arch_source != "torchvision":
        logging.info("Detected %s checkpoint; using %s model implementation.", arch_source, arch_source)

    # Try to infer architecture from checkpoint if not explicitly set and we have a checkpoint
    inferred_arch = None
    if preload_state is not None:
        inferred_arch = infer_architecture_from_checkpoint(preload_state)
        if inferred_arch and inferred_arch != args.arch:
            logging.warning(
                "Checkpoint appears to be for %s, but --arch is set to %s. "
                "Using inferred architecture %s.",
                inferred_arch, args.arch, inferred_arch
            )
            args.arch = inferred_arch

    if arch_source == "hankyul2":
        teacher_core = load_hankyul_efficientnet_v2_4ch(
            num_classes=num_classes,
            arch=args.arch,
        )
    else:
        teacher_core = load_efficientnet_v2_4ch(
            num_classes=num_classes,
            arch=args.arch,
            pretrained=not args.no_pretrained,
        )
    print(f'num parameters: {sum(p.numel() for p in teacher_core.parameters())}')
    noisy_wrapper: WrappedNoisyModel | None = None

    if init_checkpoint_path and init_state is not None and not args.test_only:
        print(f'==> Loading initial weights from {init_checkpoint_path}')
        init_state = adapt_state_dict_input_channels(init_state, teacher_core)
        teacher_core.load_state_dict(init_state)
    teacher_core = teacher_core.to(device)

    if mismatch_levels:
        net = WrappedNoisyModel(teacher_core, mismatch_levels, mismatch_type).to(device)
        noisy_wrapper = net
        logging.warning(
            "Mismatch-aware training enabled with noise levels %s (%s noise).",
            mismatch_levels,
            mismatch_type,
        )
    else:
        net = teacher_core

    if args.test_only:
        print(f'==> Evaluating checkpoint from {checkpoint_path}')
        state_dict = preload_state or load_checkpoint_state_dict(checkpoint_path, device)
        state_dict = adapt_state_dict_input_channels(state_dict, teacher_core)
        try:
            teacher_core.load_state_dict(state_dict, strict=True)
        except RuntimeError as e:
            # If strict loading fails, try non-strict with warnings
            logging.warning("Strict state_dict loading failed, attempting non-strict load.")
            missing_keys, unexpected_keys = teacher_core.load_state_dict(state_dict, strict=False)
            if missing_keys:
                logging.warning("Missing keys in checkpoint: %s", missing_keys[:10])
                if len(missing_keys) > 10:
                    logging.warning("... and %d more missing keys", len(missing_keys) - 10)
            if unexpected_keys:
                logging.warning("Unexpected keys in checkpoint: %s", unexpected_keys[:10])
                if len(unexpected_keys) > 10:
                    logging.warning("... and %d more unexpected keys", len(unexpected_keys) - 10)
            if not missing_keys and not unexpected_keys:
                # If non-strict worked perfectly, suppress the original error
                pass
            else:
                # Re-raise if there are still issues
                raise RuntimeError(
                    f"Failed to load checkpoint. Missing {len(missing_keys)} keys, "
                    f"unexpected {len(unexpected_keys)} keys. Original error: {e}"
                ) from e
        acc, top5 = evaluate(net, testloader, criterion, device, dataset_name=args.dataset)
        if top5 is not None:
            print(f'Test top1={acc:.2f}%, top5={top5:.2f}%')
        else:
            print(f'Test acc={acc:.2f}%')
        return

    micro_batch_size = args.batch // args.batch_split
    if micro_batch_size < 1:
        raise ValueError("batch // batch_split must be >= 1")

    trainloader = DataLoader(
        train_set,
        batch_size=micro_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == 'cuda',
    )

    optimizer = optim.SGD(teacher_core.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.nsc, gamma=args.gamma)

    manual_noise_schedule: Callable[[int], float] | None = None
    ramp_epochs = args.mismatch_ramp_epochs if mismatch_levels else 0
    if noisy_wrapper and ramp_epochs > 0:
        start_level = (
            args.mismatch_ramp_start if args.mismatch_ramp_start is not None else mismatch_levels[0]
        )
        target_level = mismatch_levels[-1]
        manual_noise_schedule = build_linear_ramp_schedule(start_level, target_level, ramp_epochs)
        logging.info(
            "Applying linear mismatch ramp: start=%.4f target=%.4f over %d epochs.",
            start_level,
            target_level,
            ramp_epochs,
        )
    elif args.mismatch_ramp_start is not None and noisy_wrapper and ramp_epochs == 0:
        start_level = args.mismatch_ramp_start
        noisy_wrapper.set_manual_noise_level(start_level)
        logging.warning(
            "mismatch_ramp_start=%.4f provided but mismatch_ramp_epochs=0; using constant mismatch level.",
            start_level,
        )

    best_acc = 0.0
    best_top5 = None
    best_epoch = -1
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.ne):
        if manual_noise_schedule and noisy_wrapper:
            scheduled_level = manual_noise_schedule(epoch)
            noisy_wrapper.set_manual_noise_level(scheduled_level)
            logging.info("Epoch %d mismatch std=%.4f", epoch, scheduled_level)
        train_one_epoch(net, trainloader, optimizer, criterion, device, epoch, args)
        scheduler.step()

        print('==> Evaluating..')
        acc, top5 = evaluate(net, testloader, criterion, device, dataset_name=args.dataset)
        if acc > best_acc:
            best_acc = acc
            best_top5 = top5
            best_epoch = epoch
            torch.save(
                {
                    'net': teacher_core.state_dict(),
                    'acc': acc,
                    'epoch': epoch,
                    'mismatch_levels': mismatch_levels,
                    'mismatch_type': mismatch_type,
                    'mismatch_ramp_start': args.mismatch_ramp_start,
                    'mismatch_ramp_epochs': args.mismatch_ramp_epochs,
                },
                checkpoint_path,
            )
            if top5 is not None:
                print(f'Best checkpoint updated at epoch {epoch} with top1={acc:.2f}%, top5={top5:.2f}%')
            else:
                print(f'Best checkpoint updated at epoch {epoch} with acc={acc:.2f}%')

    if best_top5 is not None:
        print(f'Training finished. Best top1={best_acc:.2f}% top5={best_top5:.2f}% at epoch {best_epoch}')
    else:
        print(f'Training finished. Best acc={best_acc:.2f}% at epoch {best_epoch}')


if __name__ == "__main__":
    main()

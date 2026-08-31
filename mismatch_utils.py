from typing import List, Tuple

import torch
import torch.nn as nn


ADDITIVE_SCALE_MODES = ("max_abs", "rms", "max_sqrt")


def is_filter_weight(module: nn.Module, parameter_name: str) -> bool:
    return parameter_name == "weight" and isinstance(
        module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)
    )


def additive_mismatch_scale(
    tensor: torch.Tensor,
    mode: str,
    module: nn.Module = None,
) -> torch.Tensor:
    if tensor.numel() == 0:
        raise ValueError("Cannot compute an additive mismatch scale for an empty tensor.")
    if mode == "max_abs":
        return tensor.abs().max()
    if mode == "rms":
        return tensor.square().mean().sqrt()
    if mode == "max_sqrt":
        # Paper covariance per filter: max(abs(W_f)) * sigma^2 * I.
        # The broadcast standard-deviation scale is therefore sqrt(max(abs(W_f))).
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            reduce_dims = tuple(range(1, tensor.ndim))
        elif isinstance(module, nn.ConvTranspose2d):
            reduce_dims = (0,) + tuple(range(2, tensor.ndim))
        else:
            raise ValueError(
                "max_sqrt requires a Conv2d, ConvTranspose2d, or Linear weight module."
            )
        return tensor.abs().amax(dim=reduce_dims, keepdim=True).sqrt()
    raise ValueError(f"Unsupported additive scale mode: {mode}")


def _scale_unique_parameters(
    named_parameters: List[Tuple[str, torch.nn.Parameter]], gain: float
) -> List[Tuple[str, Tuple[int, ...], int]]:
    records = []
    seen = set()
    with torch.no_grad():
        for name, parameter in named_parameters:
            if id(parameter) in seen:
                continue
            seen.add(id(parameter))
            parameter.mul_(gain)
            records.append((name, tuple(parameter.shape), parameter.numel()))
    return records


def apply_pcn_ff_gain(model: nn.Module, gain: float) -> List[Tuple[str, Tuple[int, ...], int]]:
    selected = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if name.startswith("PcConvs.") and name.endswith(".FFconv.weight")
    ]
    if not selected:
        raise ValueError("No PCN FFconv weights were found for FF-gain scaling.")
    return _scale_unique_parameters(selected, gain)


def apply_wrn_ff_gain(model: nn.Module, gain: float) -> List[Tuple[str, Tuple[int, ...], int]]:
    selected = []
    for name, module in model.named_modules():
        is_stem = name == "conv1"
        is_block_first_conv = name.endswith(".conv1") and module.__class__.__name__ == "Conv2d"
        if is_stem or is_block_first_conv:
            parent_name = name.rpartition(".")[0]
            if is_stem or any(part.startswith("layer") for part in parent_name.split(".")):
                selected.append((f"{name}.weight", module.weight))
    if not selected:
        raise ValueError("No WRN stem/block conv1 weights were found for FF-gain scaling.")
    return _scale_unique_parameters(selected, gain)

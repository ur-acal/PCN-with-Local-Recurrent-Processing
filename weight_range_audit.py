import csv
import math
import os
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn


CONV_TYPES = (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)
LINEAR_TYPES = (nn.Linear,)
PRIMARY_TYPES = CONV_TYPES + LINEAR_TYPES


@dataclass
class TensorRangeStats:
    max_abs: float
    rms: float
    kappa: float
    all_zero: bool


def tensor_range_stats(tensor: torch.Tensor) -> TensorRangeStats:
    with torch.no_grad():
        x = tensor.detach()
        if not x.is_floating_point():
            x = x.float()
        x = x.double()
        max_abs = float(x.abs().max().item()) if x.numel() else 0.0
        rms = float(torch.sqrt(torch.mean(x * x)).item()) if x.numel() else 0.0
        all_zero = max_abs == 0.0
        kappa = math.nan if all_zero or rms == 0.0 else max_abs / rms
    return TensorRangeStats(max_abs=max_abs, rms=rms, kappa=kappa, all_zero=all_zero)


def _shape_str(tensor: torch.Tensor) -> str:
    return "x".join(str(x) for x in tensor.shape)


def _row(
    *,
    dataset: str,
    model_condition: str,
    model_name: str,
    semantic_order_index: int,
    tensor_name: str,
    module_type: str,
    tensor: torch.Tensor,
    extra: Optional[Dict] = None,
) -> Dict:
    stats = tensor_range_stats(tensor)
    row = {
        "dataset": dataset,
        "model_condition": model_condition,
        "model_name": model_name,
        "semantic_order_index": semantic_order_index,
        "tensor_name": tensor_name,
        "module_type": module_type,
        "tensor_shape": _shape_str(tensor),
        "num_elements": int(tensor.numel()),
        "max_abs": stats.max_abs,
        "rms": stats.rms,
        "kappa": stats.kappa,
        "all_zero": stats.all_zero,
    }
    if extra:
        row.update(extra)
    return row


def split_primary_and_nonprimary(
    rows: Iterable[Tuple[str, nn.Module, str, torch.Tensor, int, Dict]],
    *,
    dataset: str,
    model_condition: str,
    model_name: str,
) -> Tuple[List[Dict], List[Dict]]:
    primary_rows: List[Dict] = []
    nonprimary_rows: List[Dict] = []
    primary_order = 1
    nonprimary_order = 1
    for tensor_name, module, local_name, tensor, semantic_hint, extra in rows:
        module_type = type(module).__name__ if module is not None else "None"
        is_primary = isinstance(module, PRIMARY_TYPES) and local_name == "weight"
        if is_primary:
            primary_rows.append(
                _row(
                    dataset=dataset,
                    model_condition=model_condition,
                    model_name=model_name,
                    semantic_order_index=primary_order if semantic_hint <= 0 else semantic_hint,
                    tensor_name=tensor_name,
                    module_type=module_type,
                    tensor=tensor,
                    extra=extra,
                )
            )
            primary_order += 1
        else:
            nonprimary_rows.append(
                _row(
                    dataset=dataset,
                    model_condition=model_condition,
                    model_name=model_name,
                    semantic_order_index=nonprimary_order,
                    tensor_name=tensor_name,
                    module_type=module_type,
                    tensor=tensor,
                    extra=extra,
                )
            )
            nonprimary_order += 1
    return primary_rows, nonprimary_rows


def collect_wrn_weight_range_audit_rows(
    model: nn.Module,
    *,
    dataset: str,
    model_name: str,
    should_noise_param: Callable[[str, torch.Tensor], bool],
    param_to_module: Dict[str, nn.Module],
) -> Tuple[List[Dict], List[Dict]]:
    selected = []
    order = 1
    for name, param in model.named_parameters():
        if not should_noise_param(name, param):
            continue
        module = param_to_module.get(name)
        local_name = name.rsplit(".", 1)[-1]
        selected.append(
            (
                name,
                module,
                local_name,
                param,
                order,
                {"selection_source": "FixedMismatchHelper._should_noise_param"},
            )
        )
        if isinstance(module, PRIMARY_TYPES) and local_name == "weight":
            order += 1
    return split_primary_and_nonprimary(selected, dataset=dataset, model_condition="WRN", model_name=model_name)


def collect_pcn_ode_weight_range_audit_rows(
    model: nn.Module,
    *,
    dataset: str,
    model_name: str,
    include_linear: bool = True,
) -> Tuple[List[Dict], List[Dict]]:
    selected = []
    order = 1

    first_conv = getattr(model, "first_conv", None)
    if isinstance(first_conv, nn.Conv2d):
        if getattr(first_conv, "weight", None) is not None:
            selected.append(
                (
                    "first_conv.weight",
                    first_conv,
                    "weight",
                    first_conv.weight,
                    order,
                    {"pcn_layer_index": "", "pcn_role": "first_conv", "selection_source": "PCNet.add_noise_conv_name"},
                )
            )
            order += 1
        if getattr(first_conv, "bias", None) is not None:
            selected.append(
                (
                    "first_conv.bias",
                    first_conv,
                    "bias",
                    first_conv.bias,
                    0,
                    {"pcn_layer_index": "", "pcn_role": "first_conv_bias", "selection_source": "PCNet.add_noise_conv_name"},
                )
            )

    pc_convs = getattr(model, "PcConvs", [])
    for layer_idx, block in enumerate(pc_convs):
        ff = getattr(block, "FFconv", None)
        if ff is not None and getattr(ff, "weight", None) is not None:
            selected.append(
                (
                    f"PcConvs.{layer_idx}.FFconv.weight",
                    ff,
                    "weight",
                    ff.weight,
                    order,
                    {"pcn_layer_index": layer_idx, "pcn_role": "FFconv", "selection_source": "ODEBlockPC.add_noise"},
                )
            )
            order += 1

        fb = getattr(block, "FBconv", None)
        if fb is not None and not bool(getattr(block, "tie_weights", False)) and getattr(fb, "weight", None) is not None:
            selected.append(
                (
                    f"PcConvs.{layer_idx}.FBconv.weight",
                    fb,
                    "weight",
                    fb.weight,
                    order,
                    {"pcn_layer_index": layer_idx, "pcn_role": "FBconv", "selection_source": "ODEBlockPC.add_noise"},
                )
            )
            order += 1

        bypass = getattr(block, "bypass", None)
        if bypass is not None and not bool(getattr(block, "tie_bp", False)) and getattr(bypass, "weight", None) is not None:
            selected.append(
                (
                    f"PcConvs.{layer_idx}.bypass.weight",
                    bypass,
                    "weight",
                    bypass.weight,
                    order,
                    {"pcn_layer_index": layer_idx, "pcn_role": "bypass", "selection_source": "ODEBlockPC.add_noise"},
                )
            )
            order += 1

        b0 = getattr(block, "b0", None)
        if b0 is not None and len(b0) > 0 and not torch.allclose(b0[0], torch.zeros_like(b0[0])):
            selected.append(
                (
                    f"PcConvs.{layer_idx}.b0.0",
                    block,
                    "b0.0",
                    b0[0],
                    0,
                    {"pcn_layer_index": layer_idx, "pcn_role": "b0", "selection_source": "ODEBlockPC.add_noise_nonzero_b0"},
                )
            )

    if include_linear:
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                prefix = f"{name}." if name else ""
                if getattr(module, "weight", None) is not None:
                    selected.append(
                        (
                            f"{prefix}weight",
                            module,
                            "weight",
                            module.weight,
                            order,
                            {"pcn_layer_index": "", "pcn_role": "classifier", "selection_source": "PCNet.add_noise_noise_to_linear"},
                        )
                    )
                    order += 1
                if getattr(module, "bias", None) is not None:
                    selected.append(
                        (
                            f"{prefix}bias",
                            module,
                            "bias",
                            module.bias,
                            0,
                            {"pcn_layer_index": "", "pcn_role": "classifier_bias", "selection_source": "PCNet.add_noise_noise_to_linear"},
                        )
                    )

    return split_primary_and_nonprimary(selected, dataset=dataset, model_condition="PCN", model_name=model_name)


def nonprimary_csv_path(primary_csv_path: str) -> str:
    root, ext = os.path.splitext(primary_csv_path)
    return f"{root}_nonprimary{ext or '.csv'}"


def save_audit_csv(primary_csv_path: str, primary_rows: List[Dict], nonprimary_rows: Optional[List[Dict]] = None) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(primary_csv_path)), exist_ok=True)
    _write_rows(primary_csv_path, primary_rows)
    if nonprimary_rows is not None:
        _write_rows(nonprimary_csv_path(primary_csv_path), nonprimary_rows)
    return primary_csv_path


def _write_rows(path: str, rows: List[Dict]):
    fieldnames = [
        "dataset",
        "model_condition",
        "model_name",
        "semantic_order_index",
        "tensor_name",
        "module_type",
        "tensor_shape",
        "num_elements",
        "max_abs",
        "rms",
        "kappa",
        "all_zero",
    ]
    extra_keys = []
    for row in rows:
        for key in row:
            if key not in fieldnames and key not in extra_keys:
                extra_keys.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames + extra_keys)
        writer.writeheader()
        writer.writerows(rows)

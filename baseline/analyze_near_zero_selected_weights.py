#!/usr/bin/env python3
"""Compare near-zero selected weights under per-tensor RMS additive mismatch."""

import argparse
import csv
import math
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baseline.run_baseline import (  # noqa: E402
    FixedMismatchHelper,
    build_model,
    get_baseline_config,
    load_model_weights,
)


ARCHITECTURES = ("WRN_16_2", "WRN_16_4", "WRN_28_2", "WRN_28_4")
LEVELS = (0.25, 0.5, 0.75, 1.0)
NOISE_STD_DIVISORS = (1, 3, 6)
ABSOLUTE_THRESHOLDS = (1e-5, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pcn_primary_csv",
        type=Path,
        default=ROOT.parent / "ScAN-PCN/logs/kappa_audit/all_primary.csv",
    )
    parser.add_argument(
        "--pcn_nonprimary_csv",
        type=Path,
        default=ROOT.parent / "ScAN-PCN/logs/kappa_audit/all_nonprimary.csv",
    )
    parser.add_argument(
        "--wrn_checkpoint_root",
        type=Path,
        default=ROOT / "checkpoint/baselines_wd1e3_pilot",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=ROOT / "logs/near_zero_weight_analysis_cifar100",
    )
    return parser.parse_args()


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def pcn_selected_tensors(args, architecture):
    rows = []
    for path in (args.pcn_primary_csv, args.pcn_nonprimary_csv):
        rows.extend(
            row
            for row in read_csv(path)
            if row["dataset"] == "cifar100"
            and row["paired_arch"] == architecture
            and row["model_condition"] == "PCN"
            and "PCNetNoBatchNorm" in row["model_name"]
        )
    if not rows:
        raise ValueError(f"No pure PCNetNoBatchNorm audit rows for {architecture}")
    checkpoints = {row["checkpoint_path"] for row in rows}
    if len(checkpoints) != 1:
        raise ValueError(f"Ambiguous PCN checkpoints for {architecture}: {checkpoints}")

    checkpoint = checkpoints.pop()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload["net"]
    output = []
    seen = set()
    for row in rows:
        name = row["tensor_name"]
        if name in seen:
            raise ValueError(f"Duplicate PCN tensor {architecture}:{name}")
        seen.add(name)
        output.append(
            {
                "name": name,
                "module_type": row["module_type"],
                "tensor": state[name].detach().cpu(),
                "checkpoint_path": checkpoint,
            }
        )
    return output


def wrn_checkpoint_path(root, architecture):
    model_name = f"wrn_{architecture.removeprefix('WRN_').lower()}_cifar"
    run_name = f"custom_noresize_cifar100_{model_name}"
    return (
        root
        / "cifar100"
        / "custom_noresize"
        / model_name
        / run_name
        / f"{run_name}_best_ckpt.pth"
    )


def wrn_selected_tensors(args, architecture):
    model_name = f"wrn_{architecture.removeprefix('WRN_').lower()}_cifar"
    checkpoint = wrn_checkpoint_path(args.wrn_checkpoint_root, architecture)
    cfg = get_baseline_config(
        model_name=model_name,
        pretrained=False,
        case="custom_noresize",
        prefer_resize=False,
        extra_overrides=None,
    )
    model = build_model(model_name, cfg, num_classes=100).cpu().eval()
    load_model_weights(model, str(checkpoint), torch.device("cpu"))
    helper = FixedMismatchHelper(
        model,
        noise_sigma=0.0,
        noise_type="additive",
        noise_to_norm=False,
        include_buffers=True,
        additive_scale_mode="rms",
    )
    output = []
    for name, parameter in model.named_parameters():
        if helper._should_noise_param(name, parameter):
            module = helper._param_to_module.get(name)
            output.append(
                {
                    "name": name,
                    "module_type": type(module).__name__ if module else "",
                    "tensor": parameter.detach().cpu(),
                    "checkpoint_path": str(checkpoint.resolve()),
                }
            )
    return output


def distribution_metrics(values, normalized_abs, rms, level):
    abs_values = values.abs()
    nonzero = abs_values > 0
    noise_std = level * rms
    near_zero = {
        f"near_zero_fraction_abs_le_noise_std_div_{divisor}": float(
            (normalized_abs <= level / divisor).double().mean().item()
        )
        for divisor in NOISE_STD_DIVISORS
    }
    if noise_std == 0.0:
        sign_flip = 0.0
        noise_exceeds = 0.0
    else:
        scaled = abs_values / (math.sqrt(2.0) * noise_std)
        sign_flip_terms = 0.5 * torch.erfc(scaled)
        sign_flip = (
            float(sign_flip_terms[nonzero].mean().item())
            if bool(nonzero.any())
            else 0.0
        )
        noise_exceeds = float(torch.erfc(scaled).mean().item())
    mean_abs = float(abs_values.mean().item())
    expected_abs_noise_over_mean_abs = (
        noise_std * math.sqrt(2.0 / math.pi) / mean_abs
        if mean_abs > 0.0
        else 0.0
    )
    return {
        "mismatch_level": level,
        "noise_std": noise_std,
        **near_zero,
        "expected_sign_flip_fraction_nonzero": sign_flip,
        "expected_fraction_abs_noise_gt_abs_weight": noise_exceeds,
        "expected_mean_abs_noise_over_mean_abs_weight": expected_abs_noise_over_mean_abs,
    }


def absolute_threshold_metrics(values):
    abs_values = values.abs()
    return {
        f"fraction_abs_le_{threshold:.0e}".replace("-", "m"): float(
            (abs_values <= threshold).double().mean().item()
        )
        for threshold in ABSOLUTE_THRESHOLDS
    }


def base_metrics(values):
    values = values.detach().double().reshape(-1)
    abs_values = values.abs()
    rms = float(values.square().mean().sqrt().item())
    normalized = abs_values / rms if rms > 0.0 else torch.zeros_like(abs_values)
    quantiles = torch.quantile(normalized, torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9], dtype=torch.double))
    return values, normalized, {
        "num_elements": values.numel(),
        "mean_abs": float(abs_values.mean().item()),
        "median_abs": float(abs_values.median().item()),
        "rms": rms,
        "mean_abs_over_rms": float(normalized.mean().item()) if rms > 0.0 else math.nan,
        "normalized_abs_q10": float(quantiles[0].item()),
        "normalized_abs_q25": float(quantiles[1].item()),
        "normalized_abs_q50": float(quantiles[2].item()),
        "normalized_abs_q75": float(quantiles[3].item()),
        "normalized_abs_q90": float(quantiles[4].item()),
        "exact_zero_fraction": float((abs_values == 0).double().mean().item()),
        "all_zero": rms == 0.0,
    }


def analyze_condition(architecture, condition, tensors):
    tensor_rows = []
    scope_values = {"conv_linear_weights": [], "all_selected": []}
    scope_normalized = {"conv_linear_weights": [], "all_selected": []}
    checkpoints = {item["checkpoint_path"] for item in tensors}
    if len(checkpoints) != 1:
        raise ValueError(f"Expected one checkpoint for {architecture} {condition}")
    checkpoint = next(iter(checkpoints))

    for item in tensors:
        values, normalized, base = base_metrics(item["tensor"])
        is_primary = item["tensor"].ndim >= 2
        for level in LEVELS:
            tensor_rows.append(
                {
                    "dataset": "cifar100",
                    "architecture": architecture,
                    "model_condition": condition,
                    "scope": "conv_linear_weights" if is_primary else "selected_1d",
                    "tensor_name": item["name"],
                    "module_type": item["module_type"],
                    "tensor_shape": "x".join(str(x) for x in item["tensor"].shape),
                    **base,
                    **absolute_threshold_metrics(values),
                    **distribution_metrics(values, normalized, base["rms"], level),
                    "checkpoint_path": item["checkpoint_path"],
                }
            )
        scope_values["all_selected"].append(values)
        scope_normalized["all_selected"].append(normalized)
        if is_primary:
            scope_values["conv_linear_weights"].append(values)
            scope_normalized["conv_linear_weights"].append(normalized)

    summary_rows = []
    for scope in ("conv_linear_weights", "all_selected"):
        values = torch.cat(scope_values[scope])
        normalized = torch.cat(scope_normalized[scope])
        _, _, base = base_metrics(values)
        normalized_quantiles = torch.quantile(
            normalized,
            torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9], dtype=torch.double),
        )
        base.update(
            {
                "normalized_abs_q10": float(normalized_quantiles[0].item()),
                "normalized_abs_q25": float(normalized_quantiles[1].item()),
                "normalized_abs_q50": float(normalized_quantiles[2].item()),
                "normalized_abs_q75": float(normalized_quantiles[3].item()),
                "normalized_abs_q90": float(normalized_quantiles[4].item()),
            }
        )
        tensor_count = sum(
            1 for item in tensors if scope == "all_selected" or item["tensor"].ndim >= 2
        )
        # The near-zero threshold remains per-tensor RMS. `normalized` preserves that.
        for level in LEVELS:
            abs_values = values.abs()
            nonzero = abs_values > 0
            scaled = normalized / (math.sqrt(2.0) * level)
            sign_flip_terms = 0.5 * torch.erfc(scaled)
            summary_rows.append(
                {
                    "dataset": "cifar100",
                    "architecture": architecture,
                    "model_condition": condition,
                    "scope": scope,
                    "selected_tensor_count": tensor_count,
                    **base,
                    **absolute_threshold_metrics(values),
                    "mismatch_level": level,
                    **{
                        f"near_zero_fraction_abs_le_noise_std_div_{divisor}": float(
                            (normalized <= level / divisor).double().mean().item()
                        )
                        for divisor in NOISE_STD_DIVISORS
                    },
                    "expected_sign_flip_fraction_nonzero": float(
                        sign_flip_terms[nonzero].mean().item()
                    ) if bool(nonzero.any()) else 0.0,
                    "expected_fraction_abs_noise_gt_abs_weight": float(
                        torch.erfc(scaled).mean().item()
                    ),
                    "expected_mean_abs_noise_over_mean_abs_weight": float(
                        level
                        * math.sqrt(2.0 / math.pi)
                        * torch.cat(
                            [
                                torch.full_like(norm, float(tensor.square().mean().sqrt().item()))
                                for tensor, norm in zip(scope_values[scope], scope_normalized[scope])
                            ]
                        ).mean().item()
                        / base["mean_abs"]
                    ),
                    "checkpoint_path": checkpoint,
                }
            )
    return tensor_rows, summary_rows


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path, rows):
    primary = [row for row in rows if row["scope"] == "conv_linear_weights"]
    lines = [
        "# CIFAR-100 selected-weight near-zero analysis",
        "",
        "Clean checkpoints only. The primary scope contains exactly the selected Conv/Linear weight tensors; selected classifier biases are retained in `summary.csv` under `all_selected`.",
        "",
        "For RMS additive mismatch on tensor `l`, the elementwise noise standard deviation is `s_l = sigma * RMS(W_l)`. The three relative near-zero columns use `s_l`, `s_l / 3`, and `s_l / 6`. Expected rates are analytic under independent standard-normal mismatch; no noisy inference was run.",
        "",
        "| Pair | Model | sigma | global RMS | mean |W| | mean |W| / RMS | median |W| / tensor RMS | |W| <= s | |W| <= s/3 | |W| <= s/6 | E[sign flip] (nonzero) | E[|noise| > |W|] | E|noise| / mean|W| |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in primary:
        lines.append(
            "| {architecture} | {model_condition} | {mismatch_level:.2f} | {rms:.6f} | {mean_abs:.6f} | {mean_abs_over_rms:.4f} | {normalized_abs_q50:.4f} | {near_zero_fraction_abs_le_noise_std_div_1:.2%} | {near_zero_fraction_abs_le_noise_std_div_3:.2%} | {near_zero_fraction_abs_le_noise_std_div_6:.2%} | {expected_sign_flip_fraction_nonzero:.2%} | {expected_fraction_abs_noise_gt_abs_weight:.2%} | {expected_mean_abs_noise_over_mean_abs_weight:.4f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Common absolute thresholds",
            "",
            "These thresholds are identical for PCN and WRN and do not depend on tensor or model RMS.",
            "",
            "| Pair | Model | <= 1e-5 | <= 1e-4 | <= 1e-3 | <= 3e-3 | <= 1e-2 | <= 3e-2 |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in primary:
        if row["mismatch_level"] != LEVELS[0]:
            continue
        lines.append(
            "| {architecture} | {model_condition} | {fraction_abs_le_1em05:.2%} | {fraction_abs_le_1em04:.2%} | {fraction_abs_le_1em03:.2%} | {fraction_abs_le_3em03:.2%} | {fraction_abs_le_1em02:.2%} | {fraction_abs_le_3em02:.2%} |".format(**row)
        )
    lines.extend(
        [
            "",
            "Notes:",
            "- `global RMS` is over all selected Conv/Linear weight elements and describes absolute model scale.",
            "- Near-zero and expected event rates normalize each element by its own tensor RMS, matching the implemented mismatch scale.",
            "- `E[sign flip]` excludes exact-zero weights because zero has no sign; `E[|noise| > |W|]` includes them.",
            "- Equal expected relative Frobenius perturbation (`approximately sigma`) does not imply equal elementwise effects when normalized weight distributions differ.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tensor_rows = []
    summary_rows = []
    for architecture in ARCHITECTURES:
        for condition, tensors in (
            ("PCN", pcn_selected_tensors(args, architecture)),
            ("WRN_WD1e3_unfused", wrn_selected_tensors(args, architecture)),
        ):
            condition_tensor_rows, condition_summary_rows = analyze_condition(
                architecture, condition, tensors
            )
            tensor_rows.extend(condition_tensor_rows)
            summary_rows.extend(condition_summary_rows)
    write_csv(args.output_dir / "per_tensor_by_level.csv", tensor_rows)
    write_csv(args.output_dir / "summary.csv", summary_rows)
    write_markdown(args.output_dir / "summary.md", summary_rows)
    print(f"Wrote {len(tensor_rows)} tensor-level rows and {len(summary_rows)} summaries")


if __name__ == "__main__":
    main()

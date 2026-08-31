#!/usr/bin/env python3
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baseline.run_baseline import (
    FixedMismatchHelper,
    build_model,
    get_baseline_config,
    load_model_weights,
)


PAIRS = ("WRN_16_2", "WRN_16_4", "WRN_28_2", "WRN_28_4")
LEVELS = (0.25, 0.5, 0.75, 1.0)
TRIALS = (0, 1, 2)


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
        default=ROOT / "logs/rms_relative_perturbation_audit_cifar100",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--base_seed", type=int, default=123)
    return parser.parse_args()


def load_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def shape_from_row(row):
    return tuple(int(value) for value in row["tensor_shape"].split("x") if value)


def mismatch_seed(base_seed, level, trial):
    return base_seed + 1000 * int(round(level * 1e6)) + trial


def tensor_metrics(clean, noisy):
    clean64 = clean.detach().double()
    delta64 = noisy.detach().double() - clean64
    clean_norm = float(torch.linalg.vector_norm(clean64).item())
    delta_norm = float(torch.linalg.vector_norm(delta64).item())
    relative = delta_norm / clean_norm if clean_norm else math.nan
    rms = float(clean64.square().mean().sqrt().item())
    return rms, clean_norm, delta_norm, relative


def pcn_selected_rows(args, architecture):
    rows = []
    for path in (args.pcn_primary_csv, args.pcn_nonprimary_csv):
        for row in load_csv(path):
            if (
                row["dataset"] == "cifar100"
                and row["paired_arch"] == architecture
                and row["model_condition"] == "PCN"
                and "PCNetNoBatchNorm" in row["model_name"]
            ):
                rows.append(row)
    primary = sorted(
        (row for row in rows if row["tensor_name"] != "linear.bias"),
        key=lambda row: int(float(row["semantic_order_index"])),
    )
    biases = [row for row in rows if row["tensor_name"] == "linear.bias"]
    if len(biases) != 1:
        raise ValueError(f"Expected one PCN classifier bias for {architecture}")
    return primary + biases


def replay_pcn(args, architecture, device):
    selected = pcn_selected_rows(args, architecture)
    output = []
    for level in LEVELS:
        for trial in TRIALS:
            seed = mismatch_seed(args.base_seed, level, trial)
            generator = torch.Generator(device=device).manual_seed(seed)
            for row in selected:
                shape = shape_from_row(row)
                numel = int(row["num_elements"])
                rms = float(row["rms"])
                clean_norm = rms * math.sqrt(numel)
                noise = torch.randn(shape, device=device, dtype=torch.float32, generator=generator)
                delta_norm = float((noise.double() * (level * rms)).norm().item())
                relative = delta_norm / clean_norm if clean_norm else math.nan
                output.append(
                    {
                        "dataset": "cifar100",
                        "architecture": architecture,
                        "model_condition": "PCN",
                        "realization_kind": "deterministic_replay_not_historical",
                        "mismatch_level": level,
                        "trial": trial,
                        "mismatch_seed": seed,
                        "tensor_name": row["tensor_name"],
                        "module_type": row["module_type"],
                        "tensor_shape": row["tensor_shape"],
                        "num_elements": numel,
                        "clean_rms": rms,
                        "clean_frobenius_norm": clean_norm,
                        "delta_frobenius_norm": delta_norm,
                        "realized_relative_frobenius": relative,
                        "requested_sigma": level,
                        "relative_over_sigma": relative / level,
                        "selection_source": row["selection_source"],
                        "checkpoint_path": row["checkpoint_path"],
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


def audit_wrn(args, architecture, device):
    model_name = f"wrn_{architecture.removeprefix('WRN_').lower()}_cifar"
    checkpoint = wrn_checkpoint_path(args.wrn_checkpoint_root, architecture)
    cfg = get_baseline_config(
        model_name=model_name,
        pretrained=False,
        case="custom_noresize",
        prefer_resize=False,
        extra_overrides=None,
    )
    model = build_model(model_name, cfg, num_classes=100).to(device).eval()
    load_model_weights(model, str(checkpoint), device)
    helper = FixedMismatchHelper(
        model,
        noise_sigma=0.0,
        noise_type="additive",
        noise_to_norm=False,
        include_buffers=True,
        additive_scale_mode="rms",
    )
    helper.snapshot_clean_state()
    selected_names = [
        name for name, parameter in model.named_parameters()
        if helper._should_noise_param(name, parameter)
    ]
    output = []
    for level in LEVELS:
        for trial in TRIALS:
            helper.restore_clean_state()
            clean = {
                name: parameter.detach().clone()
                for name, parameter in model.named_parameters()
                if name in selected_names
            }
            helper.noise_sigma = level
            helper.seed = mismatch_seed(args.base_seed, level, trial)
            helper.add_noise()
            modules = helper._param_to_module
            for name, parameter in model.named_parameters():
                if name not in clean:
                    continue
                rms, clean_norm, delta_norm, relative = tensor_metrics(clean[name], parameter)
                module = modules.get(name)
                output.append(
                    {
                        "dataset": "cifar100",
                        "architecture": architecture,
                        "model_condition": "WRN_WD1e3_unfused",
                        "realization_kind": "historical_seed_reconstruction",
                        "mismatch_level": level,
                        "trial": trial,
                        "mismatch_seed": helper.seed,
                        "tensor_name": name,
                        "module_type": type(module).__name__ if module is not None else "",
                        "tensor_shape": "x".join(str(value) for value in parameter.shape),
                        "num_elements": parameter.numel(),
                        "clean_rms": rms,
                        "clean_frobenius_norm": clean_norm,
                        "delta_frobenius_norm": delta_norm,
                        "realized_relative_frobenius": relative,
                        "requested_sigma": level,
                        "relative_over_sigma": relative / level,
                        "selection_source": "FixedMismatchHelper._should_noise_param",
                        "checkpoint_path": str(checkpoint.resolve()),
                    }
                )
    helper.restore_clean_state()
    return output


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[
            (
                row["architecture"],
                row["model_condition"],
                float(row["mismatch_level"]),
                int(row["trial"]),
            )
        ].append(row)

    trial_rows = []
    for (architecture, condition, level, trial), items in sorted(grouped.items()):
        clean_sq = sum(float(row["clean_frobenius_norm"]) ** 2 for row in items)
        delta_sq = sum(float(row["delta_frobenius_norm"]) ** 2 for row in items)
        tensor_ratios = np.asarray(
            [float(row["realized_relative_frobenius"]) for row in items],
            dtype=float,
        )
        trial_rows.append(
            {
                "architecture": architecture,
                "model_condition": condition,
                "mismatch_level": level,
                "trial": trial,
                "selected_tensor_count": len(items),
                "selected_parameter_count": sum(int(row["num_elements"]) for row in items),
                "model_relative_frobenius": math.sqrt(delta_sq / clean_sq),
                "model_relative_over_sigma": math.sqrt(delta_sq / clean_sq) / level,
                "tensor_relative_mean": float(tensor_ratios.mean()),
                "tensor_relative_min": float(tensor_ratios.min()),
                "tensor_relative_max": float(tensor_ratios.max()),
            }
        )

    summary_rows = []
    keys = sorted({
        (row["architecture"], row["model_condition"], float(row["mismatch_level"]))
        for row in trial_rows
    })
    for architecture, condition, level in keys:
        items = [
            row for row in trial_rows
            if (
                row["architecture"],
                row["model_condition"],
                float(row["mismatch_level"]),
            ) == (architecture, condition, level)
        ]
        model_values = np.asarray([row["model_relative_frobenius"] for row in items])
        summary_rows.append(
            {
                "architecture": architecture,
                "model_condition": condition,
                "mismatch_level": level,
                "trials": len(items),
                "selected_tensor_count": items[0]["selected_tensor_count"],
                "selected_parameter_count": items[0]["selected_parameter_count"],
                "model_relative_mean": float(model_values.mean()),
                "model_relative_std": float(model_values.std()),
                "model_relative_over_sigma_mean": float(
                    np.mean([row["model_relative_over_sigma"] for row in items])
                ),
                "tensor_relative_min_across_trials": min(
                    row["tensor_relative_min"] for row in items
                ),
                "tensor_relative_max_across_trials": max(
                    row["tensor_relative_max"] for row in items
                ),
            }
        )
    return trial_rows, summary_rows


def render_markdown(summary_rows):
    by_key = {
        (row["architecture"], row["model_condition"], float(row["mismatch_level"])): row
        for row in summary_rows
    }
    lines = [
        "# RMS Relative-Perturbation Audit",
        "",
        "WRN values reconstruct the recorded historical mismatch seeds. PCN values are deterministic mismatch-only replays because the completed PCN evaluator did not record or set trial seeds.",
        "",
        "| Pair | RMS level | PCN model relative norm | WRN model relative norm | PCN / sigma | WRN / sigma |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for architecture in PAIRS:
        for level in LEVELS:
            pcn = by_key[(architecture, "PCN", level)]
            wrn = by_key[(architecture, "WRN_WD1e3_unfused", level)]
            lines.append(
                f"| {architecture} | {level:g} | "
                f"{pcn['model_relative_mean']:.6f} +/- {pcn['model_relative_std']:.6f} | "
                f"{wrn['model_relative_mean']:.6f} +/- {wrn['model_relative_std']:.6f} | "
                f"{pcn['model_relative_over_sigma_mean']:.6f} | "
                f"{wrn['model_relative_over_sigma_mean']:.6f} |"
            )
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for architecture in PAIRS:
        rows.extend(replay_pcn(args, architecture, device))
        rows.extend(audit_wrn(args, architecture, device))
    trial_rows, summary_rows = summarize(rows)
    write_csv(args.output_dir / "per_tensor.csv", rows)
    write_csv(args.output_dir / "per_trial.csv", trial_rows)
    write_csv(args.output_dir / "summary.csv", summary_rows)
    (args.output_dir / "summary.md").write_text(render_markdown(summary_rows))
    print(args.output_dir / "summary.md")


if __name__ == "__main__":
    main()


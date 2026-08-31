#!/usr/bin/env python3
"""Audit tensorwise scaling for a PCN/WRN max-vs-RMS accuracy reversal."""

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", default="WRN_28_4")
    parser.add_argument("--dataset", default="cifar100")
    parser.add_argument("--max_level", type=float, default=0.1)
    parser.add_argument("--rms_level", type=float, default=1.0)
    parser.add_argument("--base_seed", type=int, default=123)
    parser.add_argument(
        "--accuracy_csv",
        type=Path,
        default=ROOT
        / "logs/combined_pcn_wrn_mismatch_finaldrop025_complete/full_accuracy_comparison.csv",
    )
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
        default=ROOT / "checkpoint/baselines_wd1e3_finaldrop025",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=ROOT / "logs/max_rms_reversal_cifar100_wrn28_4",
    )
    return parser.parse_args()


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def shape_from_text(text):
    return tuple(int(value) for value in text.split("x") if value)


def stable_tensor_seed(base_seed, model_condition, tensor_name):
    digest = hashlib.sha256(f"{model_condition}:{tensor_name}".encode()).digest()
    return base_seed + int.from_bytes(digest[:4], "little")


def add_realized_metrics(row, max_level, rms_level, base_seed):
    shape = shape_from_text(row["tensor_shape"])
    generator = torch.Generator(device="cpu")
    generator.manual_seed(stable_tensor_seed(base_seed, row["model_condition"], row["tensor_name"]))
    standard_normal = torch.randn(shape, generator=generator, dtype=torch.float32)
    normal_norm = float(torch.linalg.vector_norm(standard_normal.double()).item())
    numel = int(row["num_elements"])
    max_abs = float(row["max_abs"])
    rms = float(row["rms"])
    clean_norm = rms * math.sqrt(numel)
    max_delta_norm = max_level * max_abs * normal_norm
    rms_delta_norm = rms_level * rms * normal_norm
    max_relative = max_delta_norm / clean_norm
    rms_relative = rms_delta_norm / clean_norm
    row.update(
        {
            "controlled_noise_seed": stable_tensor_seed(
                base_seed, row["model_condition"], row["tensor_name"]
            ),
            "clean_frobenius_norm": clean_norm,
            "max_delta_frobenius_norm": max_delta_norm,
            "rms_delta_frobenius_norm": rms_delta_norm,
            "max_realized_relative_frobenius": max_relative,
            "rms_realized_relative_frobenius": rms_relative,
            "max_over_rms_relative": max_relative / rms_relative,
            "max_expected_relative": max_level * float(row["kappa"]),
            "rms_expected_relative": rms_level,
        }
    )
    return row


def load_pcn_rows(args):
    selected = []
    for path in (args.pcn_primary_csv, args.pcn_nonprimary_csv):
        for row in read_csv(path):
            if (
                row["dataset"] == args.dataset
                and row["paired_arch"] == args.architecture
                and row["model_condition"] == "PCN"
                and "PCNetNoBatchNorm" in row["model_name"]
            ):
                selected.append(dict(row))
    if not selected:
        raise ValueError("No matching PCNetNoBatchNorm audit rows found.")
    checkpoint_paths = {row["checkpoint_path"] for row in selected}
    if len(checkpoint_paths) != 1:
        raise ValueError(f"PCN rows reference multiple checkpoints: {checkpoint_paths}")
    checkpoint = Path(next(iter(checkpoint_paths)))
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    primary = sorted(
        (row for row in selected if row["tensor_name"] != "linear.bias"),
        key=lambda row: int(float(row["semantic_order_index"])),
    )
    biases = [row for row in selected if row["tensor_name"] == "linear.bias"]
    if len(biases) != 1:
        raise ValueError(f"Expected one selected PCN classifier bias, found {len(biases)}")
    rows = primary + biases
    for order, row in enumerate(rows, start=1):
        row["order"] = order
        row["model_condition"] = "PCNetNoBatchNorm"
        row["selection_source"] = row.get("selection_source", "PCN mismatch selector")
    return rows, checkpoint


def wrn_checkpoint_path(root, dataset, architecture):
    model_name = f"wrn_{architecture.removeprefix('WRN_').lower()}_cifar"
    run_name = f"custom_noresize_{dataset}_{model_name}"
    return (
        root
        / dataset
        / "custom_noresize"
        / model_name
        / run_name
        / f"{run_name}_best_ckpt.pth"
    )


def load_wrn_rows(args):
    model_name = f"wrn_{args.architecture.removeprefix('WRN_').lower()}_cifar"
    checkpoint = wrn_checkpoint_path(
        args.wrn_checkpoint_root, args.dataset, args.architecture
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
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
        additive_scale_mode="max_abs",
    )

    rows = []
    for name, parameter in model.named_parameters():
        if not helper._should_noise_param(name, parameter):
            continue
        values = parameter.detach().double()
        max_abs = float(values.abs().max().item())
        rms = float(values.square().mean().sqrt().item())
        module = helper._param_to_module.get(name)
        rows.append(
            {
                "order": len(rows) + 1,
                "dataset": args.dataset,
                "paired_arch": args.architecture,
                "model_condition": "WD1e-3 final-drop0.25 WRN unfused",
                "model_name": model_name,
                "tensor_name": name,
                "module_type": type(module).__name__ if module is not None else "",
                "tensor_shape": "x".join(str(value) for value in parameter.shape),
                "num_elements": parameter.numel(),
                "max_abs": max_abs,
                "rms": rms,
                "kappa": max_abs / rms if rms else math.nan,
                "all_zero": rms == 0.0,
                "selection_source": "FixedMismatchHelper._should_noise_param(noise_to_norm=False)",
                "checkpoint_path": str(checkpoint.resolve()),
            }
        )
    if not rows:
        raise ValueError("WRN mismatch selector returned no parameters.")
    return rows, checkpoint


def accuracy_row(args, mismatch_type, level):
    matches = [
        row
        for row in read_csv(args.accuracy_csv)
        if row["dataset"] == args.dataset
        and row["architecture"] == args.architecture
        and row["mismatch_type"] == mismatch_type
        and math.isclose(float(row["level"]), level, rel_tol=0.0, abs_tol=1e-12)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one accuracy row for {mismatch_type} level {level}, found {len(matches)}"
        )
    row = matches[0]
    return {
        "mismatch_type": mismatch_type,
        "level": level,
        "pcn_accuracy": float(row["PCNetNoBatchNorm"]),
        "wrn_accuracy": float(
            row["WD1e-3 final-drop0.25 WRN unfused recal-BN"]
        ),
        "trial_counts": row["trial counts"],
    }


def summarize_model(rows):
    kappas = np.asarray([float(row["kappa"]) for row in rows], dtype=float)
    clean_sq = sum(float(row["clean_frobenius_norm"]) ** 2 for row in rows)
    max_delta_sq = sum(float(row["max_delta_frobenius_norm"]) ** 2 for row in rows)
    rms_delta_sq = sum(float(row["rms_delta_frobenius_norm"]) ** 2 for row in rows)
    return {
        "model_condition": rows[0]["model_condition"],
        "selected_tensor_count": len(rows),
        "selected_parameter_count": sum(int(row["num_elements"]) for row in rows),
        "kappa_mean": float(kappas.mean()),
        "kappa_median": float(np.median(kappas)),
        "kappa_min": float(kappas.min()),
        "kappa_max": float(kappas.max()),
        "max_model_relative_frobenius": math.sqrt(max_delta_sq / clean_sq),
        "rms_model_relative_frobenius": math.sqrt(rms_delta_sq / clean_sq),
        "tensor_count_max_stronger_than_rms": sum(
            float(row["max_over_rms_relative"]) > 1.0 for row in rows
        ),
    }


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows):
    lines = [
        "| order | tensor | type | shape | max_abs | rms | kappa | rel(max 0.10) | rel(RMS 1.00) | max/RMS |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['order']} | `{row['tensor_name']}` | {row['module_type']} | "
            f"`{row['tensor_shape']}` | {float(row['max_abs']):.6g} | "
            f"{float(row['rms']):.6g} | {float(row['kappa']):.4f} | "
            f"{float(row['max_realized_relative_frobenius']):.4f} | "
            f"{float(row['rms_realized_relative_frobenius']):.4f} | "
            f"{float(row['max_over_rms_relative']):.4f} |"
        )
    return lines


def render_markdown(args, accuracy_rows, summaries, rows_by_model, checkpoints):
    lines = [
        "# Max-Abs versus RMS Reversal Audit: CIFAR-100 WRN-28-4",
        "",
        "This is a read-only clean-checkpoint audit of the exact parameter selectors used by each evaluator.",
        "The max-abs and RMS columns use the same controlled Gaussian realization within each tensor.",
        "",
        "## Observed reversal",
        "",
        "| mismatch | level | PCNetNoBatchNorm | final WRN unfused recal-BN | PCN - WRN | trials |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in accuracy_rows:
        lines.append(
            f"| {row['mismatch_type']} | {row['level']:g} | {row['pcn_accuracy']:.2f} | "
            f"{row['wrn_accuracy']:.2f} | {row['pcn_accuracy'] - row['wrn_accuracy']:+.2f} | "
            f"{row['trial_counts']} |"
        )
    lines.extend(
        [
            "",
            "## Network summary",
            "",
            "| model | tensors | parameters | mean kappa | median kappa | min | max | rel(max 0.10) | rel(RMS 1.00) | tensors max > RMS |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for summary in summaries:
        lines.append(
            f"| {summary['model_condition']} | {summary['selected_tensor_count']} | "
            f"{summary['selected_parameter_count']} | {summary['kappa_mean']:.4f} | "
            f"{summary['kappa_median']:.4f} | {summary['kappa_min']:.4f} | "
            f"{summary['kappa_max']:.4f} | {summary['max_model_relative_frobenius']:.4f} | "
            f"{summary['rms_model_relative_frobenius']:.4f} | "
            f"{summary['tensor_count_max_stronger_than_rms']}/{summary['selected_tensor_count']} |"
        )

    for condition, rows in rows_by_model.items():
        lines.extend(["", f"## {condition}: all selected tensors", ""])
        lines.extend(markdown_table(rows))

    lines.extend(
        [
            "",
            "## Candidate layers",
            "",
            "The strongest candidates are tensors with the largest `max/RMS` ratio, equivalently the largest kappa.",
            "These receive the largest extra relative perturbation under max-abs 0.10 compared with RMS 1.00.",
        ]
    )
    for condition, rows in rows_by_model.items():
        top = sorted(rows, key=lambda row: float(row["kappa"]), reverse=True)[:6]
        lines.extend(["", f"### {condition}", ""])
        lines.extend(markdown_table(top))

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "At these two reported levels, `max/RMS = 0.10 * kappa` for every tensor. "
            "Therefore kappa above 10 means max-abs 0.10 is stronger than RMS 1.00 for that tensor; "
            "kappa below 10 means it is weaker.",
            "",
            "This audit localizes different perturbation allocation but does not establish functional causality. "
            "A one-tensor-at-a-time mismatch intervention is required to determine which candidates cause the accuracy reversal.",
            "",
            "## Checkpoints",
            "",
            f"- PCN: `{checkpoints['PCNetNoBatchNorm']}`",
            f"- WRN: `{checkpoints['WD1e-3 final-drop0.25 WRN unfused']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pcn_rows, pcn_checkpoint = load_pcn_rows(args)
    wrn_rows, wrn_checkpoint = load_wrn_rows(args)
    rows_by_model = {
        "PCNetNoBatchNorm": pcn_rows,
        "WD1e-3 final-drop0.25 WRN unfused": wrn_rows,
    }
    all_rows = []
    for rows in rows_by_model.values():
        for row in rows:
            all_rows.append(
                add_realized_metrics(row, args.max_level, args.rms_level, args.base_seed)
            )

    accuracy_rows = [
        accuracy_row(args, "additive_max", args.max_level),
        accuracy_row(args, "additive_rms", args.rms_level),
    ]
    summaries = [summarize_model(rows) for rows in rows_by_model.values()]
    checkpoints = {
        "PCNetNoBatchNorm": str(pcn_checkpoint.resolve()),
        "WD1e-3 final-drop0.25 WRN unfused": str(wrn_checkpoint.resolve()),
    }

    write_csv(args.output_dir / "per_tensor.csv", all_rows)
    write_csv(args.output_dir / "model_summary.csv", summaries)
    (args.output_dir / "summary.md").write_text(
        render_markdown(args, accuracy_rows, summaries, rows_by_model, checkpoints)
    )
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "dataset": args.dataset,
                "architecture": args.architecture,
                "max_level": args.max_level,
                "rms_level": args.rms_level,
                "base_seed": args.base_seed,
                "controlled_same_noise_within_tensor_across_scale_modes": True,
                "accuracy_csv": str(args.accuracy_csv.resolve()),
                "checkpoints": checkpoints,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(args.output_dir / "summary.md")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate a presentation table for WRN BN-recalibration runs."""
import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baseline.run_baseline import (  # noqa: E402
    FixedMismatchHelper,
    build_model,
    get_baseline_config,
    infer_num_classes,
    load_model_weights,
    maybe_fold_norms,
)
from baseline.run_wrn_bn_recalibration_experiment import wrn_label_to_model_name  # noqa: E402
from weight_range_audit import collect_wrn_weight_range_audit_rows, save_audit_csv  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--aggregate_csv", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--pcn_kappa_csv", default="logs/kappa_audit/all_primary.csv")
    p.add_argument(
        "--unfolded_recal_aggregate_csv",
        default=None,
        help="Optional aggregate CSV from the unfused WRN BN-recalibration run to add the no-fold all-BN recalibrated accuracy column.",
    )
    p.add_argument("--checkpoint_root", default="checkpoint/baselines")
    p.add_argument("--case", default="custom_noresize")
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--mismatch_parameter_policy",
        choices=["existing", "bn_fold_no_mismatch_bias"],
        default="existing",
    )
    return p.parse_args()


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, object]]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fnum(value: object, digits: int = 2) -> str:
    x = float(value)
    if math.isnan(x):
        return "nan"
    return f"{x:.{digits}f}"


def checkpoint_path(root: Path, dataset: str, case: str, model_name: str) -> Path:
    run_name = f"{case}_{dataset}_{model_name}"
    return root / dataset / case / model_name / run_name / f"{run_name}_best_ckpt.pth"


def module_param_to_module(model) -> Dict[str, torch.nn.Module]:
    mapping = {}
    for module_name, module in model.named_modules():
        prefix = f"{module_name}." if module_name else ""
        for local_name, _ in module.named_parameters(recurse=False):
            mapping[f"{prefix}{local_name}"] = module
    return mapping


def collect_folded_wrn_kappa(
    aggregate_rows: Iterable[Dict[str, str]],
    *,
    checkpoint_root: Path,
    case: str,
    device: torch.device,
    policy: str,
) -> List[Dict[str, object]]:
    seen: List[Tuple[str, str]] = []
    for row in aggregate_rows:
        key = (row["dataset"], row["architecture"])
        if key not in seen:
            seen.append(key)

    all_primary: List[Dict[str, object]] = []
    all_nonprimary: List[Dict[str, object]] = []
    for dataset, arch in seen:
        model_name = wrn_label_to_model_name(arch)
        cfg = get_baseline_config(
            model_name=model_name,
            pretrained=False,
            case=case,
            prefer_resize=False,
            extra_overrides=None,
        )
        model = build_model(model_name, cfg, num_classes=infer_num_classes(dataset, None)).to(device)
        load_model_weights(model, checkpoint_path(checkpoint_root, dataset, case, model_name), device)
        if policy == "bn_fold_no_mismatch_bias":
            model = maybe_fold_norms(model, mode="wrn_preact_no_mismatch_bias").to(device)
        helper = FixedMismatchHelper(
            model=model,
            noise_sigma=0.0,
            noise_type="additive",
            noise_to_norm=False,
            include_buffers=True,
            exclude_param_names=getattr(model, "_mismatch_excluded_param_names", set()),
        )
        primary, nonprimary = collect_wrn_weight_range_audit_rows(
            model,
            dataset=dataset,
            model_name=model_name,
            should_noise_param=helper._should_noise_param,
            param_to_module=module_param_to_module(model),
        )
        for out in primary + nonprimary:
            out["paired_arch"] = arch
            out["mismatch_parameter_policy"] = policy
            out["excluded_mismatch_params"] = ";".join(sorted(getattr(model, "_mismatch_excluded_param_names", set())))
        all_primary.extend(primary)
        all_nonprimary.extend(nonprimary)

    return all_primary, all_nonprimary


def rows_by_pair(rows: Iterable[Dict[str, str]], condition: str) -> Dict[Tuple[str, str], List[Dict[str, str]]]:
    out: Dict[Tuple[str, str], List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("model_condition") == condition:
            out[(row["dataset"], row.get("paired_arch", row.get("architecture", "")))].append(row)
    for values in out.values():
        values.sort(key=lambda r: int(float(r["semantic_order_index"])))
    return out


def comparison_label(pcn_rows: List[Dict[str, str]], wrn_rows: List[Dict[str, str]]) -> str:
    pcn = np.array([float(r["kappa"]) for r in pcn_rows], dtype=float)
    wrn = np.array([float(r["kappa"]) for r in wrn_rows], dtype=float)
    if not len(pcn) or not len(wrn):
        return "kappa comparison unavailable"
    p_med = float(np.median(pcn))
    w_med = float(np.median(wrn))
    p_mean = float(np.mean(pcn))
    w_mean = float(np.mean(wrn))
    if p_med < 0.9 * w_med and p_mean < 0.9 * w_mean:
        return "PCN has broadly lower kappa across selected layers."
    if w_med < 0.9 * p_med and w_mean < 0.9 * p_mean:
        return "WRN has lower kappa."
    if abs(p_med - w_med) <= 0.1 * max(p_med, w_med):
        return "PCN and WRN have similar kappa distributions."
    return "differences are concentrated in a small number of layers."


def markdown_kappa_table(rows: List[Dict[str, str]]) -> List[str]:
    lines = [
        "| order | parameter name | module type | shape | max_abs | rms | kappa |",
        "|---:|---|---|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {order} | `{name}` | {module} | `{shape}` | {max_abs} | {rms} | {kappa} |".format(
                order=int(float(row["semantic_order_index"])),
                name=row["tensor_name"],
                module=row["module_type"],
                shape=row["tensor_shape"],
                max_abs=fnum(row["max_abs"], 6),
                rms=fnum(row["rms"], 6),
                kappa=fnum(row["kappa"], 6),
            )
        )
    return lines


def summary_row(name: str, rows: List[Dict[str, str]]) -> str:
    kappas = np.array([float(r["kappa"]) for r in rows], dtype=float)
    return (
        f"| {name} | {len(rows)} | {np.median(kappas):.6g} | {np.mean(kappas):.6g} | "
        f"{np.min(kappas):.6g} | {np.max(kappas):.6g} |"
    )


def aggregate_key(row: Dict[str, str]) -> Tuple[str, str, str, float]:
    return (row["dataset"], row["architecture"], row["mismatch_type"], float(row["mismatch_level"]))


def generate_markdown(
    aggregate_rows: List[Dict[str, str]],
    pcn_by_pair: Dict[Tuple[str, str], List[Dict[str, str]]],
    wrn_by_pair: Dict[Tuple[str, str], List[Dict[str, str]]],
    policy: str,
    unfolded_recal_by_key: Dict[Tuple[str, str, str, float], Dict[str, str]] | None = None,
) -> str:
    if unfolded_recal_by_key:
        missing = [aggregate_key(row) for row in aggregate_rows if aggregate_key(row) not in unfolded_recal_by_key]
        if missing:
            preview = ", ".join(str(key) for key in missing[:5])
            raise ValueError(f"Missing unfolded recalibration rows for {len(missing)} aggregate rows: {preview}")

    grouped: Dict[Tuple[str, str, str], List[Dict[str, str]]] = defaultdict(list)
    for row in aggregate_rows:
        grouped[(row["dataset"], row["architecture"], row["mismatch_type"])].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda r: float(r["mismatch_level"]))

    include_unfolded_recal = bool(unfolded_recal_by_key)
    lines = [
        "# WRN BN Recalibration Presentation Table - 5120 Calibration Samples - bn_fold_no_mismatch_bias",
        "",
        "Accuracy values are percentages. Calibration uses 5120 train images with batch size 128, so the calibration pass has exactly 40 full batches.",
        "",
        "****************************************************************************",
        "WRN folded recal-BN acc: Fold foldable BNs into their previous convolutions",
        "and then recalibrate the rest BNs.",
        "",
        "WRN unfused recal-BN acc: Add mismatch first without BN folding",
        "and then recalibrate all BN modules.",
        "****************************************************************************",
        "",
        f"Mismatch parameter policy: `{policy}`. Supported WRN preactivation Conv/BN pairs are folded before mismatch; folded Conv bias parameters induced by BN folding are excluded from mismatch.",
        "",
    ]

    last_pair = None
    for dataset, arch, mismatch_type in sorted(grouped):
        pair = (dataset, arch)
        if pair != last_pair:
            if last_pair is not None:
                lines.extend(["", "---", ""])
            pcn_rows = pcn_by_pair.get(pair, [])
            wrn_rows = wrn_by_pair.get(pair, [])
            lines.extend(
                [
                    f"## {dataset} {arch} kappa audit",
                    "",
                    "PCN/NODE selected Conv/Linear weights:",
                ]
            )
            lines.extend(markdown_kappa_table(pcn_rows) if pcn_rows else ["PCN/NODE kappa rows unavailable."])
            lines.extend(["", "WRN selected Conv/Linear weights after configured parameter policy:"])
            lines.extend(markdown_kappa_table(wrn_rows) if wrn_rows else ["WRN kappa rows unavailable."])
            if pcn_rows and wrn_rows:
                lines.extend(
                    [
                        "",
                        "| model | layers | median kappa | mean kappa | min | max |",
                        "|---|---:|---:|---:|---:|---:|",
                        summary_row("PCN/NODE", pcn_rows),
                        summary_row("WRN", wrn_rows),
                        "",
                        f"Kappa comparison: {comparison_label(pcn_rows, wrn_rows)}",
                    ]
                )
            last_pair = pair

        if include_unfolded_recal:
            header = "| mismatch level | trials | PCN/NODE acc | WRN folded frozen-BN acc | WRN folded recal-BN acc | WRN unfused recal-BN acc | folded BN recovery | remaining PCN gap folded recal |"
            sep = "|---:|---:|---:|---:|---:|---:|---:|---:|"
        else:
            header = "| mismatch level | trials | PCN/NODE acc | WRN folded frozen-BN acc | WRN folded recal-BN acc | BN recovery | remaining PCN gap |"
            sep = "|---:|---:|---:|---:|---:|---:|---:|"
        lines.extend(["", f"## {dataset} {arch} {mismatch_type}", "", header, sep])
        for row in grouped[(dataset, arch, mismatch_type)]:
            base = (
                f"| {float(row['mismatch_level']):.2f} | {int(float(row['num_trials']))} | "
                f"{fnum(row['pcn_node_mean_accuracy'])} | "
                f"{fnum(row['paired_wrn_frozen_bn_mean_accuracy'])} | "
                f"{fnum(row['wrn_recalibrated_bn_mean_accuracy'])} | "
            )
            if include_unfolded_recal:
                unfolded = unfolded_recal_by_key[aggregate_key(row)]
                lines.append(
                    base
                    + f"{fnum(unfolded['wrn_recalibrated_bn_mean_accuracy'])} | "
                    + f"{fnum(row['mean_paired_bn_recovery'])} | "
                    + f"{fnum(row['remaining_pcn_gap'])} |"
                )
            else:
                lines.append(
                    base
                    + f"{fnum(row['mean_paired_bn_recovery'])} | "
                    + f"{fnum(row['remaining_pcn_gap'])} |"
                )
    lines.append("")
    return "\n".join(lines)


def main():
    args = parse_args()
    aggregate_csv = Path(args.aggregate_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    aggregate_rows = read_csv(aggregate_csv)
    unfolded_recal_by_key = None
    if args.unfolded_recal_aggregate_csv:
        unfolded_rows = read_csv(Path(args.unfolded_recal_aggregate_csv))
        unfolded_recal_by_key = {aggregate_key(row): row for row in unfolded_rows}

    primary, nonprimary = collect_folded_wrn_kappa(
        aggregate_rows,
        checkpoint_root=Path(args.checkpoint_root),
        case=args.case,
        device=torch.device(args.device),
        policy=args.mismatch_parameter_policy,
    )
    save_audit_csv(str(output_dir / "folded_wrn_kappa_primary.csv"), primary, nonprimary)

    pcn_rows = read_csv(Path(args.pcn_kappa_csv))
    pcn_by_pair = rows_by_pair(pcn_rows, "PCN")
    wrn_by_pair = rows_by_pair(primary, "WRN")

    presentation_rows = []
    if unfolded_recal_by_key:
        missing = [aggregate_key(row) for row in aggregate_rows if aggregate_key(row) not in unfolded_recal_by_key]
        if missing:
            preview = ", ".join(str(key) for key in missing[:5])
            raise ValueError(f"Missing unfolded recalibration rows for {len(missing)} aggregate rows: {preview}")

    for row in aggregate_rows:
        out = {
            "dataset": row["dataset"],
            "architecture": row["architecture"],
            "mismatch_type": row["mismatch_type"],
            "mismatch_level": row["mismatch_level"],
            "num_trials": row["num_trials"],
            "pcn_node_mean_accuracy": row["pcn_node_mean_accuracy"],
            "paired_wrn_frozen_bn_mean_accuracy": row["paired_wrn_frozen_bn_mean_accuracy"],
            "wrn_recalibrated_bn_mean_accuracy": row["wrn_recalibrated_bn_mean_accuracy"],
            "mean_paired_bn_recovery": row["mean_paired_bn_recovery"],
            "remaining_pcn_gap": row["remaining_pcn_gap"],
            "pcn_node_column": row["pcn_node_column"],
            "mismatch_parameter_policy": row.get("mismatch_parameter_policy", args.mismatch_parameter_policy),
        }
        if unfolded_recal_by_key:
            unfolded = unfolded_recal_by_key[aggregate_key(row)]
            out["wrn_unfolded_recalibrated_bn_mean_accuracy"] = unfolded["wrn_recalibrated_bn_mean_accuracy"]
        presentation_rows.append(out)
    write_csv(output_dir / "full_presentation_table.csv", presentation_rows)

    md = generate_markdown(aggregate_rows, pcn_by_pair, wrn_by_pair, args.mismatch_parameter_policy, unfolded_recal_by_key)
    (output_dir / "full_presentation_table.md").write_text(md)
    print(f"Wrote {output_dir / 'full_presentation_table.md'}")
    print(f"Wrote {output_dir / 'full_presentation_table.csv'}")
    print(f"Wrote {output_dir / 'folded_wrn_kappa_primary.csv'}")


if __name__ == "__main__":
    main()

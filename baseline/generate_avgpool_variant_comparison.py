#!/usr/bin/env python3
"""Merge the WRN-28-2 pooling study into selected established comparison columns."""

import argparse
import csv
import hashlib
import json
import math
import pickle
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = ROOT / "logs/combined_pcn_wrn_mismatch_finaldrop025_complete/full_accuracy_comparison.csv"
DEFAULT_STUDY = ROOT / "logs/wrn28_2_avgpool_study"
DEFAULT_OUTPUT = ROOT / "logs/combined_pcn_wrn_mismatch_avgpool_variants_complete"
DEFAULT_PCN_MAX_SQRT = (
    ROOT
    / "logs/pcn_legacy_no_x_cifar100_max_sqrt_local/cifar100/WRN_28_2/max_sqrt_additive/result.pkl"
)

OLD_COLUMNS = [
    "PCNetNoBatchNorm",
    "WRN unfused recal-BN",
    "WD1e-3 WRN unfused recal-BN",
    "WD1e-3 final-drop0.25 WRN unfused recal-BN",
]
NEW_MODELS = {
    "AvgPool BN": "wrn_28_2_cifar_avgpool",
    "Stride-2 BN": "wrn_28_2_cifar_avgpool_shortcut",
    "AvgPool BN-free": "wrn_28_2_cifar_nobn_avgpool",
    "Stride-2 BN-free": "wrn_28_2_cifar_nobn_avgpool_shortcut",
}
FAMILIES = {
    "additive_max": ("Corrected max-scaled additive", "max_additive"),
    "additive_rms": ("RMS-scaled additive", "rms_additive"),
    "additive_max_sqrt": ("Per-filter max-sqrt additive", "max_sqrt_additive"),
    "multiplicative": ("Multiplicative", "multiplicative"),
}
PAIRS = [(dataset, "WRN_28_2") for dataset in ("cifar10", "cifar100")]


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path, text):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def atomic_csv(path, rows, columns):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def load_base(path):
    index = {}
    for row in read_csv(path):
        key = (row["dataset"], row["architecture"], row["mismatch_type"], float(row["level"]))
        if key in index:
            raise ValueError(f"Duplicate established result: {key}")
        index[key] = row
    return index


def study_value(row, model_name):
    field = "wrn_nobn_mean_accuracy" if "_nobn_" in model_name else "wrn_recalibrated_bn_mean_accuracy"
    return f"{float(row[field]):.2f}"


def load_study(root):
    index = {}
    sources = []
    for dataset in ("cifar10", "cifar100"):
        for label, model_name in NEW_MODELS.items():
            for report_type, (_, family) in FAMILIES.items():
                path = root / "evaluation" / dataset / model_name / family / "full_aggregate.csv"
                if not path.is_file():
                    raise FileNotFoundError(path)
                sources.append(path)
                for row in read_csv(path):
                    if int(row["num_trials"]) != 10:
                        raise ValueError(f"Expected 10 trials in {path}")
                    key = (dataset, "WRN_28_2", report_type, float(row["mismatch_level"]), label)
                    if key in index:
                        raise ValueError(f"Duplicate pooling-study result: {key}")
                    index[key] = study_value(row, model_name)
    return index, sources


def load_pcn_max_sqrt(path):
    with path.open("rb") as handle:
        result = pickle.load(handle)
    time_entries = list(result.values())
    if len(time_entries) != 1:
        raise ValueError(f"Expected one t_end result in {path}")
    methods = time_entries[0]["noise_acc_spec"]
    if len(methods) != 1:
        raise ValueError(f"Expected one solver result in {path}")
    values = next(iter(methods.values()))
    return {
        float(level): f"{sum(accuracies) / len(accuracies):.2f}"
        for level, accuracies in values.items()
    }


def render_table(lines, rows, columns):
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("|" + "|".join("---:" for _ in columns) + "|")
    for row in rows:
        lines.append("| " + " | ".join(row[column] for column in columns) + " |")
    lines.append("")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_csv", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--study_root", type=Path, default=DEFAULT_STUDY)
    parser.add_argument("--pcn_max_sqrt", type=Path, default=DEFAULT_PCN_MAX_SQRT)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    base = load_base(args.base_csv)
    study, study_sources = load_study(args.study_root)
    pcn_max_sqrt = load_pcn_max_sqrt(args.pcn_max_sqrt)
    report_columns = ["level", "trial counts", *OLD_COLUMNS, *NEW_MODELS]
    csv_columns = ["dataset", "architecture", "mismatch_type", *report_columns]
    output_rows = []
    lines = [
        "# PCNetNoBatchNorm and WRN Pooling-Variant Mismatch Comparison",
        "",
        "Accuracy values are percentages. The four established columns are copied directly from the existing complete report CSV; that source is not modified.",
        "",
        "The new BN models use WD=1e-3, final-feature dropout 0.25, and unfused-BN recalibration after each mismatch realization. The new BN-free models use their searched dataset-specific recipe with WD=1e-3 and final-feature dropout 0.25. Every new result uses 10 trials.",
        "",
        "`AvgPool` uses average-pool downsampling in both the main and shortcut paths. `Stride-2` retains stride-2 main-path convolutions while using average-pool shortcuts. New-model cells outside WRN-28-2 are N/A because only WRN-28-2 variants were trained.",
        "",
        "The older WRN conditions were not evaluated under max-sqrt and are N/A in that section. The CIFAR-100 PCNetNoBatchNorm max-sqrt values come from its existing 10-trial result; CIFAR-10 is unavailable.",
        "",
    ]

    for dataset, architecture in PAIRS:
        lines.extend(["****************************************************************************", "", f"# {dataset} {architecture}", ""])
        report_types = ["additive_max", "additive_rms", "multiplicative"]
        if architecture == "WRN_28_2":
            report_types.insert(2, "additive_max_sqrt")
        for report_type in report_types:
            title, _ = FAMILIES[report_type]
            base_levels = {
                level for ds, arch, kind, level in base
                if (ds, arch, kind) == (dataset, architecture, report_type)
            }
            study_levels = {
                level for ds, arch, kind, level, _ in study
                if (ds, arch, kind) == (dataset, architecture, report_type)
            }
            levels = base_levels | study_levels
            if report_type == "additive_max_sqrt" and dataset == "cifar100":
                levels |= set(pcn_max_sqrt)
            rows = []
            for level in sorted(levels):
                established = base.get((dataset, architecture, report_type, level))
                row = {
                    "level": f"{level:g}",
                    "trial counts": established["trial counts"] if established else "new WRNs=10",
                    **{column: established[column] if established else "N/A" for column in OLD_COLUMNS},
                    **{
                        label: study.get((dataset, architecture, report_type, level, label), "N/A")
                        for label in NEW_MODELS
                    },
                }
                if report_type == "additive_max_sqrt" and dataset == "cifar100":
                    row["PCNetNoBatchNorm"] = pcn_max_sqrt.get(level, "N/A")
                if established and any(row[column] != established[column] for column in OLD_COLUMNS):
                    raise AssertionError("An established result was changed")
                rows.append(row)
                output_rows.append({
                    "dataset": dataset,
                    "architecture": architecture,
                    "mismatch_type": report_type,
                    **row,
                })
            lines.extend([f"## {title}", ""])
            render_table(lines, rows, report_columns)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = args.output_dir / "full_accuracy_comparison.md"
    csv_path = args.output_dir / "full_accuracy_comparison.csv"
    atomic_text(markdown_path, "\n".join(lines) + "\n")
    atomic_csv(csv_path, output_rows, csv_columns)

    source_paths = [args.base_csv, args.pcn_max_sqrt, *study_sources]
    manifest = {
        "established_source": str(args.base_csv.resolve()),
        "established_columns": OLD_COLUMNS,
        "new_models": NEW_MODELS,
        "report_rows": len(output_rows),
        "sources": [
            {"path": str(path.resolve()), "sha256": sha256(path)} for path in source_paths
        ],
    }
    atomic_text(args.output_dir / "report_manifest.json", json.dumps(manifest, indent=2) + "\n")
    print(markdown_path)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate the CIFAR-100 WRN-28-2 report for an independent mismatch seed."""

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "logs/combined_pcn_wrn_mismatch_avgpool_variants_complete/full_accuracy_comparison.csv"
RUN_ROOT = ROOT / "logs/wrn28_2_cifar100_seed456_replication"
OUTPUT = ROOT / "logs/combined_pcn_wrn28_2_cifar100_seed456"

COLUMNS = [
    "PCNetNoBatchNorm (existing reference)",
    "WRN unfused recal-BN",
    "WD1e-3 WRN unfused recal-BN",
    "WD1e-3 final-drop0.25 WRN unfused recal-BN",
    "AvgPool BN",
    "Stride-2 BN",
    "AvgPool BN-free",
    "Stride-2 BN-free",
]
CONDITIONS = {
    "WRN unfused recal-BN": ("wrn_unfused_recal_bn", "bn"),
    "WD1e-3 WRN unfused recal-BN": ("wrn_wd1e3_unfused_recal_bn", "bn"),
    "WD1e-3 final-drop0.25 WRN unfused recal-BN": (
        "wrn_wd1e3_finaldrop025_unfused_recal_bn", "bn"
    ),
    "AvgPool BN": ("avgpool_bn", "bn"),
    "Stride-2 BN": ("stride2_bn", "bn"),
    "AvgPool BN-free": ("avgpool_bnfree", "bnfree"),
    "Stride-2 BN-free": ("stride2_bnfree", "bnfree"),
}
FAMILIES = {
    "additive_max": ("Corrected max-scaled additive", "max_additive"),
    "additive_rms": ("RMS-scaled additive", "rms_additive"),
    "multiplicative": ("Multiplicative", "multiplicative"),
}


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path, content):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content)
    temporary.replace(path)


def atomic_csv(path, rows, columns):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def load_reference():
    return {
        (row["mismatch_type"], float(row["level"])): row["PCNetNoBatchNorm"]
        for row in read_csv(REFERENCE)
        if row["dataset"] == "cifar100" and row["architecture"] == "WRN_28_2"
    }


def load_new_results():
    index = {}
    sources = []
    for column, (condition, kind) in CONDITIONS.items():
        for report_type, (_, family) in FAMILIES.items():
            aggregate = RUN_ROOT / "results" / condition / family / "full_aggregate.csv"
            per_trial = RUN_ROOT / "results" / condition / family / "full_per_trial.csv"
            if not aggregate.is_file() or not per_trial.is_file():
                raise FileNotFoundError(aggregate if not aggregate.is_file() else per_trial)
            sources.extend([aggregate, per_trial])
            trial_rows = read_csv(per_trial)
            zero_seeds = sorted(
                int(row["mismatch_seed"])
                for row in trial_rows
                if float(row["mismatch_level"]) == 0
            )
            if zero_seeds != list(range(456, 466)):
                raise ValueError(f"Unexpected base mismatch seed in {per_trial}")
            for row in read_csv(aggregate):
                if int(row["num_trials"]) != 10:
                    raise ValueError(f"Expected 10 trials in {aggregate}")
                accuracy_field = (
                    "wrn_recalibrated_bn_mean_accuracy"
                    if kind == "bn"
                    else "wrn_nobn_mean_accuracy"
                )
                key = (report_type, float(row["mismatch_level"]), column)
                index[key] = f"{float(row[accuracy_field]):.2f}"
    return index, sources


def render_table(lines, rows):
    columns = ["level", "trial counts", *COLUMNS]
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("|" + "|".join("---:" for _ in columns) + "|")
    for row in rows:
        lines.append("| " + " | ".join(row[column] for column in columns) + " |")
    lines.append("")


def main():
    reference = load_reference()
    results, sources = load_new_results()
    lines = [
        "# CIFAR-100 WRN-28-2 Independent-Seed Mismatch Comparison",
        "",
        "All seven WRN columns were rerun with mismatch base seed 456 and 10 trials. BN columns use unfused-BN recalibration with 5,120 calibration images and calibration subset seed 20240618.",
        "",
        "PCNetNoBatchNorm was not rerun and is included only as an existing reference from the prior complete report. Max-sqrt is omitted because it was not part of this independent-seed run.",
        "",
    ]
    output_rows = []
    for report_type, (title, _) in FAMILIES.items():
        levels = sorted({level for kind, level, _ in results if kind == report_type})
        table_rows = []
        for level in levels:
            row = {
                "level": f"{level:g}",
                "trial counts": "WRNs=10; PCN=existing",
                "PCNetNoBatchNorm (existing reference)": reference.get((report_type, level), "N/A"),
                **{
                    column: results[(report_type, level, column)]
                    for column in CONDITIONS
                },
            }
            table_rows.append(row)
            output_rows.append({"mismatch_type": report_type, **row})
        lines.extend([f"## {title}", ""])
        render_table(lines, table_rows)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    markdown = OUTPUT / "full_accuracy_comparison.md"
    csv_path = OUTPUT / "full_accuracy_comparison.csv"
    atomic_text(markdown, "\n".join(lines) + "\n")
    atomic_csv(csv_path, output_rows, ["mismatch_type", "level", "trial counts", *COLUMNS])
    manifest = {
        "dataset": "cifar100",
        "architecture": "WRN_28_2",
        "wrn_mismatch_seed": 456,
        "wrn_trials": 10,
        "pcn_reference_rerun": False,
        "reference": {"path": str(REFERENCE.resolve()), "sha256": sha256(REFERENCE)},
        "sources": [{"path": str(path.resolve()), "sha256": sha256(path)} for path in sources],
    }
    atomic_text(OUTPUT / "report_manifest.json", json.dumps(manifest, indent=2) + "\n")
    print(markdown)


if __name__ == "__main__":
    main()

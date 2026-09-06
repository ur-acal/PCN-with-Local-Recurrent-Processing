#!/usr/bin/env python3
"""Add the CIFAR-100 BN-free/no-convolution-bias runs to two WRN reports."""

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_SOURCE = (
    ROOT
    / "logs/combined_pcn_wrn_mismatch_finaldrop025_complete/full_accuracy_comparison.csv"
)
SEED456_SOURCE = (
    ROOT / "logs/combined_pcn_wrn28_2_cifar100_seed456/full_accuracy_comparison.csv"
)
RUN_ROOT = ROOT / "logs/wrn_nobn_no_bias_mismatch_cifar100/results"
TRAIN_ROOT = ROOT / "logs/wrn_nobn_no_bias_wd1e3_finaldrop025_cifar100"
COMPLETE_OUTPUT = (
    ROOT / "logs/combined_pcn_wrn_mismatch_nobias_cifar100_complete"
)
SEED456_OUTPUT = (
    ROOT / "logs/combined_pcn_wrn28_2_cifar100_seed456_nobias"
)

NEW_COLUMNS = {
    "BN-free/no-conv-bias, WD5e-4/drop0": "original_wd5e4_drop0",
    "BN-free/no-conv-bias, WD1e-3/final-drop0.25": "wd1e3_finaldrop025",
}
ARCHITECTURES = ("WRN_16_2", "WRN_16_4", "WRN_28_2", "WRN_28_4")
MODEL_NAMES = {
    architecture: architecture.lower() + "_cifar_nobn_no_bias"
    for architecture in ARCHITECTURES
}
FAMILIES = {
    "additive_max": ("Corrected max-scaled additive", "max_additive"),
    "additive_rms": ("RMS-scaled additive", "rms_additive"),
    "multiplicative": ("Multiplicative", "multiplicative"),
}
COLLAPSED = {("wd1e3_finaldrop025", "WRN_16_4")}
SEED456_REFERENCE_COLUMNS = [
    "PCNetNoBatchNorm (existing reference)",
    "WRN unfused recal-BN",
    "WD1e-3 final-drop0.25 WRN unfused recal-BN",
]


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


def load_new_results():
    index = {}
    sources = []
    for column, recipe in NEW_COLUMNS.items():
        for architecture in ARCHITECTURES:
            if (recipe, architecture) in COLLAPSED:
                continue
            model_name = MODEL_NAMES[architecture]
            for report_type, (_, family) in FAMILIES.items():
                path = RUN_ROOT / recipe / model_name / family / "full_aggregate.csv"
                if not path.is_file():
                    raise FileNotFoundError(path)
                sources.append(path)
                rows = read_csv(path)
                if not rows or any(int(row["num_trials"]) != 10 for row in rows):
                    raise ValueError(f"Expected nonempty 10-trial aggregate: {path}")
                for row in rows:
                    key = (architecture, report_type, float(row["mismatch_level"]), column)
                    if key in index:
                        raise ValueError(f"Duplicate no-bias result: {key}")
                    index[key] = f'{float(row["wrn_nobn_mean_accuracy"]):.2f}'
    return index, sources


def render_table(lines, rows, columns):
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("|" + "|".join("---:" for _ in columns) + "|")
    for row in rows:
        lines.append("| " + " | ".join(row[column] for column in columns) + " |")
    lines.append("")


def new_trial_note(architecture, report_type, level, results):
    notes = []
    for column, recipe in NEW_COLUMNS.items():
        short_name = "no-bias WD5e-4" if recipe == "original_wd5e4_drop0" else "no-bias WD1e-3/drop0.25"
        if (recipe, architecture) in COLLAPSED:
            notes.append(f"{short_name}=collapsed")
        elif (architecture, report_type, level, column) in results:
            notes.append(f"{short_name}=10")
        else:
            notes.append(f"{short_name}=N/A")
    return "; ".join(notes)


def generate_complete(results, sources):
    source_rows = [
        row for row in read_csv(CANONICAL_SOURCE) if row["dataset"] == "cifar100"
    ]
    if not source_rows:
        raise ValueError("Canonical report contains no CIFAR-100 rows")
    source_columns = list(source_rows[0])
    output_columns = [*source_columns, *NEW_COLUMNS]
    output_rows = []
    for source in source_rows:
        architecture = source["architecture"]
        report_type = source["mismatch_type"]
        level = float(source["level"])
        row = dict(source)
        row["trial counts"] = (
            source["trial counts"] + "; "
            + new_trial_note(architecture, report_type, level, results)
        )
        for column, recipe in NEW_COLUMNS.items():
            if (recipe, architecture) in COLLAPSED:
                row[column] = "N/A (collapsed)"
            else:
                row[column] = results.get(
                    (architecture, report_type, level, column), "N/A"
                )
        for column in source_columns:
            if column != "trial counts" and row[column] != source[column]:
                raise AssertionError(f"Canonical value changed: {column}")
        output_rows.append(row)

    lines = [
        "# CIFAR-100 PCNetNoBatchNorm and WRN No-Bias Mismatch Comparison",
        "",
        "All established columns and values are copied from the canonical complete report; that source is unchanged. Two BN-free WRN variants with no convolution biases are appended. Both retain the standard learned shortcuts and stride-2 main-path convolutions; only the final linear layer has a bias.",
        "",
        "The first new variant uses the original WRN recipe (WD=5e-4, final dropout 0). The second changes only WD to 1e-3 and final-feature dropout to 0.25. New mismatch results use base seed 123 and 10 trials.",
        "",
        "The WD=1e-3/final-drop0.25 WRN-16-4 collapsed at epoch 80 with a nonfinite loss. Its checkpoint and derived mismatch CSVs are invalid and are deliberately reported as `N/A (collapsed)`.",
        "",
    ]
    table_columns = ["level", "trial counts", *source_columns[5:], *NEW_COLUMNS]
    for architecture in ARCHITECTURES:
        lines.extend(["****************************************************************************", "", f"# CIFAR-100 {architecture}", ""])
        for report_type, (title, _) in FAMILIES.items():
            rows = [
                row for row in output_rows
                if row["architecture"] == architecture
                and row["mismatch_type"] == report_type
            ]
            if not rows:
                raise ValueError(f"Missing canonical section: {architecture} {report_type}")
            lines.extend([f"## {title}", ""])
            render_table(lines, rows, table_columns)

    COMPLETE_OUTPUT.mkdir(parents=True, exist_ok=True)
    atomic_text(COMPLETE_OUTPUT / "full_accuracy_comparison.md", "\n".join(lines) + "\n")
    atomic_csv(COMPLETE_OUTPUT / "full_accuracy_comparison.csv", output_rows, output_columns)
    write_manifest(
        COMPLETE_OUTPUT,
        CANONICAL_SOURCE,
        sources,
        {"scope": "cifar100_all_architectures", "mismatch_seed": 123},
    )


def generate_seed456(results, sources):
    source_rows = read_csv(SEED456_SOURCE)
    output_columns = [
        "mismatch_type", "level", "trial counts",
        *SEED456_REFERENCE_COLUMNS, *NEW_COLUMNS,
    ]
    output_rows = []
    for source in source_rows:
        report_type = source["mismatch_type"]
        level = float(source["level"])
        row = {
            "mismatch_type": report_type,
            "level": source["level"],
            "trial counts": source["trial counts"] + "; no-bias WRNs=10",
            **{column: source[column] for column in SEED456_REFERENCE_COLUMNS},
            **{
                column: results.get(("WRN_28_2", report_type, level, column), "N/A")
                for column in NEW_COLUMNS
            },
        }
        output_rows.append(row)

    lines = [
        "# CIFAR-100 WRN-28-2 Seed-456 and No-Bias Mismatch Comparison",
        "",
        "The three reference columns are copied unchanged from the existing seed-456 report. Its WRN columns use mismatch base seed 456; PCNetNoBatchNorm remains the existing seed-123 reference. The two appended BN-free/no-convolution-bias WRNs also use base seed 123 and 10 trials.",
        "",
        "The no-bias models retain learned shortcuts and stride-2 main-path convolutions. Their recipes differ only in WD/final dropout: 5e-4/0 versus 1e-3/0.25.",
        "",
    ]
    table_columns = ["level", "trial counts", *SEED456_REFERENCE_COLUMNS, *NEW_COLUMNS]
    for report_type, (title, _) in FAMILIES.items():
        rows = [row for row in output_rows if row["mismatch_type"] == report_type]
        if not rows:
            raise ValueError(f"Missing seed-456 section: {report_type}")
        lines.extend([f"## {title}", ""])
        render_table(lines, rows, table_columns)

    SEED456_OUTPUT.mkdir(parents=True, exist_ok=True)
    atomic_text(SEED456_OUTPUT / "full_accuracy_comparison.md", "\n".join(lines) + "\n")
    atomic_csv(SEED456_OUTPUT / "full_accuracy_comparison.csv", output_rows, output_columns)
    write_manifest(
        SEED456_OUTPUT,
        SEED456_SOURCE,
        sources,
        {
            "scope": "cifar100_wrn28_2",
            "reference_wrn_mismatch_seed": 456,
            "pcn_and_nobias_mismatch_seed": 123,
        },
    )


def write_manifest(output, reference, sources, metadata):
    collapse_log = TRAIN_ROOT / "train_logs/wrn_16_4_cifar_nobn_no_bias.log"
    paths = [reference, *sources, collapse_log]
    manifest = {
        **metadata,
        "reference": str(reference.resolve()),
        "collapsed_run": "wd1e3_finaldrop025/WRN_16_4",
        "sources": [
            {"path": str(path.resolve()), "sha256": sha256(path)} for path in paths
        ],
    }
    atomic_text(output / "report_manifest.json", json.dumps(manifest, indent=2) + "\n")


def main():
    results, sources = load_new_results()
    generate_complete(results, sources)
    generate_seed456(results, sources)
    print(COMPLETE_OUTPUT / "full_accuracy_comparison.md")
    print(SEED456_OUTPUT / "full_accuracy_comparison.md")


if __name__ == "__main__":
    main()

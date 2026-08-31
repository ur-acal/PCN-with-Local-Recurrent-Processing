#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAIRS = [
    (dataset, architecture)
    for dataset in ("cifar10", "cifar100")
    for architecture in ("WRN_16_2", "WRN_16_4", "WRN_28_2", "WRN_28_4")
]
NEW_COLUMNS = [
    "WD1e-3 final-drop0.25 WRN folded frozen-BN",
    "WD1e-3 final-drop0.25 WRN folded recal-BN",
    "WD1e-3 final-drop0.25 WRN unfused recal-BN",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Add WD=1e-3, final-dropout=0.25 WRN results to the validated combined report."
    )
    parser.add_argument(
        "--base_csv",
        type=Path,
        default=ROOT / "logs/combined_pcn_wrn_mismatch_full/full_accuracy_comparison.csv",
    )
    parser.add_argument(
        "--mismatch_root",
        type=Path,
        default=ROOT / "logs/wrn_wd1e3_finaldrop025_mismatch_full",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=ROOT / "logs/combined_pcn_wrn_mismatch_finaldrop025",
    )
    return parser.parse_args()


def load_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def atomic_text(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=path.parent) as handle:
        handle.write(content)
        temporary = handle.name
    os.replace(temporary, path)


def atomic_csv(path, rows):
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", newline="", delete=False, dir=path.parent) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        temporary = handle.name
    os.replace(temporary, path)


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def condition_for(report_type):
    if report_type == "additive_max":
        return "max_mul", "additive", "max_abs"
    if report_type == "additive_rms":
        return "rms", "additive", "rms"
    if report_type == "multiplicative":
        return "max_mul", "multiplicative", "max_abs"
    raise ValueError(f"Unsupported report mismatch type: {report_type}")


def load_result_index(root):
    indexes = {}
    source_paths = []
    for family in ("max_mul",):
        expected_types = {"additive", "multiplicative"}
        expected_scale = "max_abs"
        for mode in ("folded", "unfolded"):
            index = {}
            for dataset, architecture in PAIRS:
                path = root / family / mode / dataset / architecture / "full_aggregate.csv"
                rows = load_csv(path)
                source_paths.append(path)
                for row in rows:
                    mismatch_type = row["mismatch_type"]
                    if mismatch_type not in expected_types:
                        raise ValueError(f"Unexpected mismatch type in {path}: {mismatch_type}")
                    if row.get("additive_scale_mode", expected_scale) != expected_scale:
                        raise ValueError(f"Unexpected additive scale mode in {path}")
                    if row.get("pcn_node_mean_accuracy", "") not in ("", None):
                        raise ValueError(f"WRN-only result unexpectedly contains a PCN reference: {path}")
                    key = (
                        row["dataset"],
                        row["architecture"],
                        mismatch_type,
                        float(row["mismatch_level"]),
                    )
                    if key in index:
                        raise ValueError(f"Duplicate result key in {path}: {key}")
                    index[key] = row
            indexes[(family, mode)] = index
    return indexes, source_paths


def format_accuracy(value):
    return f"{float(value):.2f}"


def append_markdown_table(lines, rows):
    headers = list(rows[0])
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---:" for _ in headers]) + "|")
    for row in rows:
        lines.append("| " + " | ".join(str(row[column]) for column in headers) + " |")
    lines.append("")


def render_markdown(rows):
    lines = [
        "# PCNetNoBatchNorm and WRN Mismatch Comparison",
        "",
        "Accuracy values are percentages. This extends the validated combined report with WRNs trained using weight decay 1e-3 and final-feature dropout 0.25 in PCN order: final BN, dropout, ReLU, global average pool.",
        "",
        "The new WRN evaluator outputs contain no PCN reference columns. PCN values below are inherited unchanged from the validated base report.",
        "",
        "RMS mismatch was not run for the final-dropout WRNs; those new cells are marked N/A.",
        "",
    ]
    for dataset, architecture in PAIRS:
        lines.extend(["****************************************************************************", "", f"# {dataset} {architecture}", ""])
        selected = [row for row in rows if row["dataset"] == dataset and row["architecture"] == architecture]
        for mismatch_type, title in (
            ("additive_max", "Corrected max-scaled additive"),
            ("additive_rms", "RMS-scaled additive"),
            ("multiplicative", "Multiplicative"),
        ):
            table_rows = []
            for row in selected:
                if row["mismatch_type"] != mismatch_type:
                    continue
                table_rows.append(
                    {
                        key: value
                        for key, value in row.items()
                        if key not in {"dataset", "architecture", "mismatch_type"}
                    }
                )
            if not table_rows:
                raise ValueError(f"Missing report rows for {(dataset, architecture, mismatch_type)}")
            lines.extend([f"## {title}", ""])
            append_markdown_table(lines, table_rows)
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    base_rows = load_csv(args.base_csv)
    indexes, source_paths = load_result_index(args.mismatch_root)

    expected_base_pairs = set(PAIRS)
    actual_base_pairs = {(row["dataset"], row["architecture"]) for row in base_rows}
    if actual_base_pairs != expected_base_pairs:
        raise ValueError("Base report pair coverage does not match the eight target checkpoints")

    output_rows = []
    seen = set()
    for base in base_rows:
        report_type = base["mismatch_type"]
        base_key = (
            base["dataset"],
            base["architecture"],
            report_type,
            float(base["level"]),
        )
        if base_key in seen:
            raise ValueError(f"Duplicate base report row: {base_key}")
        seen.add(base_key)

        row = dict(base)
        if report_type == "additive_rms":
            for column in NEW_COLUMNS:
                row[column] = "N/A"
            output_rows.append(row)
            continue

        family, raw_type, expected_scale = condition_for(report_type)
        key = (
            base["dataset"],
            base["architecture"],
            raw_type,
            float(base["level"]),
        )
        folded = indexes[(family, "folded")].get(key)
        unfolded = indexes[(family, "unfolded")].get(key)
        if folded is None or unfolded is None:
            raise ValueError(f"Missing final-dropout WRN result for {(report_type, key)}")
        if folded.get("additive_scale_mode", expected_scale) != expected_scale:
            raise ValueError(f"Scale mismatch for {(report_type, key)}")

        row[NEW_COLUMNS[0]] = format_accuracy(folded["paired_wrn_frozen_bn_mean_accuracy"])
        row[NEW_COLUMNS[1]] = format_accuracy(folded["wrn_recalibrated_bn_mean_accuracy"])
        row[NEW_COLUMNS[2]] = format_accuracy(unfolded["wrn_recalibrated_bn_mean_accuracy"])
        output_rows.append(row)

    if len(output_rows) != len(base_rows):
        raise ValueError("Output row count differs from the validated base report")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "full_accuracy_comparison.csv"
    markdown_path = args.output_dir / "full_accuracy_comparison.md"
    atomic_csv(csv_path, output_rows)
    atomic_text(markdown_path, render_markdown(output_rows))

    manifest_sources = [args.base_csv, *source_paths]
    manifest = {
        "base_report": str(args.base_csv.resolve()),
        "mismatch_root": str(args.mismatch_root.resolve()),
        "report_rows": len(output_rows),
        "new_columns": NEW_COLUMNS,
        "finaldrop_rms_evaluated": False,
        "sources": [
            {
                "path": str(path.resolve()),
                "sha256": file_hash(path),
                "size_bytes": path.stat().st_size,
            }
            for path in manifest_sources
        ],
    }
    atomic_text(
        args.output_dir / "report_manifest.json",
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    print(markdown_path)


if __name__ == "__main__":
    main()


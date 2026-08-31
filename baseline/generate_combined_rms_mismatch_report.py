#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import math
import os
import pickle
import tempfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"
RMS_ROOT = LOGS / "rms_additive_full"
OUT_DIR = LOGS / "combined_pcn_wrn_mismatch_full"
ORIGINAL_DIR = LOGS / "wrn_bn_recalibration_full_5120_bn_fold_no_mismatch_bias"
WD_DIR = LOGS / "wrn_wd1e3_bn_recalibration_full_5120_bn_fold_no_mismatch_bias"
NOBN_DIR = LOGS / "wrn_nobn_cifar10_cifar100_full"
PAIRS = [
    (dataset, architecture)
    for dataset in ("cifar10", "cifar100")
    for architecture in ("WRN_16_2", "WRN_16_4", "WRN_28_2", "WRN_28_4")
]


def load_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def keyed(rows, source="<rows>"):
    output = {}
    for row in rows:
        item_key = (row["dataset"], row["architecture"], row["mismatch_type"], float(row["mismatch_level"]))
        if item_key in output:
            raise ValueError(f"Duplicate key {item_key} in {source}")
        output[item_key] = row
    return output



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repair_rms_wrn_derived", action="store_true")
    parser.add_argument("--skip_repair_plots", action="store_true")
    return parser.parse_args()


def atomic_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=path.parent) as handle:
        handle.write(text)
        temporary = handle.name
    os.replace(temporary, path)


def atomic_csv(path, rows):
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    fields = list(rows[0])
    for row in rows[1:]:
        fields.extend(key for key in row if key not in fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", newline="", delete=False, dir=path.parent) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
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


def pcn_stats(mode, dataset, architecture):
    values = load_pcn_result(mode, dataset, architecture)
    return {
        level: {"mean": float(np.mean(trials)), "std": float(np.std(trials)),
                "trials": len(trials),
                "source": str(RMS_ROOT / "pcn" / mode / dataset / architecture / "result.pkl")}
        for level, trials in values.items()
    }


def repaired_interpretation(rows):
    recovery = np.mean([float(row["mean_paired_bn_recovery"]) for row in rows])
    gap = np.mean([float(row["remaining_pcn_gap"]) for row in rows])
    if recovery < 1:
        return "negligible BN recovery"
    if gap <= 0:
        return "recalibrated WRN matches or exceeds PCN"
    if gap < 1:
        return "most of the PCN advantage explained by BN-statistic recalibration"
    return "partial BN recovery with substantial PCN advantage remaining"


def repair_rms_wrn_derived(make_plots=True):
    from baseline.run_wrn_bn_recalibration_experiment import make_plots
    manifest = []
    for family in ("wrn_original", "wrn_wd1e3"):
        for mode in ("folded", "unfolded"):
            for dataset, architecture in PAIRS:
                directory = RMS_ROOT / family / mode / dataset / architecture
                per_path, aggregate_path = directory / "full_per_trial.csv", directory / "full_aggregate.csv"
                per_rows, aggregate = load_csv(per_path), load_csv(aggregate_path)
                reference = pcn_stats("rms", dataset, architecture)
                for row in per_rows:
                    if row["mismatch_type"] != "additive" or row["additive_scale_mode"] != "rms":
                        raise ValueError(f"Wrong condition in {per_path}")
                    stats = reference[float(row["mismatch_level"])]
                    row.update({"pcn_node_best_column": f"PCNetNoBatchNorm:{architecture}",
                                "pcn_node_best_mean_accuracy": stats["mean"],
                                "pcn_node_best_std_accuracy": stats["std"],
                                "pcn_reference_scale_mode": "rms",
                                "pcn_reference_source": stats["source"]})
                    if "old_wrn_frozen_mean_accuracy" in row:
                        row["old_wrn_frozen_mean_accuracy"] = ""
                for row in aggregate:
                    if row["mismatch_type"] != "additive" or row["additive_scale_mode"] != "rms":
                        raise ValueError(f"Wrong condition in {aggregate_path}")
                    stats = reference[float(row["mismatch_level"])]
                    row.update({"pcn_node_column": f"PCNetNoBatchNorm:{architecture}",
                                "pcn_node_mean_accuracy": stats["mean"],
                                "pcn_node_std_accuracy": stats["std"],
                                "remaining_pcn_gap": stats["mean"] - float(row["wrn_recalibrated_bn_mean_accuracy"]),
                                "pcn_reference_scale_mode": "rms",
                                "pcn_reference_source": stats["source"]})
                    if row.get("wrn_all_bn_recal_before_fold_mean_accuracy", "") != "":
                        row["remaining_pcn_gap_all_bn_recal_before_fold"] = stats["mean"] - float(row["wrn_all_bn_recal_before_fold_mean_accuracy"])
                before_per, before_aggregate = file_hash(per_path), file_hash(aggregate_path)
                atomic_csv(per_path, per_rows)
                atomic_csv(aggregate_path, aggregate)
                summary_path = directory / "full_summary.json"
                summary = json.loads(summary_path.read_text())
                summary.update({"interpretation_label": repaired_interpretation(aggregate),
                                "pcn_reference_scale_mode": "rms",
                                "pcn_reference_source": next(iter(reference.values()))["source"],
                                "derived_outputs_repaired": True})
                if make_plots:
                    summary["corrected_comparison_plots"] = make_plots(directory, aggregate, per_rows)
                atomic_text(summary_path, json.dumps(summary, indent=2, sort_keys=True) + "\n")
                manifest.append({"family": family, "mode": mode, "dataset": dataset,
                                 "architecture": architecture, "per_trial_csv": str(per_path),
                                 "per_trial_sha256_before": before_per,
                                 "per_trial_sha256_after": file_hash(per_path),
                                 "aggregate_csv": str(aggregate_path),
                                 "aggregate_sha256_before": before_aggregate,
                                 "aggregate_sha256_after": file_hash(aggregate_path),
                                 "pcn_reference_source": next(iter(reference.values()))["source"]})
    manifest_path = RMS_ROOT / "rms_wrn_derived_repair_manifest.csv"
    atomic_csv(manifest_path, manifest)
    return manifest_path


def load_pcn_result(mode, dataset, architecture):
    path = RMS_ROOT / "pcn" / mode / dataset / architecture / "result.pkl"
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if len(payload) != 1:
        raise ValueError(f"Expected one t_end in {path}, found {len(payload)}")
    result = next(iter(payload.values()))["noise_acc_spec"]
    if len(result) != 1:
        raise ValueError(f"Expected one thermal-noise condition in {path}, found {list(result)}")
    by_level = next(iter(result.values()))
    return {float(level): [float(value) for value in values] for level, values in by_level.items()}


def add_rms_row(destination, row, source):
    if row.get("mismatch_type") != "additive" or row.get("additive_scale_mode") != "rms":
        raise ValueError(f"Wrong mismatch condition in {source}")
    item_key = (row["dataset"], row["architecture"], float(row["mismatch_level"]))
    if item_key in destination:
        raise ValueError(f"Duplicate RMS key {item_key} in {source}")
    destination[item_key] = row


def load_rms_wrn_family(family):
    folded = {}
    unfolded = {}
    for dataset, architecture in PAIRS:
        folded_path = RMS_ROOT / family / "folded" / dataset / architecture / "full_aggregate.csv"
        unfolded_path = RMS_ROOT / family / "unfolded" / dataset / architecture / "full_aggregate.csv"
        for row in load_csv(folded_path):
            add_rms_row(folded, row, folded_path)
        for row in load_csv(unfolded_path):
            add_rms_row(unfolded, row, unfolded_path)
    return folded, unfolded


def load_rms_nobn():
    output = {}
    for dataset, architecture in PAIRS:
        path = RMS_ROOT / "wrn_nobn" / dataset / architecture / "full_aggregate.csv"
        for row in load_csv(path):
            add_rms_row(output, row, path)
    return output



def validate_rms_sources(sources, original, pcn_by_pair):
    expected = {
        (dataset, architecture, level)
        for dataset, architecture in PAIRS
        for ds, arch, kind, level in original
        if (ds, arch, kind) == (dataset, architecture, "additive")
    }
    for dataset, architecture in PAIRS:
        expected_levels = {level for ds, arch, level in expected if (ds, arch) == (dataset, architecture)}
        if expected_levels and set(pcn_by_pair[(dataset, architecture)]) != expected_levels:
            raise ValueError(f"PCN RMS levels differ for {(dataset, architecture)}")
    for name, source in sources:
        if set(source) != expected:
            raise ValueError(f"{name} RMS keys differ from the additive experiment grid")
        for (dataset, architecture, level), row in source.items():
            expected_mean = float(np.mean(pcn_by_pair[(dataset, architecture)][level]))
            if row.get("pcn_reference_scale_mode") != "rms":
                raise ValueError(f"Unrepaired PCN reference metadata in {name}: {(dataset, architecture, level)}")
            actual = float(row["pcn_node_mean_accuracy"])
            if not math.isclose(actual, expected_mean, rel_tol=0.0, abs_tol=1e-10):
                raise ValueError(f"Stale PCN reference in {name}: {(dataset, architecture, level)}")


def load_kappa_audit():
    specifications = (
        ("PCNetNoBatchNorm", ROOT.parent / "ScAN-PCN/logs/kappa_audit/all_primary.csv", "PCN"),
        ("WRN original folded", ORIGINAL_DIR / "folded_wrn_kappa_primary.csv", "WRN"),
        ("WRN WD1e-3 folded", WD_DIR / "folded_wrn_kappa_primary.csv", "WRN"),
    )
    rows, paths, seen = [], [], set()
    for family, path, condition in specifications:
        for row in load_csv(path):
            if row.get("model_condition") != condition:
                continue
            if (row.get("dataset"), row.get("paired_arch")) not in PAIRS:
                continue
            if family == "PCNetNoBatchNorm" and "PCNetNoBatchNorm" not in row["model_name"]:
                raise ValueError(f"Wrong PCN model in {path}")
            item_key = (family, row["dataset"], row["paired_arch"], row["tensor_name"])
            if item_key in seen:
                raise ValueError(f"Duplicate kappa row {item_key}")
            seen.add(item_key)
            output = dict(row)
            output["model_family"] = family
            output["source_csv"] = str(path)
            rows.append(output)
        paths.append(path)
    expected = {(family, dataset, architecture) for family, _, _ in specifications for dataset, architecture in PAIRS}
    actual = {(row["model_family"], row["dataset"], row["paired_arch"]) for row in rows}
    if actual != expected:
        raise ValueError(f"Kappa coverage mismatch: {sorted(expected - actual)}")
    return rows, paths


def render_kappa(rows):
    lines = ["# Layerwise Kappa Audit", "", "Kappa is max_abs / RMS.", ""]
    tick = chr(96)
    for dataset, architecture in PAIRS:
        lines.extend(["****************************************************************************", "", f"# {dataset} {architecture}", ""])
        for family in ("PCNetNoBatchNorm", "WRN original folded", "WRN WD1e-3 folded"):
            selected = sorted(
                (row for row in rows if row["dataset"] == dataset and row["paired_arch"] == architecture and row["model_family"] == family),
                key=lambda row: int(float(row["semantic_order_index"])),
            )
            lines.extend([f"## {family}", "", "| order | parameter name | module type | shape | max_abs | rms | kappa |", "|---:|---|---|---|---:|---:|---:|"])
            for row in selected:
                lines.append(
                    f"| {int(float(row['semantic_order_index']))} | {tick}{row['tensor_name']}{tick} | {row['module_type']} | "
                    f"{tick}{row['tensor_shape']}{tick} | {float(row['max_abs']):.6f} | {float(row['rms']):.6f} | {float(row['kappa']):.6f} |"
                )
            values = np.asarray([float(row["kappa"]) for row in selected])
            lines.extend(["", "| layers | median kappa | mean kappa | min | max |", "|---:|---:|---:|---:|---:|",
                          f"| {len(values)} | {np.median(values):.6f} | {np.mean(values):.6f} | {np.min(values):.6f} | {np.max(values):.6f} |", ""])
    return "\n".join(lines) + "\n"


def number(value):
    if value in (None, ""):
        return "N/A"
    return f"{float(value):.2f}"


def old_values(source, dataset, architecture, mismatch_type, level):
    row = source[(dataset, architecture, mismatch_type, level)]
    return {
        "folded_frozen": row["paired_wrn_frozen_bn_mean_accuracy"],
        "folded_recal": row["wrn_recalibrated_bn_mean_accuracy"],
        "unfolded_recal": row["wrn_unfolded_recalibrated_bn_mean_accuracy"],
        "trials": row["num_trials"],
    }


def rms_values(folded, unfolded, dataset, architecture, level):
    folded_row = folded[(dataset, architecture, level)]
    unfolded_row = unfolded[(dataset, architecture, level)]
    return {
        "folded_frozen": folded_row["paired_wrn_frozen_bn_mean_accuracy"],
        "folded_recal": folded_row["wrn_recalibrated_bn_mean_accuracy"],
        "unfolded_recal": unfolded_row["wrn_recalibrated_bn_mean_accuracy"],
        "trials": folded_row["num_trials"],
    }


def append_markdown_table(lines, rows):
    headers = [
        "level", "PCN/WRN trials", "PCNetNoBatchNorm", "WRN folded frozen-BN",
        "WRN folded recal-BN", "WRN unfused recal-BN", "WD1e-3 WRN folded frozen-BN",
        "WD1e-3 WRN folded recal-BN", "WD1e-3 WRN unfused recal-BN", "BN-free WRN",
    ]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---:"] * len(headers)) + "|")
    for row in rows:
        lines.append("| " + " | ".join(str(row[header]) for header in headers) + " |")
    lines.append("")


def extract_kappa_sections(path):
    lines = path.read_text().splitlines()
    output = []
    index = 0
    while index < len(lines):
        if lines[index].startswith("## ") and lines[index].endswith(" kappa audit"):
            end = index + 1
            while end < len(lines) and not (
                lines[end].startswith("## ") and not lines[end].endswith(" kappa audit")
            ):
                end += 1
            output.extend(lines[index:end])
            output.append("")
            index = end
        else:
            index += 1
    return output


def main():
    args = parse_args()
    if args.repair_rms_wrn_derived:
        repair_rms_wrn_derived(make_plots=not args.skip_repair_plots)
    original_path = ORIGINAL_DIR / "full_presentation_table.csv"
    wd_path = WD_DIR / "full_presentation_table.csv"
    nobn_path = NOBN_DIR / "full_presentation_table.csv"
    original = keyed(load_csv(original_path), original_path)
    wd = keyed(load_csv(wd_path), wd_path)
    nobn = keyed(load_csv(nobn_path), nobn_path)
    if set(original) != set(wd) or set(original) != set(nobn):
        raise ValueError("Legacy original, WD, and BN-free table keys differ")
    rms_original_folded, rms_original_unfolded = load_rms_wrn_family("wrn_original")
    rms_wd_folded, rms_wd_unfolded = load_rms_wrn_family("wrn_wd1e3")
    rms_nobn = load_rms_nobn()
    pcn_rms_by_pair = {(dataset, architecture): load_pcn_result("rms", dataset, architecture) for dataset, architecture in PAIRS}
    validate_rms_sources(
        [("original folded", rms_original_folded), ("original unfolded", rms_original_unfolded),
         ("WD folded", rms_wd_folded), ("WD unfolded", rms_wd_unfolded)],
        original,
        pcn_rms_by_pair,
    )
    expected_rms_keys = set(rms_original_folded)
    if set(rms_nobn) != expected_rms_keys:
        raise ValueError("BN-free RMS keys differ from the additive experiment grid")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    markdown = [
        "# PCNetNoBatchNorm and WRN Mismatch Comparison",
        "",
        "Accuracy values are percentages. Max-additive PCN values use the corrected classifier additive formula. RMS-additive uses per-tensor RMS scaling for every model family. Existing multiplicative and WRN max-additive results are reused unchanged.",
        "",
        "Folded WRN conditions exclude convolution biases induced by BN folding. BN-free WRNs exclude convolution biases. Classifier biases remain selected as in the existing evaluators.",
        "",
    ]
    csv_rows = []

    for dataset, architecture in PAIRS:
        markdown.extend(["****************************************************************************", "", f"# {dataset} {architecture}", ""])
        additive_levels = sorted(
            level for ds, arch, kind, level in original if ds == dataset and arch == architecture and kind == "additive"
        )
        multiplicative_levels = sorted(
            level for ds, arch, kind, level in original if ds == dataset and arch == architecture and kind == "multiplicative"
        )
        pcn_max = load_pcn_result("max_abs", dataset, architecture)
        pcn_rms = load_pcn_result("rms", dataset, architecture)

        for label, mismatch_type, levels in (
            ("Corrected max-scaled additive", "additive_max", additive_levels),
            ("RMS-scaled additive", "additive_rms", additive_levels),
            ("Multiplicative", "multiplicative", multiplicative_levels),
        ):
            markdown.extend([f"## {label}", ""])
            table_rows = []
            for level in levels:
                if mismatch_type == "additive_rms":
                    pcn_values = pcn_rms[level]
                    standard = rms_values(rms_original_folded, rms_original_unfolded, dataset, architecture, level)
                    wd_values = rms_values(rms_wd_folded, rms_wd_unfolded, dataset, architecture, level)
                    nobn_row = rms_nobn[(dataset, architecture, level)]
                else:
                    old_kind = "additive" if mismatch_type == "additive_max" else "multiplicative"
                    source_row = original[(dataset, architecture, old_kind, level)]
                    pcn_values = pcn_max[level] if mismatch_type == "additive_max" else [float(source_row["pcn_node_mean_accuracy"])]
                    standard = old_values(original, dataset, architecture, old_kind, level)
                    wd_values = old_values(wd, dataset, architecture, old_kind, level)
                    nobn_row = nobn[(dataset, architecture, old_kind, level)]
                pcn_trials = len(pcn_values) if mismatch_type != "multiplicative" else int(float(source_row["num_trials"]))

                row = {
                    "level": f"{level:g}",
                    "PCN/WRN trials": f"{pcn_trials}/{standard['trials']}",
                    "PCNetNoBatchNorm": number(sum(pcn_values) / len(pcn_values)),
                    "WRN folded frozen-BN": number(standard["folded_frozen"]),
                    "WRN folded recal-BN": number(standard["folded_recal"]),
                    "WRN unfused recal-BN": number(standard["unfolded_recal"]),
                    "WD1e-3 WRN folded frozen-BN": number(wd_values["folded_frozen"]),
                    "WD1e-3 WRN folded recal-BN": number(wd_values["folded_recal"]),
                    "WD1e-3 WRN unfused recal-BN": number(wd_values["unfolded_recal"]),
                    "BN-free WRN": number(nobn_row["wrn_nobn_mean_accuracy"]),
                }
                table_rows.append(row)
                csv_rows.append({"dataset": dataset, "architecture": architecture, "mismatch_type": mismatch_type, **row})
            append_markdown_table(markdown, table_rows)

    accuracy_path = OUT_DIR / "full_accuracy_comparison.md"
    atomic_text(accuracy_path, "\n".join(markdown) + "\n")
    atomic_csv(OUT_DIR / "full_accuracy_comparison.csv", csv_rows)

    kappa_rows, kappa_paths = load_kappa_audit()
    atomic_csv(OUT_DIR / "full_kappa_audit.csv", kappa_rows)
    atomic_text(OUT_DIR / "full_kappa_audit.md", render_kappa(kappa_rows))

    source_paths = [
        ("legacy_original", original_path),
        ("legacy_wd1e3", wd_path),
        ("legacy_nobn", nobn_path),
        ("rms_repair_manifest", RMS_ROOT / "rms_wrn_derived_repair_manifest.csv"),
    ]
    for pcn_mode in ("max_abs", "rms"):
        for dataset, architecture in PAIRS:
            source_paths.append((
                f"pcn_{pcn_mode}_{dataset}_{architecture}",
                RMS_ROOT / "pcn" / pcn_mode / dataset / architecture / "result.pkl",
            ))
    for family in ("wrn_original", "wrn_wd1e3"):
        for mode in ("folded", "unfolded"):
            for dataset, architecture in PAIRS:
                source_paths.append((
                    f"{family}_{mode}_{dataset}_{architecture}",
                    RMS_ROOT / family / mode / dataset / architecture / "full_aggregate.csv",
                ))
    for dataset, architecture in PAIRS:
        source_paths.append((
            f"wrn_nobn_rms_{dataset}_{architecture}",
            RMS_ROOT / "wrn_nobn" / dataset / architecture / "full_aggregate.csv",
        ))
    source_paths.extend(("kappa", path) for path in kappa_paths)
    atomic_csv(
        OUT_DIR / "source_manifest.csv",
        [{"role": role, "path": str(path.resolve()), "sha256": file_hash(path),
          "size_bytes": path.stat().st_size} for role, path in source_paths],
    )
    atomic_text(
        OUT_DIR / "report_manifest.json",
        json.dumps({
            "report_rows": len(csv_rows),
            "pcn_model_condition": "PCNetNoBatchNorm",
            "mismatch_types": ["additive_max", "additive_rms", "multiplicative"],
        }, indent=2, sort_keys=True) + "\n",
    )
    print(accuracy_path)
    print(OUT_DIR / "full_kappa_audit.md")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import csv
import hashlib
import json
import math
import os
import pickle
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"
BASE_DIR = LOGS / "combined_pcn_wrn_mismatch_full"
FINALDROP_ROOT = LOGS / "wrn_wd1e3_finaldrop025_mismatch_full"
EXTENDED_ROOT = LOGS / "rms_cifar100_remaining_pairs"
PILOT_ROOT = LOGS / "rms_level_pilot_cifar100_WRN_28_2"
OUTPUT_DIR = LOGS / "combined_pcn_wrn_mismatch_finaldrop025_complete"
WITH1ST_ROOT = LOGS / "pcn_with1stconv_mismatch_slurm"
WITH1ST_ARCHIVE = LOGS / "pcn_with1stconv_mismatch_slurm.tar.gz"

PAIRS = [
    (dataset, architecture)
    for dataset in ("cifar10", "cifar100")
    for architecture in ("WRN_16_2", "WRN_16_4", "WRN_28_2", "WRN_28_4")
]
BASE_COLUMNS = [
    "PCNetNoBatchNorm",
    "WRN folded frozen-BN",
    "WRN folded recal-BN",
    "WRN unfused recal-BN",
    "WD1e-3 WRN folded frozen-BN",
    "WD1e-3 WRN folded recal-BN",
    "WD1e-3 WRN unfused recal-BN",
    "BN-free WRN",
]
FINALDROP_COLUMN = "WD1e-3 final-drop0.25 WRN unfused recal-BN"
WITH1ST_COLUMN = "PCNetWith1stConv (new SLURM)"
REPORT_COLUMNS = [
    "level", "trial counts", BASE_COLUMNS[0], WITH1ST_COLUMN, *BASE_COLUMNS[1:], FINALDROP_COLUMN
]


def read_csv(path):
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


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def format_accuracy(value):
    if value in (None, "", "N/A"):
        return "N/A"
    return f"{float(value):.2f}"


def load_base():
    path = BASE_DIR / "full_accuracy_comparison.csv"
    rows = read_csv(path)
    required = {"dataset", "architecture", "mismatch_type", "level", "PCN/WRN trials", *BASE_COLUMNS}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Unexpected validated base schema: {path}")
    index = {}
    for row in rows:
        key = (row["dataset"], row["architecture"], row["mismatch_type"], float(row["level"]))
        if key in index:
            raise ValueError(f"Duplicate validated base key: {key}")
        index[key] = row
    if {(key[0], key[1]) for key in index} != set(PAIRS):
        raise ValueError("Validated base report does not cover all eight model pairs")
    return index, path


def add_finaldrop_rows(index, source_paths, family, expected_types, expected_scale):
    for dataset, architecture in PAIRS:
        path = FINALDROP_ROOT / family / "unfolded" / dataset / architecture / "full_aggregate.csv"
        source_paths.append(path)
        for row in read_csv(path):
            mismatch_type = row["mismatch_type"]
            if mismatch_type not in expected_types:
                raise ValueError(f"Unexpected mismatch type in {path}: {mismatch_type}")
            if row["additive_scale_mode"] != expected_scale:
                raise ValueError(f"Unexpected scale mode in {path}")
            if row["mismatch_parameter_policy"] != "existing":
                raise ValueError(f"Final-dropout result is not unfused: {path}")
            if row.get("pcn_node_mean_accuracy", "") not in ("", None):
                raise ValueError(f"WRN result contains copied PCN values: {path}")
            key = (dataset, architecture, mismatch_type, float(row["mismatch_level"]))
            if key in index:
                raise ValueError(f"Duplicate final-dropout key: {key}")
            index[key] = {
                "accuracy": float(row["wrn_recalibrated_bn_mean_accuracy"]),
                "trials": int(float(row["num_trials"])),
                "source": path,
            }


def load_finaldrop():
    index = {}
    source_paths = []
    add_finaldrop_rows(index, source_paths, "max_mul", {"additive", "multiplicative"}, "max_abs")
    add_finaldrop_rows(index, source_paths, "rms", {"additive"}, "rms")

    for dataset, architecture in PAIRS:
        max_levels = {
            level for ds, arch, kind, level in index
            if (ds, arch, kind) == (dataset, architecture, "additive")
            and index[(ds, arch, kind, level)]["source"].parts[-5] == "max_mul"
        }
        mul_levels = {
            level for ds, arch, kind, level in index
            if (ds, arch, kind) == (dataset, architecture, "multiplicative")
        }
        rms_levels = {
            level for ds, arch, kind, level in index
            if (ds, arch, kind) == (dataset, architecture, "additive")
            and index[(ds, arch, kind, level)]["source"].parts[-5] == "rms"
        }
        if max_levels != {level / 100 for level in range(11)}:
            raise ValueError(f"Unexpected final-dropout max-additive grid: {(dataset, architecture, max_levels)}")
        if mul_levels != {level / 20 for level in range(9)}:
            raise ValueError(f"Unexpected final-dropout multiplicative grid: {(dataset, architecture, mul_levels)}")
        if rms_levels != {0.25, 0.5, 0.75, 1.0, 1.25}:
            raise ValueError(f"Unexpected final-dropout RMS grid: {(dataset, architecture, rms_levels)}")
    return index, source_paths


def load_pcn_pickle(path):
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if len(payload) != 1:
        raise ValueError(f"Expected one t_end in {path}")
    noise_spec = next(iter(payload.values()))["noise_acc_spec"]
    if len(noise_spec) != 1:
        raise ValueError(f"Expected one noise condition in {path}")
    by_level = next(iter(noise_spec.values()))
    return {
        float(level): [float(value) for value in values]
        for level, values in by_level.items()
    }

def load_with1stconv():
    conditions = {
        "max_additive": {
            "report_type": "additive_max",
            "levels": {level / 100 for level in range(11)},
            "marker": "'mismatch_type': 'add', 'additive_scale_mode': 'max_abs'",
        },
        "rms_additive": {
            "report_type": "additive_rms",
            "levels": {0.25, 0.5, 0.75, 1.0, 1.25},
            "marker": "'mismatch_type': 'add', 'additive_scale_mode': 'rms'",
        },
        "multiplicative": {
            "report_type": "multiplicative",
            "levels": {level / 20 for level in range(9)},
            "marker": "'mismatch_type': 'mul'",
        },
    }
    expected_pairs = {
        ("cifar10", "WRN_16_2"),
        ("cifar10", "WRN_28_4"),
        ("cifar100", "WRN_16_2"),
        ("cifar100", "WRN_16_4"),
        ("cifar100", "WRN_28_2"),
        ("cifar100", "WRN_28_4"),
    }
    index = {}
    source_paths = []
    observed_conditions = set()
    for path in sorted(WITH1ST_ROOT.glob("*/*/*/result.pkl")):
        dataset, architecture, condition, _ = path.relative_to(WITH1ST_ROOT).parts
        if condition not in conditions:
            raise ValueError(f"Unexpected With1stConv condition directory: {path}")
        if (dataset, architecture) not in expected_pairs:
            raise ValueError(f"Unexpected With1stConv model pair: {path}")
        specification = conditions[condition]
        run_log = path.with_name("run.log")
        log_text = run_log.read_text(errors="replace")
        if "TIMMPCNetWith1stConv" not in log_text:
            raise ValueError(f"Run is not PCNetWith1stConv: {run_log}")
        if specification["marker"] not in log_text:
            raise ValueError(f"Mismatch metadata does not match directory: {run_log}")
        if dataset == "cifar100" and "_C100_" not in log_text:
            raise ValueError(f"CIFAR-100 marker missing from model name: {run_log}")
        if dataset == "cifar10" and "_C100_" in log_text:
            raise ValueError(f"CIFAR-10 run contains a CIFAR-100 model: {run_log}")
        if any(marker in log_text for marker in ("Traceback", "CUDA out of memory", "RuntimeError")):
            raise ValueError(f"Run log contains an error marker: {run_log}")

        by_level = load_pcn_pickle(path)
        if set(by_level) != specification["levels"]:
            raise ValueError(f"Unexpected With1stConv level grid: {path}")
        for level, values in by_level.items():
            expected_trials = 1 if level == 0.0 else 10
            if len(values) != expected_trials:
                raise ValueError(f"Unexpected With1stConv trial count: {(path, level, len(values))}")
            key = (dataset, architecture, specification["report_type"], level)
            if key in index:
                raise ValueError(f"Duplicate With1stConv report key: {key}")
            index[key] = {
                "accuracy": sum(values) / len(values),
                "trials": len(values),
                "source": path,
            }
        observed_conditions.add((dataset, architecture, condition))
        source_paths.extend([path, run_log])

    expected_conditions = {
        (dataset, architecture, condition)
        for dataset, architecture in expected_pairs
        for condition in conditions
    }
    if observed_conditions != expected_conditions:
        raise ValueError(
            f"With1stConv coverage mismatch: missing={sorted(expected_conditions - observed_conditions)}"
        )
    return index, source_paths

def load_extended_rms():
    manifest = json.loads((EXTENDED_ROOT / "run_manifest.json").read_text())
    if manifest["dataset"] != "cifar100" or manifest["trials"] != 3:
        raise ValueError("Unexpected extended RMS manifest")
    if manifest["additive_scale_mode"] != "rms":
        raise ValueError("Extended experiment is not RMS-additive")

    index = {}
    summary_path = EXTENDED_ROOT / "summary.csv"
    for row in read_csv(summary_path):
        key = ("cifar100", row["architecture"], float(row["rms_level"]))
        if key in index:
            raise ValueError(f"Duplicate extended RMS key: {key}")
        index[key] = {
            "pcn": float(row["pcn_mean_accuracy"]),
            "pcn_trials": 3,
            "wd_unfused": float(row["wrn_wd1e3_unfused_recal_bn_mean_accuracy"]),
            "wd_trials": 3,
        }

    pilot_pcn_path = PILOT_ROOT / "pcn" / "result.pkl"
    pilot_wrn_path = PILOT_ROOT / "wrn_wd1e3_unfolded_recal_nonzero" / "full_aggregate.csv"
    pilot_pcn = load_pcn_pickle(pilot_pcn_path)
    pilot_wrn = {
        float(row["mismatch_level"]): row
        for row in read_csv(pilot_wrn_path)
    }
    expected_pilot_levels = {0.25, 0.5, 0.75, 1.0, 1.25, 1.5}
    if set(pilot_pcn) - {0.0} != expected_pilot_levels or set(pilot_wrn) != expected_pilot_levels:
        raise ValueError("Unexpected WRN-28-2 extended RMS pilot grid")
    for level in expected_pilot_levels:
        wrn_row = pilot_wrn[level]
        if wrn_row["additive_scale_mode"] != "rms" or int(float(wrn_row["num_trials"])) != 3:
            raise ValueError(f"Unexpected WRN-28-2 pilot metadata at {level}")
        if len(pilot_pcn[level]) != 3:
            raise ValueError(f"Unexpected PCN WRN-28-2 trial count at {level}")
        replacement = {
            "pcn": sum(pilot_pcn[level]) / len(pilot_pcn[level]),
            "pcn_trials": 3,
            "wd_unfused": float(wrn_row["wrn_recalibrated_bn_mean_accuracy"]),
            "wd_trials": 3,
        }
        key = ("cifar100", "WRN_28_2", level)
        if key in index:
            for field in ("pcn", "wd_unfused"):
                if not math.isclose(index[key][field], replacement[field], rel_tol=0.0, abs_tol=1e-9):
                    raise ValueError(f"Extended summary disagrees with WRN-28-2 pilot: {(level, field)}")
        index[key] = replacement

    expected = {
        ("cifar100", architecture, level)
        for architecture in ("WRN_16_2", "WRN_16_4", "WRN_28_4")
        for level in (0.25, 0.5, 0.75, 1.0)
    }
    expected |= {
        ("cifar100", "WRN_28_2", level)
        for level in expected_pilot_levels
    }
    if set(index) != expected:
        raise ValueError(f"Extended RMS coverage mismatch: {sorted(expected - set(index))}")
    return index, [EXTENDED_ROOT / "run_manifest.json", summary_path, pilot_pcn_path, pilot_wrn_path]


def finaldrop_value(finaldrop, dataset, architecture, report_type, level):
    if report_type == "additive_max":
        raw_type, family = "additive", "max_mul"
    elif report_type == "additive_rms":
        raw_type, family = "additive", "rms"
    elif report_type == "multiplicative":
        raw_type, family = "multiplicative", "max_mul"
    else:
        raise ValueError(report_type)
    item = finaldrop.get((dataset, architecture, raw_type, level))
    if item is None or item["source"].parts[-5] != family:
        return None
    return item


def render_table(lines, rows):
    lines.append("| " + " | ".join(REPORT_COLUMNS) + " |")
    lines.append("|" + "|".join(["---:"] * len(REPORT_COLUMNS)) + "|")
    for row in rows:
        lines.append("| " + " | ".join(row[column] for column in REPORT_COLUMNS) + " |")
    lines.append("")


def make_base_row(base_row, with1st_item, finaldrop_item):
    row = {
        "level": f"{float(base_row['level']):g}",
        "trial counts": f"PCN/legacy WRNs={base_row['PCN/WRN trials']}",
    }
    row.update({column: format_accuracy(base_row[column]) for column in BASE_COLUMNS})
    row[WITH1ST_COLUMN] = "N/A"
    if with1st_item is not None:
        row[WITH1ST_COLUMN] = format_accuracy(with1st_item["accuracy"])
        row["trial counts"] += f"; with1st={with1st_item['trials']}"
    if finaldrop_item is None:
        row[FINALDROP_COLUMN] = "N/A"
    else:
        row[FINALDROP_COLUMN] = format_accuracy(finaldrop_item["accuracy"])
        row["trial counts"] += f"; final-drop={finaldrop_item['trials']}"
    return row


def make_extended_row(dataset, architecture, level, extended, with1st_item, finaldrop_item):
    values = {column: "N/A" for column in BASE_COLUMNS}
    counts = []
    extended_item = extended.get((dataset, architecture, level))
    if extended_item is not None:
        values["PCNetNoBatchNorm"] = format_accuracy(extended_item["pcn"])
        values["WD1e-3 WRN unfused recal-BN"] = format_accuracy(extended_item["wd_unfused"])
        counts.extend([
            f"PCN={extended_item['pcn_trials']}",
            f"WD1e3-unfused={extended_item['wd_trials']}",
        ])
    with1st_value = "N/A"
    if with1st_item is not None:
        with1st_value = format_accuracy(with1st_item["accuracy"])
        counts.append(f"with1st={with1st_item['trials']}")
    final_value = "N/A"
    if finaldrop_item is not None:
        final_value = format_accuracy(finaldrop_item["accuracy"])
        counts.append(f"final-drop={finaldrop_item['trials']}")
    return {
        "level": f"{level:g}",
        "trial counts": "; ".join(counts),
        **values,
        WITH1ST_COLUMN: with1st_value,
        FINALDROP_COLUMN: final_value,
    }


def main():
    base, base_path = load_base()
    finaldrop, finaldrop_paths = load_finaldrop()
    with1st, with1st_paths = load_with1stconv()
    extended, extended_paths = load_extended_rms()

    markdown = [
        "# Complete PCNetNoBatchNorm, PCNetWith1stConv, and WRN Mismatch Comparison",
        "",
        "Accuracy values are percentages. Every value is read from the result file for that model family; WRN CSV PCN-reference fields are not used.",
        "",
        "The final-dropout condition is WD=1e-3 WRN with final-feature dropout 0.25 and unfused-BN recalibration. Max-additive and multiplicative use 10 trials. Its RMS-additive grid is 0.25 through 1.25 with 10 trials.",
        "",
        "The newest PCNetWith1stConv SLURM archive covers six pairs with 10 nonzero trials. CIFAR-10 WRN-16-4 and WRN-28-2 are absent and marked N/A.",
        "",
        "For RMS level zero, With1stConv and final-dropout clean values are reused from their max-additive zero-mismatch runs because their RMS queues intentionally skipped zero.",
        "",
        "Extended CIFAR-100 RMS rows attach the earlier 3-trial PCNetNoBatchNorm and WD=1e-3 unfused recal-BN runs. WRN-28-2 additionally has 3-trial levels 1.25 and 1.5. N/A means that exact model and mismatch level was not run.",
        "",
    ]
    output_rows = []

    for dataset, architecture in PAIRS:
        markdown.extend(["****************************************************************************", "", f"# {dataset} {architecture}", ""])
        for report_type, title in (
            ("additive_max", "Corrected max-scaled additive"),
            ("additive_rms", "RMS-scaled additive"),
            ("multiplicative", "Multiplicative"),
        ):
            base_levels = sorted(
                level for ds, arch, kind, level in base
                if (ds, arch, kind) == (dataset, architecture, report_type)
            )
            levels = set(base_levels)
            if report_type == "additive_rms":
                levels.update(
                    level for ds, arch, level in extended
                    if (ds, arch) == (dataset, architecture)
                )
                levels.update(
                    level for ds, arch, kind, level in finaldrop
                    if (ds, arch, kind) == (dataset, architecture, "additive")
                    and finaldrop[(ds, arch, kind, level)]["source"].parts[-5] == "rms"
                )
            table_rows = []
            for level in sorted(levels):
                base_row = base.get((dataset, architecture, report_type, level))
                final_item = finaldrop_value(finaldrop, dataset, architecture, report_type, level)
                with1st_item = with1st.get((dataset, architecture, report_type, level))
                if report_type == "additive_rms" and level == 0.0 and final_item is None:
                    final_item = finaldrop_value(finaldrop, dataset, architecture, "additive_max", 0.0)
                if report_type == "additive_rms" and level == 0.0 and with1st_item is None:
                    with1st_item = with1st.get((dataset, architecture, "additive_max", 0.0))
                if base_row is not None:
                    row = make_base_row(base_row, with1st_item, final_item)
                else:
                    row = make_extended_row(dataset, architecture, level, extended, with1st_item, final_item)
                table_rows.append(row)
                output_rows.append({
                    "dataset": dataset,
                    "architecture": architecture,
                    "mismatch_type": report_type,
                    **row,
                })
            markdown.extend([f"## {title}", ""])
            render_table(markdown, table_rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    markdown_path = OUTPUT_DIR / "full_accuracy_comparison.md"
    csv_path = OUTPUT_DIR / "full_accuracy_comparison.csv"
    atomic_text(markdown_path, "\n".join(markdown) + "\n")
    atomic_csv(csv_path, output_rows)

    source_paths = [
        base_path,
        BASE_DIR / "source_manifest.csv",
        BASE_DIR / "report_manifest.json",
        *finaldrop_paths,
        *with1st_paths,
        WITH1ST_ARCHIVE,
        *extended_paths,
    ]
    unique_paths = list(dict.fromkeys(path.resolve() for path in source_paths))
    atomic_csv(
        OUTPUT_DIR / "source_manifest.csv",
        [
            {
                "path": str(path),
                "sha256": sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in unique_paths
        ],
    )
    atomic_text(
        OUTPUT_DIR / "report_manifest.json",
        json.dumps(
            {
                "pcn_model_condition": "PCNetNoBatchNorm",
                "report_rows": len(output_rows),
                "finaldrop_condition": "wd1e3_finaldrop025_unfused_recal_bn",
                "finaldrop_trials": 10,
                "with1stconv_condition": "new_slurm_archive",
                "with1stconv_available_pairs": 6,
                "with1stconv_nonzero_trials": 10,
                "extended_cifar100_rms_trials": 3,
                "mismatch_types": ["additive_max", "additive_rms", "multiplicative"],
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
    )
    print(markdown_path)


if __name__ == "__main__":
    main()

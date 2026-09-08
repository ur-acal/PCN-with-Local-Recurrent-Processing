#!/usr/bin/env python3
"""Assemble the paper's max-additive/multiplicative evidence without rerunning models."""

import csv
import io
import json
import math
import pickle
import statistics
import subprocess
import tarfile
from collections import Counter
from pathlib import Path

from generate_pcn_wrn_robustness_summary import (
    ALL_COMPLETE_WRNS, CANONICAL, FAMILIES, MAXPOOL_PILOT, NO_BIAS,
    OUTPUT, PCN, POOLING, ROOT, atomic_text, fmt, md_table, number,
    read_csv, selected_rows, sha256,
)
from generate_pcn_withx_top3_report import ARCHIVE, EARLIER_C100_28_2, FAMILY_DIRS


FIRST = "PCNetWith1stConv (new SLURM)"
WITH_PC = "PCNetNoBatchNorm ODEBlockPC t1.75"
WITH_X = "PCNetNoBatchNorm ODEBlockXInit t1.75"
POOLS = ["AvgPool BN", "Stride-2 BN", "AvgPool BN-free", "Stride-2 BN-free"]
NOBIAS = ["BN-free/no-conv-bias, WD5e-4/drop0", "BN-free/no-conv-bias, WD1e-3/final-drop0.25"]
MODELS = [PCN, FIRST, WITH_PC, WITH_X] + ALL_COMPLETE_WRNS + POOLS + NOBIAS
SEED_REPEAT = ROOT / "logs/combined_pcn_wrn28_2_cifar100_seed456/full_accuracy_comparison.csv"


def key(row):
    return row["dataset"], row["architecture"], row["mismatch_type"], float(row["level"])


def csv_text(rows, fields):
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue()


def training_artifact_inventory():
    """Index resolved WRN recipes; file existence is not a checkpoint identity audit."""
    roots = [
        ("Original WRN", "checkpoint/baselines"),
        ("WD1e-3 WRN", "checkpoint/baselines_wd1e3_pilot"),
        ("WD1e-3 final-drop0.25 WRN", "checkpoint/baselines_wd1e3_finaldrop025"),
        ("BN-free WRN CIFAR-10", "checkpoint/baselines_nobn_best_recipe/cifar10"),
        ("BN-free WRN CIFAR-100", "checkpoint/baselines_nobn_cifar100_searched_recipe"),
        ("Four pooling variants", "checkpoint/baselines_wrn28_2_avgpool_study"),
        (NOBIAS[0], "logs/wrn_nobn_no_bias_cifar100/checkpoints"),
        (NOBIAS[1], "logs/wrn_nobn_no_bias_wd1e3_finaldrop025_cifar100/checkpoints"),
    ]
    entries = []
    standard_names = {f"wrn_{d}_{w}_cifar" for d in (16, 28) for w in (2, 4)}
    for family, directory in roots:
        for config in sorted((ROOT / directory).rglob("baseline_config.json")):
            if directory in {"checkpoint/baselines", "checkpoint/baselines_wd1e3_pilot"} and config.parent.name not in standard_names:
                continue
            entries.append(dict(
                family=family, config=str(config.relative_to(ROOT)), config_sha256=sha256(config),
                checkpoint_files=[str(p.relative_to(ROOT)) for p in sorted(config.parent.rglob("*_best_ckpt.pth"))],
                eligible_result=not (family == NOBIAS[1] and config.parent.name == "wrn_16_4_cifar_nobn_no_bias"),
            ))
    return entries


def main():
    sources, provenance = {}, []
    rows = {key(row): dict(row) for row in selected_rows(CANONICAL)}
    assert len(rows) == 160, "Expected eight pairs, eleven additive and nine multiplicative levels"
    assert len(selected_rows(CANONICAL)) == len(rows), "Duplicate canonical rows"

    def source(path):
        relative = str(path.relative_to(ROOT))
        sources.setdefault(relative, sha256(path))
        return relative

    def record(k, model, display, path, selector, trials, cohort="primary"):
        value = number(display)
        if value is None:
            return
        assert math.isfinite(value) and 0 <= value <= 100
        provenance.append(dict(
            cohort=cohort, dataset=k[0], architecture=k[1], mismatch_type=k[2],
            level=f"{k[3]:g}", model=model, accuracy_percent=display,
            trials=trials, source=source(path), selector=selector,
        ))

    for k, row in rows.items():
        for model in [PCN, FIRST] + ALL_COMPLETE_WRNS:
            record(k, model, row.get(model), CANONICAL, model, row["trial counts"])

    for path, additions in [(POOLING, POOLS), (NO_BIAS, NOBIAS)]:
        seen = set()
        for incoming in selected_rows(path):
            k = key(incoming)
            assert k not in seen, (path, k)
            seen.add(k)
            target = rows[k]
            # Appended reports can carry copied references; check them, never import them.
            for model in [PCN, FIRST] + ALL_COMPLETE_WRNS:
                if number(incoming.get(model)) is not None and number(target.get(model)) is not None:
                    assert incoming[model] == target[model], (path, k, model)
            for model in additions:
                target[model] = incoming.get(model, "N/A")
                record(k, model, target[model], path, model, incoming["trial counts"])

    def add_pcn(handle, dataset, arch, family, model, path, member=""):
        payload = pickle.load(handle)
        assert len(payload) == 1 and float(next(iter(payload))) == 1.75
        trials = next(iter(payload.values()))["noise_acc_spec"]["Johnson"]
        expected_levels = {k[3] for k in rows if k[:3] == (dataset, arch, family)}
        assert {float(level) for level in trials} == expected_levels
        for level, values in trials.items():
            level = float(level)
            assert len(values) == (1 if level == 0 else 10)
            assert all(math.isfinite(float(v)) and 0 <= float(v) <= 100 for v in values)
            k = dataset, arch, family, level
            assert model not in rows[k], (k, model, "duplicate result")
            rows[k][model] = fmt(statistics.mean(values))
            selector = f"{member}::t_end=1.75/noise_acc_spec/Johnson/{level:g}"
            record(k, model, rows[k][model], path, selector, len(values))

    with tarfile.open(ARCHIVE) as archive:
        for member in archive.getmembers():
            if not member.isfile() or not member.name.endswith("/result.pkl"):
                continue
            parts = Path(member.name).parts
            i = next(i for i, part in enumerate(parts) if part in {"cifar10", "cifar100"})
            dataset, tag, condition = parts[i:i + 3]
            if condition not in FAMILY_DIRS:
                continue
            suffix = "_ODEBlockXInit_t1p75"
            assert tag.endswith(suffix)
            with archive.extractfile(member) as handle:
                add_pcn(handle, dataset, tag[:-len(suffix)], FAMILY_DIRS[condition], WITH_X, ARCHIVE, member.name)
    for model, directory in [(WITH_X, EARLIER_C100_28_2), (WITH_PC, EARLIER_C100_28_2.with_name("WRN_28_2_ODEBlockPC"))]:
        for condition, family in FAMILY_DIRS.items():
            path = directory / condition / "result.pkl"
            with path.open("rb") as handle:
                add_pcn(handle, "cifar100", "WRN_28_2", family, model, path)

    counts = Counter(p["model"] for p in provenance)
    assert counts[WITH_X] == 140 and counts[WITH_PC] == 20
    assert counts[FIRST] == 120
    assert counts[NOBIAS[0]] == 80 and counts[NOBIAS[1]] == 60
    assert all(number(row.get(NOBIAS[1])) is None for k, row in rows.items() if k[:2] == ("cifar100", "WRN_16_4"))
    aliases = {model: f"P{i}" for i, model in enumerate(MODELS[:4])}
    aliases.update({model: f"W{i}" for i, model in enumerate(MODELS[4:])})
    lines = ["# All Available PCN/WRN Max-Additive and Multiplicative Results", "",
             "Paper-scope consolidation; existing reports are not edited. Accuracy is top-1 percent, shown to two decimals. "
             "Nonzero primary entries are ten-trial means; N/A means unavailable or excluded, never zero accuracy. "
             "Zero-level trial counts differ between sources and are preserved in the provenance CSV. "
             "Zero-level recal-BN results are post-recalibration accuracy, not the checkpoint's training best accuracy.", "",
             "See [paper handoff](paper_handoff_max_mul.md) for definitions, training, evaluation, limitations and missing work. "
             "[Cell provenance](paper_max_mul_provenance.csv) gives each source column or archive member; "
             "[source manifest](paper_max_mul_manifest.json) records hashes and generation-code revision.", "",
             "## Model Legend", ""]
    descriptions = {
        PCN: "Legacy ODEXInitFFFB; no BN, no convolution biases; all eight pairs.",
        FIRST: "Legacy PCNetWith1stConv; conventional biased stem, no conv-bias mismatch; six pairs.",
        WITH_PC: "Input-driven RHS, FF-initialized state; CIFAR-100 WRN-28-2 only.",
        WITH_X: "Input-driven RHS, input-initialized state; seven pairs; CIFAR-10 WRN-28-4 TRAINING FAILED (latest retry: user-reported CUDA OOM).",
    }
    for model in ALL_COMPLETE_WRNS:
        descriptions[model] = "All eight pairs; BN mode is an evaluation condition, not a separately trained model."
    descriptions["BN-free WRN"] = "Convolution biases enabled; dataset-specific recipe; all eight pairs."
    for model in POOLS:
        descriptions[model] = "WRN-28-2 only, both datasets; BN variants use unfused recal-BN; BN-free variants have conv biases."
    descriptions[NOBIAS[0]] = "CIFAR-100 all four pairs; classifier bias retained."
    descriptions[NOBIAS[1]] = "CIFAR-100 three valid pairs; collapsed WRN-16-4 excluded; classifier bias retained."
    md_table(lines, ["ID", "Variant / evaluation", "Coverage / note"],
             [[aliases[m], m, descriptions[m]] for m in MODELS], ["---"] * 3)
    lines += ["## Primary Results", "",
              "Configured base seed 123 for the main campaigns; historical provenance is not proof of identical RNG streams. "
              "Only corrected PCN max-additive references are included. Original CSV display values are copied unchanged. "
             "The two newly added PCN columns are calculated from their result pickles, without substituting legacy values.", ""]
    lines += ["**Failed training:** CIFAR-10 WRN-28-4-sized PCNetNoBatchNorm / ODEBlockXInit / t_end=1.75. "
              "The latest remote retry also failed with CUDA OOM (user report); warmup was already five epochs. "
              "It is not pending evaluation. All its cells remain N/A. Solver-related graph-memory growth is suspected, "
              "not established from an inspected OOM traceback.", ""]
    for dataset in ["cifar10", "cifar100"]:
        for arch in ["WRN_16_2", "WRN_16_4", "WRN_28_2", "WRN_28_4"]:
            for family, title in FAMILIES.items():
                lines += [f"### {dataset.upper()} {arch.replace('_', '-')} / {title}", ""]
                matching = [k for k in sorted(rows) if k[:3] == (dataset, arch, family)]
                md_table(lines, ["Level"] + [aliases[m] for m in MODELS],
                         [[f"{k[3]:g}"] + [rows[k].get(m, "N/A") for m in MODELS] for k in matching])

    lines += ["## Separate Seed-456 WRN Repeat", "",
              "CIFAR-100 WRN-28-2 only, seven WRNs, ten trials per level. "
              "PCN was not rerun in this campaign, so no seed-456 PCN result is claimed. "
              "These measurements are not averaged with the primary results.", ""]
    repeat_models = ["WRN unfused recal-BN", "WD1e-3 WRN unfused recal-BN", "WD1e-3 final-drop0.25 WRN unfused recal-BN"] + POOLS
    for family, title in FAMILIES.items():
        lines += [f"### {title}", ""]
        selected = [r for r in read_csv(SEED_REPEAT) if r["mismatch_type"] == family]
        selected.sort(key=lambda r: float(r["level"]))
        md_table(lines, ["Level"] + [aliases[m] for m in repeat_models],
                 [[r["level"]] + [r[m] for m in repeat_models] for r in selected])
        for r in selected:
            for model in repeat_models:
                record(("cifar100", "WRN_28_2", family, float(r["level"])), model, r[model], SEED_REPEAT, model, 10, "seed456")

    lines += ["## Limited MaxPool-Shortcut Pilot", "",
              "CIFAR-10 WRN-16-2 BN-free, conv biases enabled. Only the two downsampling shortcuts were replaced; "
              "this is not the all-shortcuts AvgPool variant. Separate training recipe; only the five recorded "
              "multiplicative levels below are available, ten trials each. Do not rank it as a complete-family control.", ""]
    pilot = read_csv(MAXPOOL_PILOT)
    md_table(lines, ["Level", "Legacy PCN reference", "BN-free MaxPool-shortcut pilot"],
             [[r["mismatch_level"], rows[("cifar10", "WRN_16_2", "multiplicative", float(r["mismatch_level"]))][PCN], fmt(float(r["wrn_nobn_mean_accuracy"]))] for r in pilot])
    for r in pilot:
        record((r["dataset"], r["architecture"], r["mismatch_type"], float(r["mismatch_level"])),
               "BN-free MaxPool-shortcut pilot", fmt(float(r["wrn_nobn_mean_accuracy"])), MAXPOOL_PILOT,
               "wrn_nobn_mean_accuracy", r["num_trials"], "limited_pilot")

    fields = ["dataset", "architecture", "mismatch_type", "level"] + MODELS
    flat = [{**dict(zip(fields[:4], [*k[:3], f"{k[3]:g}"])), **{m: rows[k].get(m, "N/A") for m in MODELS}} for k in sorted(rows)]
    assert len({tuple(p[f] for f in ["cohort", "dataset", "architecture", "mismatch_type", "level", "model"]) for p in provenance}) == len(provenance)
    # Detect changed inputs before publishing a mixed snapshot.
    assert all(sha256(ROOT / p) == digest for p, digest in sources.items())
    OUTPUT.mkdir(parents=True, exist_ok=True)
    atomic_text(OUTPUT / "all_pcn_wrn_max_mul.md", "\n".join(lines) + "\n")
    atomic_text(OUTPUT / "all_pcn_wrn_max_mul.csv", csv_text(flat, fields))
    atomic_text(OUTPUT / "paper_max_mul_provenance.csv", csv_text(provenance, list(provenance[0])))
    manifest = dict(
        scope=list(FAMILIES), primary_rows=len(rows), primary_cell_counts=dict(counts),
        provenance_cells=len(provenance), source_sha256=sources,
        generator_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        generator_sha256=sha256(Path(__file__)),
        wrn_training_artifacts=training_artifact_inventory(),
        artifact_note="Resolved-config locations and existing checkpoint paths, not proof of historical checkpoint hashes. Failed adjusted no-bias WRN-16-4 is explicitly ineligible.",
        pcn_training_failures=[dict(
            dataset="cifar10", architecture="WRN_28_4", model_class="PCNetNoBatchNorm",
            ode_block="ODEBlockXInit", t_end=1.75, warmup_epochs=5,
            status="training_failed", latest_reported_error="CUDA OOM",
            evidence="User reports latest remote retry failed; latest OOM traceback not inspected locally.",
            suspected_cause="Numerical instability causing excessive solver steps and retained autograd memory; unconfirmed.",
            evaluation_status="unavailable_training_failed", accuracy=None,
        )],
        generator_note="Generation snapshot only, NOT verified historical training or evaluation revisions. Source CSVs may preserve only aggregate precision.",
        regeneration="python baseline/generate_paper_max_mul_bundle.py",
    )
    atomic_text(OUTPUT / "paper_max_mul_manifest.json", json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"primary_rows": len(rows), "provenance_cells": len(provenance), "primary_coverage": dict(counts)}, indent=2))


if __name__ == "__main__":
    main()

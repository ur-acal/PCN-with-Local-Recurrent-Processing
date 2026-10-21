#!/usr/bin/env python3
"""Generate the With-X PCN report from the consolidated paper values."""

import csv
import statistics
from collections import defaultdict
from pathlib import Path

from generate_pcn_wrn_robustness_summary import ROOT, md_table


EVIDENCE = ROOT.parent / "papers/hardware-native-neural-ode/shared/evidence/generic-mismatch"
VALUES = EVIDENCE / "robustness_report_values.csv"
OUTPUT = EVIDENCE / "pcn_withx_vs_top3_wrn_max_mul.md"

# Retained for generate_paper_max_mul_bundle.py, which loads the raw With-X data.
ARCHIVE = ROOT / "logs/pcn_odeblockxinit_t1p75_mismatch_results.tar.gz"
EARLIER_C100_28_2 = ROOT / "logs/pcn_x_minus_fb_mismatch_local/cifar100/WRN_28_2_ODEBlockXInit"
FAMILY_DIRS = {"max_additive": "additive_max", "multiplicative": "multiplicative"}

SEVERE = {"max_additive": (0.08, 0.09, 0.10),
          "multiplicative": (0.30, 0.35, 0.40)}
TITLES = {"max_additive": "Max additive", "multiplicative": "Multiplicative"}
ARCHITECTURES = ("WRN_16_2", "WRN_16_4", "WRN_28_2", "WRN_28_4")
LABELS = {
    "W0-R": "std_BN_wd5e-4_lr1e-1_fd0_recal",
    "W1-R": "std_BN_wd1e-3_lr1e-1_fd0_recal",
    "W2-R": "std_BN_wd1e-3_lr1e-1_fd0p25_recal",
    "R4-R": "avgSC_BN_wd1e-3_lr1e-1_fd0p25_recal",
    "R7-R": "avgMain_BN_wd1e-3_lr1e-1_fd0p25_recal",
    "R10-R": "std_BN_wd1e-3_lr1e-2_fd0p25_ptinit_recal",
    "R14-R": "maxMain_BN_wd1e-3_lr1e-1_fd0p25_recal",
    "R17-R": "avgMain_BN_wd1e-3_lr1e-2_fd0p25_ptinit_recal",
    "R18-R": "maxMain_BN_wd1e-3_lr1e-2_fd0p25_ptinit_recal",
}


def fmt(value, signed=False):
    if value is None:
        return "N/A"
    return f"{value:+.2f}" if signed else f"{value:.2f}"


def load_values(path=VALUES):
    values = {}
    with path.open() as stream:
        for row in csv.DictReader(stream):
            if row["mismatch_type"] not in SEVERE:
                continue
            key = (row["dataset"], row["architecture"], row["mismatch_type"],
                   round(float(row["level"]), 8), row["variant"])
            if key in values:
                raise ValueError(f"Duplicate value: {key}")
            values[key] = float(row["accuracy"])
    return values


def select_top3(values, dataset, mismatch_type):
    """Rank complete recalibrated-BN WRNs over four sizes and three severe levels."""
    variants = {key[4] for key in values if key[0] == dataset and key[2] == mismatch_type}
    variants = {variant for variant in variants
                if variant.endswith("-R") and not variant.startswith("PB-")}
    ranked = []
    for variant in variants:
        points = [values.get((dataset, architecture, mismatch_type, level, variant))
                  for architecture in ARCHITECTURES for level in SEVERE[mismatch_type]]
        if all(point is not None for point in points):
            ranked.append((statistics.mean(points), variant))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    if len(ranked) < 3:
        raise ValueError(f"Fewer than three complete WRNs for {dataset}/{mismatch_type}")
    return ranked[:3]


def build_report(values):
    selections = {(dataset, mismatch_type): select_top3(values, dataset, mismatch_type)
                  for dataset in ("cifar10", "cifar100") for mismatch_type in SEVERE}
    pair_rows = []
    level_sections = []
    aggregate = defaultdict(list)

    for dataset in ("cifar10", "cifar100"):
        for architecture in ARCHITECTURES:
            for mismatch_type in SEVERE:
                selected = [variant for _, variant in selections[dataset, mismatch_type]]
                levels = sorted({key[3] for key in values
                                 if key[:3] == (dataset, architecture, mismatch_type)})
                rows = []
                p0_gaps, p3_gaps = [], []
                for level in levels:
                    p0 = values.get((dataset, architecture, mismatch_type, level, "P0"))
                    p3 = values.get((dataset, architecture, mismatch_type, level, "P3"))
                    wrns = [values.get((dataset, architecture, mismatch_type, level, variant))
                            for variant in selected]
                    best = max(wrns) if all(value is not None for value in wrns) else None
                    p0_gap = p0 - best if p0 is not None and best is not None else None
                    p3_gap = p3 - best if p3 is not None and best is not None else None
                    rows.append([fmt(level), fmt(p0), fmt(p3), *[fmt(value) for value in wrns],
                                 fmt(p0_gap, True), fmt(p3_gap, True)])
                    if level > 0 and p0_gap is not None:
                        p0_gaps.append(p0_gap)
                    if level > 0 and p3 is not None:
                        p3_gaps.append(p3_gap)
                        aggregate[mismatch_type].append(p3 - p0)
                pair_rows.extend([
                    [dataset, architecture.replace("_", "-"), "PCN_noX", TITLES[mismatch_type],
                     fmt(statistics.mean(p0_gaps)) if p0_gaps else "N/A",
                     f"{sum(gap > 0 for gap in p0_gaps)}/{len(p0_gaps)}" if p0_gaps else "N/A"],
                    [dataset, architecture.replace("_", "-"), "PCN_X_XInit", TITLES[mismatch_type],
                     fmt(statistics.mean(p3_gaps)) if p3_gaps else "N/A",
                     f"{sum(gap > 0 for gap in p3_gaps)}/{len(p3_gaps)}" if p3_gaps else "N/A"],
                ])
                headers = ["Level", "PCN_noX", "PCN_X_XInit",
                           *[LABELS.get(variant, variant) for variant in selected],
                           "PCN_noX - best WRN", "PCN_X_XInit - best WRN"]
                section = [f"### {dataset.upper()} {architecture.replace('_', '-')} - {TITLES[mismatch_type]}", ""]
                md_table(section, headers, rows)
                level_sections.extend(section)

    common_deltas = aggregate["max_additive"] + aggregate["multiplicative"]
    lines = [
        "# PCN_noX and PCN_X_XInit vs Fixed Top-Three WRNs", "",
        "## Scope", "",
        "This report is generated from the consolidated paper values. For each dataset and mismatch type, "
        "complete recalibrated-BN WRNs are ranked by mean accuracy across all four architectures and all "
        "three severe mismatch levels. The resulting top three remain fixed across architectures and levels. "
        "This is the same selection rule used by the main PCN/WRN report.", "",
        "PCN_X_XInit is available for seven of eight dataset/architecture pairs; CIFAR-10 WRN-28-4 training "
        "failed and remains N/A. Comparisons involving PCN_X_XInit therefore use only its seven-pair coverage.", "",
        f"Across its matched nonzero points, PCN_X_XInit is higher than PCN_noX at "
        f"**{sum(value > 0 for value in common_deltas)}/{len(common_deltas)}** points and changes mean "
        f"accuracy by **{statistics.mean(common_deltas):+.2f} points**.", "",
        "## Fixed WRN selections", "",
    ]
    selection_rows = []
    for (dataset, mismatch_type), ranked in selections.items():
        selection_rows.append([dataset, TITLES[mismatch_type],
                               *[f"{LABELS.get(variant, variant)} ({score:.2f})"
                                 for score, variant in ranked]])
    md_table(lines, ["Dataset", "Mismatch", "Rank 1", "Rank 2", "Rank 3"], selection_rows)
    lines.extend(["## Pair summary", ""])
    md_table(lines, ["Dataset", "Architecture", "PCN", "Mismatch",
                     "Mean gap vs best selected WRN", "Wins"], pair_rows)
    lines.extend(["Means exclude level zero. A win is a strictly positive accuracy gap.", "",
                  "## Full level-by-level results", "", *level_sections,
                  "## Data source", "", f"- `{VALUES}`", "",
                  f"Generator: `{Path(__file__).resolve()}`. No inference or training is performed.", ""])
    return "\n".join(lines).rstrip() + "\n"


def main():
    values = load_values()
    OUTPUT.write_text(build_report(values))
    print(OUTPUT)


if __name__ == "__main__":
    main()

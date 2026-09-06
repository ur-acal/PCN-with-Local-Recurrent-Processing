#!/usr/bin/env python3
"""Compare legacy and input-driven PCNs with the three strongest WRN controls."""

import io
import pickle
import statistics
import tarfile
from collections import defaultdict
from pathlib import Path

from generate_pcn_wrn_robustness_summary import (
    CANONICAL,
    FAMILIES,
    PCN,
    ROOT,
    TOP_THREE,
    fmt,
    md_table,
    number,
    rank_complete_wrns,
    selected_rows,
)


ARCHIVE = ROOT / "logs/pcn_odeblockxinit_t1p75_mismatch_results.tar.gz"
EARLIER_C100_28_2 = ROOT / "logs/pcn_x_minus_fb_mismatch_local/cifar100/WRN_28_2_ODEBlockXInit"
OUTPUT = ROOT / "logs/pcn_wrn_robustness_summary/pcn_withx_vs_top3_wrn_max_mul.md"
FAMILY_DIRS = {
    "max_additive": "additive_max",
    "multiplicative": "multiplicative",
}


def result_means(handle):
    payload = pickle.load(handle)
    if len(payload) != 1:
        raise AssertionError(f"Expected one t_end result, found {list(payload)}")
    result = next(iter(payload.values()))
    trials = result["noise_acc_spec"]["Johnson"]
    means = {}
    for level, values in trials.items():
        expected = 1 if float(level) == 0 else 10
        if len(values) != expected:
            raise AssertionError(
                f"Level {level} has {len(values)} trials; expected {expected}"
            )
        means[float(level)] = statistics.mean(values)
    return means


def load_withx_results():
    results = {}
    with tarfile.open(ARCHIVE, "r:gz") as archive:
        for member in archive.getmembers():
            if not member.isfile() or not member.name.endswith("/result.pkl"):
                continue
            parts = Path(member.name).parts
            try:
                dataset_index = next(
                    index
                    for index, part in enumerate(parts)
                    if part in {"cifar10", "cifar100"}
                )
            except StopIteration:
                continue
            dataset = parts[dataset_index]
            result_tag = parts[dataset_index + 1]
            condition = parts[dataset_index + 2]
            if condition not in FAMILY_DIRS:
                continue
            suffix = "_ODEBlockXInit_t1p75"
            if not result_tag.endswith(suffix):
                raise AssertionError(f"Unexpected result tag: {result_tag}")
            architecture = result_tag[: -len(suffix)]
            extracted = archive.extractfile(member)
            if extracted is None:
                raise AssertionError(f"Could not read {member.name}")
            results[(dataset, architecture, FAMILY_DIRS[condition])] = result_means(
                io.BytesIO(extracted.read())
            )

    for condition, family in FAMILY_DIRS.items():
        path = EARLIER_C100_28_2 / condition / "result.pkl"
        with path.open("rb") as handle:
            results[("cifar100", "WRN_28_2", family)] = result_means(handle)

    expected_pairs = {
        (dataset, architecture, family)
        for dataset in ("cifar10", "cifar100")
        for architecture in ("WRN_16_2", "WRN_16_4", "WRN_28_2", "WRN_28_4")
        for family in FAMILIES
        if not (dataset == "cifar10" and architecture == "WRN_28_4")
    }
    if set(results) != expected_pairs:
        raise AssertionError(
            f"Unexpected with-X coverage; missing={expected_pairs - set(results)}, "
            f"extra={set(results) - expected_pairs}"
        )
    return results


def grouped(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["architecture"], row["mismatch_type"])].append(row)
    for values in groups.values():
        values.sort(key=lambda row: float(row["level"]))
    return groups


def mean(values):
    return statistics.mean(values) if values else None


def build_report(rows, ranking, withx):
    groups = grouped(rows)
    summary_rows = []
    legacy_common_gaps = []
    withx_gaps = []
    withx_legacy_deltas = []
    family_stats = defaultdict(lambda: defaultdict(list))

    for key, values in sorted(groups.items()):
        dataset, architecture, family = key
        clean = next(row for row in values if float(row["level"]) == 0)
        nonzero = [row for row in values if float(row["level"]) > 0]
        best_clean = max(number(clean[column]) for column in TOP_THREE)
        best_values = [max(number(row[column]) for column in TOP_THREE) for row in nonzero]
        legacy_values = [number(row[PCN]) for row in nonzero]
        legacy_gaps = [pcn - best for pcn, best in zip(legacy_values, best_values)]

        summary_rows.append(
            [
                dataset,
                architecture.replace("_", "-"),
                "Legacy no-input dynamics",
                FAMILIES[family],
                fmt(number(clean[PCN])),
                fmt(best_clean),
                fmt(mean(legacy_values)),
                fmt(mean(best_values)),
                fmt(mean(legacy_gaps), signed=True),
                f"{sum(gap > 0 for gap in legacy_gaps)}/{len(legacy_gaps)}",
                fmt(legacy_gaps[-1], signed=True),
            ]
        )

        withx_levels = withx.get(key)
        if withx_levels is None:
            summary_rows.append(
                [
                    dataset,
                    architecture.replace("_", "-"),
                    "ODEBlockXInit with input drive",
                    FAMILIES[family],
                    "N/A",
                    fmt(best_clean),
                    "N/A",
                    fmt(mean(best_values)),
                    "N/A",
                    "N/A",
                    "N/A",
                ]
            )
            continue

        expected_levels = {float(row["level"]) for row in values}
        if set(withx_levels) != expected_levels:
            raise AssertionError(f"Level mismatch for {key}")
        withx_values = [withx_levels[float(row["level"])] for row in nonzero]
        current_withx_gaps = [pcn - best for pcn, best in zip(withx_values, best_values)]
        current_legacy_gaps = [pcn - best for pcn, best in zip(legacy_values, best_values)]
        deltas = [new - old for new, old in zip(withx_values, legacy_values)]
        withx_gaps.extend(current_withx_gaps)
        legacy_common_gaps.extend(current_legacy_gaps)
        withx_legacy_deltas.extend(deltas)
        family_stats[family]["withx_gaps"].extend(current_withx_gaps)
        family_stats[family]["legacy_gaps"].extend(current_legacy_gaps)
        family_stats[family]["deltas"].extend(deltas)

        summary_rows.append(
            [
                dataset,
                architecture.replace("_", "-"),
                "ODEBlockXInit with input drive",
                FAMILIES[family],
                fmt(withx_levels[0.0]),
                fmt(best_clean),
                fmt(mean(withx_values)),
                fmt(mean(best_values)),
                fmt(mean(current_withx_gaps), signed=True),
                f"{sum(gap > 0 for gap in current_withx_gaps)}/{len(current_withx_gaps)}",
                fmt(current_withx_gaps[-1], signed=True),
            ]
        )

    lines = [
        "# Legacy and Input-Driven PCNetNoBatchNorm vs the Three Most Robust WRNs",
        "",
        "## Scope and answer",
        "",
        "This report uses corrected tensor-max additive mismatch and multiplicative mismatch. Every nonzero value is the mean of 10 trials; zero mismatch is a single clean evaluation. The WRN comparator is the strongest of the same top-three recalibrated-BN variants selected independently at each level.",
        "",
        f"The input-driven `ODEBlockXInit` PCN is available for seven of eight dataset/architecture pairs and wins **{sum(gap > 0 for gap in withx_gaps)}/{len(withx_gaps)}** matched nonzero points against the best WRN, with a mean gap of **{mean(withx_gaps):+.2f} points**. On exactly those same points, the legacy PCN wins **{sum(gap > 0 for gap in legacy_common_gaps)}/{len(legacy_common_gaps)}** with a mean gap of **{mean(legacy_common_gaps):+.2f} points**.",
        "",
        f"Relative to the legacy PCN itself, `ODEBlockXInit` changes mean nonzero accuracy by **{mean(withx_legacy_deltas):+.2f} points** and is higher at **{sum(delta > 0 for delta in withx_legacy_deltas)}/{len(withx_legacy_deltas)}** matched points. CIFAR-10 WRN-28-4 is `N/A` because its input-driven checkpoint failed during training; no result was inferred or substituted.",
        "",
        "## Architecture differences",
        "",
    ]
    md_table(
        lines,
        ["Aspect", "Legacy PCNetNoBatchNorm", "Input-driven PCNetNoBatchNorm", "WRN counterpart"],
        [
            ["Basic computation", "FF Conv2d and FB ConvTranspose2d in each ODE layer.", "Same FF/FB operators.", "Two Conv2d operations per residual block plus an additive shortcut."],
            ["Dynamics", "`dy/dt = FF(ReLU(FB(y)))` (`ODEXInitFFFB`).", "`dy/dt = FF(ReLU(x - FB(y)))` (`ODEBlockXInit`).", "Finite feed-forward residual computation."],
            ["State initialization", "Input `x` initializes `y`; channel changes use deterministic copy/padding.", "Same.", "Block input follows the identity or projection shortcut."],
            ["Integration", "Adaptive Dopri5, `t_end=1.75`, tolerance `1e-4`.", "Same.", "No ODE integration or recurrent weight reuse."],
            ["Widths/depth", "WRN-matched widths; 7 or 13 PC layers.", "Same.", "WRN-16 or WRN-28 with widths `16k`, `32k`, `64k`."],
            ["Downsampling", "Separate 2x2 MaxPool after two designated PC layers.", "Same.", "Stride-2 main convolution and learned projection at stages 2 and 3."],
            ["Normalization", "No BatchNorm.", "No BatchNorm.", "Pre-activation BatchNorm plus final BatchNorm."],
            ["Biases", "FF/FB convolutions have no bias; classifier has bias.", "Same.", "Convolutions have no bias; classifier has bias."],
            ["Head", "Final dropout 0.25, ReLU, global average pool, linear classifier.", "Same.", "Final BN, optional final dropout, ReLU, global average pool, linear classifier."],
        ],
    )
    lines.extend(
        [
            "The only architectural difference between the two PCN columns is the ODE vector field: the newer model includes the layer input `x` as a constant drive. They are independently trained checkpoints, not two evaluation modes applied to one checkpoint.",
            "",
            "## Training-recipe differences",
            "",
            "Both PCN variants use the same recorded recipe: 300 epochs, batch size 128, SGD momentum 0.9, initial LR 0.01, weight decay 1e-3, five warmup epochs, cosine decay, final-feature dropout 0.25, and the same TIMM CIFAR augmentations. The top-three WRNs use initial LR 0.1, explicit WRN initialization, seed 4096, and earlier checkpoint evaluation; their weight decay/final dropout combinations are shown below.",
            "",
        ]
    )
    md_table(
        lines,
        ["WRN rank", "Variant", "Weight decay", "Final dropout", "Mismatch-time BN"],
        [
            ["1", TOP_THREE[0], "1e-3", "0.25", "Unfused, recalibrated per level/trial"],
            ["2", TOP_THREE[1], "1e-3", "0", "Unfused, recalibrated per level/trial"],
            ["3", TOP_THREE[2], "5e-4", "0", "Unfused, recalibrated per level/trial"],
        ],
    )
    lines.extend(
        [
            "## Evaluation alignment",
            "",
            "All PCN and WRN columns use `P' = P + sigma * max(abs(P)) * epsilon` for max-additive mismatch and `P' = P * (1 + sigma * epsilon)` for multiplicative mismatch. Eligible convolution/FF/FB weights, classifier weight, and classifier bias are perturbed; convolution biases are absent. PCN linear-layer additive mismatch uses the corrected implementation.",
            "",
            "All use base seed 123 and 10 trials per nonzero level. Random draws are deterministic but not elementwise paired across architectures because tensor shapes and traversal differ. WRN BN parameters are unperturbed, and running statistics are recalibrated on 5,120 training samples after each perturbation; PCN has no BN.",
            "",
            "## WRN robustness ranking",
            "",
        ]
    )
    md_table(
        lines,
        ["Rank", "WRN variant", "Mean nonzero accuracy", "Mean clean retention", "Points"],
        [
            [str(index + 1), item["name"], fmt(item["mean"]), f"{item['retention']:.2f}%", str(item["points"])]
            for index, item in enumerate(ranking[:3])
        ],
    )
    lines.extend(["## Input-driven coverage summary", ""])
    family_rows = []
    for family in ("additive_max", "multiplicative"):
        stats = family_stats[family]
        family_rows.append(
            [
                FAMILIES[family],
                fmt(mean(stats["legacy_gaps"]), signed=True),
                f"{sum(gap > 0 for gap in stats['legacy_gaps'])}/{len(stats['legacy_gaps'])}",
                fmt(mean(stats["withx_gaps"]), signed=True),
                f"{sum(gap > 0 for gap in stats['withx_gaps'])}/{len(stats['withx_gaps'])}",
                fmt(mean(stats["deltas"]), signed=True),
                f"{sum(delta > 0 for delta in stats['deltas'])}/{len(stats['deltas'])}",
            ]
        )
    md_table(
        lines,
        ["Mismatch", "Legacy gap vs best WRN", "Legacy wins", "With-X gap vs best WRN", "With-X wins", "With-X - legacy", "With-X wins vs legacy"],
        family_rows,
    )
    lines.extend(["## Pair-level summary", ""])
    md_table(
        lines,
        ["Dataset", "Pair", "PCN variant", "Mismatch", "PCN clean", "Best-WRN clean", "PCN mean", "Best-WRN mean", "PCN gap", "PCN wins", "Endpoint gap"],
        summary_rows,
    )
    lines.extend(
        [
            "Means exclude level zero. Endpoint is 0.10 for max-additive and 0.40 for multiplicative mismatch.",
            "",
            "## Full level-by-level results",
            "",
        ]
    )
    for key, values in sorted(groups.items()):
        dataset, architecture, family = key
        withx_levels = withx.get(key)
        table_rows = []
        for row in values:
            level = float(row["level"])
            best = max(number(row[column]) for column in TOP_THREE)
            legacy = number(row[PCN])
            new = withx_levels.get(level) if withx_levels is not None else None
            table_rows.append(
                [
                    f"{level:g}",
                    fmt(legacy),
                    fmt(new),
                    row[TOP_THREE[0]],
                    row[TOP_THREE[1]],
                    row[TOP_THREE[2]],
                    fmt(legacy - best, signed=True),
                    fmt(new - best, signed=True) if new is not None else "N/A",
                ]
            )
        lines.extend(
            [
                f"### {dataset.upper()} {architecture.replace('_', '-')} - {FAMILIES[family]}",
                "",
            ]
        )
        md_table(
            lines,
            ["Level", "Legacy PCN", "With-X PCN", "WD1e-3/drop0.25", "WD1e-3/drop0", "WD5e-4/drop0", "Legacy - best WRN", "With-X - best WRN"],
            table_rows,
        )

    lines.extend(
        [
            "## Data sources",
            "",
            f"- Legacy PCN and WRNs: `{CANONICAL.relative_to(ROOT)}`",
            f"- Six new input-driven PCNs: `{ARCHIVE.relative_to(ROOT)}`",
            f"- Earlier CIFAR-100 WRN-28-2-sized input-driven PCN: `{EARLIER_C100_28_2.relative_to(ROOT)}`",
            "",
            "The existing report and source result files were not modified.",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def main():
    rows = selected_rows(CANONICAL)
    ranking = rank_complete_wrns(rows)
    withx = load_withx_results()
    OUTPUT.write_text(build_report(rows, ranking, withx))
    print(OUTPUT)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate focused PCN-versus-WRN robustness summaries from recorded CSVs."""

import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "logs/pcn_wrn_robustness_summary"
CANONICAL = (
    ROOT
    / "logs/combined_pcn_wrn_mismatch_finaldrop025_complete/full_accuracy_comparison.csv"
)
POOLING = (
    ROOT
    / "logs/combined_pcn_wrn_mismatch_avgpool_variants_complete/full_accuracy_comparison.csv"
)
NO_BIAS = (
    ROOT
    / "logs/combined_pcn_wrn_mismatch_nobias_cifar100_complete/full_accuracy_comparison.csv"
)
MAXPOOL_PILOT = (
    ROOT
    / "logs/wrn_nobn_hparam_winners_mul_mismatch/maxpool_shortcut/full_aggregate.csv"
)

PCN = "PCNetNoBatchNorm"
FAMILIES = {
    "additive_max": "Max-absolute additive",
    "multiplicative": "Multiplicative",
}
TOP_THREE = [
    "WD1e-3 final-drop0.25 WRN unfused recal-BN",
    "WD1e-3 WRN unfused recal-BN",
    "WRN unfused recal-BN",
]
ALL_COMPLETE_WRNS = [
    "WRN folded frozen-BN",
    "WRN folded recal-BN",
    "WRN unfused recal-BN",
    "WD1e-3 WRN folded frozen-BN",
    "WD1e-3 WRN folded recal-BN",
    "WD1e-3 WRN unfused recal-BN",
    "BN-free WRN",
    "WD1e-3 final-drop0.25 WRN unfused recal-BN",
]
BNFREE_GROUPS = [
    (
        "Standard BN-free WRN",
        CANONICAL,
        ["BN-free WRN"],
    ),
    (
        "Parameter-free shortcut variants",
        POOLING,
        ["AvgPool BN-free", "Stride-2 BN-free"],
    ),
    (
        "No-convolution-bias variants",
        NO_BIAS,
        [
            "BN-free/no-conv-bias, WD5e-4/drop0",
            "BN-free/no-conv-bias, WD1e-3/final-drop0.25",
        ],
    ),
]


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt(value, signed=False):
    if value is None:
        return "N/A"
    return f"{value:+.2f}" if signed else f"{value:.2f}"


def md_table(lines, headers, rows, align=None):
    lines.append("| " + " | ".join(headers) + " |")
    if align is None:
        align = ["---"] + ["---:"] * (len(headers) - 1)
    lines.append("|" + "|".join(align) + "|")
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    lines.append("")


def atomic_text(path, content):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content)
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_rows(path):
    return [row for row in read_csv(path) if row["mismatch_type"] in FAMILIES]


def clean_index(rows, columns):
    result = {}
    for row in rows:
        if float(row["level"]) != 0:
            continue
        for column in columns:
            value = number(row.get(column))
            if value is not None:
                result[(row["dataset"], row["architecture"], row["mismatch_type"], column)] = value
    return result


def rank_complete_wrns(rows):
    clean = clean_index(rows, ALL_COMPLETE_WRNS)
    ranking = []
    for column in ALL_COMPLETE_WRNS:
        values = []
        retention = []
        for row in rows:
            if float(row["level"]) == 0:
                continue
            value = number(row.get(column))
            base = clean.get(
                (row["dataset"], row["architecture"], row["mismatch_type"], column)
            )
            if value is None or base is None:
                break
            values.append(value)
            retention.append(100.0 * value / base)
        if len(values) == 144:
            ranking.append(
                {
                    "name": column,
                    "mean": statistics.mean(values),
                    "retention": statistics.mean(retention),
                    "points": len(values),
                }
            )
    ranking.sort(key=lambda item: item["mean"], reverse=True)
    observed = [item["name"] for item in ranking[:3]]
    if observed != TOP_THREE:
        raise AssertionError(f"Unexpected top-three WRN ranking: {observed}")
    return ranking


def grouped(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["architecture"], row["mismatch_type"])].append(row)
    for values in groups.values():
        values.sort(key=lambda row: float(row["level"]))
    return groups


def report_one(rows, ranking):
    groups = grouped(rows)
    summary_rows = []
    all_gaps = []
    family_wins = defaultdict(int)
    family_points = defaultdict(int)
    for (dataset, architecture, family), values in sorted(groups.items()):
        nonzero = [row for row in values if float(row["level"]) > 0]
        gaps = [
            number(row[PCN]) - max(number(row[column]) for column in TOP_THREE)
            for row in nonzero
        ]
        all_gaps.extend(gaps)
        family_wins[family] += sum(gap > 0 for gap in gaps)
        family_points[family] += len(gaps)
        clean = next(row for row in values if float(row["level"]) == 0)
        summary_rows.append(
            [
                dataset,
                architecture.replace("_", "-"),
                FAMILIES[family],
                fmt(number(clean[PCN])),
                fmt(max(number(clean[column]) for column in TOP_THREE)),
                fmt(statistics.mean(number(row[PCN]) for row in nonzero)),
                fmt(statistics.mean(max(number(row[column]) for column in TOP_THREE) for row in nonzero)),
                fmt(statistics.mean(gaps), signed=True),
                f"{sum(gap > 0 for gap in gaps)}/{len(gaps)}",
                fmt(gaps[-1], signed=True),
            ]
        )

    lines = [
        "# PCNetNoBatchNorm vs the Three Most Robust WRN Baselines",
        "",
        "## Scope and answer",
        "",
        "This report uses only corrected tensor-max additive mismatch and multiplicative mismatch. Accuracy is in percentage points. Every nonzero entry is the mean of 10 trials; zero-mismatch entries are retained only as clean references.",
        "",
        f"**Within this experiment, PCNetNoBatchNorm is broadly more robust, but not universally better.** Against the strongest of the top three WRNs independently at every nonzero level, PCN wins **{sum(gap > 0 for gap in all_gaps)}/{len(all_gaps)}** points and has a macro mean advantage of **{statistics.mean(all_gaps):+.2f} points**. It wins **{family_wins['additive_max']}/{family_points['additive_max']}** max-additive points and **{family_wins['multiplicative']}/{family_points['multiplicative']}** multiplicative points.",
        "",
        "The result is strongest on CIFAR-10 and on the larger CIFAR-100 counterparts. It is not absolute: at severe max-additive mismatch, PCN loses to the best WRN for CIFAR-100 WRN-16-2 and WRN-28-2, and there are several low-mismatch losses where a WRN starts from higher clean accuracy. This establishes robustness under these tested perturbations; it does not by itself isolate recurrence as the cause.",
        "",
        "## Architecture differences",
        "",
    ]
    md_table(
        lines,
        ["Aspect", "PCNetNoBatchNorm", "WRN counterpart"],
        [
            ["Basic computation", "Each PC layer has an FF Conv2d and an FB ConvTranspose2d inside an ODE vector field.", "Each residual block has two distinct 3x3 Conv2d operations and an additive shortcut."],
            ["Dynamics", "Legacy no-x ODEXInitFFFB: `dy/dt = FF(ReLU(FB(y)))`; adaptive Dopri5, `t_end=1.75`, tolerance `1e-4`.", "Finite feed-forward residual computation; no ODE solve or recurrent weight reuse."],
            ["State/input path", "The layer input initializes `y`; channel increases use deterministic copy/padding. The vector field itself does not use `x`.", "The block input reaches the output through the identity or learned projection shortcut."],
            ["Channel schedule", "Matches the paired WRN widths: stem output 16, then `16k`, `32k`, `64k` for `k=2` or `4`.", "Stem width 16 followed by three WRN stages at `16k`, `32k`, `64k`."],
            ["Nominal depth", "7 PC layers for WRN-16 counterparts; 13 for WRN-28 counterparts.", "Depth 16 has 6 two-conv blocks; depth 28 has 12 two-conv blocks, plus the stem."],
            ["Trainable spatial kernels", "14 for depth-16 pairs and 26 for depth-28 pairs, counting one FF and one FB kernel per PC layer.", "16 and 28 respectively: stem, two kernels per block, and three learned 1x1 projections."],
            ["Second/feedback kernel shape", "FB maps `Cout -> Cin`; its parameters are separate from FF.", "The second block convolution maps `Cout -> Cout`."],
            ["Shortcuts/bypass", "No learned bypass convolution in these checkpoints.", "Three learned 1x1 projection shortcuts, one at the first block of each width stage."],
            ["Spatial downsampling", "A separate 2x2 max pool follows two designated PC layers.", "Stride 2 is inside the main-path convolution and corresponding projection at stages 2 and 3."],
            ["Normalization", "No BatchNorm anywhere.", "Pre-activation BatchNorm in each block plus a final BatchNorm."],
            ["Biases", "FF/FB convolutions have no bias; classifier has bias. Registered zero `b0` is unused by ODEXInitFFFB.", "Convolutions and projection shortcuts have no bias; classifier has bias."],
            ["Head", "Final-feature dropout, ReLU, global average pool, linear classifier.", "Final BatchNorm, optional final dropout, ReLU, global average pool, linear classifier."],
        ],
    )
    lines.extend(
        [
            "The paired models therefore share the broad channel expansion and output head, but they do not have identical operators or parameterization. PCN generally has fewer parameters because its FB kernels map back to `Cin` and it has no learned projection shortcuts; it also reuses each FF/FB pair for an adaptive number of ODE evaluations.",
            "",
            "## Training-recipe differences",
            "",
            "All models use 300 epochs, batch size 128, SGD momentum 0.9, cosine decay to `1e-6`, five warmup epochs from `1e-5`, and the same TIMM CIFAR augmentation base: RandAugment `m9-mstd0.5-inc1`, label smoothing 0.1, mixup 0.2, cutmix 1.0, color jitter 0.1, random erasing 0.25, horizontal flip 0.5, and CIFAR mean/std normalization. None uses distillation.",
            "",
        ]
    )
    md_table(
        lines,
        ["Setting", "PCNetNoBatchNorm", "WRN", "WD1e-3 WRN", "WD1e-3/final-drop0.25 WRN"],
        [
            ["Initial learning rate", "0.01", "0.1", "0.1", "0.1"],
            ["Weight decay", "1e-3", "5e-4", "1e-3", "1e-3"],
            ["Block dropout", "Not used", "0", "0", "0"],
            ["Final-feature dropout", "0.25", "0", "0", "0.25"],
            ["Global training seed", "Not explicitly fixed", "4096", "4096", "4096"],
            ["Checkpoint evaluation window", "Begins after the first 150 epochs; every 5 epochs", "Begins after epoch 70; every 5 epochs", "Same", "Same"],
            ["Initialization", "PyTorch defaults for Conv2d, ConvTranspose2d, and Linear", "Kaiming-normal fan-out Conv; BN gamma 1/beta 0; Linear N(0,0.01)", "Same", "Same"],
        ],
    )
    lines.extend(
        [
            "## Evaluation alignment",
            "",
            "For max-additive mismatch, both paths apply `P' = P + sigma * max(abs(P)) * epsilon`; for multiplicative mismatch, both apply `P' = P * (1 + sigma * epsilon)`, with independent standard-normal entries. Convolution/FF/FB weights, classifier weight, and classifier bias are perturbed. Convolution biases are absent in these PCN and WRN models. The corrected PCN evaluator does perturb the linear layer additively.",
            "",
            "WRN BatchNorm parameters and buffers are not perturbed. For every mismatch level and trial, each top-three WRN is restored, perturbed, and then its unfused BN running statistics are recalibrated using 5,120 training samples before testing. PCN has no BN to recalibrate. Both use base mismatch seed 123, but draws are not elementwise paired because tensor shapes and traversal differ.",
            "",
            "## WRN robustness ranking",
            "",
            "Ranking includes only variants with all 144 nonzero points across both datasets, all four architectures, and both mismatch families. Mean accuracy is the primary rank; clean-normalized retention is shown as a check. Both metrics give the same top three.",
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
    lines.extend(["## Pair-level summary", ""])
    md_table(
        lines,
        ["Dataset", "Pair", "Mismatch", "PCN clean", "Best-WRN clean", "PCN mean", "Best-WRN mean", "PCN gap", "PCN wins", "Endpoint gap"],
        summary_rows,
    )
    lines.extend(
        [
            "`Best-WRN` is selected independently at each level from the top three, making the comparison conservative for PCN. Means exclude level zero; endpoint is 0.10 for max-additive and 0.40 for multiplicative.",
            "",
            "## Full level-by-level results",
            "",
        ]
    )
    for (dataset, architecture, family), values in sorted(groups.items()):
        lines.extend(
            [
                f"### {dataset.upper()} {architecture.replace('_', '-')} - {FAMILIES[family]}",
                "",
            ]
        )
        table_rows = []
        for row in values:
            best = max(number(row[column]) for column in TOP_THREE)
            table_rows.append(
                [
                    f"{float(row['level']):g}",
                    row[PCN],
                    row[TOP_THREE[0]],
                    row[TOP_THREE[1]],
                    row[TOP_THREE[2]],
                    fmt(number(row[PCN]) - best, signed=True),
                ]
            )
        md_table(
            lines,
            ["Level", "PCN", "WD1e-3/drop0.25", "WD1e-3/drop0", "WD5e-4/drop0", "PCN - best WRN"],
            table_rows,
        )
    lines.extend(
        [
            "## Data source",
            "",
            f"All numerical values come from `{CANONICAL.relative_to(ROOT)}`. Existing source files were not modified.",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def comparison_summaries(path, columns, references):
    rows = selected_rows(path)
    groups = grouped(rows)
    result = []
    for key, values in sorted(groups.items()):
        dataset, architecture, family = key
        reference_values = references[key]
        reference_by_level = {
            float(row["level"]): row for row in reference_values
        }
        nonzero = [row for row in values if float(row["level"]) > 0]
        clean = next(row for row in values if float(row["level"]) == 0)
        clean_reference = reference_by_level[0.0]
        for column in columns:
            pairs = []
            for row in nonzero:
                reference = reference_by_level[float(row["level"])]
                pcn = number(row.get(PCN))
                variant = number(row.get(column))
                top = max(number(reference[name]) for name in TOP_THREE)
                frozen = number(reference["WRN folded frozen-BN"])
                if pcn is not None and variant is not None:
                    pairs.append((pcn, top, frozen, variant))
            pcn_clean = number(clean.get(PCN))
            variant_clean = number(clean.get(column))
            if not pairs or pcn_clean is None or variant_clean is None:
                continue
            if pcn_clean != number(clean_reference[PCN]):
                raise AssertionError(f"PCN reference mismatch for {key}")
            pcn_gaps = [pcn - variant for pcn, _, _, variant in pairs]
            frozen_gaps = [variant - frozen for _, _, frozen, variant in pairs]
            result.append(
                {
                    "dataset": dataset,
                    "architecture": architecture.replace("_", "-"),
                    "variant": column,
                    "family": FAMILIES[family],
                    "family_key": family,
                    "pcn_clean": pcn_clean,
                    "top_clean": max(number(clean_reference[name]) for name in TOP_THREE),
                    "frozen_clean": number(clean_reference["WRN folded frozen-BN"]),
                    "variant_clean": variant_clean,
                    "pcn_mean": statistics.mean(pcn for pcn, _, _, _ in pairs),
                    "top_mean": statistics.mean(top for _, top, _, _ in pairs),
                    "frozen_mean": statistics.mean(frozen for _, _, frozen, _ in pairs),
                    "variant_mean": statistics.mean(variant for _, _, _, variant in pairs),
                    "variant_frozen_gap": statistics.mean(frozen_gaps),
                    "variant_frozen_wins": f"{sum(gap > 0 for gap in frozen_gaps)}/{len(frozen_gaps)}",
                    "pcn_variant_gap": statistics.mean(pcn_gaps),
                    "pcn_wins": f"{sum(gap > 0 for gap in pcn_gaps)}/{len(pcn_gaps)}",
                }
            )
    return result


def report_two():
    canonical_rows = selected_rows(CANONICAL)
    references = grouped(canonical_rows)
    group_rows = []
    for group_name, path, columns in BNFREE_GROUPS:
        for row in comparison_summaries(path, columns, references):
            row["group"] = group_name
            group_rows.append(row)

    canonical = [row for row in group_rows if row["group"] == "Standard BN-free WRN"]
    pooling = [row for row in group_rows if row["group"] == "Parameter-free shortcut variants"]
    no_bias = [row for row in group_rows if row["group"] == "No-convolution-bias variants"]

    def count_wins(rows, field):
        won = total = 0
        for row in rows:
            left, right = row[field].split("/")
            won += int(left)
            total += int(right)
        return won, total

    canonical_wins = count_wins(canonical, "pcn_wins")
    pooling_wins = count_wins(pooling, "pcn_wins")
    no_bias_wins = count_wins(no_bias, "pcn_wins")
    bnfree_frozen_wins = count_wins(canonical, "variant_frozen_wins")
    bn_effect_rows = []
    for family_key, family_name in FAMILIES.items():
        family_rows = [row for row in canonical if row["family_key"] == family_key]
        wins = count_wins(family_rows, "variant_frozen_wins")
        bn_effect_rows.append(
            [
                family_name,
                fmt(statistics.mean(row["frozen_mean"] for row in family_rows)),
                fmt(statistics.mean(row["variant_mean"] for row in family_rows)),
                fmt(statistics.mean(row["variant_frozen_gap"] for row in family_rows), signed=True),
                f"{wins[0]}/{wins[1]}",
            ]
        )
    bnfree_recal_rows = []
    for family_key, family_name in FAMILIES.items():
        family_points = [
            row
            for row in canonical_rows
            if row["mismatch_type"] == family_key and float(row["level"]) > 0
        ]
        references_to_compare = [
            ("Best top-three at each point", None),
            *[(name, name) for name in TOP_THREE],
        ]
        for label, column in references_to_compare:
            reference_values = [
                max(number(row[name]) for name in TOP_THREE)
                if column is None
                else number(row[column])
                for row in family_points
            ]
            bnfree_values = [number(row["BN-free WRN"]) for row in family_points]
            differences = [
                bnfree - reference
                for bnfree, reference in zip(bnfree_values, reference_values)
            ]
            bnfree_recal_rows.append(
                [
                    family_name,
                    label,
                    fmt(statistics.mean(reference_values)),
                    fmt(statistics.mean(bnfree_values)),
                    fmt(statistics.mean(differences), signed=True),
                    f"{sum(value > 0 for value in differences)}/{len(differences)}",
                ]
            )

    context_columns = [
        PCN,
        *TOP_THREE,
        "WRN folded frozen-BN",
        "BN-free WRN",
    ]
    clean = clean_index(canonical_rows, context_columns)
    context_rows = []
    for column in context_columns:
        clean_values = [
            value
            for (dataset, architecture, family, name), value in clean.items()
            if name == column
        ]
        values = []
        retention = []
        for row in canonical_rows:
            if float(row["level"]) == 0:
                continue
            value = number(row[column])
            base = clean[(row["dataset"], row["architecture"], row["mismatch_type"], column)]
            values.append(value)
            retention.append(100.0 * value / base)
        context_rows.append(
            [
                column,
                fmt(statistics.mean(clean_values)),
                fmt(statistics.mean(values)),
                f"{statistics.mean(retention):.2f}%",
                str(len(values)),
            ]
        )
    lines = [
        "# PCNetNoBatchNorm vs BN-Free WRN Variants",
        "",
        "## Scope and answer",
        "",
        "This report uses corrected tensor-max additive and multiplicative mismatch only. Means exclude zero mismatch; every reported mismatch value is a 10-trial mean.",
        "",
        "**The current evidence does not support the absence of BatchNorm as the main explanation for PCN robustness.** PCN itself has no BN, but removing BN from WRN does not reproduce PCN's result. PCN wins **%d/%d** matched nonzero points against standard BN-free WRNs, **%d/%d** against the two WRN-28-2 parameter-free-shortcut BN-free variants, and **%d/%d** against valid no-convolution-bias variants." % (*canonical_wins, *pooling_wins, *no_bias_wins),
        "",
        "BN still has a measurable effect: the standard BN-free WRN beats the recorded non-recalibrated `WRN folded frozen-BN` at **%d/%d** points, especially under max-additive mismatch. The decisive countercheck is that it performs worse than every top-three unfused recalibrated-BN WRN under both mismatch families. Against the best recalibrated result at each point, BN-free trails by **9.70 points** for max-additive and **5.09 points** for multiplicative mismatch, winning only **12/80** and **11/64** points. Thus BN handling cannot be ignored, but **lack of BN is neither sufficient nor the dominant explanation for PCN's lead**. More credit belongs to PCN-specific architecture/parameterization and training differences, which still require further controls to isolate." % bnfree_frozen_wins,
        "",
        "The recorded baseline without BN recalibration is `WRN folded frozen-BN`. There is no separate unfused frozen-BN column in these experiments.",
        "",
        "## Complete-coverage context",
        "",
        "This table uses all 144 nonzero points across both datasets, all four architecture pairs, and both mismatch families. It places the BN-free result beside PCN, the previously selected top-three WRNs, and the no-recalibration baseline.",
        "",
    ]
    md_table(
        lines,
        ["Model/evaluation condition", "Mean clean accuracy", "Mean nonzero accuracy", "Mean clean retention", "Points"],
        context_rows,
    )
    lines.extend([
        "## Direct BN-removal check",
        "",
        "This is the closest available direct check: standard BN-free WRN versus original WRN with folded/frozen BN and no recalibration. Positive differences favor BN removal.",
        "",
    ])
    md_table(
        lines,
        ["Mismatch", "Frozen-BN mean", "BN-free mean", "BN-free - frozen", "BN-free wins"],
        bn_effect_rows,
    )
    lines.extend([
        "## BN-free versus recalibrated BN",
        "",
        "This is the missing decisive comparison. Negative differences mean the standard BN-free WRN is less robust than the recalibrated-BN reference. `Best top-three at each point` takes the highest of the three recalibrated WRNs separately at every matched level.",
        "",
    ])
    md_table(
        lines,
        ["Mismatch", "Recalibrated-BN reference", "Recal-BN mean", "BN-free mean", "BN-free - recal-BN", "BN-free wins"],
        bnfree_recal_rows,
    )
    lines.extend([
        "## WRN references",
        "",
        "The first three rows are the previously selected top-three robust WRNs. They share the standard WRN architecture: BatchNorm, original stride-2 main convolutions, and learned 1x1 transition shortcuts. The fourth row is the available baseline without BN recalibration.",
        "",
    ])
    md_table(
        lines,
        ["Reference", "Training difference from original WRN", "Mismatch-time BN handling"],
        [
            ["WD1e-3 final-drop0.25 WRN", "WD 1e-3 and final-feature dropout 0.25.", "Unfused BN recalibrated independently after each perturbation."],
            ["WD1e-3 WRN", "WD 1e-3; final-feature dropout remains 0.", "Unfused BN recalibrated independently after each perturbation."],
            ["Original WRN", "WD 5e-4 and final-feature dropout 0.", "Unfused BN recalibrated independently after each perturbation."],
            ["Original WRN, no recalibration", "Same original WRN training recipe.", "Folded eligible Conv-BN pairs; remaining BN statistics frozen, with no recalibration."],
        ],
    )
    lines.extend([
        "## BN-free models and training",
        "",
        "All variants use 300 epochs, batch size 128, SGD momentum 0.9, cosine scheduling, five warmup epochs, seed 4096, and no BatchNorm. Convolution weights use Kaiming-normal fan-out initialization and linear weights use N(0,0.01); enabled convolution biases retain their PyTorch initialization. Unless stated otherwise, convolution and classifier biases are enabled.",
        "",
    ])
    md_table(
        lines,
        ["Variant", "Architecture", "Training recipe", "Coverage"],
        [
            ["Standard BN-free WRN", "Original stride-2 main convolutions and three learned 1x1 shortcuts; convolution biases enabled.", "CIFAR-10: LR 0.1, WD 1e-4, clip 2, bias LR x0.5, zero bias WD, block dropout 0.1, final dropout 0, RandAugment m9, no mixup/cutmix, smoothing 0.1, erasing 0.1. CIFAR-100: LR 0.1, WD 5e-4, no clipping, ordinary bias LR/WD, block/final dropout 0, RandAugment m7, mixup 0.1, cutmix 0.5, smoothing 0.05, erasing 0.1.", "CIFAR-10/100; WRN-16-2, 16-4, 28-2, 28-4"],
            ["AvgPool BN-free", "All main convolutions use stride 1; stage transitions use AvgPool. Learned shortcuts are replaced by channel padding and AvgPool where spatial downsampling is needed.", "Dataset-specific standard BN-free recipe above, except WD 1e-3 and final-feature dropout 0.25.", "CIFAR-10/100; WRN-28-2 only"],
            ["Stride-2 BN-free", "Original stride-2 main convolutions; learned shortcuts replaced by channel padding and AvgPool.", "Same as AvgPool BN-free.", "CIFAR-10/100; WRN-28-2 only"],
            ["BN-free/no-conv-bias, WD5e-4/drop0", "Original stride-2 main path and learned 1x1 shortcuts; every convolution bias disabled; classifier bias retained.", "Original WRN `custom_noresize` recipe unchanged: LR 0.1, WD 5e-4, block/final dropout 0, standard m9 TIMM augmentations.", "CIFAR-100; all four pairs"],
            ["BN-free/no-conv-bias, WD1e-3/final-drop0.25", "Same no-BN/no-convolution-bias architecture.", "Only WD 1e-3 and final-feature dropout 0.25 differ from the preceding no-bias recipe.", "CIFAR-100; WRN-16-2, 28-2, 28-4. WRN-16-4 collapsed and is excluded."],
            ["Earlier max-pool-shortcut pilot", "Standard BN-free WRN-16-2, but only the two stride-2 learned shortcuts are replaced by MaxPool/channel padding.", "Its independently searched CIFAR-10 recipe; not a full-family controlled run.", "CIFAR-10 WRN-16-2; multiplicative levels 0, 0.1, 0.2, 0.3, 0.4 only"],
        ],
    )
    lines.extend(
        [
            "## Robustness summary",
            "",
            "Every mean excludes level zero. `Best recal-WRN` selects the strongest of the top three recalibrated-BN WRNs independently at each level. Positive BN-free minus frozen-BN values show where removing BN beats the recorded no-recalibration baseline. Coverage differs by group and must not be pooled as if every variant were tested on every pair.",
            "",
        ]
    )
    headers = ["Group", "Dataset", "Pair", "BN-free variant", "Mismatch", "PCN mean", "Best recal-WRN", "Frozen-BN", "BN-free", "BN-free - frozen", "BN-free wins", "PCN - BN-free", "PCN wins"]
    md_table(
        lines,
        headers,
        [
            [
                row["group"], row["dataset"], row["architecture"], row["variant"], row["family"],
                fmt(row["pcn_mean"]), fmt(row["top_mean"]), fmt(row["frozen_mean"]),
                fmt(row["variant_mean"]), fmt(row["variant_frozen_gap"], signed=True),
                row["variant_frozen_wins"], fmt(row["pcn_variant_gap"], signed=True), row["pcn_wins"],
            ]
            for row in group_rows
        ],
    )

    pilot_rows = read_csv(MAXPOOL_PILOT)
    canonical_map = {
        float(row["level"]): number(row[PCN])
        for row in canonical_rows
        if row["dataset"] == "cifar10"
        and row["architecture"] == "WRN_16_2"
        and row["mismatch_type"] == "multiplicative"
    }
    lines.extend(
        [
            "## Earlier max-pool-shortcut pilot",
            "",
            "This limited pilot is kept separate because it lacks a max-additive run and uses a separately selected recipe. Replacing the two learned downsampling shortcuts with MaxPool did not improve robustness.",
            "",
        ]
    )
    md_table(
        lines,
        ["Multiplicative level", "PCN", "BN-free MaxPool shortcut", "PCN gap"],
        [
            [
                f"{float(row['mismatch_level']):g}",
                fmt(canonical_map.get(float(row["mismatch_level"]))),
                fmt(number(row["wrn_nobn_mean_accuracy"])),
                fmt(
                    canonical_map.get(float(row["mismatch_level"]))
                    - number(row["wrn_nobn_mean_accuracy"]),
                    signed=True,
                ),
            ]
            for row in pilot_rows
        ],
    )
    lines.extend(
        [
            "## Interpretation",
            "",
            "1. Removing BN often improves robustness over folded/frozen BN, particularly for max-additive mismatch, so BN treatment does affect the measured result.",
            "2. That improvement is not consistent across all points, does not beat the strongest recalibrated-BN WRNs globally, and does not close the PCN gap.",
            "3. Parameter-free shortcuts, removing stride-2 convolutions, and removing convolution biases also fail to reproduce PCN robustness. The WD/final-drop aligned no-bias WRN-16-4 collapsed, demonstrating additional optimization instability.",
            "4. The defensible conclusion is therefore not that BN has zero effect. It is that PCN robustness cannot be explained primarily by the absence of BN; recurrence and the remaining PCN-specific differences are stronger candidates.",
            "",
            "## Data sources",
            "",
            f"- `{CANONICAL.relative_to(ROOT)}`",
            f"- `{POOLING.relative_to(ROOT)}`",
            f"- `{NO_BIAS.relative_to(ROOT)}`",
            f"- `{MAXPOOL_PILOT.relative_to(ROOT)}`",
            "",
            "Existing source files were not modified.",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def main():
    canonical_rows = selected_rows(CANONICAL)
    ranking = rank_complete_wrns(canonical_rows)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    first = OUTPUT / "pcn_vs_top3_wrn_max_mul.md"
    second = OUTPUT / "pcn_vs_bnfree_wrn_max_mul.md"
    atomic_text(first, report_one(canonical_rows, ranking))
    atomic_text(second, report_two())
    sources = [CANONICAL, POOLING, NO_BIAS, MAXPOOL_PILOT]
    manifest = {
        "reports": [str(first.resolve()), str(second.resolve())],
        "mismatch_families": list(FAMILIES),
        "top_three": TOP_THREE,
        "ranking_rule": "macro mean accuracy over all 144 nonzero complete-coverage points",
        "sources": [
            {"path": str(path.resolve()), "sha256": sha256(path)} for path in sources
        ],
    }
    atomic_text(OUTPUT / "report_manifest.json", json.dumps(manifest, indent=2) + "\n")
    print(first)
    print(second)


if __name__ == "__main__":
    main()

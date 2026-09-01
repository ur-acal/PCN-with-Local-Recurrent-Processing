#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


LEVELS = (0.0, 0.02, 0.03, 0.05, 0.07)
PAPER_VANILLA = {0.0: 79.31, 0.02: 77.24, 0.03: 71.95, 0.05: 56.21, 0.07: 47.09}
PAPER_RWP = {0.02: 78.26, 0.03: 77.00, 0.05: 72.43, 0.07: 65.27}
PAPER_SAM = {0.02: 79.49, 0.03: 77.13, 0.05: 68.11, 0.07: 60.24}


def read_results(path: Path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {float(row["noise_level"]): row["acc"] for row in rows}


def mean_accuracy(value: str):
    return float(value.split("±", 1)[0].rstrip("%"))


def format_paper(values, level):
    return f"{values[level]:.2f}" if level in values else "N/A"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", type=Path, required=True)
    args = parser.parse_args()

    folded_csv = args.output_root / "folded_frozen_bn" / "summary_adapt_noresize_scratch_cifar100_resnet18.csv"
    recal_csv = args.output_root / "unfused_recal_bn" / "summary_adapt_noresize_scratch_cifar100_resnet18_bn_recal.csv"
    folded = read_results(folded_csv)
    recal = read_results(recal_csv)

    lines = [
        "# CIFAR-100 ResNet-18 `max_sqrt` mismatch reproduction",
        "",
        "The local checkpoint has 79.88% recorded clean accuracy. Mismatch uses the paper's per-output-filter scale. "
        "Folded evaluation fuses all 20 Conv-BN pairs and excludes fused biases; unfused evaluation recalibrates "
        "all BN running statistics on 5120 unlabeled training images after each fixed mismatch realization.",
        "",
        "| sigma | Paper vanilla | Folded frozen-BN | Unfused recal-BN | Paper Quadratic RWP | Paper Quadratic SAM |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for level in LEVELS:
        lines.append(
            f"| {level:g} | {PAPER_VANILLA[level]:.2f} | {folded[level]} | {recal[level]} | "
            f"{format_paper(PAPER_RWP, level)} | {format_paper(PAPER_SAM, level)} |"
        )

    nonzero = LEVELS[1:]
    folded_mae = sum(abs(mean_accuracy(folded[level]) - PAPER_VANILLA[level]) for level in nonzero) / len(nonzero)
    recal_mae = sum(abs(mean_accuracy(recal[level]) - PAPER_VANILLA[level]) for level in nonzero) / len(nonzero)
    closest = "folded frozen-BN" if folded_mae <= recal_mae else "unfused recal-BN"
    lines.extend([
        "",
        f"Mean absolute deviation from the paper vanilla curve over nonzero levels: folded frozen-BN "
        f"{folded_mae:.2f} points; unfused recal-BN {recal_mae:.2f} points. Closest condition: **{closest}**.",
        "",
        "Paper source: https://arxiv.org/abs/2602.07320",
    ])
    (args.output_root / "summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

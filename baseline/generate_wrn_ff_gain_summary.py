#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate_csv", type=Path, required=True)
    parser.add_argument("--output_md", type=Path, required=True)
    args = parser.parse_args()

    with args.aggregate_csv.open(newline="") as handle:
        rows = sorted(csv.DictReader(handle), key=lambda row: float(row["ff_gain"]))

    best_frozen = max(rows, key=lambda row: float(row["paired_wrn_frozen_bn_mean_accuracy"]))
    best_recal = max(rows, key=lambda row: float(row["wrn_recalibrated_bn_mean_accuracy"]))
    lines = [
        "# CIFAR-100 WRN-28-2 FF-gain sweep",
        "",
        "Model: WD=1e-3, final-feature dropout=0.25 WRN-28-2. The gain scales the stem "
        "and first convolution in every residual block. No random mismatch is applied. "
        "Recal-BN uses the fixed 5120-image unlabeled training subset.",
        "",
        "| FF gain | Frozen-BN accuracy | Recal-BN accuracy | Recalibration change |",
        "|---:|---:|---:|---:|",
    ]
    for row in rows:
        frozen = float(row["paired_wrn_frozen_bn_mean_accuracy"])
        recal = float(row["wrn_recalibrated_bn_mean_accuracy"])
        lines.append(f"| {float(row['ff_gain']):.2f} | {frozen:.2f} | {recal:.2f} | {recal - frozen:+.2f} |")

    lines.extend([
        "",
        f"Best frozen-BN accuracy: **{float(best_frozen['paired_wrn_frozen_bn_mean_accuracy']):.2f}%** "
        f"at gain **{float(best_frozen['ff_gain']):.2f}**.",
        "",
        f"Best recal-BN accuracy: **{float(best_recal['wrn_recalibrated_bn_mean_accuracy']):.2f}%** "
        f"at gain **{float(best_recal['ff_gain']):.2f}**.",
    ])
    args.output_md.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

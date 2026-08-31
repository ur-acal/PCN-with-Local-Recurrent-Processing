#!/usr/bin/env python3
import argparse
import csv
import pickle
from pathlib import Path

import numpy as np


PAIRS = ("WRN_16_2", "WRN_16_4", "WRN_28_2", "WRN_28_4")
LEVELS = (0.25, 0.5, 0.75, 1.0)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--existing_pilot_root", type=Path, required=True)
    return parser.parse_args()


def load_pcn(path):
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    block = next(iter(payload.values()))["noise_acc_spec"]
    values = next(iter(block.values()))
    return {float(level): [float(value) for value in trials] for level, trials in values.items()}


def load_wrn(path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {float(row["mismatch_level"]): row for row in rows}


def format_stats(mean, std):
    return f"{mean:.2f} +/- {std:.2f}"


def main():
    args = parse_args()
    output_rows = []
    for architecture in PAIRS:
        if architecture == "WRN_28_2":
            pcn_path = args.existing_pilot_root / "pcn/result.pkl"
            wrn_path = (
                args.existing_pilot_root
                / "wrn_wd1e3_unfolded_recal_nonzero/full_aggregate.csv"
            )
        else:
            pcn_path = args.root / "pcn" / architecture / "result.pkl"
            wrn_path = args.root / "wrn" / architecture / "full_aggregate.csv"

        pcn = load_pcn(pcn_path)
        wrn = load_wrn(wrn_path)

        for level in LEVELS:
            pcn_trials = np.asarray(pcn[level], dtype=float)
            wrn_row = wrn[level]
            if len(pcn_trials) != 3 or int(wrn_row["num_trials"]) != 3:
                raise ValueError(f"Expected three trials for {architecture} level {level}")
            pcn_mean = float(pcn_trials.mean())
            pcn_std = float(pcn_trials.std())
            wrn_mean = float(wrn_row["wrn_recalibrated_bn_mean_accuracy"])
            wrn_std = float(wrn_row["wrn_recalibrated_bn_std_accuracy"])
            output_rows.append(
                {
                    "architecture": architecture,
                    "rms_level": level,
                    "pcn_mean_accuracy": pcn_mean,
                    "pcn_std_accuracy": pcn_std,
                    "wrn_wd1e3_unfused_recal_bn_mean_accuracy": wrn_mean,
                    "wrn_wd1e3_unfused_recal_bn_std_accuracy": wrn_std,
                    "pcn_minus_wrn": pcn_mean - wrn_mean,
                    "pcn_source": str(pcn_path.resolve()),
                    "wrn_source": str(wrn_path.resolve()),
                }
            )

    csv_path = args.root / "summary.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)

    lines = [
        "# CIFAR-100 RMS-Additive Pilot",
        "",
        "Accuracy is mean +/- population standard deviation over three trials. WRN is the WD=1e-3 unfused recal-BN condition.",
        "",
        "| Pair | RMS level | PCN accuracy | WRN accuracy | PCN - WRN |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in output_rows:
        lines.append(
            f"| {row['architecture']} | {row['rms_level']:g} | "
            f"{format_stats(row['pcn_mean_accuracy'], row['pcn_std_accuracy'])} | "
            f"{format_stats(row['wrn_wd1e3_unfused_recal_bn_mean_accuracy'], row['wrn_wd1e3_unfused_recal_bn_std_accuracy'])} | "
            f"{row['pcn_minus_wrn']:+.2f} |"
        )
    markdown_path = args.root / "summary.md"
    markdown_path.write_text("\n".join(lines) + "\n")
    print(markdown_path)


if __name__ == "__main__":
    main()


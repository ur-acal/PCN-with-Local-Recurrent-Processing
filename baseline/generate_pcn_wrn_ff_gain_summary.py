#!/usr/bin/env python3
import argparse
import csv
import pickle
from pathlib import Path


PCN_SERIES = (
    ("legacy", "PCN legacy, no `-x`"),
    ("odeblockpc", "PCN `ODEBlockPC`, with `-x`"),
    ("odeblockxinit", "PCN `ODEBlockXInit`, with `-x`"),
)


def load_accuracy(path: Path) -> float:
    with path.open("rb") as handle:
        result = pickle.load(handle)
    values = []
    for time_result in result.values():
        for level_map in time_result["noise_acc_spec"].values():
            for accuracies in level_map.values():
                values.extend(float(value) for value in accuracies)
    if len(values) != 1:
        raise ValueError(f"Expected one deterministic accuracy in {path}, found {values}")
    return values[0]


def load_pcn_series(primary: Path, reference: Path, expected_gains) -> dict:
    rows = {}
    for root in (primary, reference):
        for path in root.glob("gain_*/result.pkl"):
            gain = float(path.parent.name.removeprefix("gain_").replace("p", "."))
            if gain in expected_gains and gain not in rows:
                rows[gain] = load_accuracy(path)
    missing = sorted(expected_gains - rows.keys())
    if missing:
        raise ValueError(f"Missing gains {missing} under {primary} and {reference}")
    return rows


def load_wrn_series(primary: Path, reference: Path, expected_gains) -> dict:
    rows = {}
    for path in (primary, reference):
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                gain = float(row["ff_gain"])
                if gain in expected_gains and gain not in rows:
                    rows[gain] = (
                        float(row["paired_wrn_frozen_bn_mean_accuracy"]),
                        float(row["wrn_recalibrated_bn_mean_accuracy"]),
                    )
    missing = sorted(expected_gains - rows.keys())
    if missing:
        raise ValueError(f"Missing WRN gains {missing} in {primary} and {reference}")
    return rows


def main():
    parser = argparse.ArgumentParser()
    for key, _ in PCN_SERIES:
        parser.add_argument(f"--{key}_root", type=Path, required=True)
        parser.add_argument(f"--{key}_reference_root", type=Path, required=True)
    parser.add_argument("--wrn_csv", type=Path, required=True)
    parser.add_argument("--wrn_reference_csv", type=Path, required=True)
    parser.add_argument("--gains", required=True)
    parser.add_argument("--output_csv", type=Path, required=True)
    parser.add_argument("--output_md", type=Path, required=True)
    args = parser.parse_args()

    gains = sorted({float(value) for value in args.gains.split(",")})
    expected_gains = set(gains)
    pcn = {}
    for key, _ in PCN_SERIES:
        pcn[key] = load_pcn_series(
            getattr(args, f"{key}_root"),
            getattr(args, f"{key}_reference_root"),
            expected_gains,
        )
    wrn = load_wrn_series(args.wrn_csv, args.wrn_reference_csv, expected_gains)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "ff_gain",
            "pcn_legacy_no_x_accuracy",
            "pcn_odeblockpc_with_x_accuracy",
            "pcn_odeblockxinit_with_x_accuracy",
            "wrn_frozen_bn_accuracy",
            "wrn_recal_bn_accuracy",
        ])
        for gain in gains:
            writer.writerow([
                f"{gain:.2f}",
                f"{pcn['legacy'][gain]:.2f}",
                f"{pcn['odeblockpc'][gain]:.2f}",
                f"{pcn['odeblockxinit'][gain]:.2f}",
                f"{wrn[gain][0]:.2f}",
                f"{wrn[gain][1]:.2f}",
            ])

    labels = [label for _, label in PCN_SERIES]
    lines = [
        "# CIFAR-100 WRN-28-2-sized extended FF-gain comparison",
        "",
        "The gain is applied once to the selected feedforward convolutional weights. "
        "No random mismatch is applied. Each PCN value is one deterministic evaluation. "
        "WRN uses the WD=1e-3, final-dropout=0.25 checkpoint; recal-BN uses the fixed "
        "5120-image training subset after applying each gain.",
        "",
        f"| FF gain | {labels[0]} | {labels[1]} | {labels[2]} | WRN frozen BN | WRN recal-BN |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for gain in gains:
        lines.append(
            f"| {gain:.2f} | {pcn['legacy'][gain]:.2f} | {pcn['odeblockpc'][gain]:.2f} | "
            f"{pcn['odeblockxinit'][gain]:.2f} | {wrn[gain][0]:.2f} | {wrn[gain][1]:.2f} |"
        )

    lines.extend(["", "## Endpoints relative to gain 1.0", ""])
    lines.extend([
        "| Model/condition | Gain=1.00 | Gain=0.50 change | Gain=2.00 change |",
        "|---|---:|---:|---:|",
    ])
    for key, label in PCN_SERIES:
        base = pcn[key][1.0]
        lines.append(
            f"| {label} | {base:.2f} | {pcn[key][0.5] - base:+.2f} | {pcn[key][2.0] - base:+.2f} |"
        )
    for index, label in ((0, "WRN frozen BN"), (1, "WRN recal-BN")):
        base = wrn[1.0][index]
        lines.append(
            f"| {label} | {base:.2f} | {wrn[0.5][index] - base:+.2f} | "
            f"{wrn[2.0][index] - base:+.2f} |"
        )

    args.output_md.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

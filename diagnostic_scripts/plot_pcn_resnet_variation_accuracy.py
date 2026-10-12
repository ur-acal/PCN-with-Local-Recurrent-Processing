#!/usr/bin/env python3
"""Plot PCN and ResNet-110 accuracy under hardware variation.

The PCN FS_V2_T1 values are the 10-trial means reported in the CiFAIR-10 and
CiFAIR-100 ``coupler_monte_v2`` 45-corner summaries.  The ResNet-110 values
are the full-dataset FS_V2_T1 evaluations using the full-range and legacy
narrow-range coupler curve banks, respectively.

The default PCN ideal value (64.44%) is the experiment value supplied for
this comparison.  The exact matching pretraining checkpoint is not retained
locally; the retained physical fine-tuning checkpoint records 60.16%, while
a separate unitless-pretraining checkpoint records 64.33%.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-label", default="CiFAIR-100")
    parser.add_argument("--pcn-ideal", type=float, default=64.44)
    parser.add_argument("--pcn-fs-v2-t1", type=float, default=58.912)
    parser.add_argument(
        "--plot-pcn-raw",
        action="store_true",
        help="Plot the pretrained, non-hardware-adapted PCN series.",
    )
    parser.add_argument("--pcn-raw-full-range", type=float)
    parser.add_argument("--pcn-raw-legacy", type=float)
    parser.add_argument("--resnet-ideal", type=float, default=63.20)
    parser.add_argument("--resnet-full-range", type=float, default=2.70)
    parser.add_argument("--resnet-legacy", type=float, default=1.03)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/pcn_resnet_fs_v2_t1_variation_accuracy.png"),
        help="PNG output path; a PDF with the same stem is also generated.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.plot_pcn_raw and (
            args.pcn_raw_full_range is None or args.pcn_raw_legacy is None):
        raise ValueError(
            "--plot-pcn-raw requires --pcn-raw-full-range and "
            "--pcn-raw-legacy.")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    blue = "#2563a6"
    red = "#c93c37"
    dark_yellow = "#c58a00"

    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    ax.axvspan(1.0, 2.18, color="#eeeeee", zorder=0)
    ax.axvline(1.0, color="#7d858d", linewidth=1.4, zorder=1)

    # The x positions show increasing variation conceptually; they are not a
    # calibrated numerical variation scale, so numeric x ticks are omitted.
    if args.plot_pcn_raw:
        # The raw and adapted PCNs share the same digital starting point.  Do
        # not place a yellow marker at x=0 so the shared blue point stays clear.
        ax.plot(
            [0.0, 1.0, 2.0],
            [args.pcn_ideal, args.pcn_raw_full_range, args.pcn_raw_legacy],
            linestyle="--",
            linewidth=2.2,
            marker="D",
            markevery=[1, 2],
            markersize=7,
            color=dark_yellow,
            label="PCN raw",
            zorder=2,
        )

    ax.plot(
        [0.0, 1.0],
        [args.pcn_ideal, args.pcn_fs_v2_t1],
        linestyle="--",
        linewidth=2.2,
        marker="o",
        markersize=8,
        color=blue,
        label="PCN adapted" if args.plot_pcn_raw else "PCN",
        zorder=3,
    )
    ax.plot(
        [0.0, 1.0, 2.0],
        [args.resnet_ideal, args.resnet_full_range, args.resnet_legacy],
        linestyle="--",
        linewidth=2.2,
        marker="s",
        markersize=7,
        color=red,
        label="ResNet-110",
        zorder=3,
    )

    ax.set_xlim(-0.18, 2.18)
    maximum_accuracy = max(
        args.pcn_ideal,
        args.pcn_fs_v2_t1,
        args.resnet_ideal,
        args.resnet_full_range,
        args.resnet_legacy,
    )
    if args.plot_pcn_raw:
        maximum_accuracy = max(
            maximum_accuracy,
            args.pcn_raw_full_range,
            args.pcn_raw_legacy,
        )
    ax.set_ylim(0, math.ceil(maximum_accuracy / 10.0) * 10.0 + 5.0)
    ax.set_xticks([0.0, 1.0, 2.0])
    ax.set_xticklabels([
        "0 non-ideality",
        "residual non-ideality",
        "original non-ideality",
    ])
    ax.set_xlabel("Variation", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title(
        f"{args.dataset_label} Accuracy Under Hardware Variation",
        fontsize=14,
        pad=14,
    )
    ax.grid(axis="y", color="#d7dce2", linewidth=0.8, alpha=0.8)
    ax.legend(loc="lower left", frameon=False, fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    pdf_output = args.output.with_suffix(".pdf")
    fig.savefig(pdf_output, bbox_inches="tight")
    plt.close(fig)
    print(args.output.resolve())
    print(pdf_output.resolve())


if __name__ == "__main__":
    main()

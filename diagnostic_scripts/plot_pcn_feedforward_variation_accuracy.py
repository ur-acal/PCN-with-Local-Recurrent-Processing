#!/usr/bin/env python3
"""Plot raw and hardware-adapted PCN/feedforward-CNN accuracy.

The conceptual x axis represents increasing hardware variation rather than a
calibrated numerical scale.  Dashed segments connect each ideal pretrained
model to its hardware-adapted result.  Solid segments connect the raw
pretrained model's full-range and legacy-coupler results.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-label", default="CiFAIR-100")

    parser.add_argument("--pcn-ideal", type=float, default=64.44)
    parser.add_argument("--pcn-adapted", type=float, default=58.912)
    parser.add_argument("--pcn-raw-middle", type=float, default=5.11)
    parser.add_argument("--pcn-raw-right", type=float, default=1.22)

    parser.add_argument("--feedforward-label", default="WRN-28-2*")
    parser.add_argument("--feedforward-ideal", type=float, default=63.87)
    parser.add_argument("--feedforward-adapted", type=float, default=30.608)
    parser.add_argument("--feedforward-raw-middle", type=float, default=1.0254)
    parser.add_argument("--feedforward-raw-right", type=float, default=1.1837)
    parser.add_argument(
        "--adapted-star-legend",
        action="store_true",
        help=(
            "Mark adapted points with stars and use separate raw/adapted "
            "legend entries."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/pcn_wrn28_c27_fs_v2_t1_variation_accuracy_cifair100.png"
        ),
        help="PNG output path; a PDF with the same stem is also generated.",
    )
    return parser.parse_args()


def plot_model(
    ax: plt.Axes,
    *,
    ideal: float,
    adapted: float,
    raw_middle: float,
    raw_right: float,
    color: str,
    marker: str,
    label: str,
    adapted_star: bool,
) -> None:
    if adapted_star:
        ax.plot(
            [0.0, 1.0],
            [ideal, adapted],
            linestyle="--",
            linewidth=2.2,
            color=color,
            zorder=3,
        )
        ax.plot(
            [0.0],
            [ideal],
            linestyle="none",
            marker=marker,
            markersize=8,
            color=color,
            zorder=4,
        )
        ax.plot(
            [1.0],
            [adapted],
            linestyle="none",
            marker="*",
            markersize=14,
            color=color,
            zorder=4,
        )
    else:
        ax.plot(
            [0.0, 1.0],
            [ideal, adapted],
            linestyle="--",
            linewidth=2.2,
            marker=marker,
            markersize=8,
            color=color,
            label=label,
            zorder=3,
        )
    ax.plot(
        [1.0, 2.0],
        [raw_middle, raw_right],
        linestyle="-",
        linewidth=2.2,
        marker=marker,
        markersize=8,
        color=color,
        zorder=3,
    )


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    blue = "#2563a6"
    red = "#c93c37"

    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    ax.axvspan(1.0, 2.18, color="#eeeeee", zorder=0)
    ax.axvline(1.0, color="#7d858d", linewidth=1.4, zorder=1)

    plot_model(
        ax,
        ideal=args.pcn_ideal,
        adapted=args.pcn_adapted,
        raw_middle=args.pcn_raw_middle,
        raw_right=args.pcn_raw_right,
        color=blue,
        marker="o",
        label="PCN",
        adapted_star=args.adapted_star_legend,
    )
    plot_model(
        ax,
        ideal=args.feedforward_ideal,
        adapted=args.feedforward_adapted,
        raw_middle=args.feedforward_raw_middle,
        raw_right=args.feedforward_raw_right,
        color=red,
        marker="s",
        label=args.feedforward_label,
        adapted_star=args.adapted_star_legend,
    )

    maximum_accuracy = max(
        args.pcn_ideal,
        args.pcn_adapted,
        args.pcn_raw_middle,
        args.pcn_raw_right,
        args.feedforward_ideal,
        args.feedforward_adapted,
        args.feedforward_raw_middle,
        args.feedforward_raw_right,
    )
    ax.set_xlim(-0.18, 2.18)
    ax.set_ylim(0, math.ceil(maximum_accuracy / 10.0) * 10.0 + 5.0)
    ax.set_xticks([0.0, 1.0, 2.0])
    ax.set_xticklabels([
        "Ideal",
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
    if args.adapted_star_legend:
        legend_handles = [
            Line2D(
                [0], [0], color=blue, linewidth=2.2, marker="o",
                markersize=8, label="PCN raw"),
            Line2D(
                [0], [0], color=blue, linewidth=2.2, marker="*",
                markersize=12, label="PCN adapted"),
            Line2D(
                [0], [0], color=red, linewidth=2.2, marker="s",
                markersize=8, label=f"{args.feedforward_label} raw"),
            Line2D(
                [0], [0], color=red, linewidth=2.2, marker="*",
                markersize=12, label=f"{args.feedforward_label} adapted"),
        ]
        ax.legend(
            handles=legend_handles, loc="lower left", frameon=False,
            fontsize=11, handlelength=2.0)
    else:
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

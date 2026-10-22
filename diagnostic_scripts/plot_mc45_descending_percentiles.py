#!/usr/bin/env python3
"""Descending percentile curves from corner means, with separate ideal anchors.

All series contribute exactly 45 observations regardless of trials per corner.
Monotonic PCHIP interpolation uses every sorted corner mean; marker count only
changes display density. The ideal 0% anchor is excluded from interpolation.
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator
from matplotlib.ticker import PercentFormatter

REPO = Path(__file__).resolve().parents[1]
DEFAULT_SERIES = [
    dict(label="PCN adapted", model="PCN", color="#202020",
         directory=REPO / "results/coupler_monte_v2_CiFAIR100_qf1_noENOB_default45"),
    dict(label="PCN raw, improved circuit", model="PCN", color="#e68613",
         directory=REPO / "results/tt_singleton_50ep_CiFAIR100_pcn_coupler_monte_v2_mc45"),
    dict(label="PCN raw, original circuit", model="PCN", color="#e53935",
         directory=REPO / "results/tt_singleton_50ep_CiFAIR100_pcn_coupler_monte_mc45"),
    dict(label="WRN* raw, original circuit", model="WRN", color="#a50f15",
         directory=REPO / "results/tt_singleton_50ep_CiFAIR100_wrn_c27_coupler_monte_fixedTiming15ns_vdd0p1_qinv5_mc45"),
]


def load_corner_means(directory):
    directory = Path(directory)
    paths = ([directory / "corner_trials.csv"] if (directory / "corner_trials.csv").is_file()
             else sorted(directory.glob("shard_*/corner_trials.csv")))
    grouped, seen = {}, set()
    for path in paths:
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                key = (row["corner"], int(row["trial_index"]))
                if key in seen:
                    raise ValueError(f"Duplicate trial {key} in {directory}")
                seen.add(key)
                value = float(row["accuracy"])
                if not np.isfinite(value) or not 0 <= value <= 100:
                    raise ValueError(f"Invalid accuracy {value} in {directory}")
                grouped.setdefault(row["corner"], []).append(value)
    expected = {f"{p}_V{v}_T{t}" for p in ("TT", "FF", "SS", "FS", "SF")
                for v in range(3) for t in range(3)}
    if set(grouped) != expected or len({len(v) for v in grouped.values()}) != 1:
        raise ValueError(f"Expected 45 complete corners with equal trial counts: {directory}")
    return {k: float(np.mean(v)) for k, v in grouped.items()}, len(next(iter(grouped.values())))


def plot_descending_percentiles(series, output_dir, marker_counts=(10, 15),
                                ideal_accuracies=None, relative_to=None,
                                dataset_label="CiFAIR-100"):
    """Generate independent PDF/PNG figures and exact data for each marker count."""
    ideal_accuracies = ideal_accuracies or {"PCN": 64.44, "WRN": 63.87}
    if relative_to is not None and (not np.isfinite(relative_to) or relative_to <= 0):
        raise ValueError("Relative accuracy reference must be finite and positive")
    scale = 1.0 if relative_to is None else 100.0 / relative_to
    accuracy_column = "accuracy_percent" if relative_to is None else "relative_accuracy_percent"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared = []
    ranks = np.arange(1, 46) * 100.0 / 45
    for spec in series:
        means, trials = load_corner_means(spec["directory"])
        ordered = sorted(means.items(), key=lambda pair: (-pair[1], pair[0]))
        ordered = [(corner, value * scale) for corner, value in ordered]
        values = np.array([value for _, value in ordered])
        fit = PchipInterpolator(ranks, values, extrapolate=False)
        grid = np.linspace(ranks[0], 100, 2001)
        assert np.all(np.diff(fit(grid)) <= 1e-9), spec["label"]
        assert np.allclose(fit(ranks), values, atol=1e-10)
        ideal = ideal_accuracies[spec["model"]] * scale
        if ideal < values[0]:
            raise ValueError(f"Ideal accuracy below highest corner: {spec['label']}")
        prepared.append((spec, ordered, values, fit, trials, ideal))

    with (output_dir / "ranked_corner_means.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["series", "rank", "percentile", "corner", accuracy_column])
        for spec, ordered, _, _, _, _ in prepared:
            writer.writerows([spec["label"], i+1, f"{ranks[i]:.8f}", corner, f"{value:.4f}"]
                             for i, (corner, value) in enumerate(ordered))
    outputs = []
    for count in marker_counts:
        if count < 2:
            raise ValueError("At least two percentile markers are required")
        marker_x = np.linspace(ranks[0], 100, count)
        with plt.rc_context({"font.size": 10, "pdf.fonttype": 42, "ps.fonttype": 42}):
            fig, ax = plt.subplots(figsize=(8.2, 5.1), layout="constrained")
            marker_rows = []
            for spec, _, values, fit, _, ideal in prepared:
                color = spec["color"]
                ax.plot(grid, fit(grid), color=color, linewidth=1.9, label=spec["label"])
                ax.plot([0, ranks[0]], [ideal, values[0]], color=color,
                        linestyle="--", linewidth=1.3)
                ax.scatter(marker_x, fit(marker_x), color=color, s=23, zorder=4,
                           edgecolors="white", linewidths=0.35, clip_on=False)
                ax.scatter([0], [ideal], facecolors="white", edgecolors=color,
                           s=26, linewidths=1.1, zorder=5, clip_on=False)
                marker_rows.append([spec["label"], "ideal", 0, ideal])
                marker_rows.extend([spec["label"], "percentile", float(x), float(y)]
                                   for x, y in zip(marker_x, fit(marker_x)))
            upper = max(70, np.ceil(max(ideal_accuracies.values()) / 10) * 10)
            ax.set(xlim=(0, 100), ylim=(0, upper if relative_to is None else 102),
                   xlabel="Corner percentile (%)",
                   ylabel="Accuracy (%)" if relative_to is None else "Relative Accuracy",
                   title=f"{dataset_label}: descending corner accuracy")
            if relative_to is not None:
                ax.set_yticks(np.arange(0, 101, 20))
                ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
            ax.set_xticks(np.arange(0, 101, 20))
            ax.grid(axis="y", color="0.9", linewidth=0.8)
            ax.spines[["top", "right"]].set_visible(False)
            ax.legend(loc="best", ncols=2, fontsize=8, frameon=False,
                      columnspacing=1.2, handlelength=2)
            ax.text(0.5, -0.19,
                    "Corners ranked highest to lowest; 0% marks ideal pretrained accuracy.",
                    transform=ax.transAxes, ha="center", fontsize=8)
            stem = f"descending_corner_percentiles_{count}_dots"
            for extension in ("pdf", "png"):
                path = output_dir / f"{stem}.{extension}"
                fig.savefig(path, dpi=240, bbox_inches="tight")
                outputs.append(str(path))
            plt.close(fig)
        with (output_dir / f"{stem}_markers.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["series", "point_type", "percentile", accuracy_column])
            writer.writerows(marker_rows)
    metadata = dict(
        dataset=dataset_label,
        percentile_definition="rank / 45 * 100; rank 1 is the highest corner accuracy",
        interpolation="monotonic PCHIP through all 45 corner means; ideal anchor excluded",
        percentile_marker_counts=list(marker_counts), additional_ideal_marker=True,
        relative_accuracy_reference_percent=relative_to,
        series=[dict(label=s["label"], source=str(Path(s["directory"]).resolve()),
                     corners=45, trials_per_corner=trials, ideal_accuracy_percent=ideal / scale,
                     mean_accuracy_percent=float(values.mean()) / scale,
                     plotted_ideal=ideal, plotted_mean=float(values.mean()))
                for s, _, values, _, trials, ideal in prepared])
    (output_dir / "plot_sources.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=REPO / "results/mc45_descending_corner_percentiles")
    parser.add_argument("--dots", nargs="+", type=int, default=[10, 15])
    parser.add_argument("--dataset", choices=("cifar10", "cifar100"), default="cifar100")
    parser.add_argument("--pcn-ideal", type=float)
    parser.add_argument("--wrn-ideal", type=float)
    parser.add_argument("--pcn-adapted-dir", type=Path,
                        help="Override the multi-curve adapted PCN reference.")
    parser.add_argument("--pcn-improved-dir", type=Path,
                        help="Override improved-circuit PCN corner results.")
    parser.add_argument("--pcn-original-dir", type=Path,
                        help="Override original-circuit PCN corner results.")
    parser.add_argument("--wrn-original-dir", type=Path,
                        help="Override original-circuit WRN corner results.")
    parser.add_argument("--exclude-wrn", action="store_true",
                        help="Plot only the three PCN series.")
    parser.add_argument("--relative-to", type=float,
                        help="Digital accuracy in percent used as the shared 100%% reference.")
    args = parser.parse_args()
    series = [dict(spec) for spec in DEFAULT_SERIES]
    if args.dataset == "cifar10":
        # Never silently borrow CiFAIR-100 corner data for a CiFAIR-10 figure.
        if not all((args.pcn_adapted_dir, args.pcn_improved_dir,
                    args.pcn_original_dir, args.wrn_original_dir or args.exclude_wrn)):
            parser.error("CiFAIR-10 requires explicit result directories for every plotted series")
    if args.pcn_adapted_dir is not None:
        series[0]["directory"] = args.pcn_adapted_dir
    if args.pcn_improved_dir is not None:
        series[1]["directory"] = args.pcn_improved_dir
    if args.pcn_original_dir is not None:
        series[2]["directory"] = args.pcn_original_dir
    if args.wrn_original_dir is not None:
        series[3]["directory"] = args.wrn_original_dir
    if args.exclude_wrn:
        series = [spec for spec in series if spec["model"] != "WRN"]
    defaults = (89.18, 88.83) if args.dataset == "cifar10" else (64.44, 63.87)
    ideals = {"PCN": args.pcn_ideal if args.pcn_ideal is not None else defaults[0],
              "WRN": args.wrn_ideal if args.wrn_ideal is not None else defaults[1]}
    for path in plot_descending_percentiles(series, args.output_dir, args.dots,
                                            ideals, relative_to=args.relative_to,
                                            dataset_label="CiFAIR-10" if args.dataset == "cifar10" else "CiFAIR-100"):
        print(path)


if __name__ == "__main__":
    main()

"""Plot a completed standalone MC45 CSV using the established violin style."""
import argparse
import csv
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import plot_mc45_summary_figures as style


def generate_plots(result_dir, prefix="C100_PCN_cascode_1022_digital"):
    result_dir = Path(result_dir)
    trials = {}
    with (result_dir / "corner_trials.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            trials.setdefault(row["corner"], []).append(
                (int(row["trial_index"]), float(row["accuracy"])))
    expected = {
        f"{process}_V{voltage}_T{temperature}"
        for process in style.PROCESS_ORDER
        for voltage in range(3) for temperature in range(3)}
    if set(trials) != expected:
        raise ValueError("Expected exactly 45 corners")
    results = {}
    for corner in sorted(expected):
        rows = sorted(trials[corner])
        if [index for index, _ in rows] != list(range(10)):
            raise ValueError(f"Expected 10 unique trials for {corner}")
        values = np.array([value for _, value in rows])
        if not np.isfinite(values).all() or ((values < 0) | (values > 100)).any():
            raise ValueError(f"Invalid accuracy for {corner}")
        results[corner] = dict(mean=float(values.mean()), std=float(values.std()))

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42,
                         "ps.fonttype": 42})
    output = result_dir / "plots"
    means = np.array([entry["mean"] for entry in results.values()])
    limits = (np.floor(means.min()) - 0.5, np.ceil(means.max()) + 0.5)
    ordered = sorted(results, key=lambda corner: -results[corner]["mean"])
    ranks = np.arange(1, 46)
    for fmt in ("pdf", "png"):
        style.plot_violin(results, str(output), prefix,
                          limits, "TT_V1_T1", style.DEFAULT_DOT_BINS, fmt)
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        ax.errorbar(ranks, [results[c]["mean"] for c in ordered],
                    yerr=[results[c]["std"] for c in ordered],
                    fmt="o", markersize=3.5, capsize=2,
                    color=style.BASE_COLOR, linewidth=0.9)
        index = ordered.index("TT_V1_T1")
        ax.scatter(ranks[index], results["TT_V1_T1"]["mean"], s=105,
                   marker="*", color=style.HIGHLIGHT_COLOR,
                   edgecolors=style.HIGHLIGHT_EDGE_COLOR, linewidths=0.7,
                   zorder=5, label="TT_V1_T1 corner")
        ax.axhline(means.mean(), color=style.SUMMARY_COLOR,
                   linestyle=(0, (5, 3)), linewidth=1,
                   label=f"Mean: {means.mean():.2f}%")
        ax.set(xlabel="Corner rank (highest to lowest accuracy)",
               ylabel="Accuracy (%)", title="Corner Accuracy: Mean ± Trial SD",
               xlim=(0, 46))
        ax.grid(axis="y", alpha=0.22)
        ax.legend(framealpha=1, fontsize=9)
        style.save_figure(fig, str(output),
                          f"{prefix}_ranked_errorbars", fmt)
    print(f"45 corners × 10 trials; mean accuracy: {means.mean():.4f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--prefix", default="C100_PCN_cascode_1022_digital")
    args = parser.parse_args()
    generate_plots(args.result_dir, args.prefix)

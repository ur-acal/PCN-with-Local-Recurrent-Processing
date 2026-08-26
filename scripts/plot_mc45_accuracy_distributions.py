#!/usr/bin/env python3
"""Plot complementary views of two MC45 corner-accuracy distributions."""

import argparse
import csv
import glob
import os

import matplotlib.pyplot as plt
import numpy as np


PROCESS_ORDER = ("TT", "FF", "SS", "FS", "SF")
PROCESS_COLORS = {
    "TT": "#7A5195",
    "FF": "#2F6B9A",
    "SS": "#D95F02",
    "FS": "#1B9E77",
    "SF": "#E6AB02",
}
BASE_COLOR = "#3569A8"
PATCH_COLOR = "#D66B35"
BASE_LABEL = "No measured pooling\n(train and inference)"
PATCH_LABEL = "Patched-Gaussian measured pooling\n(no pooling-aware training)"
IDEAL_POOLING_LABEL = "Ideal pooling"
NONIDEAL_POOLING_LABEL = "Gaussian-based non-ideal pooling"


def load_corner_results(root):
    rows = []
    for path in sorted(glob.glob(os.path.join(root, "shard_*", "corner_trials.csv"))):
        with open(path, newline="") as handle:
            rows.extend(csv.DictReader(handle))

    keys = [(row["corner"], int(row["trial_index"])) for row in rows]
    if len(rows) != 225 or len(set(keys)) != 225:
        raise ValueError(
            "{} must contain 225 unique corner/trial results; found {}/{}."
            .format(root, len(rows), len(set(keys))))

    grouped = {}
    for row in rows:
        grouped.setdefault(row["corner"], []).append(
            (int(row["trial_index"]), float(row["accuracy"])))

    if len(grouped) != 45 or any(len(values) != 5 for values in grouped.values()):
        raise ValueError("{} must contain five trials for each of 45 corners.".format(root))

    output = {}
    for corner, values in grouped.items():
        trials = np.array([value for _, value in sorted(values)], dtype=float)
        output[corner] = {
            "trials": trials,
            "mean": float(trials.mean()),
            "std": float(trials.std()),
            "process": corner.split("_", 1)[0],
        }
    return output


def save_figure(fig, output_dir, name):
    path = os.path.join(output_dir, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(path)


def normal_pdf(x, mean, std):
    return np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))


def draw_mean_std(ax, position, values):
    mean = values.mean()
    std = values.std()
    line_length = 2.0 * std
    data_min = values.min()
    data_max = values.max()
    if line_length > data_max - data_min:
        raise ValueError("The 2-sigma marker cannot fit inside the violin range.")
    line_lower = min(max(mean - std, data_min), data_max - line_length)
    line_upper = line_lower + line_length
    ax.vlines(position, line_lower, line_upper, color="#222222",
              linewidth=3.0, alpha=0.85)
    ax.hlines(mean, position - 0.15, position + 0.15, color="#222222",
              linewidth=1.7)


def violin_jitter(body, position, values, rng):
    vertices = body.get_paths()[0].vertices
    y_coordinates = np.unique(vertices[:, 1])
    half_widths = np.array([
        np.max(np.abs(vertices[vertices[:, 1] == y, 0] - position))
        for y in y_coordinates
    ])
    widths_at_values = np.interp(values, y_coordinates, half_widths)
    return rng.uniform(-1.0, 1.0, size=len(values)) * widths_at_values * 0.8


def plot_histogram(base, patch, output_dir):
    base_means = np.array([entry["mean"] for entry in base.values()])
    patch_means = np.array([entry["mean"] for entry in patch.values()])
    lower = np.floor(min(base_means.min(), patch_means.min()))
    upper = np.ceil(max(base_means.max(), patch_means.max()))
    bins = np.linspace(lower, upper, 14)
    x = np.linspace(lower, upper, 500)

    configurations = (
        (base_means, BASE_COLOR, BASE_LABEL.replace("\n", " "),
         "corner_mean_histogram_gaussian_fit_no_measured_pooling.pdf"),
        (patch_means, PATCH_COLOR, PATCH_LABEL.replace("\n", " "),
         "corner_mean_histogram_gaussian_fit_patched_gaussian.pdf"),
    )
    for means, color, label, filename in configurations:
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        mean, std = means.mean(), means.std()
        ax.hist(means, bins=bins, density=True, alpha=0.28, color=color,
                edgecolor=color, linewidth=1.0)
        ax.plot(x, normal_pdf(x, mean, std), color=color, linewidth=2.2,
                label="Gaussian fit: μ={:.2f}, σ={:.2f}".format(mean, std))
        ax.plot(means, np.full_like(means, -0.002), "|", color=color,
                markersize=8, markeredgewidth=1.0)
        ax.set_xlim(lower, upper)
        ax.set_title("Corner Accuracy Distribution")
        ax.set_xlabel("Corner mean accuracy (%)")
        ax.set_ylabel("Probability density")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.22)
        save_figure(fig, output_dir, filename)

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    combined_labels = (IDEAL_POOLING_LABEL, NONIDEAL_POOLING_LABEL)
    for (means, color, _, _), label in zip(configurations, combined_labels):
        mean, std = means.mean(), means.std()
        ax.hist(means, bins=bins, density=True, alpha=0.28, color=color,
                edgecolor=color, linewidth=1.0)
        ax.plot(x, normal_pdf(x, mean, std), color=color, linewidth=2.2,
                label="{}: μ={:.2f}, σ={:.2f}".format(label, mean, std))
        ax.plot(means, np.full_like(means, -0.002), "|", color=color,
                markersize=8, markeredgewidth=1.0)
    ax.set_xlim(lower, upper)
    ax.set_title("Corner Accuracy Distribution")
    ax.set_xlabel("Corner mean accuracy (%)")
    ax.set_ylabel("Probability density")
    ax.legend(fontsize=8.5)
    ax.grid(axis="y", alpha=0.22)
    save_figure(fig, output_dir, "corner_mean_histogram_gaussian_fit.pdf")


def plot_ecdf(base, patch, output_dir):
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for data, color, label in (
            (base, BASE_COLOR, BASE_LABEL.replace("\n", " ")),
            (patch, PATCH_COLOR, PATCH_LABEL.replace("\n", " "))):
        values = np.sort([entry["mean"] for entry in data.values()])
        fraction = np.arange(1, len(values) + 1) / len(values)
        ax.step(values, fraction, where="post", linewidth=2.2,
                color=color, label=label)
        ax.scatter(values, fraction, s=13, color=color, alpha=0.65)

    for threshold in (55, 60):
        ax.axvline(threshold, color="#666666", linestyle="--", linewidth=0.8,
                   alpha=0.55)
    ax.set_title("Low-accuracy tail across PVT corners")
    ax.set_xlabel("Corner mean accuracy threshold (%)")
    ax.set_ylabel("Fraction of corners at or below threshold")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8.5, loc="upper left")
    ax.grid(alpha=0.22)
    save_figure(fig, output_dir, "corner_mean_ecdf.pdf")


def plot_violin(base, patch, output_dir):
    base_values = np.array([entry["mean"] for entry in base.values()])
    patch_values = np.array([entry["mean"] for entry in patch.values()])
    lower = min(base_values.min(), patch_values.min()) - 1.0
    upper = max(base_values.max(), patch_values.max()) + 1.0
    rng = np.random.default_rng(20260806)

    configurations = (
        (base_values, BASE_COLOR, BASE_LABEL,
         "corner_mean_violin_strip_no_measured_pooling.pdf"),
        (patch_values, PATCH_COLOR, PATCH_LABEL,
         "corner_mean_violin_strip_patched_gaussian.pdf"),
    )
    for values, color, label, filename in configurations:
        fig, ax = plt.subplots(figsize=(4.8, 5.0))
        parts = ax.violinplot([values], positions=(1,), widths=0.72,
                              showmeans=False, showmedians=False,
                              showextrema=False)
        body = parts["bodies"][0]
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.25)

        jitter = violin_jitter(body, 1, values, rng)
        ax.scatter(1 + jitter, values, s=25, color=color, alpha=0.72,
                   edgecolors="white", linewidths=0.35)
        draw_mean_std(ax, 1, values)

        ax.set_xlim(0.5, 1.5)
        ax.set_ylim(lower, upper)
        ax.set_xticks(())
        ax.set_ylabel("Corner mean accuracy (%)")
        ax.set_title("Corner Accuracy Distribution")
        ax.grid(axis="y", alpha=0.22)
        save_figure(fig, output_dir, filename)

    fig, ax = plt.subplots(figsize=(7.1, 5.0))
    for position, (values, color, _, _) in enumerate(configurations, start=1):
        parts = ax.violinplot([values], positions=(position,), widths=0.72,
                              showmeans=False, showmedians=False,
                              showextrema=False)
        body = parts["bodies"][0]
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.25)
        jitter = violin_jitter(body, position, values, rng)
        ax.scatter(position + jitter, values, s=25, color=color, alpha=0.72,
                   edgecolors="white", linewidths=0.35)
        draw_mean_std(ax, position, values)

    ax.set_ylim(lower, upper)
    ax.set_xticks((1, 2), (IDEAL_POOLING_LABEL, NONIDEAL_POOLING_LABEL))
    ax.set_ylabel("Corner mean accuracy (%)")
    ax.set_title("Corner Accuracy Distribution")
    ax.grid(axis="y", alpha=0.22)
    save_figure(fig, output_dir, "corner_mean_violin_strip.pdf")


def plot_paired(base, patch, output_dir):
    if set(base) != set(patch):
        raise ValueError("The paired comparison requires identical corner IDs.")
    corners = sorted(base)
    x = np.array([base[corner]["mean"] for corner in corners])
    y = np.array([patch[corner]["mean"] for corner in corners])
    lower = np.floor(min(x.min(), y.min())) - 0.5
    upper = np.ceil(max(x.max(), y.max())) + 0.5

    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    for process in PROCESS_ORDER:
        indices = [index for index, corner in enumerate(corners)
                   if base[corner]["process"] == process]
        ax.scatter(x[indices], y[indices], s=43, alpha=0.82,
                   color=PROCESS_COLORS[process], label=process,
                   edgecolors="white", linewidths=0.45)
    ax.plot((lower, upper), (lower, upper), color="#333333", linestyle="--",
            linewidth=1.0, label="No change")

    largest = np.argsort(np.abs(y - x))[-6:]
    for index in largest:
        ax.annotate(corners[index], (x[index], y[index]), xytext=(4, 4),
                    textcoords="offset points", fontsize=7.5)

    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("No measured pooling: corner mean accuracy (%)")
    ax.set_ylabel("Patched-Gaussian measured pooling: corner mean accuracy (%)")
    ax.set_title("Paired comparison of the same 45 corners")
    ax.legend(title="Process", fontsize=8, title_fontsize=8.5)
    ax.grid(alpha=0.2)
    save_figure(fig, output_dir, "paired_corner_mean_scatter.pdf")


def plot_ranked_errorbars(base, patch, output_dir):
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 7.0), sharex=True, sharey=True)
    configurations = (
        (axes[0], base, BASE_COLOR, BASE_LABEL.replace("\n", " ")),
        (axes[1], patch, PATCH_COLOR, PATCH_LABEL.replace("\n", " ")),
    )
    for ax, data, color, label in configurations:
        ordered = sorted(data.items(), key=lambda item: item[1]["mean"])
        ranks = np.arange(1, len(ordered) + 1)
        means = np.array([entry["mean"] for _, entry in ordered])
        stds = np.array([entry["std"] for _, entry in ordered])
        ax.errorbar(ranks, means, yerr=stds, fmt="o", markersize=3.5,
                    color=color, ecolor=color, elinewidth=0.8, capsize=1.5,
                    alpha=0.78)
        ax.plot(ranks, means, color=color, linewidth=0.8, alpha=0.55)
        for rank, (corner, entry) in zip(ranks[:5], ordered[:5]):
            ax.annotate(corner, (rank, entry["mean"]), xytext=(3, -3),
                        textcoords="offset points", fontsize=7.2,
                        rotation=35, ha="left", va="top")
        ax.set_title(label, fontsize=10.5)
        ax.set_ylabel("Accuracy (%)")
        ax.grid(axis="y", alpha=0.22)
    axes[1].set_xlabel("Corner rank, lowest to highest mean accuracy")
    fig.suptitle("Per-corner mean accuracy and within-corner trial standard deviation",
                 fontsize=12)
    fig.tight_layout()
    save_figure(fig, output_dir, "ranked_corner_mean_std.pdf")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_dir", required=True)
    parser.add_argument("--patched_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    plt.rcParams.update({
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    baseline = load_corner_results(args.baseline_dir)
    patched = load_corner_results(args.patched_dir)
    plot_histogram(baseline, patched, args.output_dir)
    plot_violin(baseline, patched, args.output_dir)


if __name__ == "__main__":
    main()

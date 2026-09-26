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
OUTPUT_FORMAT = "pdf"
ACCURACY_MIN = None
ACCURACY_MAX = None
VIOLIN_DOT_BINS = 6
SUMMARY_COLOR = "#4A4A4A"
MIN_DOTS_PER_BIN = 2
BIN_WIDTH_REGULARIZATION = 0.02
BIN_SEPARATION_REGULARIZATION = 0.05
ADJACENT_BIN_EDGE_DOTS = 3
MIN_CROSS_BIN_DISTANCE = 0.025


def load_corner_results(root):
    rows = []
    for path in sorted(glob.glob(os.path.join(root, "shard_*", "corner_trials.csv"))):
        with open(path, newline="") as handle:
            rows.extend(csv.DictReader(handle))

    keys = [(row["corner"], int(row["trial_index"])) for row in rows]
    if not rows or len(set(keys)) != len(rows):
        raise ValueError(
            "{} must contain unique corner/trial results; found {}/{}."
            .format(root, len(rows), len(set(keys))))

    grouped = {}
    for row in rows:
        grouped.setdefault(row["corner"], []).append(
            (int(row["trial_index"]), float(row["accuracy"])))

    trial_counts = {len(values) for values in grouped.values()}
    if len(grouped) != 45 or len(trial_counts) != 1:
        raise ValueError(
            "{} must contain the same nonzero number of trials for each of "
            "45 corners.".format(root))

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
    name = os.path.splitext(name)[0] + "." + OUTPUT_FORMAT
    path = os.path.join(output_dir, name)
    fig.savefig(path, format=OUTPUT_FORMAT, bbox_inches="tight")
    plt.close(fig)
    print(path)


def accuracy_limits(lower, upper):
    if ACCURACY_MIN is not None:
        lower = ACCURACY_MIN
    if ACCURACY_MAX is not None:
        upper = ACCURACY_MAX
    return lower, upper


def normal_pdf(x, mean, std):
    return np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))


def draw_mean_std(ax, position, values):
    mean = values.mean()
    std = values.std()
    line_lower = mean - std
    line_upper = mean + std
    cap_half_width = 0.035
    ax.axhline(mean, color=SUMMARY_COLOR, linestyle=(0, (5, 3)),
               linewidth=1.0, zorder=3.5)
    ax.annotate("mean", xy=(0.98, mean), xycoords=("axes fraction", "data"),
                xytext=(0, 3), textcoords="offset points", ha="right",
                va="bottom", color="black", fontsize=plt.rcParams["font.size"],
                zorder=5)
    ax.vlines(position, line_lower, line_upper, color=SUMMARY_COLOR,
              linewidth=1.0, zorder=4)
    ax.hlines((line_lower, line_upper),
              position - cap_half_width, position + cap_half_width,
              color=SUMMARY_COLOR, linewidth=1.0, zorder=4)


def mark_mean_tick(ax, mean):
    lower, upper = ax.get_ylim()
    minimum_separation = 0.035 * (upper - lower)
    ticks = [tick for tick in ax.get_yticks()
             if lower <= tick <= upper and
             abs(tick - mean) >= minimum_separation]
    ticks.append(mean)
    ticks.sort()
    labels = ["{:.2f}".format(tick) if np.isclose(tick, mean)
              else "{:g}".format(tick) for tick in ticks]
    ax.set_yticks(ticks, labels)
    for tick, label in zip(ticks, ax.get_yticklabels()):
        if np.isclose(tick, mean):
            label.set_color(SUMMARY_COLOR)
            label.set_fontweight("semibold")


def _cumulative_violin_area(y_coordinates, half_widths):
    segment_areas = (np.diff(y_coordinates) *
                     (half_widths[:-1] + half_widths[1:]) * 0.5)
    return np.concatenate(([0.0], np.cumsum(segment_areas)))


def optimize_violin_bins(values, y_coordinates, half_widths, num_bins,
                         y_axis_span):
    """Partition sorted values so dot counts match integrated violin area.

    For fixed K, candidate boundaries are the midpoints between adjacent
    accuracies. Dynamic programming minimizes

        sum_j (observed_fraction_j - violin_mass_j)^2 / violin_mass_j

    plus a small bin-width regularizer and a cross-bin separation penalty.
    Every bin contains at least MIN_DOTS_PER_BIN points.
    """
    num_values = len(values)
    if num_bins < 1:
        raise ValueError("The number of violin dot bins must be positive.")
    if num_values < num_bins * MIN_DOTS_PER_BIN:
        raise ValueError(
            "{} values cannot fill {} bins with at least {} dots each."
            .format(num_values, num_bins, MIN_DOTS_PER_BIN))

    sorted_values = np.sort(values)
    sorted_indices = np.argsort(values, kind="stable")
    widths_at_values = np.interp(values, y_coordinates, half_widths)
    edges = np.empty(num_values + 1, dtype=float)
    edges[0] = y_coordinates[0]
    edges[-1] = y_coordinates[-1]
    edges[1:-1] = 0.5 * (sorted_values[:-1] + sorted_values[1:])

    cumulative_area = _cumulative_violin_area(y_coordinates, half_widths)
    total_area = cumulative_area[-1]
    total_range = edges[-1] - edges[0]

    def area_at(value):
        return np.interp(value, y_coordinates, cumulative_area)

    def segment_cost(left, right):
        observed_fraction = (right - left) / num_values
        violin_mass = ((area_at(edges[right]) - area_at(edges[left])) /
                       total_area)
        mass_cost = ((observed_fraction - violin_mass) ** 2 /
                     max(violin_mass, np.finfo(float).eps))
        width_fraction = (edges[right] - edges[left]) / total_range
        width_cost = BIN_WIDTH_REGULARIZATION * (
            width_fraction - 1.0 / num_bins) ** 2
        return mass_cost + width_cost

    layouts = {}

    def segment_layout(left, right):
        key = (left, right)
        if key not in layouts:
            members = sorted_indices[left:right]
            usable_half_width = 0.8 * widths_at_values[members].min()
            slots = np.linspace(-usable_half_width, usable_half_width,
                                len(members))
            layouts[key] = dict(zip(members, slots))
        return layouts[key]

    def separation_cost(previous_left, boundary, right):
        lower_layout = segment_layout(previous_left, boundary)
        upper_layout = segment_layout(boundary, right)
        lower_edge = sorted_indices[
            max(previous_left, boundary - ADJACENT_BIN_EDGE_DOTS):boundary]
        upper_edge = sorted_indices[
            boundary:min(right, boundary + ADJACENT_BIN_EDGE_DOTS)]
        penalty = 0.0
        for lower_index in lower_edge:
            for upper_index in upper_edge:
                dx = upper_layout[upper_index] - lower_layout[lower_index]
                dy = ((values[upper_index] - values[lower_index]) /
                      y_axis_span)
                distance = np.hypot(dx, dy)
                shortfall = max(
                    0.0, 1.0 - distance / MIN_CROSS_BIN_DISTANCE)
                penalty += shortfall ** 2
        return BIN_SEPARATION_REGULARIZATION * penalty

    costs = np.full(
        (num_bins + 1, num_values + 1, num_values + 1), np.inf)
    parents = np.full(
        (num_bins + 1, num_values + 1, num_values + 1), -1, dtype=int)

    last_first_bin = num_values - (num_bins - 1) * MIN_DOTS_PER_BIN
    for right in range(MIN_DOTS_PER_BIN, last_first_bin + 1):
        costs[1, 0, right] = segment_cost(0, right)

    for bins_used in range(2, num_bins + 1):
        first_left = (bins_used - 1) * MIN_DOTS_PER_BIN
        last_left = num_values - (
            num_bins - bins_used + 1) * MIN_DOTS_PER_BIN
        for left in range(first_left, last_left + 1):
            if np.isclose(sorted_values[left - 1], sorted_values[left]):
                continue
            first_right = left + MIN_DOTS_PER_BIN
            last_right = num_values - (
                num_bins - bins_used) * MIN_DOTS_PER_BIN
            for right in range(first_right, last_right + 1):
                for previous_left in range(0, left):
                    previous_cost = costs[
                        bins_used - 1, previous_left, left]
                    if not np.isfinite(previous_cost):
                        continue
                    candidate = (previous_cost + segment_cost(left, right) +
                                 separation_cost(
                                     previous_left, left, right))
                    if candidate < costs[bins_used, left, right]:
                        costs[bins_used, left, right] = candidate
                        parents[bins_used, left, right] = previous_left

    final_left = int(np.argmin(costs[num_bins, :, num_values]))
    objective = costs[num_bins, final_left, num_values]
    if not np.isfinite(objective):
        raise RuntimeError("Could not construct optimized violin bins.")

    cuts = [num_values]
    right = num_values
    left = final_left
    for bins_used in range(num_bins, 0, -1):
        cuts.append(left)
        if bins_used > 1:
            previous_left = parents[bins_used, left, right]
            right, left = left, previous_left
    cuts.reverse()
    boundaries = edges[np.asarray(cuts)]
    return boundaries, float(objective)


def deterministic_violin_offsets(body, position, values, num_bins,
                                 y_axis_span):
    vertices = body.get_paths()[0].vertices
    y_coordinates = np.unique(vertices[:, 1])
    half_widths = np.array([
        np.max(np.abs(vertices[vertices[:, 1] == y, 0] - position))
        for y in y_coordinates
    ])
    widths_at_values = np.interp(values, y_coordinates, half_widths)
    bin_edges, objective = optimize_violin_bins(
        values, y_coordinates, half_widths, num_bins, y_axis_span)
    bin_indices = np.digitize(values, bin_edges[1:-1])
    offsets = np.zeros(len(values), dtype=float)

    for bin_index in range(num_bins):
        members = np.flatnonzero(bin_indices == bin_index)
        if len(members) <= 1:
            continue
        members = members[np.argsort(values[members], kind="stable")]
        usable_half_width = 0.8 * widths_at_values[members].min()
        offsets[members] = np.linspace(
            -usable_half_width, usable_half_width, len(members))
    return offsets, bin_edges, objective


def plot_histogram(base, patch, output_dir):
    base_means = np.array([entry["mean"] for entry in base.values()])
    patch_means = np.array([entry["mean"] for entry in patch.values()])
    lower = np.floor(min(base_means.min(), patch_means.min()))
    upper = np.ceil(max(base_means.max(), patch_means.max()))
    bins = np.linspace(lower, upper, 14)
    x = np.linspace(lower, upper, 500)
    view_lower, view_upper = accuracy_limits(lower, upper)

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
        ax.set_xlim(view_lower, view_upper)
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
    ax.set_xlim(view_lower, view_upper)
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


def plot_violin(base, patch, output_dir, num_dot_bins):
    base_values = np.array([entry["mean"] for entry in base.values()])
    patch_values = np.array([entry["mean"] for entry in patch.values()])
    lower = min(base_values.min(), patch_values.min()) - 1.0
    upper = max(base_values.max(), patch_values.max()) + 1.0
    view_lower, view_upper = accuracy_limits(lower, upper)
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

        jitter, bin_edges, objective = deterministic_violin_offsets(
            body, 1, values, num_dot_bins, view_upper - view_lower)
        print("Optimized {}-bin violin layout: boundaries={}, objective={:.8g}"
              .format(num_dot_bins,
                      ",".join("{:.6g}".format(value)
                               for value in bin_edges),
                      objective))
        ax.scatter(1 + jitter, values, s=25, color=color, alpha=0.72,
                   edgecolors="white", linewidths=0.35)
        draw_mean_std(ax, 1, values)

        ax.set_xlim(0.5, 1.5)
        ax.set_ylim(view_lower, view_upper)
        ax.set_xticks(())
        ax.set_ylabel("Corner mean accuracy (%)")
        ax.set_title("Corner Accuracy Distribution")
        ax.grid(axis="y", alpha=0.22)
        mark_mean_tick(ax, values.mean())
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
        jitter, _, _ = deterministic_violin_offsets(
            body, position, values, num_dot_bins,
            view_upper - view_lower)
        ax.scatter(position + jitter, values, s=25, color=color, alpha=0.72,
                   edgecolors="white", linewidths=0.35)
        draw_mean_std(ax, position, values)

    ax.set_ylim(view_lower, view_upper)
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
    global ACCURACY_MAX, ACCURACY_MIN, OUTPUT_FORMAT

    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_dir", required=True)
    parser.add_argument("--patched_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--output_format", choices=("pdf", "svg"),
                        default="pdf")
    parser.add_argument("--plot_kind",
                        choices=("all", "histogram", "violin"),
                        default="all")
    parser.add_argument("--accuracy_min", type=float)
    parser.add_argument("--accuracy_max", type=float)
    parser.add_argument("--violin_dot_bins", type=int,
                        default=VIOLIN_DOT_BINS)
    args = parser.parse_args()

    if (args.accuracy_min is not None and
            args.accuracy_max is not None and
            args.accuracy_min >= args.accuracy_max):
        parser.error("--accuracy_min must be less than --accuracy_max")

    OUTPUT_FORMAT = args.output_format
    ACCURACY_MIN = args.accuracy_min
    ACCURACY_MAX = args.accuracy_max

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
    if args.plot_kind in ("all", "histogram"):
        plot_histogram(baseline, patched, args.output_dir)
    if args.plot_kind in ("all", "violin"):
        plot_violin(baseline, patched, args.output_dir,
                    args.violin_dot_bins)


if __name__ == "__main__":
    main()

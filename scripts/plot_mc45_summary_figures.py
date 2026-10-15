#!/usr/bin/env python3
"""Plot deterministic MC45 violins and process/voltage/temperature panels."""

import argparse
import csv
import math
import os
import re

import matplotlib.pyplot as plt
import numpy as np

if __package__:
    from .generate_mc45_summary import generate_summary
else:
    from generate_mc45_summary import generate_summary


PROCESS_ORDER = ("TT", "FF", "SS", "FS", "SF")
VOLTAGE_ORDER = ("V0", "V1", "V2")
TEMPERATURE_ORDER = ("T0", "T1", "T2")
TEMPERATURE_LABELS = (r"$-20\,^\circ\mathrm{C}$",
                      r"$25\,^\circ\mathrm{C}$",
                      r"$85\,^\circ\mathrm{C}$")
VOLTAGE_LABELS = (r"$0.9\,V_{DD}$", r"$V_{DD}$", r"$1.1\,V_{DD}$")
TEMPERATURE_COLORS = ("#0072BD", "#00B050", "#FF0000")
BASE_COLOR = "#3569A8"
SUMMARY_COLOR = "#4A4A4A"
HIGHLIGHT_COLOR = "#D4AF37"
HIGHLIGHT_EDGE_COLOR = "#7A5B00"
REFERENCE_RED = "#CC0000"
REFERENCE_GREEN = "#009E49"
# The pre-optimization deterministic figure reused the historical histogram's
# 13 equal-height bins. Keep that exact partition so every legacy dot retains
# its horizontal position.
DEFAULT_DOT_BINS = 13


def ensure_summary(path):
    """Generate the aggregate summary from sibling shards when absent."""
    if os.path.isfile(path):
        return
    result_root = os.path.dirname(os.path.abspath(path))
    generated_path = generate_summary(result_root, output_path=path)
    print("Generated missing MC45 summary: {}".format(generated_path))


def load_summary(path):
    """Read the 45 corner means/stds from a generated summary Markdown file."""
    row_pattern = re.compile(
        r"^\|\s*((?:TT|FF|SS|FS|SF)_V[012]_T[012])\s*\|"
        r"\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*\|"
        r"\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*\|\s*$")
    results = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            match = row_pattern.match(line.rstrip())
            if match:
                corner, mean, std = match.groups()
                results[corner] = {"mean": float(mean), "std": float(std)}

    expected = {
        "{}_{}_{}".format(process, voltage, temperature)
        for process in PROCESS_ORDER
        for voltage in VOLTAGE_ORDER
        for temperature in TEMPERATURE_ORDER
    }
    missing = sorted(expected - set(results))
    extra = sorted(set(results) - expected)
    if missing or extra:
        raise ValueError(
            "{} does not contain exactly the expected 45 corners; missing={}, "
            "extra={}".format(path, missing, extra))
    return results


def load_corner_trials(summary_path):
    """Load the individual MC trials stored beside an aggregate summary."""
    root = os.path.dirname(os.path.abspath(summary_path))
    paths = []
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name, "corner_trials.csv")
        if name.startswith("shard_") and os.path.isfile(path):
            paths.append(path)
    if not paths:
        raise ValueError("No shard_*/corner_trials.csv files under {}".format(
            root))

    trials = {}
    for path in paths:
        with open(path, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                corner = row["corner"]
                trials.setdefault(corner, []).append(
                    (int(row["trial_index"]), float(row["accuracy"])))

    for corner, rows in trials.items():
        rows.sort(key=lambda item: item[0])
        indices = [item[0] for item in rows]
        if indices != list(range(len(rows))):
            raise ValueError(
                "{} has non-contiguous/duplicate trial indices: {}".format(
                    corner, indices))
    return trials


def _violin_half_widths(body, position, values):
    vertices = body.get_paths()[0].vertices
    y_coordinates = np.unique(vertices[:, 1])
    half_widths = np.array([
        np.max(np.abs(vertices[vertices[:, 1] == y, 0] - position))
        for y in y_coordinates
    ])
    return np.interp(values, y_coordinates, half_widths)


def deterministic_fixed_bin_offsets(body, position, values, num_bins):
    """Reproduce the pre-optimization equal-height-bin dot layout.

    Values remain at their exact vertical coordinates. Within each fixed
    vertical bin, points are sorted by accuracy and placed at evenly spaced
    horizontal positions bounded by the narrowest violin width in that bin.
    """
    if num_bins < 1:
        raise ValueError("num_bins must be positive")
    widths_at_values = _violin_half_widths(body, position, values)
    bin_edges = np.linspace(values.min(), values.max(), num_bins + 1)
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
    return offsets


def draw_mean_std(ax, position, values):
    mean = values.mean()
    std = values.std()
    lower = mean - std
    upper = mean + std
    cap_half_width = 0.035
    ax.axhline(mean, color=SUMMARY_COLOR, linestyle=(0, (5, 3)),
               linewidth=1.0, zorder=3.5)
    ax.annotate("mean", xy=(0.98, mean), xycoords=("axes fraction", "data"),
                xytext=(0, 3), textcoords="offset points", ha="right",
                va="bottom", color="black", fontsize=plt.rcParams["font.size"],
                zorder=5)
    ax.vlines(position, lower, upper, color=SUMMARY_COLOR,
              linewidth=1.0, zorder=4)
    ax.hlines((lower, upper), position - cap_half_width,
              position + cap_half_width, color=SUMMARY_COLOR,
              linewidth=1.0, zorder=4)


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


def save_figure(fig, output_dir, filename, output_format):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename + "." + output_format)
    fig.savefig(path, format=output_format, bbox_inches="tight")
    plt.close(fig)
    print(path)


def plot_violin(results, output_dir, prefix, limits, highlight_corner,
                num_bins, output_format):
    corners = list(results)
    values = np.array([results[corner]["mean"] for corner in corners])
    highlight_index = corners.index(highlight_corner)

    fig, ax = plt.subplots(figsize=(4.8, 5.0))
    parts = ax.violinplot([values], positions=(1,), widths=0.72,
                          showmeans=False, showmedians=False,
                          showextrema=False)
    body = parts["bodies"][0]
    body.set_facecolor(BASE_COLOR)
    body.set_edgecolor(BASE_COLOR)
    body.set_alpha(0.25)

    offsets = deterministic_fixed_bin_offsets(body, 1, values, num_bins)
    ordinary = np.arange(len(values)) != highlight_index
    ax.scatter(1 + offsets[ordinary], values[ordinary], s=25,
               color=BASE_COLOR, alpha=0.72, edgecolors="white",
               linewidths=0.35, zorder=3)
    ax.scatter(1 + offsets[highlight_index], values[highlight_index], s=105,
               marker="*", color=HIGHLIGHT_COLOR,
               edgecolors=HIGHLIGHT_EDGE_COLOR, linewidths=0.7, zorder=6,
               label="{} corner".format(highlight_corner))
    draw_mean_std(ax, 1, values)

    ax.set_xlim(0.5, 1.5)
    ax.set_ylim(*limits)
    ax.set_xticks(())
    ax.set_ylabel("Corner mean accuracy (%)")
    ax.set_title("Corner Accuracy Distribution")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(loc="upper right", frameon=True, framealpha=1.0,
              edgecolor="#A0A0A0", facecolor="white", fontsize=9,
              handletextpad=0.4)
    mark_mean_tick(ax, values.mean())

    range_tag = "range{:g}_{:g}".format(*limits).replace(".", "p")
    highlight_tag = re.sub(r"[^a-z0-9]+", "_", highlight_corner.lower()).strip("_")
    filename = "{}_corner_mean_violin_deterministic_dagger_{}_{}_star".format(
        prefix, range_tag, highlight_tag)
    save_figure(fig, output_dir, filename, output_format)


def _create_pvt_axes(layout, height_scale=1.0):
    if layout == "horizontal":
        fig, axes = plt.subplots(
            1, 5, figsize=(15.5, 4.8 * height_scale), sharey=True)
        return fig, list(axes)

    # Both compact layouts use a six-column GridSpec. Three panels occupy
    # two columns each; the shorter two-panel row occupies columns 1:3 and
    # 3:5, centering it without a visible empty subplot.
    fig = plt.figure(figsize=(10.0, 7.0 * height_scale))
    grid = fig.add_gridspec(2, 6, hspace=0.38, wspace=0.50)
    three_panel_row = [fig.add_subplot(grid[0, 0:2]),
                       fig.add_subplot(grid[0, 2:4]),
                       fig.add_subplot(grid[0, 4:6])]
    two_panel_row = [fig.add_subplot(grid[1, 1:3]),
                     fig.add_subplot(grid[1, 3:5])]
    if layout == "3x2":
        return fig, three_panel_row + two_panel_row

    # 2x3 means two panels on the first row and three on the second.
    fig.clear()
    grid = fig.add_gridspec(2, 6, hspace=0.38, wspace=0.50)
    two_panel_row = [fig.add_subplot(grid[0, 1:3]),
                     fig.add_subplot(grid[0, 3:5])]
    three_panel_row = [fig.add_subplot(grid[1, 0:2]),
                       fig.add_subplot(grid[1, 2:4]),
                       fig.add_subplot(grid[1, 4:6])]
    return fig, two_panel_row + three_panel_row


def plot_pvt_errorbars(results, trials, output_dir, prefix, layout,
                       output_format, red_reference, green_reference,
                       fixed_y_limits=None, height_scale=1.0):
    missing_trials = sorted(set(results) - set(trials))
    if missing_trials:
        raise ValueError("Missing per-trial data for {}".format(
            missing_trials))
    trial_values = np.array([
        accuracy for corner in results for _, accuracy in trials[corner]
    ])
    reference_values = np.array([red_reference, green_reference],
                                dtype=float)
    if fixed_y_limits is None:
        y_limits = (
            math.floor(min(trial_values.min(), reference_values.min())),
            math.ceil(max(trial_values.max(), reference_values.max())))
    else:
        y_limits = tuple(fixed_y_limits)
    fig, axes = _create_pvt_axes(layout, height_scale)
    voltage_positions = np.arange(len(VOLTAGE_ORDER), dtype=float)
    temperature_offsets = (-0.18, 0.0, 0.18)

    for process, ax in zip(PROCESS_ORDER, axes):
        for temperature, temperature_label, offset, color in zip(
                TEMPERATURE_ORDER, TEMPERATURE_LABELS, temperature_offsets,
                TEMPERATURE_COLORS):
            corner_means = []
            corner_stds = []
            for voltage_index, voltage in enumerate(VOLTAGE_ORDER):
                corner = "{}_{}_{}".format(process, voltage, temperature)
                entry = results[corner]
                corner_means.append(entry["mean"])
                corner_stds.append(entry["std"])
                values = np.array([accuracy for _, accuracy in trials[corner]])
                ax.scatter(
                    np.full(len(values),
                            voltage_positions[voltage_index] + offset),
                    values, s=15, marker="o", color=color, alpha=0.82,
                    edgecolors="none", zorder=3)
            ax.errorbar(voltage_positions + offset, corner_means,
                        yerr=corner_stds, fmt="_", markersize=11,
                        markeredgewidth=2.0, color=color, ecolor=color,
                        elinewidth=1.6, capsize=4.0, capthick=1.6,
                        label=temperature_label if process == PROCESS_ORDER[0]
                        else None, zorder=4)

        ax.set_xticks(voltage_positions, VOLTAGE_LABELS)
        ax.set_xlabel(process, labelpad=12, fontsize=12)
        ax.set_xlim(-0.45, 2.45)
        ax.set_ylim(*y_limits)
        ax.grid(axis="y", alpha=0.22)
        ax.axhline(red_reference, color=REFERENCE_RED,
                   linestyle=(0, (5, 4)), linewidth=1.5, zorder=1)
        ax.axhline(green_reference, color=REFERENCE_GREEN,
                   linestyle=(0, (5, 4)), linewidth=1.5, zorder=1)

    axes[0].set_ylabel("Accuracy (%)")
    for ax in axes[1:]:
        ax.tick_params(axis="y", which="both", left=False,
                       labelleft=False)

    label_offset = 0.018 * (y_limits[1] - y_limits[0])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.97 if layout == "horizontal" else 0.97))
    if layout == "horizontal":
        fig.subplots_adjust(top=0.86, bottom=0.19, left=0.055,
                            right=0.99, wspace=0.0)
    else:
        fig.subplots_adjust(top=0.90, bottom=0.09, left=0.075,
                            right=0.98)

    # Figure-level labels remain visible while spanning several subplot
    # patches; axis-level text would be hidden by the neighboring axes.
    axis_box = axes[0].get_position()
    def figure_y(data_y):
        fraction = (data_y - y_limits[0]) / (y_limits[1] - y_limits[0])
        return axis_box.y0 + fraction * axis_box.height

    fig.text(
        0.5, figure_y(red_reference + label_offset),
        "Digital baseline: {:.2f}%".format(red_reference),
        color=REFERENCE_RED, ha="center", va="bottom", fontsize=9, zorder=20)
    fig.text(
        0.5, figure_y(green_reference + label_offset),
        "Hardware target model without hardware non-ideality: {:.2f}%".format(
            green_reference),
        color=REFERENCE_GREEN, ha="center", va="bottom", fontsize=9,
        zorder=20)

    filename = "{}_corner_mean_pvt_errorbars_{}".format(prefix, layout)
    save_figure(fig, output_dir, filename, output_format)


def plot_isolated_errorbar(results, trials, output_dir, prefix, corner,
                           output_format, y_limits=None, height_scale=1.0):
    """Render one corner's trials and mean/std error bar on a blank page."""
    if corner not in results or corner not in trials:
        raise ValueError("Unknown or missing corner data: {}".format(corner))
    values = np.array([accuracy for _, accuracy in trials[corner]])
    mean = results[corner]["mean"]
    std = results[corner]["std"]

    # Match the full horizontal plot's panel height and vertical margins so
    # identical accuracy differences have identical rendered distances.
    fig, ax = plt.subplots(
        figsize=(2.9, 4.8 * height_scale), facecolor="white")
    ax.set_facecolor("white")
    x = np.zeros(len(values))
    color = TEMPERATURE_COLORS[TEMPERATURE_ORDER.index(corner[-2:])]
    ax.scatter(x, values, s=15, marker="o", color=color, alpha=0.82,
               edgecolors="none", zorder=3)
    ax.errorbar([0.0], [mean], yerr=[std], fmt="_", markersize=11,
                markeredgewidth=2.0, color=color, ecolor=color,
                elinewidth=1.6, capsize=4.0, capthick=1.6, zorder=4)

    ax.set_xlim(-0.22, 0.22)
    if y_limits is None:
        half_span = max(np.max(np.abs(values - mean)), std) * 1.35
        y_limits = (mean - half_span, mean + half_span)
    ax.set_ylim(*y_limits)
    ax.axis("off")
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.19, top=0.86)

    os.makedirs(output_dir, exist_ok=True)
    filename = "{}_{}_errorbar_only.{}".format(
        prefix, corner.lower(), output_format)
    path = os.path.join(output_dir, filename)
    fig.savefig(path, format=output_format, facecolor="white")
    plt.close(fig)
    print(path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True,
                        help="MC45 summary.md containing all 45 corner rows")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--violin_range", nargs=2, type=float,
                        action="append", metavar=("MIN", "MAX"),
                        help="Repeat to generate multiple violin y-axis ranges")
    parser.add_argument("--pvt_layout",
                        choices=("horizontal", "1x5", "3x2", "3-2",
                                 "2x3", "2-3"),
                        default="horizontal")
    parser.add_argument("--highlight_corner", default="TT_V1_T1")
    parser.add_argument("--violin_dot_bins", type=int,
                        default=DEFAULT_DOT_BINS)
    parser.add_argument("--output_format", choices=("pdf", "svg", "png"),
                        default="pdf")
    parser.add_argument("--red_reference", type=float,
                        help="Red dashed reference accuracy")
    parser.add_argument("--green_reference", type=float,
                        help="Green dashed reference accuracy")
    parser.add_argument("--pvt_y_range", nargs=2, type=float,
                        metavar=("MIN", "MAX"),
                        help="Fixed shared y-range for the PVT plot")
    parser.add_argument("--pvt_height_scale", type=float, default=1.0,
                        help="Scale the PVT figure height without changing width")
    parser.add_argument("--isolated_corner",
                        help="Also render only this corner's trials/error bar")
    parser.add_argument("--isolated_y_range", nargs=2, type=float,
                        metavar=("MIN", "MAX"),
                        help="Use this full-plot y-range for the isolated bar")
    return parser.parse_args()


def main():
    args = parse_args()
    for lower, upper in args.violin_range or ():
        if lower >= upper:
            raise ValueError("Each violin range must have MIN < MAX")

    plt.rcParams.update({
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    ensure_summary(args.summary)
    results = load_summary(args.summary)
    trials = load_corner_trials(args.summary)
    if args.highlight_corner not in results:
        raise ValueError("Unknown highlight corner: {}".format(
            args.highlight_corner))

    for limits in args.violin_range or ():
        plot_violin(results, args.output_dir, args.prefix, limits,
                    args.highlight_corner, args.violin_dot_bins,
                    args.output_format)
    layout = {"1x5": "horizontal", "3-2": "3x2", "2-3": "2x3"}.get(
        args.pvt_layout, args.pvt_layout)
    if ((args.red_reference is None) !=
            (args.green_reference is None)):
        raise ValueError("Pass both --red_reference and --green_reference")
    if args.red_reference is not None:
        plot_pvt_errorbars(results, trials, args.output_dir, args.prefix,
                           layout, args.output_format, args.red_reference,
                           args.green_reference, args.pvt_y_range,
                           args.pvt_height_scale)
    if args.isolated_corner:
        plot_isolated_errorbar(results, trials, args.output_dir, args.prefix,
                               args.isolated_corner, args.output_format,
                               args.isolated_y_range,
                               args.pvt_height_scale)


if __name__ == "__main__":
    main()

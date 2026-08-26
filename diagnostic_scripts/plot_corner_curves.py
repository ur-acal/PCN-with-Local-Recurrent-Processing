#!/usr/bin/env python3
"""Plot measured nonlinear-R and ReLU Monte Carlo curves.

Copy-ready corner selections from the completed local reference runs:

CIFAR-100, results/coupler_monte_v2_cifar100_qf1
  Lowest five:  --corners "FS_V0_T0,FF_V2_T0,FF_V0_T0,FF_V0_T1,TT_V0_T0"
  Highest five: --corners "TT_V1_T2,SF_V2_T2,SF_V1_T2,FS_V1_T2,SF_V0_T2"

CIFAR-10, results/coupler_monte_v2_cifar10
  Lowest five:  --corners "FS_V0_T0,TT_V0_T0,FF_V0_T0,FF_V0_T1,SF_V0_T0"
  Highest five: --corners "SF_V2_T2,SF_V1_T2,FS_V1_T2,TT_V1_T2,SS_V1_T2"
"""

import argparse
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np

from diagnostic_config import MC45_ROOT, REPO_ROOT, V_DD, normalize_task


PROCESS_ORDER = ("TT", "FF", "SS", "FS", "SF")


def parse_corners(text):
    if not text:
        return None
    corners = []
    pattern = re.compile(r"^(TT|FF|SS|FS|SF)_V([0-2])_T([0-2])$", re.IGNORECASE)
    for value in text.split(","):
        corner = value.strip().upper()
        if not pattern.match(corner):
            raise ValueError("Invalid corner: {}".format(value))
        if corner not in corners:
            corners.append(corner)
    return corners


def corner_sort_key(corner):
    process, voltage, temperature = corner.split("_")
    return PROCESS_ORDER.index(process), int(voltage[1:]), int(temperature[1:])


def load_paired_curves(path):
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    curves = []
    for index in range(data.shape[1] // 2):
        x = data[:, 2 * index]
        y = data[:, 2 * index + 1]
        valid = np.isfinite(x) & np.isfinite(y)
        curves.append((x[valid], y[valid]))
    return curves


def mean_curve(curves):
    """Average one corner's Monte Carlo outputs on its shared input grid."""
    reference_x = curves[0][0]
    outputs = []
    for x, y in curves:
        if x.shape == reference_x.shape and np.allclose(x, reference_x):
            outputs.append(y)
        else:
            outputs.append(np.interp(reference_x, x, y))
    return reference_x, np.mean(np.stack(outputs, axis=0), axis=0)


def relu_path(corner):
    process, voltage, temperature = corner.lower().split("_")
    index = int(voltage[1:]) * 3 + int(temperature[1:])
    process_name = {"tt": "ttg", "ff": "ffg", "ss": "ssg",
                    "fs": "fsg", "sf": "sfg"}[process]
    return MC45_ROOT / "relu_monteCarlo" / "relu_{}{}.csv".format(process_name, index)


def nonlinear_r_path(corner):
    process, voltage, temperature = corner.lower().split("_")
    temperatures = (-20, 25, 85)
    return MC45_ROOT / "coupler_monte_v2" / "{}_{}_{}.csv".format(
        process, temperatures[int(temperature[1:])], int(voltage[1:]))


def header_metadata(path):
    with Path(path).open() as handle:
        header = handle.readline()
    vdd = re.search(r"(?:VDD|VDD_VALUE)=([-+0-9.eE]+)", header)
    temperature = re.search(r"temperature=([-+0-9.eE]+)", header)
    return (
        None if vdd is None else float(vdd.group(1)),
        None if temperature is None else float(temperature.group(1)))


def style_axes():
    plt.rcParams.update({
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def draw_curves(curves, xlabel, ylabel, title, output_path, x_scale=1.0,
                y_scale=1.0):
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    colors = plt.get_cmap("turbo")(np.linspace(0.02, 0.98, len(curves)))
    linewidth = 0.75 if len(curves) <= 100 else 0.25
    alpha = 0.82 if len(curves) <= 100 else 0.32
    for color, (x, y) in zip(colors, curves):
        ax.plot(x * x_scale, y * y_scale, color=color,
                linewidth=linewidth, alpha=alpha)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(output_path)


def plot_one_relu(corner, output_dir):
    path = relu_path(corner)
    curves = load_paired_curves(path)
    vdd, temperature = header_metadata(path)
    v_char = max(np.max(np.abs(curve[0])) for curve in curves)
    scale = V_DD / v_char
    title = (
        "Measured ReLU {}: {} piecewise-linear MC curves\n"
        "characterized VDD={} V, T={} °C; both axes scaled by {:.0f} to v_dd={} V"
        .format(corner, len(curves), vdd, temperature, scale, V_DD))
    draw_curves(
        curves, "Runtime input voltage (V)", "Runtime output (V)", title,
        output_dir / "relu_{}_100_curves_scaled_vdd_0p1V.pdf".format(corner),
        x_scale=scale, y_scale=scale)


def plot_one_nonlinear_r(corner, output_dir):
    path = nonlinear_r_path(corner)
    curves = load_paired_curves(path)
    vdd, temperature = header_metadata(path)
    process, voltage, temp_level = corner.split("_")
    title = (
        "Coupler {}: {} Monte Carlo gm Vin curves\n"
        "process {}, voltage level {} (VDD={} V), temperature level {} ({} °C)"
        .format(corner, len(curves), process, voltage, vdd, temp_level, temperature))
    draw_curves(
        curves, "Input voltage, Vin (V)", "Effective conductance, gm (µS)",
        title,
        output_dir / "coupler_{}_100_gm_vs_vin_curves.pdf".format(corner),
        y_scale=1e6)


def all_corner_ids():
    return [
        "{}_V{}_T{}".format(process, voltage, temperature)
        for process in PROCESS_ORDER
        for voltage in range(3)
        for temperature in range(3)
    ]


def plot_all(curve_type, output_dir, corner_means=False):
    curves = []
    for corner in all_corner_ids():
        path = relu_path(corner) if curve_type == "relu" else nonlinear_r_path(corner)
        loaded = load_paired_curves(path)
        if corner_means:
            loaded = [mean_curve(loaded)]
        if curve_type == "relu":
            scale = V_DD / max(np.max(np.abs(curve[0])) for curve in loaded)
            loaded = [(x * scale, y * scale) for x, y in loaded]
        curves.extend(loaded)
    if curve_type == "relu":
        count_description = (
            "45 corner-mean curves" if corner_means
            else "4500 piecewise-linear MC curves")
        filename = (
            "relu_all_45_corner_mean_curves_scaled_vdd_0p1V.pdf"
            if corner_means
            else "relu_all_45_corners_4500_curves_scaled_vdd_0p1V.pdf")
        draw_curves(
            curves, "Runtime input voltage (V)", "Runtime output (V)",
            "All 45 measured ReLU corners: {}".format(count_description),
            output_dir / filename)
    else:
        count_description = (
            "45 corner-mean gm Vin curves" if corner_means
            else "4500 Monte Carlo gm Vin curves")
        filename = (
            "coupler_all_45_corner_mean_gm_vs_vin_curves.pdf"
            if corner_means
            else "coupler_all_45_corners_4500_gm_vs_vin_curves.pdf")
        draw_curves(
            curves, "Input voltage, Vin (V)", "Effective conductance, gm (µS)",
            "All 45 coupler corners: {}".format(count_description),
            output_dir / filename, y_scale=1e6)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="cifar100", choices=("cifar10", "cifar100"),
                        help="Selects the documented top/bottom reference-corner set.")
    parser.add_argument("--nonlinear_R", action="store_true")
    parser.add_argument("--relu", action="store_true")
    parser.add_argument("--corners", default=None,
                        help="Case-insensitive comma-separated corner IDs.")
    parser.add_argument(
        "--corner_means", action="store_true",
        help="Plot one Monte Carlo mean curve per corner instead of all curves.")
    parser.add_argument(
        "--output_dir", default=str(REPO_ROOT / "results" / "diagnostic_hardware_curves"))
    args = parser.parse_args()
    normalize_task(args.task)
    style_axes()
    output_dir = Path(args.output_dir)
    selected_types = []
    if args.nonlinear_R:
        selected_types.append("nonlinear_R")
    if args.relu:
        selected_types.append("relu")
    if not selected_types:
        selected_types = ["nonlinear_R", "relu"]

    corners = parse_corners(args.corners)
    if args.corner_means and corners is not None:
        raise ValueError("--corner_means plots all 45 corners and cannot use --corners")
    if corners is None:
        for curve_type in selected_types:
            plot_all(curve_type, output_dir, corner_means=args.corner_means)
        return

    for corner in sorted(corners, key=corner_sort_key):
        if "nonlinear_R" in selected_types:
            plot_one_nonlinear_r(corner, output_dir)
        if "relu" in selected_types:
            plot_one_relu(corner, output_dir)


if __name__ == "__main__":
    main()


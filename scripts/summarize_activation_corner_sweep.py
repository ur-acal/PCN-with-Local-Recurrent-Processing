#!/usr/bin/env python3
"""Consolidate fixed-corner trial results into a Markdown table."""

import argparse
import csv
import math
import statistics
from pathlib import Path


PROCESSES = ("FF", "FS", "SF", "SS", "TT")
VDDS = ("0P9", "1", "1P1")
TEMPERATURES = ("M20", "25", "85")
TEMPERATURE_LABELS = {"M20": "-20 C", "25": "25 C", "85": "85 C"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--summary_path", default=None)
    return parser.parse_args()


def read_corner(input_dir, corner):
    path = input_dir / corner / "ablation_trials_partial.csv"
    if not path.exists():
        return None
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    values = [
        float(row["accuracy"])
        for row in rows
        if row["case"] == "all_known"
    ]
    if not values:
        raise ValueError("No all_known results in {}".format(path))
    return values


def corner_name(process, vdd, temperature):
    return "{}_VDD{}_T{}".format(process, vdd, temperature)


def format_cell(row):
    if row is None:
        return "—"
    if math.isclose(row["std_accuracy"], 0.0, abs_tol=5e-9):
        return "{:.4f}".format(row["mean_accuracy"])
    return "{:.4f} ± {:.4f}".format(
        row["mean_accuracy"], row["std_accuracy"])


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    summary_path = (
        Path(args.summary_path)
        if args.summary_path
        else input_dir / "summary.md"
    )

    result_by_corner = {}
    rows = []
    for process in PROCESSES:
        for vdd in VDDS:
            for temperature in TEMPERATURES:
                corner = corner_name(process, vdd, temperature)
                values = read_corner(input_dir, corner)
                if values is None:
                    continue
                row = {
                    "corner": corner,
                    "process": process,
                    "vdd": vdd.replace("P", "."),
                    "temperature_c": temperature.replace("M", "-"),
                    "mean_accuracy": statistics.fmean(values),
                    "std_accuracy": statistics.pstdev(values),
                }
                rows.append(row)
                result_by_corner[corner] = row

    if not rows:
        raise ValueError("No corner trial results found below {}".format(input_dir))

    lines = [
        "# All-on measured-activation corner sweep",
        "",
        "Spin variation, multiplicative coupler variation (`sigma=0.181`, "
        "or 18.1%), coupler noise, and DTC timing each use a representative "
        "fitted value obtained from characterized corner data. The "
        "input-dependent nonlinear-R characterization is applied. The "
        "piecewise-linear measured ReLU remains corner-specific across the 45 "
        "process/voltage/temperature corners.",
        "",
        "Each cell reports the mean top-1 accuracy in percent; when multiple "
        "trials are present, the population standard deviation follows `±`.",
        "",
        "| Process | VDD | -20 C | 25 C | 85 C |",
        "|---|---:|---:|---:|---:|",
    ]
    for process in PROCESSES:
        for vdd in VDDS:
            cells = [
                format_cell(result_by_corner.get(
                    corner_name(process, vdd, temperature)))
                for temperature in TEMPERATURES
            ]
            lines.append("| {} | {} V | {} | {} | {} |".format(
                process, vdd.replace("P", "."), *cells))
    lines.extend([
        "",
        "Evaluated corners: **{}**.".format(len(rows)),
        "",
    ])
    summary_path.write_text("\n".join(lines))
    print(summary_path.read_text(), flush=True)


if __name__ == "__main__":
    main()

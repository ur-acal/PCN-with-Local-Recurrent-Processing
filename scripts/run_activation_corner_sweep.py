#!/usr/bin/env python3
"""Run the retained all-on pulse inference for selected ReLU corners."""

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = (
    "TIMMQAT5b8aNT0p25mulTIMMPCNetNoBatchNorm_PCConvReLU6_0.0eps_"
    "ToggleODEXInitFFFB_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_"
    "C100_3K1S96C_0.25Dropout_16Layers4l5l4_2Pool_srrlDistill_a0p3_t2p0_"
    "scanGFI_6REP"
)
DEFAULT_ACTIVATION_TABLE = ROOT / "hardware_data" / "relu_current_0p2uA_all.csv"
DEFAULT_NONLINEAR_R_TABLE = ROOT / "hardware_data" / "res_vs_vin_10k_150k.csv"


def available_corners(curve_path):
    pattern = re.compile(
        r":top_(ff|fs|sf|ss|tt),.*?VDD_VALUE=([-+0-9.eE]+),"
        r"temperature=([-+0-9.eE]+)\)\s*Y", re.IGNORECASE)

    def token(value):
        number = float(value)
        text = str(int(number)) if number.is_integer() else format(number, "g")
        return text.replace("-", "M").replace(".", "P")

    with Path(curve_path).open(newline="") as handle:
        fieldnames = csv.DictReader(handle).fieldnames or ()
    corners = []
    for column in fieldnames:
        match = pattern.search(column)
        if match is None:
            continue
        corners.append("{}_VDD{}_T{}".format(
            match.group(1).upper(), token(match.group(2)), token(match.group(3))))
    if not corners:
        raise ValueError("No characterized Y columns found in {}".format(curve_path))
    return tuple(corners)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the retained all-on hardware configuration by activation corner.")
    parser.add_argument("--n_trials", type=int, default=10)
    parser.add_argument("--base_seed", type=int, default=20260721)
    parser.add_argument("--model_name", default=DEFAULT_MODEL)
    parser.add_argument("--model_dir", default="saved_ckpt")
    parser.add_argument("--expanded_w_dir", default="expanded_weights")
    parser.add_argument("--output_dir", default="results/activation_corner_reproduction")
    parser.add_argument("--activation_curve_path", default=str(DEFAULT_ACTIVATION_TABLE))
    parser.add_argument("--nonlinear_R_table", default=str(DEFAULT_NONLINEAR_R_TABLE))
    parser.add_argument("--corners", nargs="*", default=None)
    parser.add_argument("--test_bs", type=int, default=128)
    return parser.parse_args()


def main():
    args = parse_args()
    curve_path = Path(args.activation_curve_path)
    known_corners = available_corners(curve_path)
    corners = tuple(args.corners) if args.corners else known_corners
    unknown = sorted(set(corners) - set(known_corners))
    if unknown:
        raise ValueError(
            "Unknown activation corner(s): {}. Available: {}".format(
                ", ".join(unknown), ", ".join(known_corners)))

    for index, corner in enumerate(corners, start=1):
        corner_output = Path(args.output_dir) / corner
        command = [
            sys.executable,
            "-u",
            "scripts/run_toggle_nonideality_ablation.py",
            "--n_trials",
            str(args.n_trials),
            "--base_seed",
            str(args.base_seed),
            "--model_name",
            args.model_name,
            "--model_dir",
            args.model_dir,
            "--expanded_w_dir",
            args.expanded_w_dir,
            "--output_dir",
            str(corner_output),
            "--test_bs",
            str(args.test_bs),
            "--nonlinear_R_table",
            args.nonlinear_R_table,
            "--activation_curve_path",
            str(curve_path),
            "--activation_corner",
            corner,
        ]
        print(
            "[{}/{}] Evaluating {}".format(index, len(corners), corner),
            flush=True)
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run the PCN-equivalent 45-corner protocol on a physical feedforward CNN."""

import argparse
import csv
import json
import math
import random
import re
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data_utils import MC45CornerData


TRIAL_RESULT_RE = re.compile(
    r"^trial\s+(\d+):\s+accuracy=([-+0-9.eE]+)%")


def str2bool(value):
    return str(value).lower() in {"yes", "true", "t", "1"}


def optional_int(value):
    return None if str(value).strip().lower() in {"", "none", "auto"} else int(value)


def optional_bool(value):
    if str(value).strip().lower() in {"", "none", "auto"}:
        return None
    return str2bool(value)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset", choices=("cifar10", "cifar100"), default="cifar100")
    p.add_argument("--img_type", default="CiFAIR")
    p.add_argument("--output_dir", default="results/feedforward_mc45")
    p.add_argument("--expanded_weight_dir", default="expanded_weights/feedforward")
    p.add_argument("--n_trials", type=int, default=5)
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--test_bs", type=int, default=128)
    p.add_argument("--base_seed", type=int, default=20260723)
    p.add_argument("--corner_ids", nargs="+", default=None)
    p.add_argument("--fixed_relu_mc_index", type=int, default=None)
    p.add_argument("--fixed_coupler_mc_index", type=int, default=None)
    p.add_argument("--mc_45_corner_dir", default="hardware_data/mc_45_corners")
    p.add_argument("--mc_spin_variation_source", default="PVT_Monte_Carlo_Results_SPIN.csv")
    p.add_argument("--mc_dtc_pulse_width_variation_source", default="PVT_45corner_DTC_pulse_width.csv")
    p.add_argument("--mc_relu_monte_carlo_source", default="relu_monteCarlo")
    p.add_argument("--mc_coupler_nonlinear_variation_source", default="coupler_monte_v2")
    p.add_argument("--mc_coupler_nonlinear_variation_quantity", default="conductance")
    p.add_argument("--mc_coupler_nominal_R", type=float, default=50e3)
    p.add_argument("--room_temperature_c", type=float, default=25.0)
    p.add_argument("--full_45_corner_C", type=float, default=500e-15)
    p.add_argument("--physical_level", type=int, choices=(2, 3), default=3)
    p.add_argument("--v_dd", type=float, default=0.1)
    p.add_argument("--one_over_q", type=float, default=1.0)
    p.add_argument("--w_bits", type=int, default=5)
    p.add_argument("--weight_quant_factor_bits", type=int, default=1)
    p.add_argument("--enob", type=optional_int, default=8)
    p.add_argument("--toggle_timing_mode", choices=("derived", "fixed"), default="derived")
    p.add_argument("--toggle_y_time", type=float, default=5e-9)
    p.add_argument("--z_over_y_time", type=float, default=1.0)
    p.add_argument("--input_quant_bits", type=optional_int, default=None)
    p.add_argument("--center_student_input", type=optional_bool, default=None)
    p.add_argument("--enable_nonlinear_R", type=str2bool, default=True)
    p.add_argument("--enable_diff_mismatch", type=str2bool, default=False)
    p.add_argument("--nonlinear_R_curve_sharing", default="per_coupler")
    p.add_argument("--nonlinear_R_curve_sampling", default="empirical_with_replacement")
    p.add_argument("--nonlinear_R_curve_edge_chunk_size", type=int, default=65536)
    p.add_argument("--enable_spin_variation", type=str2bool, default=True)
    p.add_argument("--enable_measured_activation", type=str2bool, default=True)
    p.add_argument("--activation_curve_sharing", default="per_model",
                   choices=("per_model", "per_layer", "per_spin"))
    p.add_argument("--activation_interpolation", default="piecewise_linear")
    p.add_argument("--activation_fit_constraint", default="auto")
    p.add_argument("--activation_normalize_positive_endpoint", type=str2bool, default=False)
    p.add_argument("--compile_measured_activation", type=str2bool, default=False)
    p.add_argument("--enable_measured_pooling", type=str2bool, default=True)
    p.add_argument("--enable_summing_current_noise", type=str2bool, default=True)
    p.add_argument("--summing_current_p", type=float, default=0.6e-12)
    p.add_argument("--enable_coupler_noise", type=str2bool, default=True)
    p.add_argument("--coupler_noise_p", type=float, default=0.6e-12)
    p.add_argument("--enable_slow_summing_current", type=str2bool, default=False)
    p.add_argument("--slow_summing_current", type=float, default=2.47e-9)
    p.add_argument("--enable_slow_coupler_noise", type=str2bool, default=False)
    p.add_argument("--slow_coupler_noise", type=float, default=2.47e-9)
    p.add_argument("--enable_dtc_nonideality", type=str2bool, default=True)
    p.add_argument("--dtc_characterized_nominal_fraction", type=float, default=1.0 / 16.0)
    p.add_argument("--dtc_leading_edge_variation_std", type=float, default=0.0)
    p.add_argument("--dtc_leading_edge_jitter_std", type=float, default=0.005)
    p.add_argument("--dtc_falling_edge_jitter_std", type=float, default=0.005)
    return p.parse_args()


def bool_text(value):
    return "true" if value else "false"


def run_corner(args, corner, relu_indices):
    temperature_scale = math.sqrt(
        (float(corner["coupler"]["temperature"]) + 273.15) /
        (args.room_temperature_c + 273.15))
    nominal = args.dtc_characterized_nominal_fraction
    dtc_mean = (float(corner["dtc"]["mean"]) / 100.0) / nominal - 1.0
    dtc_std = (float(corner["dtc"]["std"]) / 100.0) / nominal
    result_dir = Path(args.output_dir) / "trial_outputs" / corner["id"]
    activation_corner = (
        "MC1" if relu_indices[0] is None else
        "MC{}".format(relu_indices[0] + 1))
    command = [
        sys.executable, "-u", "baseline/evaluate_physical_feedforward_cifar.py",
        "--model_name", args.model_name,
        "--checkpoint", args.checkpoint,
        "--dataset", args.dataset,
        "--img_type", args.img_type,
        "--batch_size", str(args.test_bs),
        "--expanded_weight_dir", args.expanded_weight_dir,
        "--result_path", str(result_dir),
        "--n_trials", str(args.n_trials),
        "--physical_level", str(args.physical_level),
        "--R", str(corner["coupler"].get("nominal_R", args.mc_coupler_nominal_R)),
        "--C", str(args.full_45_corner_C),
        "--v_dd", str(args.v_dd),
        "--one_over_q", str(args.one_over_q),
        "--w_bits", str(args.w_bits),
        "--weight_quant_factor_bits", str(args.weight_quant_factor_bits),
        "--enob", "none" if args.enob is None else str(args.enob),
        "--toggle_timing_mode", args.toggle_timing_mode,
        "--toggle_y_time", str(args.toggle_y_time),
        "--z_over_y_time", str(args.z_over_y_time),
        "--input_quant_bits", "none" if args.input_quant_bits is None else str(args.input_quant_bits),
        "--center_student_input", (
            "auto" if args.center_student_input is None
            else bool_text(args.center_student_input)),
        "--enable_nonlinear_R", bool_text(args.enable_nonlinear_R),
        "--nonlinear_R_table", corner["coupler"]["path"],
        "--nonlinear_R_mc_quantity", corner["coupler"]["quantity"],
        "--nonlinear_R_curve_sharing", args.nonlinear_R_curve_sharing,
        "--nonlinear_R_curve_sampling", args.nonlinear_R_curve_sampling,
        "--nonlinear_R_curve_bank_indices", ",".join(
            str(index) for index in (
                [args.fixed_coupler_mc_index]
                if args.fixed_coupler_mc_index is not None else
                corner["coupler"]["curve_indices"])),
        "--nonlinear_R_curve_seed", str(args.base_seed),
        "--nonlinear_R_curve_edge_chunk_size", str(args.nonlinear_R_curve_edge_chunk_size),
        "--diff_mismatch", bool_text(args.enable_diff_mismatch),
        "--enable_spin_variation", bool_text(args.enable_spin_variation),
        "--spin_variation_mean", str(corner["spin"]["mean"]),
        "--sigma_spin", str(corner["spin"]["std"]),
        "--spin_variation_seed", str(args.base_seed),
        "--enable_measured_activation", bool_text(args.enable_measured_activation),
        "--activation_curve_path", corner["relu"]["path"],
        "--activation_corner", activation_corner,
        "--activation_curve_sharing", args.activation_curve_sharing,
        "--activation_curve_seed", str(args.base_seed),
        "--activation_interpolation", args.activation_interpolation,
        "--activation_fit_constraint", args.activation_fit_constraint,
        "--activation_normalize_positive_endpoint", bool_text(
            args.activation_normalize_positive_endpoint),
        "--compile_measured_activation", bool_text(args.compile_measured_activation),
        "--enable_measured_pooling", bool_text(args.enable_measured_pooling),
        "--enable_summing_current_noise", bool_text(args.enable_summing_current_noise),
        "--summing_current_p", str(args.summing_current_p * temperature_scale),
        "--summing_noise_seed", str(args.base_seed),
        "--enable_coupler_noise", bool_text(args.enable_coupler_noise),
        "--coupler_noise_p", str(args.coupler_noise_p * temperature_scale),
        "--coupler_noise_seed", str(args.base_seed),
        "--enable_slow_summing_current", bool_text(args.enable_slow_summing_current),
        "--slow_summing_current", str(args.slow_summing_current),
        "--enable_slow_coupler_noise", bool_text(args.enable_slow_coupler_noise),
        "--slow_coupler_noise", str(args.slow_coupler_noise),
        "--enable_dtc_nonideality", bool_text(args.enable_dtc_nonideality),
        "--dtc_leading_edge_variation_std", str(args.dtc_leading_edge_variation_std),
        "--dtc_width_variation_mean", str(dtc_mean),
        "--dtc_width_variation_std", str(dtc_std),
        "--dtc_leading_edge_jitter_std", str(args.dtc_leading_edge_jitter_std),
        "--dtc_falling_edge_jitter_std", str(args.dtc_falling_edge_jitter_std),
        "--dtc_timing_seed", str(args.base_seed),
        "--data_seed", str(args.base_seed),
    ]
    if relu_indices[0] is not None:
        command.extend([
            "--activation_mc_curve_indices",
            ",".join(str(index) for index in relu_indices),
        ])
    log_path = Path(args.output_dir) / "{}.log".format(corner["id"])
    print("Running {} ({} trials)".format(
        corner["id"], args.n_trials), flush=True)
    results = []
    with log_path.open("w") as log:
        log.write("COMMAND: " + " ".join(command) + "\n")
        process = subprocess.Popen(
            command, cwd=REPO_ROOT, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:
            log.write(line)
            log.flush()
            match = TRIAL_RESULT_RE.search(line)
            if match:
                trial_index = int(match.group(1))
                accuracy = float(match.group(2))
                result = {
                    "corner": corner["id"],
                    "process": corner["process"].upper(),
                    "voltage_level": corner["voltage_level"],
                    "temperature_level": corner["temperature_level"],
                    "temperature_c": corner["coupler"]["temperature"],
                    "trial_index": trial_index,
                    "activation_curve_sharing": args.activation_curve_sharing,
                    "relu_mc_index": relu_indices[trial_index],
                    "coupler_mc_index": args.fixed_coupler_mc_index,
                    "accuracy": accuracy,
                }
                results.append(result)
                print("  trial {}: {:.4f}%".format(
                    trial_index, accuracy), flush=True)
        return_code = process.wait()
    if return_code:
        raise RuntimeError("{} failed; see {}".format(
            corner["id"], log_path))
    if len(results) != args.n_trials:
        raise RuntimeError("Expected {} results for {}, found {}; see {}".format(
            args.n_trials, corner["id"], len(results), log_path))
    return results


def write_outputs(rows, output_dir):
    fields = ("corner", "process", "voltage_level", "temperature_level",
              "temperature_c", "trial_index", "activation_curve_sharing",
              "relu_mc_index", "coupler_mc_index", "accuracy")
    with (output_dir / "corner_trials.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    grouped = {}
    for row in rows:
        grouped.setdefault(row["corner"], []).append(row["accuracy"])
    lines = ["| Corner | Per-trial accuracy (%) | Mean (%) | Std (%) |",
             "|---|---:|---:|---:|"]
    for corner, values in grouped.items():
        lines.append("| {} | {} | {:.4f} | {:.4f} |".format(
            corner, ", ".join("{:.4f}".format(v) for v in values),
            statistics.mean(values), statistics.pstdev(values)))
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.json").write_text(
        json.dumps(vars(args), indent=2, sort_keys=True) + "\n")
    catalog = MC45CornerData(
        args.mc_45_corner_dir,
        spin_variation_source=args.mc_spin_variation_source,
        dtc_pulse_width_variation_source=args.mc_dtc_pulse_width_variation_source,
        relu_monte_carlo_source=args.mc_relu_monte_carlo_source,
        coupler_nonlinear_variation_source=args.mc_coupler_nonlinear_variation_source,
        coupler_nonlinear_variation_quantity=args.mc_coupler_nonlinear_variation_quantity,
        coupler_nominal_R=args.mc_coupler_nominal_R)
    corners = list(catalog.corners)
    if args.corner_ids:
        requested = {item.upper() for item in args.corner_ids}
        corners = [corner for corner in corners if corner["id"] in requested]
        if len(corners) != len(requested):
            raise ValueError("One or more requested corner IDs are unknown.")
    tasks = []
    for corner in corners:
        corner_index = catalog.corners.index(corner)
        if (args.fixed_coupler_mc_index is not None and
                args.fixed_coupler_mc_index not in
                corner["coupler"]["curve_indices"]):
            raise ValueError(
                "Corner {} does not contain coupler MC index {}.".format(
                    corner["id"], args.fixed_coupler_mc_index))
        relu_indices = [None] * args.n_trials
        if args.fixed_relu_mc_index is not None:
            if args.activation_curve_sharing != "per_model":
                raise ValueError(
                    "fixed_relu_mc_index requires per_model activation sharing.")
            if args.fixed_relu_mc_index not in corner["relu"]["curve_indices"]:
                raise ValueError(
                    "Corner {} does not contain ReLU MC index {}.".format(
                        corner["id"], args.fixed_relu_mc_index))
            relu_indices = [args.fixed_relu_mc_index] * args.n_trials
        elif args.activation_curve_sharing == "per_model":
            if args.n_trials > len(corner["relu"]["curve_indices"]):
                raise ValueError(
                    "Corner {} has only {} ReLU curves for {} trials."
                    .format(corner["id"],
                            len(corner["relu"]["curve_indices"]),
                            args.n_trials))
            relu_indices = random.Random(
                args.base_seed + 10007 * corner_index).sample(
                    corner["relu"]["curve_indices"], args.n_trials)
        tasks.append((corner, relu_indices))
    rows = []
    if args.jobs == 1:
        for task in tasks:
            rows.extend(run_corner(args, *task))
    else:
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            futures = [executor.submit(run_corner, args, *task) for task in tasks]
            for future in as_completed(futures):
                rows.extend(future.result())
        order = {corner["id"]: index for index, corner in enumerate(corners)}
        rows.sort(key=lambda row: (order[row["corner"]], row["trial_index"]))
    write_outputs(rows, output_dir)


if __name__ == "__main__":
    main()

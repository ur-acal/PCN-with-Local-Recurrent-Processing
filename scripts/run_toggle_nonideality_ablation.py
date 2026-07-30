#!/usr/bin/env python3
"""Run toggle nonideality ablations through ode_inference.py."""

import argparse
import csv
import math
import random
import statistics
from concurrent.futures import ThreadPoolExecutor
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_utils import MC45CornerData


DEFAULT_MODEL = (
    "TIMMQAT5b8aNT0p25mulTIMMPCNetNoBatchNorm_PCConvReLU6_0.0eps_"
    "ToggleODEXInitFFFB_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_"
    "C100_3K1S96C_0.25Dropout_16Layers4l5l4_2Pool_srrlDistill_a0p3_t2p0_"
    "scanGFI_1REP"
)

CASES = {
    "ideal": {},
    "spin_only": {"spin": True},
    "coupler_only": {"coupler": True},
    "measured_activation_only": {"activation": True},
    "nonlinear_R_only": {"nonlinear_R": True},
    "summing_current_noise_only": {"current_noise": True},
    "coupler_noise_only": {"coupler_noise": True},
    "dtc_nonideality_only": {"dtc_nonideality": True},
    "all_known": {
        "spin": True,
        "coupler": True,
        "activation": True,
        "nonlinear_R": True,
        "coupler_noise": True,
        "dtc_nonideality": True,
    },
    "all_known_except_summing_current_noise": {
        "spin": True,
        "coupler": True,
        "activation": True,
        "nonlinear_R": True,
        "coupler_noise": True,
        "dtc_nonideality": True,
    },
}

DEFAULT_CASES = (
    "ideal",
    "spin_only",
    "coupler_only",
    "measured_activation_only",
    "nonlinear_R_only",
    "coupler_noise_only",
    "dtc_nonideality_only",
    "all_known",
)

RESULT_RE = re.compile(
    r"ABLATION_RESULT case=(\S+) trial_index=(\d+) accuracy=([0-9.]+)")


def bool_arg(value):
    return "true" if value else "false"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trials", type=int, default=5)
    parser.add_argument(
        "--full_45_corner_test",
        type=lambda v: v.lower() in ("yes", "true", "t", "1"),
        default=False)
    parser.add_argument(
        "--mc_45_corner_dir",
        default=str(REPO_ROOT / "hardware_data" / "mc_45_corners"))
    parser.add_argument(
        "--mc_spin_variation_source",
        default="PVT_Monte_Carlo_Results_SPIN.csv")
    parser.add_argument(
        "--mc_dtc_pulse_width_variation_source",
        default="PVT_45corner_DTC_pulse_width.csv")
    parser.add_argument(
        "--mc_relu_monte_carlo_source",
        default="relu_monteCarlo")
    parser.add_argument(
        "--mc_coupler_nonlinear_variation_source",
        default="coupler_monte")
    parser.add_argument(
        "--mc_coupler_nonlinear_variation_quantity",
        choices=("conductance", "resistance"), default=None)
    parser.add_argument("--mc_coupler_nominal_R", type=float, default=67e3)
    parser.add_argument(
        "--full_45_corner_enable_spin_variation",
        type=lambda v: v.lower() in ("yes", "true", "t", "1"),
        default=True)
    parser.add_argument(
        "--full_45_corner_enable_measured_activation",
        type=lambda v: v.lower() in ("yes", "true", "t", "1"),
        default=True)
    parser.add_argument(
        "--full_45_corner_enable_nonlinear_R",
        type=lambda v: v.lower() in ("yes", "true", "t", "1"),
        default=True)
    parser.add_argument(
        "--full_45_corner_enable_diff_mismatch",
        type=lambda v: v.lower() in ("yes", "true", "t", "1"),
        default=False)
    parser.add_argument(
        "--full_45_corner_enable_summing_current_noise",
        type=lambda v: v.lower() in ("yes", "true", "t", "1"),
        default=True)
    parser.add_argument(
        "--full_45_corner_enable_coupler_noise",
        type=lambda v: v.lower() in ("yes", "true", "t", "1"),
        default=True)
    parser.add_argument(
        "--full_45_corner_enable_dtc_nonideality",
        type=lambda v: v.lower() in ("yes", "true", "t", "1"),
        default=True)
    parser.add_argument("--corner_ids", nargs="+", default=None)
    parser.add_argument("--corner_limit", type=int, default=None)
    parser.add_argument(
        "--random_corner_selection",
        type=lambda v: v.lower() in ("yes", "true", "t", "1"),
        default=False)
    parser.add_argument(
        "--dtc_characterized_nominal_fraction", type=float, default=1.0 / 16.0)
    parser.add_argument("--room_temperature_c", type=float, default=25.0)
    parser.add_argument(
        "--full_45_corner_summing_current_p", type=float, default=0.6e-12)
    parser.add_argument(
        "--full_45_corner_C", type=float, default=1.2e-13)
    parser.add_argument("--base_seed", type=int, default=20260721)
    parser.add_argument("--model_name", default=DEFAULT_MODEL)
    parser.add_argument("--ckpt", default="best")
    parser.add_argument("--model_dir", default="saved_ckpt")
    parser.add_argument("--expanded_w_dir", default="expanded_weights")
    parser.add_argument("--output_dir", default="results/prompt2_toggle_ablation")
    parser.add_argument("--cases", nargs="+", choices=tuple(CASES), default=list(DEFAULT_CASES))
    parser.add_argument("--test_bs", type=int, default=128)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--toggle_level", type=int, choices=(2, 3), default=3)
    parser.add_argument("--odexinit_scaling_mode",
                        choices=("approx", "direct"), default="approx")
    parser.add_argument("--summing_current_p", type=float, default=18.5e-12)
    parser.add_argument("--coupler_noise_p", type=float, default=0.6e-12)
    parser.add_argument("--nonlinear_R_table", default=None)
    parser.add_argument(
        "--nonlinear_R_curve_sharing",
        choices=("shared", "per_coupler", "per_input",
                 "per_input_output"),
        default="shared")
    parser.add_argument(
        "--nonlinear_R_curve_sampling",
        choices=("empirical_with_replacement", "multivariate_gaussian"),
        default="empirical_with_replacement")
    parser.add_argument(
        "--nonlinear_R_curve_edge_chunk_size", type=int, default=65536)
    parser.add_argument("--R", type=float, default=10e3)
    parser.add_argument("--R_max", default="150e3")
    parser.add_argument("--C", type=float, default=49e-15)
    parser.add_argument("--k", type=float, default=1e3)
    parser.add_argument("--v_dd", type=float, default=0.1)
    parser.add_argument("--dtc_leading_edge_variation_std", type=float, default=0.0)
    parser.add_argument("--dtc_width_variation_std", type=float, default=0.018)
    parser.add_argument("--dtc_leading_edge_jitter_std", type=float, default=0.005)
    parser.add_argument("--dtc_falling_edge_jitter_std", type=float, default=0.005)
    parser.add_argument(
        "--activation_curve_path",
        default=str(Path(__file__).resolve().parents[1] / "hardware_data"
                    / "relu_current_0p2uA_finer.csv"))
    parser.add_argument("--activation_corner", default="TT")
    parser.add_argument(
        "--activation_interpolation",
        choices=("cubic_bspline", "piecewise_linear"),
        default="piecewise_linear")
    parser.add_argument("--activation_fit_constraint",
                        choices=("none", "nonnegative", "auto"), default="auto")
    parser.add_argument("--activation_normalize_positive_endpoint",
                        type=lambda v: v.lower() in ('yes', 'true', 't', '1'), default=False)
    parser.add_argument("--compile_measured_activation",
                        type=lambda v: v.lower() in ("yes", "true", "t", "1"), default=False)
    parser.add_argument("--measured_activation_baseline", action="store_true",
                        help="Keep measured activation enabled in every selected case.")
    return parser.parse_args()


def build_command(args, case_name):
    enabled = CASES[case_name]
    ode_block = (
        "ToggleODEXInitFFFB" if args.toggle_level == 2
        else "TogglePulseODEXInitFFFB"
    )
    ode_wrapper = (
        "ToggleWrapper1State" if "full_param" in args.ckpt
        else "ToggleQATTester1State"
    )
    return [
        sys.executable, "-u", "ode_inference.py",
        "--model_name", args.model_name,
        "--ckpt", args.ckpt,
        "--model_dir", args.model_dir,
        "--task", "cifar100",
        "--img_type", "scanGFI",
        "--test_bs", str(args.test_bs),
        "--pc_conv", "PCConvReLU6Noisy",
        "--ode_block", ode_block,
        "--ode_wrapper", ode_wrapper,
        "--method", "dopri5",
        "--tol", "1e-6",
        "--n_steps", "100",
        "--ts_scale", "1",
        "--d_start", "0",
        "--d_end", "1",
        "--n_sweep_left", "0",
        "--n_sweep_right", "1",
        "--noise_level_list", "0.0",
        "--noisy_trials", str(args.n_trials),
        "--thermal_noise", "false",
        "--sde_noise_type", "mul",
        "--mismatch_type", "mul",
        "--sweep_eps", "false",
        "--R", str(args.R),
        "--R_max", str(args.R_max),
        "--C", str(args.C),
        "--k", str(args.k),
        "--v_dd", str(args.v_dd),
        "--enob", "8",
        "--w_bits", "5",
        "--patch_node", "8",
        "--patch_stride", "8",
        "--patch_cycle", "1",
        "--patch_pad", "0",
        "--fold_scalar", "1",
        "--tie_cap", "false",
        "--one_over_q", "1",
        "--toggle_n_cycles", "5",
        "--toggle_time_split", "0.5",
        "--toggle_fast_path", "true",
        "--odexinit_scaling_mode", args.odexinit_scaling_mode,
        "--conv_only", bool_arg(args.toggle_level == 3),
        "--test_expanded", bool_arg(args.toggle_level == 3),
        "--expanded_w_dir", args.expanded_w_dir,
        "--diff_mismatch", bool_arg(enabled.get("coupler", False)),
        "--nonlinear_R", bool_arg(enabled.get("nonlinear_R", False)),
        "--nonlinear_R_table", args.nonlinear_R_table or "none",
        "--nonlinear_R_mc_quantity", "conductance",
        "--nonlinear_R_curve_sharing", args.nonlinear_R_curve_sharing,
        "--nonlinear_R_curve_sampling", args.nonlinear_R_curve_sampling,
        "--nonlinear_R_curve_seed", str(args.base_seed),
        "--nonlinear_R_curve_edge_chunk_size",
        str(args.nonlinear_R_curve_edge_chunk_size),
        "--mul_mismatch_mode", "static_mismatch",
        "--enable_spin_variation", bool_arg(enabled.get("spin", False)),
        "--sigma_spin", "0.10",
        "--spin_variation_mean", "1.0",
        "--spin_variation_seed", str(args.base_seed),
        "--enable_measured_activation",
        bool_arg(args.measured_activation_baseline or enabled.get("activation", False)),
        "--activation_curve_path", args.activation_curve_path,
        "--activation_corner", args.activation_corner,
        "--activation_interpolation", args.activation_interpolation,
        "--activation_spline_parameters", "10",
        "--activation_fit_constraint", args.activation_fit_constraint,
        "--activation_normalize_positive_endpoint",
        bool_arg(args.activation_normalize_positive_endpoint),
        "--compile_measured_activation", bool_arg(args.compile_measured_activation),
        "--enable_summing_current_noise", bool_arg(enabled.get("current_noise", False)),
        "--summing_current_p", str(args.summing_current_p),
        "--summing_noise_seed", str(args.base_seed),
        "--enable_coupler_noise", bool_arg(enabled.get("coupler_noise", False)),
        "--coupler_noise_p", str(args.coupler_noise_p),
        "--coupler_noise_seed", str(args.base_seed),
        "--enable_dtc_nonideality", bool_arg(enabled.get("dtc_nonideality", False)),
        "--dtc_leading_edge_variation_std",
        str(args.dtc_leading_edge_variation_std),
        "--dtc_width_variation_mean", "0.0",
        "--dtc_width_variation_std", str(args.dtc_width_variation_std),
        "--dtc_leading_edge_jitter_std", str(args.dtc_leading_edge_jitter_std),
        "--dtc_falling_edge_jitter_std", str(args.dtc_falling_edge_jitter_std),
        "--dtc_timing_seed", str(args.base_seed),
        "--return_init", "0",
        "--test_only", "false",
        "--ablation_single_case", "true",
        "--full_45_corner_test", "false",
        "--ablation_case_name", case_name,
        "--hardware_seed", str(args.base_seed),
        "--data_seed", str(args.base_seed),
    ]


def _set_command_arg(command, name, value):
    position = command.index(name)
    command[position + 1] = str(value)


def _corner_trial_configuration(args, corner, corner_index):
    if args.n_trials <= 0:
        raise ValueError("n_trials must be positive.")
    relu_available = corner["relu"]["curve_indices"]
    coupler_available = corner["coupler"]["curve_indices"]
    if args.n_trials > len(relu_available):
        raise ValueError(
            "Corner {} cannot sample {} ReLU curves without replacement; "
            "only {} are available.".format(
                corner["id"], args.n_trials, len(relu_available)))
    if (args.nonlinear_R_curve_sharing == "shared" and
            args.n_trials > len(coupler_available)):
        raise ValueError(
            "Corner {} cannot sample {} coupler curves without replacement; "
            "only {} are available.".format(
                corner["id"], args.n_trials, len(coupler_available)))

    relu_rng = random.Random(args.base_seed + 10007 * corner_index)
    relu_indices = relu_rng.sample(relu_available, args.n_trials)
    if args.nonlinear_R_curve_sharing == "shared":
        coupler_rng = random.Random(
            args.base_seed + 10007 * corner_index + 1)
        coupler_indices = coupler_rng.sample(
            coupler_available, args.n_trials)
    else:
        coupler_indices = None

    temperature_c = float(corner["coupler"]["temperature"])
    temperature_k = temperature_c + 273.15
    room_temperature_k = float(args.room_temperature_c) + 273.15
    if temperature_k <= 0 or room_temperature_k <= 0:
        raise ValueError("Noise-temperature scaling requires positive Kelvin values.")
    temperature_scale = math.sqrt(temperature_k / room_temperature_k)
    coupler_noise_p = args.coupler_noise_p * temperature_scale
    summing_current_p = (
        args.full_45_corner_summing_current_p * temperature_scale)

    nominal_fraction = float(args.dtc_characterized_nominal_fraction)
    if nominal_fraction <= 0:
        raise ValueError("dtc_characterized_nominal_fraction must be positive.")
    dtc_mean_fraction = float(corner["dtc"]["mean"]) / 100.0
    dtc_std_fraction = float(corner["dtc"]["std"]) / 100.0
    dtc_width_mean = dtc_mean_fraction / nominal_fraction - 1.0
    dtc_width_std = dtc_std_fraction / nominal_fraction

    return {
        "relu_indices": relu_indices,
        "coupler_indices": coupler_indices,
        "coupler_noise_p": coupler_noise_p,
        "summing_current_p": summing_current_p,
        "temperature_c": temperature_c,
        "dtc_width_mean": dtc_width_mean,
        "dtc_width_std": dtc_width_std,
    }


def build_corner_command(args, corner, corner_index):
    trial_config = _corner_trial_configuration(args, corner, corner_index)
    command = build_command(args, "all_known")
    _set_command_arg(command, "--ablation_case_name", corner["id"])
    _set_command_arg(command, "--full_45_corner_test", "true")
    _set_command_arg(command, "--C", args.full_45_corner_C)
    _set_command_arg(
        command, "--diff_mismatch",
        bool_arg(args.full_45_corner_enable_diff_mismatch))
    _set_command_arg(
        command, "--nonlinear_R",
        bool_arg(args.full_45_corner_enable_nonlinear_R))
    _set_command_arg(command, "--nonlinear_R_table", corner["coupler"]["path"])
    _set_command_arg(
        command, "--R", corner["coupler"].get("nominal_R", args.R))
    _set_command_arg(
        command, "--nonlinear_R_mc_quantity", corner["coupler"]["quantity"])
    if trial_config["coupler_indices"] is not None:
        command.extend([
            "--nonlinear_R_mc_curve_indices",
            ",".join(str(index) for index in
                     trial_config["coupler_indices"]),
        ])
    else:
        command.extend([
            "--nonlinear_R_curve_bank_indices",
            ",".join(str(index) for index in
                     corner["coupler"]["curve_indices"]),
        ])
    command.extend([
        "--activation_mc_curve_indices",
        ",".join(str(index) for index in trial_config["relu_indices"]),
    ])
    _set_command_arg(command, "--activation_curve_path", corner["relu"]["path"])
    _set_command_arg(command, "--activation_corner", "MC1")
    _set_command_arg(
        command, "--enable_spin_variation",
        bool_arg(args.full_45_corner_enable_spin_variation))
    _set_command_arg(command, "--spin_variation_mean", corner["spin"]["mean"])
    _set_command_arg(command, "--sigma_spin", corner["spin"]["std"])
    _set_command_arg(
        command, "--enable_measured_activation",
        bool_arg(args.full_45_corner_enable_measured_activation))
    _set_command_arg(
        command, "--enable_coupler_noise",
        bool_arg(args.full_45_corner_enable_coupler_noise))
    _set_command_arg(command, "--coupler_noise_p", trial_config["coupler_noise_p"])
    _set_command_arg(
        command, "--enable_summing_current_noise",
        bool_arg(args.full_45_corner_enable_summing_current_noise))
    _set_command_arg(
        command, "--summing_current_p", trial_config["summing_current_p"])
    _set_command_arg(
        command, "--enable_dtc_nonideality",
        bool_arg(args.full_45_corner_enable_dtc_nonideality))
    _set_command_arg(
        command, "--dtc_width_variation_mean",
        trial_config["dtc_width_mean"])
    _set_command_arg(
        command, "--dtc_width_variation_std",
        trial_config["dtc_width_std"])
    return command, trial_config


def run_corner(args, corner, corner_index, output_dir):
    log_path = output_dir / (corner["id"] + ".log")
    command, trial_config = build_corner_command(args, corner, corner_index)
    print("Running {} ({} trials)".format(
        corner["id"], args.n_trials), flush=True)
    results = []
    with log_path.open("w") as log_file:
        log_file.write("COMMAND: {}\n".format(" ".join(command)))
        process = subprocess.Popen(
            command, cwd=REPO_ROOT, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:
            log_file.write(line)
            match = RESULT_RE.search(line)
            if match:
                trial_index = int(match.group(2))
                result = {
                    "corner": corner["id"],
                    "process": corner["process"].upper(),
                    "voltage_level": corner["voltage_level"],
                    "temperature_level": corner["temperature_level"],
                    "temperature_c": trial_config["temperature_c"],
                    "trial_index": trial_index,
                    "relu_mc_index": trial_config["relu_indices"][trial_index],
                    "coupler_mc_index": (
                        trial_config["coupler_indices"][trial_index]
                        if trial_config["coupler_indices"] is not None
                        else None),
                    "accuracy": float(match.group(3)),
                }
                results.append(result)
                print("  trial {}: {:.4f}%".format(
                    trial_index, result["accuracy"]), flush=True)
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError("{} failed; see {}".format(corner["id"], log_path))
    if len(results) != args.n_trials:
        raise RuntimeError("Expected {} results for {}, found {}.".format(
            args.n_trials, corner["id"], len(results)))
    return results


def write_corner_outputs(results, output_dir):
    fieldnames = (
        "corner", "process", "voltage_level", "temperature_level",
        "temperature_c", "trial_index", "relu_mc_index",
        "coupler_mc_index", "accuracy")
    csv_path = output_dir / "corner_trials.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    by_corner = {}
    for row in results:
        by_corner.setdefault(row["corner"], []).append(row["accuracy"])
    lines = [
        "| Corner | Per-trial accuracy (%) | Mean (%) | Std (%) |",
        "|---|---:|---:|---:|",
    ]
    for corner, values in by_corner.items():
        lines.append("| {} | {} | {:.4f} | {:.4f} |".format(
            corner, ", ".join("{:.4f}".format(value) for value in values),
            statistics.mean(values), statistics.pstdev(values)))
    summary_path = output_dir / "summary.md"
    summary_path.write_text("\n".join(lines) + "\n")
    print("\n" + summary_path.read_text(), flush=True)
    return csv_path, summary_path


def run_case(args, case_name, output_dir):
    log_path = output_dir / (case_name + ".log")
    command = build_command(args, case_name)
    print("Running {} ({} trials)".format(case_name, args.n_trials), flush=True)
    results = []
    with log_path.open("w") as log_file:
        log_file.write("COMMAND: {}\n".format(" ".join(command)))
        process = subprocess.Popen(
            command, cwd=Path(__file__).resolve().parents[1],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1)
        for line in process.stdout:
            log_file.write(line)
            match = RESULT_RE.search(line)
            if match:
                result = {
                    "case": match.group(1),
                    "trial_index": int(match.group(2)),
                    "accuracy": float(match.group(3)),
                }
                results.append(result)
                print("  trial {}: {:.4f}%".format(
                    result["trial_index"], result["accuracy"]), flush=True)
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError("{} failed; see {}".format(case_name, log_path))
    if len(results) != args.n_trials:
        raise RuntimeError("Expected {} results for {}, found {}.".format(
            args.n_trials, case_name, len(results)))
    return results


def write_outputs(results, output_dir, case_names=None,
                  measured_activation_baseline=False):
    csv_path = output_dir / "ablation_trials.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("case", "trial_index", "accuracy"))
        writer.writeheader()
        writer.writerows(results)

    by_case = {}
    for row in results:
        by_case.setdefault(row["case"], []).append(row["accuracy"])
    ideal_mean = sum(by_case["ideal"]) / len(by_case["ideal"])

    summary_path = output_dir / "summary.md"
    lines = []
    if measured_activation_baseline:
        lines.extend([
            "Measured activation is enabled in every case. "
            "The ideal row is measured activation only.",
            "",
        ])
    lines.extend([
        "| Case | Per-trial accuracy (%) | Mean (%) | Drop from ideal (pp) |",
        "|---|---:|---:|---:|",
    ])
    for case_name in case_names or CASES:
        values = by_case[case_name]
        mean = sum(values) / len(values)
        display_name = (
            "ideal (measured activation only)"
            if measured_activation_baseline and case_name == "ideal"
            else case_name
        )
        lines.append("| {} | {} | {:.4f} | {:.4f} |".format(
            display_name, ", ".join("{:.4f}".format(v) for v in values),
            mean, ideal_mean - mean))
    summary_path.write_text("\n".join(lines) + "\n")
    print("\n" + summary_path.read_text(), flush=True)
    return csv_path, summary_path


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.full_45_corner_test:
        catalog = MC45CornerData(
            args.mc_45_corner_dir,
            spin_variation_source=args.mc_spin_variation_source,
            dtc_pulse_width_variation_source=(
                args.mc_dtc_pulse_width_variation_source),
            relu_monte_carlo_source=args.mc_relu_monte_carlo_source,
            coupler_nonlinear_variation_source=(
                args.mc_coupler_nonlinear_variation_source),
            coupler_nonlinear_variation_quantity=(
                args.mc_coupler_nonlinear_variation_quantity),
            coupler_nominal_R=args.mc_coupler_nominal_R)
        corners = list(catalog.corners)
        if args.corner_ids is not None:
            requested = {name.upper() for name in args.corner_ids}
            available = {corner["id"] for corner in corners}
            unknown = sorted(requested - available)
            if unknown:
                raise ValueError(
                    "Unknown corner IDs: {}.".format(", ".join(unknown)))
            corners = [corner for corner in corners if corner["id"] in requested]
        if args.corner_limit is not None:
            if args.corner_limit <= 0:
                raise ValueError("corner_limit must be positive.")
            if args.random_corner_selection:
                corners = random.Random(args.base_seed).sample(
                    corners, min(args.corner_limit, len(corners)))
            else:
                corners = corners[:args.corner_limit]

        results = []
        if args.jobs == 1:
            for corner in corners:
                corner_index = catalog.corners.index(corner)
                results.extend(run_corner(
                    args, corner, corner_index, output_dir))
        else:
            with ThreadPoolExecutor(max_workers=args.jobs) as executor:
                futures = [executor.submit(
                    run_corner, args, corner, catalog.corners.index(corner),
                    output_dir) for corner in corners]
                for future in futures:
                    results.extend(future.result())
        write_corner_outputs(results, output_dir)
        return

    results = []
    if args.jobs == 1:
        for case_name in args.cases:
            results.extend(run_case(args, case_name, output_dir))
    else:
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            futures = [executor.submit(run_case, args, case_name, output_dir)
                       for case_name in args.cases]
            for future in futures:
                results.extend(future.result())
    if "ideal" not in args.cases:
        print("Partial run complete; summary requires the ideal baseline case.")
        partial_path = output_dir / "ablation_trials_partial.csv"
        with partial_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=("case", "trial_index", "accuracy"))
            writer.writeheader()
            writer.writerows(results)
        return
    write_outputs(
        results, output_dir, args.cases,
        measured_activation_baseline=args.measured_activation_baseline)


if __name__ == "__main__":
    main()

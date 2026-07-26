#!/usr/bin/env python3
"""Run toggle nonideality ablations through ode_inference.py."""

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
import re
import subprocess
import sys
from pathlib import Path


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
        "--R", "10e3",
        "--R_max", "150e3",
        "--C", "49e-15",
        "--k", "1e3",
        "--v_dd", "0.1",
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
        "--mul_mismatch_mode", "static_mismatch",
        "--enable_spin_variation", bool_arg(enabled.get("spin", False)),
        "--sigma_spin", "0.10",
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
        "--dtc_width_variation_std", str(args.dtc_width_variation_std),
        "--dtc_leading_edge_jitter_std", str(args.dtc_leading_edge_jitter_std),
        "--dtc_falling_edge_jitter_std", str(args.dtc_falling_edge_jitter_std),
        "--dtc_timing_seed", str(args.base_seed),
        "--return_init", "0",
        "--test_only", "false",
        "--ablation_single_case", "true",
        "--ablation_case_name", case_name,
        "--hardware_seed", str(args.base_seed),
        "--data_seed", str(args.base_seed),
    ]


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

#!/usr/bin/env python3
"""Run the retained all-on hardware configuration for one ReLU corner."""

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
RESULT_RE = re.compile(
    r"ABLATION_RESULT case=all_known trial_index=(\d+) accuracy=([0-9.]+)")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trials", type=int, default=10)
    parser.add_argument("--base_seed", type=int, default=20260721)
    parser.add_argument("--model_name", default=DEFAULT_MODEL)
    parser.add_argument("--model_dir", default="saved_ckpt")
    parser.add_argument("--expanded_w_dir", default="expanded_weights")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--activation_curve_path", required=True)
    parser.add_argument("--activation_corner", required=True)
    parser.add_argument("--nonlinear_R_table", required=True)
    parser.add_argument("--test_bs", type=int, default=128)
    return parser.parse_args()


def build_command(args):
    seed = str(args.base_seed)
    return [
        sys.executable, "-u", "ode_inference.py",
        "--model_name", args.model_name,
        "--ckpt", "best",
        "--model_dir", args.model_dir,
        "--task", "cifar100",
        "--img_type", "scanGFI",
        "--test_bs", str(args.test_bs),
        "--pc_conv", "PCConvReLU6Noisy",
        "--ode_block", "TogglePulseODEXInitFFFB",
        "--ode_wrapper", "ToggleQATTester1State",
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
        "--R", "10000",
        "--R_max", "150000",
        "--C", "49e-15",
        "--k", "1000",
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
        "--odexinit_scaling_mode", "approx",
        "--conv_only", "true",
        "--test_expanded", "true",
        "--expanded_w_dir", args.expanded_w_dir,
        "--diff_mismatch", "true",
        "--nonlinear_R", "true",
        "--nonlinear_R_table", args.nonlinear_R_table,
        "--mul_mismatch_mode", "static_mismatch",
        "--enable_spin_variation", "true",
        "--sigma_spin", "0.10",
        "--spin_variation_seed", seed,
        "--enable_measured_activation", "true",
        "--activation_curve_path", args.activation_curve_path,
        "--activation_corner", args.activation_corner,
        "--activation_interpolation", "piecewise_linear",
        "--activation_spline_parameters", "10",
        "--activation_fit_constraint", "auto",
        "--activation_normalize_positive_endpoint", "false",
        "--compile_measured_activation", "false",
        "--enable_summing_current_noise", "false",
        "--summing_current_p", "18.5e-12",
        "--summing_noise_seed", seed,
        "--enable_coupler_noise", "true",
        "--coupler_noise_p", "0.6e-12",
        "--coupler_noise_seed", seed,
        "--enable_dtc_nonideality", "true",
        "--dtc_leading_edge_variation_std", "0.0",
        "--dtc_width_variation_std", "0.018",
        "--dtc_leading_edge_jitter_std", "0.005",
        "--dtc_falling_edge_jitter_std", "0.005",
        "--dtc_timing_seed", seed,
        "--return_init", "0",
        "--test_only", "false",
        "--ablation_single_case", "true",
        "--ablation_case_name", "all_known",
        "--hardware_seed", seed,
        "--data_seed", seed,
    ]


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "all_known.log"
    command = build_command(args)
    results = []

    print(
        "Running {}: {} trial(s)".format(
            args.activation_corner, args.n_trials),
        flush=True)
    with log_path.open("w") as log_file:
        log_file.write("COMMAND: {}\n".format(" ".join(command)))
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1)
        for line in process.stdout:
            log_file.write(line)
            match = RESULT_RE.search(line)
            if match:
                row = {
                    "case": "all_known",
                    "trial_index": int(match.group(1)),
                    "accuracy": float(match.group(2)),
                }
                results.append(row)
                print(
                    "  trial {}: {:.4f}%".format(
                        row["trial_index"], row["accuracy"]),
                    flush=True)
        return_code = process.wait()

    if return_code != 0:
        raise RuntimeError("Inference failed; see {}".format(log_path))
    if len(results) != args.n_trials:
        raise RuntimeError(
            "Expected {} results, found {} in {}".format(
                args.n_trials, len(results), log_path))

    csv_path = output_dir / "ablation_trials_partial.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("case", "trial_index", "accuracy"))
        writer.writeheader()
        writer.writerows(results)
    print("Corner results: {}".format(csv_path), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Evaluate a physically fine-tuned pre-activation feedforward CIFAR CNN."""

import argparse
import copy
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import baseline.cifar_resnet  # noqa: F401 -- registers models
from baseline.baseline_cifar_configs import (
    RGGB_DEFAULTS,
    build_model,
    get_baseline_config,
)
from feedforward_validation import FeedForwardCNNValidator
from bn_recalibration import recalibrate_batchnorm
from data_utils import MISMATCH_LEVELS_5b
from inference_utils import get_bn_calibration_data, get_test_data
from input_preprocessing import resolve_preprocessing
from measured_activation import (
    configure_feedforward_measured_activation,
    feedforward_measured_activation_factory,
)
from measured_pooling import configure_feedforward_measured_pooling
from physical_feedforward import (
    convert_wide_resnet_to_physical,
    iter_physical_wrappers,
    prepare_flattened_qat_for_pulse_inference,
)


def str2bool(value):
    return str(value).lower() in ("yes", "true", "t", "1")


def optional_int(value):
    return None if str(value).strip().lower() in {"", "none", "auto"} else int(value)


def optional_bool(value):
    if str(value).strip().lower() in {"", "none", "auto"}:
        return None
    return str2bool(value)


def index_list(value):
    if value is None or str(value).strip().lower() in {"", "none"}:
        return None
    return [int(item) for item in str(value).split(",")]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", required=True)
    p.add_argument("--wrn_depth", type=int, default=None)
    p.add_argument("--wrn_first_stage_channels", type=int, default=None)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset", choices=("cifar10", "cifar100"), default="cifar100")
    p.add_argument("--img_type", default="CiFAIR")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--expanded_weight_dir", default="./expanded_weights/feedforward")
    p.add_argument("--result_path", default="./results/feedforward_physical_eval")
    p.add_argument("--n_trials", type=int, default=1)
    p.add_argument("--physical_level", type=int, choices=(2, 3), default=3)
    p.add_argument("--R", type=float, default=50e3)
    p.add_argument("--C", type=float, default=500e-15)
    p.add_argument("--v_dd", type=float, default=0.1)
    p.add_argument("--one_over_q", type=float, default=1.0)
    p.add_argument("--w_bits", type=int, default=5)
    p.add_argument("--weight_quant_factor_bits", type=int, default=1)
    p.add_argument("--noise_level", type=float, default=0.0)
    p.add_argument("--mismatch_type", choices=("mul", "add"), default="mul")
    p.add_argument("--diff_mismatch", type=str2bool, default=False)
    p.add_argument("--toggle_timing_mode", choices=("derived", "fixed"), default="derived")
    p.add_argument("--toggle_y_time", type=float, default=5e-9)
    p.add_argument("--z_over_y_time", type=float, default=1.0)
    p.add_argument("--enob", type=optional_int, default=8)
    p.add_argument("--input_quant_bits", type=optional_int, default=None)
    p.add_argument("--center_student_input", type=optional_bool, default=None)
    p.add_argument("--enable_spin_variation", type=str2bool, default=True)
    p.add_argument("--sigma_spin", type=float, default=0.10)
    p.add_argument("--spin_variation_mean", type=float, default=1.0)
    p.add_argument("--spin_variation_seed", type=int, default=None)
    p.add_argument("--enable_summing_current_noise", type=str2bool, default=True)
    p.add_argument("--summing_current_p", type=float, default=0.6e-12)
    p.add_argument("--summing_noise_seed", type=int, default=None)
    p.add_argument("--enable_coupler_noise", type=str2bool, default=True)
    p.add_argument("--coupler_noise_p", type=float, default=0.6e-12)
    p.add_argument("--coupler_noise_seed", type=int, default=None)
    p.add_argument("--enable_slow_summing_current", type=str2bool, default=False)
    p.add_argument("--slow_summing_current", type=float, default=2.47e-9)
    p.add_argument("--enable_slow_coupler_noise", type=str2bool, default=False)
    p.add_argument("--slow_coupler_noise", type=float, default=2.47e-9)
    p.add_argument("--enable_dtc_nonideality", type=str2bool, default=True)
    p.add_argument("--dtc_leading_edge_variation_std", type=float, default=0.0)
    p.add_argument("--dtc_width_variation_mean", type=float, default=0.0)
    p.add_argument("--dtc_width_variation_std", type=float, default=0.018)
    p.add_argument("--dtc_leading_edge_jitter_std", type=float, default=0.005)
    p.add_argument("--dtc_falling_edge_jitter_std", type=float, default=0.005)
    p.add_argument("--dtc_timing_seed", type=int, default=None)
    p.add_argument("--enable_measured_activation", type=str2bool, default=True)
    p.add_argument("--activation_curve_path", default=str(
        PROJECT_ROOT / "hardware_data" / "relu_current_0p2uA_finer.csv"))
    p.add_argument("--activation_corner", default="TT")
    p.add_argument("--activation_curve_sharing", default="per_model",
                   choices=("per_model", "per_layer", "per_spin"))
    p.add_argument("--activation_curve_seed", type=int, default=None)
    p.add_argument("--activation_mc_curve_indices", type=index_list, default=None)
    p.add_argument("--activation_interpolation", default="piecewise_linear",
                   choices=("piecewise_linear", "cubic_bspline"))
    p.add_argument("--activation_spline_parameters", type=int, default=10)
    p.add_argument("--activation_fit_constraint", default="auto",
                   choices=("none", "nonnegative", "auto"))
    p.add_argument("--activation_normalize_positive_endpoint",
                   type=str2bool, default=False)
    p.add_argument("--compile_measured_activation", type=str2bool, default=False)
    p.add_argument("--enable_nonlinear_R", type=str2bool, default=True)
    p.add_argument("--nonlinear_R_table", default=str(
        PROJECT_ROOT / "hardware_data" / "mc_45_corners" / "coupler_monte"))
    p.add_argument("--nonlinear_R_mc_quantity", default="conductance",
                   choices=("conductance", "resistance"))
    p.add_argument("--nonlinear_R_curve_sharing", default="per_coupler")
    p.add_argument("--nonlinear_R_curve_sampling", default="empirical_with_replacement")
    p.add_argument("--nonlinear_R_curve_seed", type=int, default=None)
    p.add_argument("--nonlinear_R_curve_bank_indices", type=index_list, default=None)
    p.add_argument("--nonlinear_R_curve_edge_chunk_size", type=int, default=65536)
    p.add_argument("--mul_mismatch_mode", default="static_mismatch")
    p.add_argument("--enable_measured_pooling", type=str2bool, default=True)
    p.add_argument("--data_seed", type=int, default=None)
    p.add_argument("--bn_recalibrate", type=str2bool, default=False)
    p.add_argument("--bn_calibration_batch_size", type=int, default=128)
    p.add_argument("--bn_calibration_samples", type=optional_int, default=None)
    p.add_argument("--use_expanded_weights", type=str2bool, default=True)
    p.add_argument("--nonlinear_R_train_mode", default="none",
                   choices=("none", "exact_curve", "mean"))
    p.add_argument("--nonlinear_R_corner_range", default="all")
    return p.parse_args()


@torch.no_grad()
def evaluate(model, loader, device):
    total = correct = 0
    progress = tqdm(loader)
    for inputs, targets in progress:
        inputs = inputs.to(device)
        targets = targets.to(device)
        correct += model(inputs).argmax(1).eq(targets).sum().item()
        total += targets.numel()
        progress.set_description("acc={:.4f}%".format(100 * correct / total))
    return 100 * correct / total


def evaluate_once(args):
    if args.data_seed is not None:
        random.seed(args.data_seed)
        np.random.seed(args.data_seed)
        torch.manual_seed(args.data_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.data_seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg = get_baseline_config(
        args.model_name, case="custom_noresize")
    cfg["wrn_depth"] = args.wrn_depth
    cfg["wrn_first_stage_channels"] = args.wrn_first_stage_channels
    if args.img_type.lower() != "rgb":
        cfg.update(RGGB_DEFAULTS)
    model = build_model(
        args.model_name, cfg, 100 if args.dataset == "cifar100" else 10)
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("net", checkpoint)
    checkpoint_is_full_param = any(
        ".parametrizations.weight.original" in key for key in state_dict)
    model = convert_wide_resnet_to_physical(
        model, activation_factory=None,
        physical_level=args.physical_level, physical=True,
        qat=checkpoint_is_full_param,
        R=args.R, C=args.C, v_dd=args.v_dd,
        one_over_q=args.one_over_q, w_bits=args.w_bits,
        weight_quant_factor_bits=args.weight_quant_factor_bits,
        noise_level=(
            MISMATCH_LEVELS_5b
            if args.diff_mismatch and args.mismatch_type == "mul"
            else args.noise_level),
        mismatch_type=args.mismatch_type,
        toggle_timing_mode=args.toggle_timing_mode,
        toggle_y_time=args.toggle_y_time,
        z_over_y_time=args.z_over_y_time, enob=args.enob,
        enable_spin_variation=args.enable_spin_variation,
        sigma_spin=args.sigma_spin,
        spin_variation_mean=args.spin_variation_mean,
        spin_variation_seed=args.spin_variation_seed,
        enable_summing_current_noise=args.enable_summing_current_noise,
        summing_current_p=args.summing_current_p,
        summing_noise_seed=args.summing_noise_seed,
        enable_coupler_noise=args.enable_coupler_noise,
        coupler_noise_p=args.coupler_noise_p,
        coupler_noise_seed=args.coupler_noise_seed,
        enable_slow_summing_current=args.enable_slow_summing_current,
        slow_summing_current=args.slow_summing_current,
        enable_slow_coupler_noise=args.enable_slow_coupler_noise,
        slow_coupler_noise=args.slow_coupler_noise,
        enable_dtc_nonideality=args.enable_dtc_nonideality,
        dtc_leading_edge_variation_std=args.dtc_leading_edge_variation_std,
        dtc_width_variation_mean=args.dtc_width_variation_mean,
        dtc_width_variation_std=args.dtc_width_variation_std,
        dtc_leading_edge_jitter_std=args.dtc_leading_edge_jitter_std,
        dtc_falling_edge_jitter_std=args.dtc_falling_edge_jitter_std,
        dtc_timing_seed=args.dtc_timing_seed)
    model.load_state_dict(state_dict, strict=True)
    if not checkpoint_is_full_param:
        prepare_flattened_qat_for_pulse_inference(model)
    model.to(device).eval()
    if args.enable_measured_activation:
        activation_factory = feedforward_measured_activation_factory(
            args.activation_curve_path, args.v_dd,
            corner=args.activation_corner,
            curve_sharing=args.activation_curve_sharing,
            curve_seed=args.activation_curve_seed,
            normalize_positive_endpoint=(
                args.activation_normalize_positive_endpoint),
            interpolation=args.activation_interpolation,
            spline_parameters=args.activation_spline_parameters,
            fit_constraint=args.activation_fit_constraint,
            compile_evaluator=args.compile_measured_activation)
        configure_feedforward_measured_activation(model, activation_factory)
    nonlinear_R_package = None
    if args.enable_nonlinear_R and args.nonlinear_R_train_mode != "none":
        if args.physical_level != 2:
            raise ValueError(
                "nonlinear_R_train_mode is only valid for Level-2 evaluation.")
        wrappers = list(iter_physical_wrappers(model))
        nonlinear_R_package = wrappers[0].configure_nonlinear_R_training(
            args.nonlinear_R_table,
            mode=args.nonlinear_R_train_mode,
            corner_range=args.nonlinear_R_corner_range,
            quantity=args.nonlinear_R_mc_quantity,
            curve_seed=args.nonlinear_R_curve_seed)
        for wrapper in wrappers[1:]:
            wrapper.install_nonlinear_R_training_package(
                nonlinear_R_package)
    elif args.enable_nonlinear_R:
        wrappers = list(iter_physical_wrappers(model))
        nonlinear_R_package = wrappers[0].configure_nonlinear_R_inference(
            args.nonlinear_R_table,
            quantity=args.nonlinear_R_mc_quantity,
            curve_sharing=args.nonlinear_R_curve_sharing,
            curve_sampling=args.nonlinear_R_curve_sampling,
            curve_seed=args.nonlinear_R_curve_seed,
            curve_edge_chunk_size=args.nonlinear_R_curve_edge_chunk_size,
            curve_indices=args.nonlinear_R_curve_bank_indices,
            mul_mismatch_mode=args.mul_mismatch_mode)
        for wrapper in wrappers[1:]:
            wrapper.install_nonlinear_R_inference_package(
                nonlinear_R_package)
    if args.enable_measured_pooling:
        pooling_curve_path = args.nonlinear_R_table
        pooling_curve_gaussian = None
        if args.nonlinear_R_train_mode != "none":
            configure_feedforward_measured_pooling(
                model, enable_nonideality=True,
                curve_path=pooling_curve_path,
                quantity=args.nonlinear_R_mc_quantity,
                nominal_R=args.R,
                seed=args.nonlinear_R_curve_seed,
                training_curve_mode=args.nonlinear_R_train_mode,
                corner_range=args.nonlinear_R_corner_range)
        elif args.nonlinear_R_curve_sampling == "multivariate_gaussian":
            pooling_curve_path = None
            pooling_curve_gaussian = (
                None if nonlinear_R_package is None
                else nonlinear_R_package.get("curve_gaussian"))
            if pooling_curve_gaussian is None:
                raise ValueError(
                    "Gaussian measured pooling requires nonlinear-R "
                    "Gaussian curve sampling.")
        if args.nonlinear_R_train_mode == "none":
            configure_feedforward_measured_pooling(
                model, enable_nonideality=True,
                curve_path=pooling_curve_path,
                curve_gaussian=pooling_curve_gaussian,
                quantity=args.nonlinear_R_mc_quantity,
                nominal_R=args.R,
                seed=args.nonlinear_R_curve_seed,
                curve_indices=args.nonlinear_R_curve_bank_indices)

    model.to(device).eval()

    loader = get_test_data(
        test_bs=args.batch_size, img_type=args.img_type,
        task=args.dataset, shuffle=False,
        input_quant_bits=args.input_quant_bits,
        center_student_input=args.center_student_input)
    os.makedirs(args.result_path, exist_ok=True)
    if args.use_expanded_weights:
        FeedForwardCNNValidator(
            model, args.expanded_weight_dir, device, loader,
            args.result_path)
    if args.bn_recalibrate:
        calibration_loader = get_bn_calibration_data(
            bs=args.bn_calibration_batch_size,
            n_samples=args.bn_calibration_samples,
            img_type=args.img_type,
            task=args.dataset,
            input_quant_bits=args.input_quant_bits,
            center_student_input=args.center_student_input)
        stats = recalibrate_batchnorm(model, calibration_loader, device)
        print(
            "BN recalibration: batchnorms={num_batchnorms}, "
            "batches={num_batches}, samples={num_samples}".format(**stats),
            flush=True)
    if args.data_seed is not None:
        random.seed(args.data_seed)
        np.random.seed(args.data_seed)
        torch.manual_seed(args.data_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.data_seed)
    return evaluate(model, loader, device)


def trial_args(args, trial_index):
    current = copy.copy(args)
    for name in (
            "data_seed", "spin_variation_seed", "summing_noise_seed",
            "coupler_noise_seed", "dtc_timing_seed",
            "activation_curve_seed", "nonlinear_R_curve_seed"):
        value = getattr(current, name)
        if value is not None:
            setattr(current, name, int(value) + trial_index)
    if args.activation_mc_curve_indices is not None:
        current.activation_corner = "MC{}".format(
            args.activation_mc_curve_indices[trial_index] + 1)
    if args.n_trials > 1:
        current.result_path = str(Path(args.result_path) / str(trial_index))
    return current


def main():
    args = parse_args()
    if args.n_trials <= 0:
        raise ValueError("n_trials must be positive.")
    if (args.activation_mc_curve_indices is not None and
            len(args.activation_mc_curve_indices) != args.n_trials):
        raise ValueError(
            "activation_mc_curve_indices must contain one index per trial.")
    args.input_quant_bits, args.center_student_input = resolve_preprocessing(
        args.checkpoint, args.input_quant_bits, args.center_student_input)
    for index in range(args.n_trials):
        accuracy = evaluate_once(trial_args(args, index))
        print("trial {}: accuracy={:.4f}%".format(
            index, accuracy), flush=True)
        if args.n_trials == 1:
            print("accuracy={:.4f}%".format(accuracy), flush=True)


if __name__ == "__main__":
    main()

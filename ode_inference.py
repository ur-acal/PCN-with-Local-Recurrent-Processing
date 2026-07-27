import numpy as np
import torch
import torch.nn as nn
import torchvision
import os
import gc
import pickle
import argparse
import json
import logging
import random

from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm
from copy import deepcopy

from data_utils import MISMATCH_LEVELS_5b
from pc_conv import PCConvNoisy, PCConv, PartialTiedPCConv
from pc_model import PCNet, PCNetWithMiddleConv, PCN_CLASSES, PC_CONV_CLASS
from inference_utils import load_and_prepare_model, replace_transpose_conv, get_test_data, test_once
from ode_pc import make_ode_block, is_adaptive, ODEBLOCK_CLASSES, ODEWrapper_CLASSES, wrap_ode_block
from validation import Validator, snapshot_clean_mvm_mat_values, assert_mvm_mats_all_values_noised, MVMConv


import logging
log = logging.getLogger(__name__)
log.propagate = False

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run PCConvNoisy tests with custom noise parameters"
    )
    parser.add_argument("--model_dir",  type=str, required=True,
                        help="Directory containing the saved model checkpoint")
    parser.add_argument("--ckpt", type=str, default="best")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Identifier or filename of the model to load")
    parser.add_argument("--task", type=str, default="cifar10", choices=["cifar10", "cifar100"])
    parser.add_argument("--img_type", type=str, default="rgb")
    parser.add_argument("--test_bs", type=int, default=128)
    parser.add_argument("--shuffle_test", type=lambda v: v.lower() in ('yes', 'true', 't', '1'),
                        default=False, help="Shuffle test data.")
    parser.add_argument("--pc_conv", type=str, choices=list(PC_CONV_CLASS.keys())+[None],
                   default=None)
    parser.add_argument("--ode_block", type=str, choices=list(ODEBLOCK_CLASSES.keys()) + [None],
                        default=None)
    parser.add_argument("--ode_wrapper", type=lambda s: None if s.lower() in {"none", ""} else str(s),
                        choices=list(ODEWrapper_CLASSES.keys()) + [None], default=None)
    parser.add_argument("--return_init", type=str, default="0")
    parser.add_argument("--state_calib", type=str, required=False,
                        help="The calibration result of ode intermediate states for each layer")
    parser.add_argument("--R", type=float, default=1e5, help="Resistance")
    parser.add_argument("--R_max", type=lambda s: None if s.lower() in {"none", ""} else float(s),
                        default=None, help="Maximum meaningful Resistance")
    parser.add_argument("--nonlinear_R", type=lambda v: v.lower() in ('yes', 'true', 't', '1'),
                        default=False, help="R change with v_in or not")
    parser.add_argument(
        "--nonlinear_R_table",
        type=lambda s: None if s.lower() in {"none", ""} else str(s),
        default=None,
        help="Optional nonlinear-R CSV name or path; otherwise infer it from R and R_max.")
    parser.add_argument("--mul_mismatch_mode", type=str, default="scale_mismatch",
                        choices=["scale_mismatch", "static_mismatch"])
    parser.add_argument("--C", type=float, default=49e-15, help="Capacitance")
    parser.add_argument("--k", type=float, default=1e3, help="1-state hardware gain used by toggle physical models and wrappers")
    parser.add_argument("--v_dd", type=float, default=1.0, help="V_DD")
    parser.add_argument("--thermal_noise", type=lambda v: v.lower() in ('yes', 'true', 't', '1'),
                        default=True)
    parser.add_argument("--sde_noise_type", type=str, default="mul", choices=["mul", "add"],
                        help="Only useful when self.eps is set in the ODESolver class.")
    parser.add_argument("--mismatch_type", type=str, default="mul", choices=["mul", "add"],
                        help="Additive or multiplicative mismatch.")
    parser.add_argument("--noise_level_list", type=str, default="0.0,0.01,0.02,0.05")
    parser.add_argument("--sweep_eps", type=lambda v: v.lower() in ('yes', 'true', 't', '1'),
                        default=False)
    parser.add_argument("--patch_node", type=lambda s: None if s.lower() in {"none", ""} else int(s), default=None)
    parser.add_argument("--patch_stride", type=lambda s: None if s.lower() in {"none", ""} else int(s), default=None)
    parser.add_argument("--patch_cycle", type=lambda s: None if s.lower() in {"none", ""} else int(s), default=None)
    parser.add_argument("--patch_pad", type=lambda s: None if s.lower() in {"none", ""} else int(s), default=None)
    parser.add_argument("--fold_scalar", type=lambda s: None if s.lower() in {"none", ""} else int(s), default=None)
    parser.add_argument("--enob", type=lambda s: None if s.lower() in {"none", ""} else int(s),
                        default=None, help="The effective number of bits applied to the output spins.")
    parser.add_argument("--w_bits", type=int, default=8, help="weight quantized bits")
    parser.add_argument("--tie_cap", type=lambda v: v.lower() in ('yes', 'true', 't', '1'), default=False)
    parser.add_argument("--one_over_q", type=float, default=10, help="1/q")
    parser.add_argument("--toggle_n_cycles", type=lambda s: None if s.lower() in {"none", ""} else int(s),
                        default=None, help="Number of staged toggle cycles. Defaults to the layer n_steps/cls value.")
    parser.add_argument("--toggle_time_split", type=float, default=0.5,
                        help="Fraction of each t_end/N_cycles toggle cycle used by the z-stage.")
    parser.add_argument("--toggle_fast_path", type=lambda v: v.lower() in ('yes', 'true', 't', '1'),
                        default=True, help="Use direct constant-RHS updates inside Level 3 pulse slices.")
    parser.add_argument("--odexinit_scaling_mode", choices=("approx", "direct"),
                        default="approx",
                        help="Use legacy k-based or direct RC ODEXInit toggle timing.")
    parser.add_argument("--enable_spin_variation", type=lambda v: v.lower() in ('yes', 'true', 't', '1'),
                        default=False)
    parser.add_argument("--sigma_spin", type=float, default=0.10)
    parser.add_argument("--spin_variation_seed",
                        type=lambda s: None if s.lower() in {"none", ""} else int(s), default=None)
    parser.add_argument("--enable_measured_activation",
                        type=lambda v: v.lower() in ('yes', 'true', 't', '1'), default=False)
    parser.add_argument("--activation_curve_path", type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             "hardware_data", "relu_current_0p2uA_finer.csv"))
    parser.add_argument("--activation_corner", type=str, default="TT")
    parser.add_argument(
        "--activation_interpolation", type=str,
        choices=["cubic_bspline", "piecewise_linear"],
        default="piecewise_linear")
    parser.add_argument("--activation_spline_parameters", type=int, default=10)
    parser.add_argument("--activation_fit_constraint", type=str,
                        choices=["none", "nonnegative", "auto"], default="auto")
    parser.add_argument("--activation_normalize_positive_endpoint",
                        type=lambda v: v.lower() in ('yes', 'true', 't', '1'), default=False)
    parser.add_argument("--compile_measured_activation",
                        type=lambda v: v.lower() in ('yes', 'true', 't', '1'), default=False)
    parser.add_argument("--enable_summing_current_noise",
                        type=lambda v: v.lower() in ('yes', 'true', 't', '1'), default=False)
    parser.add_argument("--summing_current_p", type=float, default=18.5e-12)
    parser.add_argument("--summing_noise_seed",
                        type=lambda s: None if s.lower() in {"none", ""} else int(s), default=None)
    parser.add_argument("--enable_coupler_noise",
                        type=lambda v: v.lower() in ("yes", "true", "t", "1"), default=False)
    parser.add_argument("--coupler_noise_p", type=float, default=0.6e-12)
    parser.add_argument("--coupler_noise_seed",
                        type=lambda s: None if s.lower() in {"none", ""} else int(s), default=None)
    parser.add_argument("--enable_dtc_nonideality",
                        type=lambda v: v.lower() in ("yes", "true", "t", "1"), default=False)
    parser.add_argument("--dtc_leading_edge_variation_std", type=float, default=0.0)
    parser.add_argument("--dtc_width_variation_std", type=float, default=0.018)
    parser.add_argument("--dtc_leading_edge_jitter_std", type=float, default=0.005)
    parser.add_argument("--dtc_falling_edge_jitter_std", type=float, default=0.005)
    parser.add_argument("--dtc_timing_seed",
                        type=lambda s: None if s.lower() in {"none", ""} else int(s), default=None)
    parser.add_argument("--ablation_single_case",
                        type=lambda v: v.lower() in ('yes', 'true', 't', '1'), default=False)
    parser.add_argument("--ablation_case_name", type=str, default="")
    parser.add_argument("--hardware_seed",
                        type=lambda s: None if s.lower() in {"none", ""} else int(s), default=None)
    parser.add_argument("--data_seed", type=int, default=20260721)
    parser.add_argument("--w_quant_mode", type=str, default="min_max", help="min_max or perc")
    parser.add_argument("--w_perc", type=float, default=0.99999, help="percentile for quantization")
    parser.add_argument("--method", type=str, default="dopri5")
    parser.add_argument("--t_end", type=float, default=None, help="Stop time of the solver")
    parser.add_argument("--t_end_sf", type=float, default=1.0,
                        help="Scaling factor of the Stop time of the solver")
    parser.add_argument("--switch_period",
                        type=lambda s: None if s.lower() in {"none", ""} else float(s), default=7e-8,
                        help="Period of the switch validator. (Use switch_iter instead).")
    parser.add_argument("--switch_iter", type=int, default=2,
                        help="Number of iterations for the time interleaving method.")
    parser.add_argument("--i_leak", type=lambda s: None if s.lower() in {"none", ""} else float(s),
                        default=1e-9,
                        help="The leak current when the spin voltage is v_dd. Use with switched inference.")
    parser.add_argument("--n_steps", type=float, default=10, help="ODE solver number of steps")
    parser.add_argument("--tol", type=float, default=1e-3, help="ODE solver tolerance")
    parser.add_argument("--ts_scale", type=int, default=10,
                        help="Scale factor of time step; Applies to fixed grid methods")
    parser.add_argument("--d_start", type=float, default=0.1,
                        help="Difference between real t_end and position to start sweep")
    parser.add_argument("--d_end", type=float, default=0.1,
                        help="Difference between position to end sweep and real t_end")
    parser.add_argument("--n_sweep_left", type=int, default=5, help="Number of swept t_end")
    parser.add_argument("--n_sweep_right", type=int, default=5, help="Number of swept t_end")
    parser.add_argument("--conv_only", type=lambda v: v.lower() in ('yes', 'true', 't', '1'),
                        default=False)
    parser.add_argument("--mem_frac", type=float, default=1.0)
    parser.add_argument("--noisy_trials", type=int, default=20)
    parser.add_argument("--test_only_nl", type=float, default=0.15,
                        help="The noise level to use when test_only=True.")
    parser.add_argument("--analyze_mm", type=float, default=0.25, help="Mismatch level for analysis.")
    parser.add_argument("--analyze_mode", type=lambda s: None if s.lower() in {"none", ""} else str(s),
                        default=None, choices=[None, "none", "", "real", "clean"])
    parser.add_argument("--count_mac", type=lambda v: v.lower() in ('yes', 'true', 't', '1'),
                        default=False)
    parser.add_argument("--test_only", type=lambda v: v.lower() in ('yes','true','t','1'),
                        default=False)
    parser.add_argument("--hw_validate", type=lambda v: v.lower() in ('yes', 'true', 't', '1'),
                        default=False)
    parser.add_argument("--rec_full_traj", type=lambda v: v.lower() in ('yes', 'true', 't', '1'),
                        default=False)
    parser.add_argument("--test_expanded", type=lambda v: v.lower() in ('yes', 'true', 't', '1'),
                        default=False)
    parser.add_argument("--diff_mismatch", type=lambda v: v.lower() in ('yes', 'true', 't', '1'),
                        default=False)
    parser.add_argument("--pvt_to_origin", type=lambda v: v.lower() in ('yes', 'true', 't', '1'),
                        default=False,  help="Add weight non-ideality to the original or unrolled weights; works for test_expanded=True")
    parser.add_argument("--expanded_w_dir", type=str, default="./expanded_weights")
    parser.add_argument("--hw_val_path", type=str, default="./hw_validation_data")
    parser.add_argument("--hw_val_inp", type=lambda s: None if s.lower() in {"none", ""} else s, default="")
    parser.add_argument("--valid_samples", type=int, default=20,
                        help="Number of samples used for validation")
    parser.add_argument("--valid_select_layer", type=int, default=6,
                        help="Layer used to rank validation samples by trajectory change")
    return parser.parse_args()


def get_t_end(args):
    if "TEnd" in args.model_name:
        return float(args.model_name.split("TEnd")[0].split("_")[-1])
    elif "CLS" in args.model_name and "LRPC" in args.model_name:
        cycles = float(args.model_name.split("CLS")[0].split("_")[-1])
        lr_pc = float(args.model_name.split("LRPC")[0].split("_")[-1])
        return cycles * lr_pc
    else:
        assert args.t_end is not None
        return args.t_end


def _merge_metric_dicts(metric_dicts, weights=None):
    """
    Average metrics from multiple batches.
    Scalars and tensors are averaged.
    D_ratio is recomputed from averaged D_ode and D_one_step.
    """
    if weights is None:
        weights = [1.0 for _ in metric_dicts]

    total_w = sum(weights)
    out = {}

    keys = metric_dicts[0].keys()

    for key in keys:
        vals = [m[key] for m in metric_dicts]

        # Do not average metadata strings.
        if isinstance(vals[0], str):
            out[key] = vals[0]
            continue

        # Average tensors.
        if hasattr(vals[0], "shape"):
            acc = None
            for v, w in zip(vals, weights):
                acc = v * w if acc is None else acc + v * w
            out[key] = acc / total_w

        # Average scalars.
        elif isinstance(vals[0], (float, int)):
            out[key] = sum(v * w for v, w in zip(vals, weights)) / total_w

        else:
            out[key] = vals[0]

    # Recompute ratio from averaged values.
    if "D_ode" in out and "D_one_step" in out:
        out["D_ratio_ode_over_one_step"] = out["D_ode"] / max(out["D_one_step"], 1e-12)

    return out


def _print_metrics_for_block(layer_idx, metrics, skip_keys=None):
    print(f"\nLayer {layer_idx}")
    for key, value in metrics.items():
        if skip_keys is not None and key in skip_keys:
            continue
        print(f"{key}: {value}")


def run_ode_mismatch_analysis_multi_batch(model, test_loader, device="cuda", sigma=0.25, n_steps=50, n_seeds=10,
                                          mismatch_type="mul", add_scale="max_abs", layer_idx=None, n_batches=3,
                                          analyze_mode="clean"):
    """
    analyze_mode == "real" means that we are using propagated noisy output from previous layers as input to
    each layer.
    analyze_mode == "clean" means that we are using clean input for each layer to study the mismatch on each
    layer only.
    Structure:
        for batch in selected_batches:
            for mismatch_seed in seeds:
                analyze same batch under different mismatch seeds
        average over batch and seed

    The inner seed loop is inside ODEMismatchAnalyzer.
    This function adds the outer batch loop.
    """
    assert analyze_mode in ("real", "clean")
    analyzer = ODEMismatchAnalyzer(
        model=model, test_loader=test_loader, device=device, sigma=sigma, n_steps=n_steps,
        n_seeds=n_seeds, mismatch_type=mismatch_type, add_scale=add_scale,
    )

    batch_metrics = []
    batch_weights = []

    data_iter = iter(test_loader)

    for batch_idx in range(n_batches):
        images, labels = next(data_iter)
        images = images.to(device, non_blocking=True)

        batch_size = images.shape[0]
        batch_weights.append(float(batch_size))

        if layer_idx is None:
            if analyze_mode == "clean":
                metrics = analyzer.analyze_all_blocks(images=images)
            else:
                metrics = analyzer.analyze_all_blocks_realistic(images=images)
        else:
            if analyze_mode == "clean":
                metrics = analyzer.analyze_block(layer_idx=layer_idx, images=images)
            else:
                metrics = analyzer.analyze_block_realistic(layer_idx=layer_idx, images=images)

        batch_metrics.append(metrics)

    # Case 1: one selected layer
    if layer_idx is not None:
        merged = _merge_metric_dicts(batch_metrics, weights=batch_weights)
        _print_metrics_for_block(layer_idx, merged, skip_keys={"H", "C_from_H", "C_avg_cos",
                                                               "H_realistic", "C_from_H_realistic",
                                                               "C_avg_cos_realistic"})
        return merged

    # Case 2: all layers
    all_merged = {}

    for i in range(model.num_layers):
        layer_metrics_i = [bm[i] for bm in batch_metrics]
        all_merged[i] = _merge_metric_dicts(layer_metrics_i, weights=batch_weights)
        _print_metrics_for_block(i, all_merged[i], skip_keys={"H", "C_from_H", "C_avg_cos",
                                                              "H_realistic", "C_from_H_realistic",
                                                              "C_avg_cos_realistic"})

    return all_merged


def run_validation_data_gen(args, test_dataloader, ckpt_path, pc_conv, device):
    logging.info("----- Generating validation data for model: {} -----".format(args.model_name))
    t_end = get_t_end(args)
    unrolled_noise_level = 0.0
    noisy_params = {"noise_level": 0.0, "weight": None}
    num_layers = int(args.model_name.split("Layers")[0].split("_")[-1])
    return_init = list(map(lambda x: bool(int(x)), args.return_init.split(",")))
    return_init = return_init + [False] * (num_layers - len(return_init))
    if args.pvt_to_origin:
        # addinig non-ideality to the original weights and then expand (expanded values have the same non-ideality)
        # Todo: This needs unrolling at each forward pass, very inefficient. Thus is not used for now.
        noisy_params["noise_level"] = unrolled_noise_level
    ode_params = {"ode_block": ODEBLOCK_CLASSES[args.ode_block],
                  "return_init": return_init,
                  "t_end": t_end * args.t_end_sf, # possibly scaling the t_end to plot the spin voltage
                  "t_end_sf": args.t_end_sf,
                  "method": args.method,
                  "switch_period": args.switch_period, "n_iters": args.switch_iter, "i_leak": args.i_leak,
                  "tol": args.tol, "ts_scale": args.ts_scale, "n_steps": args.n_steps,
                  "sde_noise_type": args.sde_noise_type,
                  "patch_node": args.patch_node, "patch_stride": args.patch_stride, "patch_cycle": args.patch_cycle,
                  "patch_pad": args.patch_pad, "fold_scalar": args.fold_scalar,
                  "toggle_n_cycles": args.toggle_n_cycles, "toggle_time_split": args.toggle_time_split,
                  "toggle_fast_path": args.toggle_fast_path,
                  "odexinit_scaling_mode": args.odexinit_scaling_mode,
                  "enable_spin_variation": args.enable_spin_variation, "sigma_spin": args.sigma_spin,
                  "spin_variation_seed": args.spin_variation_seed,
                  "enable_summing_current_noise": args.enable_summing_current_noise,
                  "summing_current_p": args.summing_current_p,
                  "summing_noise_seed": args.summing_noise_seed,
                  "enable_coupler_noise": args.enable_coupler_noise,
                  "coupler_noise_p": args.coupler_noise_p,
                  "coupler_noise_seed": args.coupler_noise_seed,
                  "enable_dtc_nonideality": args.enable_dtc_nonideality,
                  "dtc_leading_edge_variation_std": args.dtc_leading_edge_variation_std,
                  "dtc_width_variation_std": args.dtc_width_variation_std,
                  "dtc_leading_edge_jitter_std": args.dtc_leading_edge_jitter_std,
                  "dtc_falling_edge_jitter_std": args.dtc_falling_edge_jitter_std,
                  "dtc_timing_seed": args.dtc_timing_seed}
    wrapper_params = {"ode_wrapper": ODEWrapper_CLASSES[args.ode_wrapper], "calib_path": args.state_calib,
                      "R": args.R, "R_max": args.R_max, "C": args.C, "k": args.k,
                      "v_dd": args.v_dd, "w_bits": args.w_bits,
                      "enob": args.enob, "w_quant_mode": args.w_quant_mode,
                      "tie_cap": args.tie_cap, "one_over_q": args.one_over_q,
                      "thermal_noise": args.thermal_noise, # Todo: Add thermal noise in validation?
                      "nonlinear_R": args.nonlinear_R, "nonlinear_R_table": args.nonlinear_R_table,
                      "mul_mismatch_mode": args.mul_mismatch_mode, # Only valid when wrapped with Validator
                      "enable_measured_activation": args.enable_measured_activation,
                      "activation_curve_path": args.activation_curve_path,
                      "activation_corner": args.activation_corner,
                      "activation_interpolation": args.activation_interpolation,
                      "activation_spline_parameters": args.activation_spline_parameters,
                      "activation_fit_constraint": args.activation_fit_constraint,
                      "activation_normalize_positive_endpoint": args.activation_normalize_positive_endpoint,
                      # offset_eps None means using Johnson noise
                      "offset_eps": None, "w_perc": args.w_perc} if args.ode_wrapper is not None else None
    saved_wrappers = {}
    net_ = load_and_prepare_model(model_path=ckpt_path, device=device, model_struct=PCNet,
                                  pc_conv_layer=pc_conv, data_parallel=False,
                                  noise_to_bn=True, noise_to_linear=True,
                                  fuse_bn=False, conv_only=args.conv_only, ode_params=ode_params,
                                  ode_wrapper_params=wrapper_params, wrappers=saved_wrappers,
                                  **noisy_params)
    logging.warning("Model input channels: {}".format(net_.ics))
    logging.warning("Model output channels: {}".format(net_.ocs))
    logging.warning("Model pooling layers: {}".format(net_.max_pool))

    valid_ins = Validator(model=net_,
                          expanded_weight_dir=os.path.join(args.expanded_w_dir, args.model_name,
                                                           args.ode_wrapper, "{}b".format(args.w_bits)),
                          device=device, test_dataloader=test_dataloader,
                          result_path=os.path.join(args.hw_val_path, args.model_name, "{}b".format(args.w_bits)),
                          wrapper=saved_wrappers["wrappers"],
                          record_full_traj=args.rec_full_traj, t_end_sf=args.t_end_sf)
    # valid_kwargs = dict(
    #     model=net_,
    #     expanded_weight_dir=os.path.join(args.expanded_w_dir, args.model_name, "{}b".format(args.w_bits)),
    #     device=device, test_dataloader=test_dataloader,
    #     result_path=os.path.join(args.hw_val_path, args.model_name, "{}b".format(args.w_bits)),
    #     wrapper=saved_wrappers["wrappers"],
    #     record_full_traj=args.rec_full_traj, t_end_sf=args.t_end_sf,
    #     # PixelBlockSwitchValidator-specific
    #     switch_period=args.switch_period,
    # )
    # valid_ins = VALIDATOR_CLASSES[args.validator](**valid_kwargs)
    logging.warning("Unroll or load expanded weights finished")
    if not args.pvt_to_origin:
        # adding non-ideality to unrolled weights
        for _blk in net_.PcConvs:
            _blk.noise_level = unrolled_noise_level
            _blk.add_noise()
    logging.warning("Test unrolled with noise level: {} (Mismatch added to unrolled weights)".format(unrolled_noise_level))
    if args.test_expanded:
        valid_ins.test_unroll()
    sample_inp = None
    if args.hw_val_inp:
        # saved as hw_validation_data/past_runs/args.model_name/xxx.pkl
        # xxx.pkl is passed in as hw_val_inp
        inp_file_name = os.path.join(args.hw_val_path, "past_runs", args.model_name, args.hw_val_inp)
        with open(inp_file_name, "rb") as fp:
            samples = pickle.load(fp)
            sample_inp = torch.from_numpy(samples["layer_0"]["inp"])
    valid_ins.gen_validate_data(wrappers=saved_wrappers["wrappers"], solver=args.method,
                                n_samples=args.valid_samples, sample_inp=sample_inp,
                                select_layer=args.valid_select_layer)


def print_quantization_level_ratios(net, q_hi):
    for layer_idx, block in enumerate(net.PcConvs):
        weights = [block.clean_params[name].detach().reshape(-1) for name in ("FFconv", "FBconv")]
        values = torch.cat(weights).abs()
        level_idx = torch.round(values * q_hi).to(torch.long).clamp_(0, q_hi)
        counts = torch.bincount(level_idx, minlength=q_hi + 1)
        ratios = counts.to(torch.float64) / counts.sum()
        ratios_str = ",".join(f"{ratio.item():.8f}" for ratio in ratios)
        print(f"QUANT_LEVEL_RATIOS layer={layer_idx} ratios={ratios_str}")


def run_test_only(args, test_dataloader, ckpt_path, pc_conv, device, return_net=False, noise_level=None):
    logging.info("----- Running one forward pass for model: {} -----".format(args.model_name))
    t_end = get_t_end(args)
    noisy_params = {"noise_level": 0.15 if noise_level is None else noise_level, "weight": None}
    num_layers = int(args.model_name.split("Layers")[0].split("_")[-1])
    return_init = list(map(lambda x: bool(int(x)), args.return_init.split(",")))
    return_init = return_init + [False] * (num_layers - len(return_init))
    if args.diff_mismatch:
        noisy_params["noise_level"] = MISMATCH_LEVELS_5b
    ode_params = {"ode_block": ODEBLOCK_CLASSES[args.ode_block], "t_end": t_end, "method": args.method,
                  "tol": args.tol, "ts_scale": args.ts_scale, "n_steps": args.n_steps, "return_init": return_init,
                  "switch_period": args.switch_period, "n_iters": args.switch_iter, "i_leak": args.i_leak,
                  "sde_noise_type": args.sde_noise_type, "mismatch_type": args.mismatch_type,
                  "patch_node": args.patch_node, "patch_stride": args.patch_stride, "patch_cycle": args.patch_cycle,
                  "patch_pad": args.patch_pad, "fold_scalar": args.fold_scalar,
                  "toggle_n_cycles": args.toggle_n_cycles, "toggle_time_split": args.toggle_time_split,
                  "toggle_fast_path": args.toggle_fast_path,
                  "odexinit_scaling_mode": args.odexinit_scaling_mode,
                  "enable_spin_variation": args.enable_spin_variation, "sigma_spin": args.sigma_spin,
                  "spin_variation_seed": args.spin_variation_seed,
                  "enable_summing_current_noise": args.enable_summing_current_noise,
                  "summing_current_p": args.summing_current_p,
                  "summing_noise_seed": args.summing_noise_seed,
                  "enable_coupler_noise": args.enable_coupler_noise,
                  "coupler_noise_p": args.coupler_noise_p,
                  "coupler_noise_seed": args.coupler_noise_seed,
                  "enable_dtc_nonideality": args.enable_dtc_nonideality,
                  "dtc_leading_edge_variation_std": args.dtc_leading_edge_variation_std,
                  "dtc_width_variation_std": args.dtc_width_variation_std,
                  "dtc_leading_edge_jitter_std": args.dtc_leading_edge_jitter_std,
                  "dtc_falling_edge_jitter_std": args.dtc_falling_edge_jitter_std,
                  "dtc_timing_seed": args.dtc_timing_seed}
    wrapper_params = {"ode_wrapper": ODEWrapper_CLASSES[args.ode_wrapper], "calib_path": args.state_calib,
                      "R": args.R, "R_max": args.R_max, "C": args.C, "k": args.k,
                      "v_dd": args.v_dd, "w_bits": args.w_bits,
                      "enob": args.enob, "tie_cap": args.tie_cap, "one_over_q": args.one_over_q,
                      "w_quant_mode": args.w_quant_mode, "thermal_noise": args.thermal_noise,
                      "nonlinear_R": args.nonlinear_R, "nonlinear_R_table": args.nonlinear_R_table,
                      "mul_mismatch_mode": args.mul_mismatch_mode, # Only valid when wrapped with Validator
                      "enable_measured_activation": args.enable_measured_activation,
                      "activation_curve_path": args.activation_curve_path,
                      "activation_corner": args.activation_corner,
                      "activation_interpolation": args.activation_interpolation,
                      "activation_spline_parameters": args.activation_spline_parameters,
                      "activation_fit_constraint": args.activation_fit_constraint,
                      "activation_normalize_positive_endpoint": args.activation_normalize_positive_endpoint,
                      # offset_eps None means using Johnson noise
                      "offset_eps": None, "w_perc": args.w_perc} if args.ode_wrapper is not None else None
    net_ = load_and_prepare_model(model_path=ckpt_path, device=device, model_struct=PCNet,
                                  pc_conv_layer=pc_conv, data_parallel=False,
                                  noise_to_bn=True, noise_to_linear=True,
                                  fuse_bn=False, conv_only=args.conv_only, ode_params=ode_params,
                                  ode_wrapper_params=wrapper_params,
                                  **noisy_params)
    logging.warning("Model Total number of parameters: {}M".format(sum(p.numel() for p in net_.parameters()) / 1e6))
    logging.warning("Model input channels: {}".format(net_.ics))
    logging.warning("Model output channels: {}".format(net_.ocs))
    logging.warning("Model pooling layers: {}".format(net_.max_pool))
    net_.eval()

    if return_net:
        return net_
    print_quantization_level_ratios(net_, 2 ** (args.w_bits - 1) - 1)
    # logging.warning("Model info: {}".format(torchinfo.summary(net_, input_size=(16,4,16,16))))
    # test_batch = next(iter(test_dataloader))[0].to(device)[:512]
    # _ = net_(test_batch)
    # _, predicted_raw = torch.max(_, 1)
    if not args.count_mac:
        test_once(net_, test_dataloader, device, args.model_name, img_type=args.img_type, task=args.task)
    # net_.recover_params()
    # test_once(net_, test_dataloader, device)

    logging.info("===== Inspecting the range of the weights =====")
    for _name, _p in net_.named_parameters():
        if "ff" in _name.lower() or args.conv_only:
            print("Name: {}, max: {}, min: {}, median: {}, mean: {}, sum's mean: {}".format(
                _name, _p.max(), _p.min(), _p.median(), _p.mean(), _p.reshape(_p.shape[0], -1).sum(-1).mean()))
        elif "fb" in _name.lower():
            print("Name: {}, max: {}, min: {}, median: {}, mean: {}, sum's mean: {}".format(
                _name, _p.max(), _p.min(), _p.median(), _p.mean(), _p.view(_p.shape[1], -1).sum(-1).mean()))
        else:
            print("Name: {}, max: {}, min: {}, median: {}, mean: {}".format(
                _name, _p.max(), _p.min(), _p.median(), _p.mean()))
    input_ranges = calibrate_input(model=net_, device=device, model_name=args.model_name,
                                   calib_bs=256,
                                   calib_samples=256, percentile=0.995,
                                   symmetric=False, save_to=None, img_type=args.img_type)
    for _t, _range in input_ranges.items():
        print("Type: {}".format(_t))
        print(_range)
        print("================================")

    if args.count_mac:
        with torch.no_grad():
            macs, params = profile(net_, inputs=(next(iter(test_dataloader))[0].to(device)[:1],), verbose=False)
            logging.warning("Model Params: {}, Solver: {}, N steps: {}, MAC Count: {}".format(
                clever_format([params], "%.1f"), args.method, args.n_steps, clever_format([macs], "%.1f")
            ))


def run_ode_inference():
    args = parse_args()
    if args.test_only:
        # set level in the very beginning before calling logging.warning, otherwise the line below will not work
        logging.basicConfig(level=logging.INFO)

    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(args.mem_frac, device=0)

    # Get pc_conv_layer to use
    pc_conv = PC_CONV_CLASS.get(args.pc_conv, PCConvNoisy)
    logging.warning("----- Using PC Conv layer: {} -----".format(pc_conv.__name__))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    test_dataloader = get_test_data(test_bs=args.test_bs, img_type=args.img_type, task=args.task)
    ckpt_path = os.path.join(args.model_dir, args.model_name, args.model_name + "_{}_ckpt.pth".format(args.ckpt))

    with torch.no_grad():
        if args.test_only:
            run_test_only(args, test_dataloader, ckpt_path, pc_conv, device, noise_level=args.test_only_nl)
            exit(0)
        elif args.hw_validate:
            run_validation_data_gen(args,
                                    get_test_data(
                                        test_bs=args.test_bs, img_type=args.img_type,
                                        task=args.task, shuffle=args.shuffle_test),
                                    ckpt_path, pc_conv, device)
            exit(0)
        elif args.analyze_mode is not None:
            net_ = run_test_only(args, test_dataloader, ckpt_path, pc_conv, device, True, 0.0)
            run_ode_mismatch_analysis_multi_batch(model=net_, test_loader=test_dataloader,
                                                  device=device, sigma=args.analyze_mm, n_steps=args.n_steps,
                                                  n_seeds=args.noisy_trials, mismatch_type=args.mismatch_type,
                                                  n_batches=args.noisy_trials, analyze_mode=args.analyze_mode)
            exit(0)

    # noise_level_list_ = [0, 0.05, 0.1, 0.15, .20, .25, .30, .35, .40] # mul
    # noise_level_list_ = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.15, 0.2] # add
    if args.sweep_eps:
        # offset_eps None means using Johnson noise
        offset_eps_list = [0.2, 0.35, 0.5, 0.65, 0.8] if args.sde_noise_type == "mul" else [None, 0.05, 0.1, 0.15, 0.2]
    else:
        offset_eps_list = [None]
    noise_level_list_ = [float(x.strip()) for x in args.noise_level_list.split(",") if x.strip()]
    noisy_trials = args.noisy_trials
    if args.diff_mismatch and args.mismatch_type == "mul":
        noise_level_list_ = [0, MISMATCH_LEVELS_5b]
    if args.test_expanded:
        # Too time-consuming, only run one mismatch level.
        # In this case we overide the input args.noise_level_list
        if not args.diff_mismatch:
            noise_level_list_ = [0, 0.15, 0.25]
    if args.ablation_single_case:
        noise_level_list_ = [MISMATCH_LEVELS_5b if args.diff_mismatch else 0.0]
    gt_t_end = get_t_end(args)

    # Get t_end_list for experiments
    # For single t_end, set d_start=0, d_end=1, n_sweep=1
    sweep_start, sweep_end = gt_t_end - args.d_start, gt_t_end + args.d_end
    t_end_before, t_end_after = torch.tensor([]), torch.tensor([])
    if args.d_start > 0 and args.n_sweep_left > 0:
        t_end_before = torch.arange(sweep_start, gt_t_end, args.d_start / args.n_sweep_left, dtype=torch.float32)
        t_end_before = t_end_before[t_end_before < gt_t_end]
    if args.d_end > 0 and args.n_sweep_right > 0:
        t_end_after = torch.arange(gt_t_end, sweep_end, args.d_end / args.n_sweep_right, dtype=torch.float32)
    t_end_list = torch.cat([t_end_before, t_end_after]).tolist()

    logging.warning("Running ODE pcn inference, method: {}, tol: {}".format(args.method, args.tol))
    num_layers = int(args.model_name.split("Layers")[0].split("_")[-1])
    return_init = list(map(lambda x: bool(int(x)), args.return_init.split(",")))
    return_init = return_init + [False] * (num_layers - len(return_init))
    acc_dict = {}
    for t_end in t_end_list:
        logging.warning("Current t_end: {}, ground truth t_end: {}".format(t_end, gt_t_end))
        ode_params = {"ode_block": ODEBLOCK_CLASSES[args.ode_block], "t_end": t_end, "method": args.method,
                      "tol": args.tol, "ts_scale": args.ts_scale, "n_steps": args.n_steps, "return_init": return_init,
                      "switch_period": args.switch_period, "n_iters": args.switch_iter, "i_leak": args.i_leak,
                      "sde_noise_type": args.sde_noise_type, "mismatch_type": args.mismatch_type,
                      "patch_node": args.patch_node, "patch_stride": args.patch_stride, "patch_cycle": args.patch_cycle,
                      "patch_pad": args.patch_pad, "fold_scalar": args.fold_scalar,
                      "toggle_n_cycles": args.toggle_n_cycles, "toggle_time_split": args.toggle_time_split,
                      "toggle_fast_path": args.toggle_fast_path,
                      "odexinit_scaling_mode": args.odexinit_scaling_mode,
                      "enable_spin_variation": args.enable_spin_variation, "sigma_spin": args.sigma_spin,
                      "spin_variation_seed": args.spin_variation_seed,
                      "enable_summing_current_noise": args.enable_summing_current_noise,
                      "summing_current_p": args.summing_current_p,
                      "summing_noise_seed": args.summing_noise_seed,
                      "enable_coupler_noise": args.enable_coupler_noise,
                      "coupler_noise_p": args.coupler_noise_p,
                      "coupler_noise_seed": args.coupler_noise_seed,
                  "enable_dtc_nonideality": args.enable_dtc_nonideality,
                  "dtc_leading_edge_variation_std": args.dtc_leading_edge_variation_std,
                  "dtc_width_variation_std": args.dtc_width_variation_std,
                  "dtc_leading_edge_jitter_std": args.dtc_leading_edge_jitter_std,
                  "dtc_falling_edge_jitter_std": args.dtc_falling_edge_jitter_std,
                  "dtc_timing_seed": args.dtc_timing_seed}
        wrapper_params = {"ode_wrapper": ODEWrapper_CLASSES[args.ode_wrapper], "calib_path": args.state_calib,
                          "R": args.R, "R_max": args.R_max, "C": args.C, "k": args.k,
                          "v_dd": args.v_dd, "w_bits": args.w_bits,
                          "enob": args.enob, "tie_cap": args.tie_cap, "one_over_q": args.one_over_q,
                          "nonlinear_R": args.nonlinear_R, "nonlinear_R_table": args.nonlinear_R_table,
                          "mul_mismatch_mode": args.mul_mismatch_mode, # Only valid when wrapped with Validator
                          "enable_measured_activation": args.enable_measured_activation,
                          "activation_curve_path": args.activation_curve_path,
                          "activation_corner": args.activation_corner,
                          "activation_interpolation": args.activation_interpolation,
                          "activation_spline_parameters": args.activation_spline_parameters,
                          "activation_fit_constraint": args.activation_fit_constraint,
                          "activation_normalize_positive_endpoint": args.activation_normalize_positive_endpoint,
                          "compile_measured_activation": args.compile_measured_activation,
                          "w_quant_mode": args.w_quant_mode, "thermal_noise": args.thermal_noise,
                          "w_perc": args.w_perc} if args.ode_wrapper is not None else None
        noise_acc_spec_all = {}
        max_real_t, min_real_t, real_t_end = t_end, t_end, t_end
        for offset_eps_ in offset_eps_list:
            wrapper_params.update({"offset_eps": offset_eps_}) if wrapper_params is not None else None
            noise_acc_spec = {}
            for noise_level in noise_level_list_:
                trials = noisy_trials
                if not args.ablation_single_case and not isinstance(noise_level, dict):
                    trials = noisy_trials if (noise_level > 0 or args.thermal_noise or
                                             args.enable_spin_variation or args.enable_summing_current_noise or
                                             args.enable_coupler_noise or args.enable_dtc_nonideality) else 1
                    trials = min(5, trials) if noise_level <= 0 and args.thermal_noise and args.test_expanded else trials
                acc_list = []
                for t in range(trials):
                    noisy_params = {"noise_level": noise_level, "weight": None}
                    if args.test_expanded:
                        # Add non-ideality to expanded weights
                        noisy_params["noise_level"] = 0.0
                    if args.hardware_seed is not None:
                        trial_seed = args.hardware_seed + t
                        random.seed(trial_seed)
                        np.random.seed(trial_seed)
                        torch.manual_seed(trial_seed)
                        if torch.cuda.is_available():
                            torch.cuda.manual_seed_all(trial_seed)
                    if args.spin_variation_seed is not None:
                        ode_params["spin_variation_seed"] = args.spin_variation_seed + t
                    if args.summing_noise_seed is not None:
                        ode_params["summing_noise_seed"] = args.summing_noise_seed + t
                    if args.coupler_noise_seed is not None:
                        ode_params["coupler_noise_seed"] = args.coupler_noise_seed + t
                    if args.dtc_timing_seed is not None:
                        ode_params["dtc_timing_seed"] = args.dtc_timing_seed + t
                    with torch.no_grad():
                        saved_wrappers = {}
                        net_ = load_and_prepare_model(model_path=ckpt_path, device=device, model_struct=PCNet,
                                                      pc_conv_layer=pc_conv, data_parallel=False,
                                                      noise_to_bn=True, noise_to_linear=True,
                                                      fuse_bn=False, conv_only=args.conv_only, ode_params=ode_params,
                                                      ode_wrapper_params=wrapper_params, wrappers=saved_wrappers,
                                                      **noisy_params)
                        if args.test_expanded:
                            # Use validator to expand the weights of the model
                            valid_ins = Validator(model=net_,
                                                  expanded_weight_dir=os.path.join(args.expanded_w_dir, args.model_name,
                                                                                   args.ode_wrapper,
                                                                                   "{}b".format(args.w_bits)),
                                                  device=device, test_dataloader=test_dataloader,
                                                  result_path=os.path.join(args.hw_val_path, args.model_name,
                                                                           "{}b".format(args.w_bits)),
                                                  wrapper=saved_wrappers["wrappers"])
                            logging.warning("Unroll or load expanded weights finished")
                            clean_params = {_name: _p.clone() for _name, _p in net_.named_parameters()}
                            clean_buffs = {_name: _buf.clone() for _name, _buf in net_.named_buffers()}
                            clean_vals = None
                            if not args.nonlinear_R:
                                clean_vals = snapshot_clean_mvm_mat_values(net_) # not part of named_parameters
                            net_ = valid_ins.model
                            # Add noise after wrapped with Validator
                            net_.noise_level = noise_level
                            for _blk in net_.PcConvs:
                                _blk.noise_level = noise_level
                                # _blk.add_noise()
                            # Use the existing model-wide mismatch path for both regular
                            # inference and ablation cases. This also applies mismatch to
                            # the model's non-PC convolutions and final linear layer.
                            net_.add_noise(noise_to_bn=True, noise_to_linear=True)
                            if isinstance(noise_level, dict) or noise_level > 0.0:
                                for _name, _p in net_.named_parameters():
                                    assert torch.allclose(_p, torch.zeros_like(_p)) or not torch.allclose(_p, clean_params[
                                        _name]), "{} noise not added".format(_name)
                                for _name, _buf in net_.named_buffers():
                                    if _name.endswith(('running_mean', 'running_var')):
                                        assert torch.allclose(_buf, torch.zeros_like(_buf)) or not torch.allclose(_buf,
                                                                                                                  clean_buffs[
                                                                                                                  _name])
                                if not args.nonlinear_R:
                                    # Now we keep a fixed mismatch matrix in the MVMConv, add that
                                    # before we do the sparse mm. The original weights are clean.
                                    assert_mvm_mats_all_values_noised(net_, clean_vals)
                    real_t_list = torch.tensor([_.integration_time[-1].cpu() for _ in net_.PcConvs])
                    max_real_t, min_real_t, avg_real_t = real_t_list.max(), real_t_list.min(), real_t_list.mean()
                    max_real_t, min_real_t, avg_real_t = f"{max_real_t.item():.4g}", f"{min_real_t.item():.4g}", f"{avg_real_t.item():.4g}"
                    real_t_end = avg_real_t
                    net_.eval()
                    total = 0
                    correct = 0

                    if args.ablation_single_case:
                        # Keep the evaluated dataset identical while hardware seeds advance.
                        random.seed(args.data_seed)
                        np.random.seed(args.data_seed)
                        torch.manual_seed(args.data_seed)
                        if torch.cuda.is_available():
                            torch.cuda.manual_seed_all(args.data_seed)
                    pbar = tqdm(enumerate(test_dataloader), total=len(test_dataloader), disable=False)
                    for batch_idx, (inputs, targets) in pbar:
                        inputs, targets = inputs.to(device), targets.to(device)
                        with torch.no_grad():
                            output_tensor = net_(inputs)
                            if torch.isnan(output_tensor).any():
                                logging.warning("=====> Output tensor contains nan values. <=====")

                        # Get the predicted class
                        _, predicted = torch.max(output_tensor, 1)
                        total += targets.size(0)
                        correct += (predicted == targets).sum().item()

                        running_acc = 100.0 * correct / total
                        pbar.set_postfix(acc=f"{running_acc:.2f}%")

                    # Calculate the accuracy
                    accuracy = 100 * correct / total
                    acc_list.append(accuracy)
                    if args.ablation_single_case:
                        print("ABLATION_RESULT case={} trial_index={} accuracy={:.8f}".format(
                            args.ablation_case_name, t, accuracy), flush=True)
                    if not args.ablation_single_case:
                        log.warning(
                            f'Test Accuracy: {accuracy:.2f}%'
                        )

                    # Cross trial clean up code
                    if args.test_expanded:
                        for _m in net_.modules():
                            if isinstance(_m, MVMConv):
                                _m.code_idx_mat = None
                                _m.mismatch_mat = None
                                _m.mat = None
                                _m.v_grid = None
                                _m.R_codes = None
                                _m.R_left = None
                                _m.R_slope = None

                        for _w in saved_wrappers["wrappers"]:
                            _blk = getattr(_w, "ode_block", None)
                            if _blk is not None:
                                if hasattr(_w, "original_forward"):
                                    _blk.forward = _w.original_forward
                                if hasattr(_w, "original_make_fn"):
                                    _blk._make_ode_fn = _w.original_make_fn
                                if hasattr(_w, "original_init_y"):
                                    _blk.init_y = _w.original_init_y
                                if hasattr(_w, "original_make_z_fn") and hasattr(_blk, "_make_z_ode_fn"):
                                    _blk._make_z_ode_fn = _w.original_make_z_fn

                        valid_ins.model = None
                        valid_ins.wrappers = None
                        valid_ins.dataloader = None

                    del net_, saved_wrappers
                    if args.test_expanded:
                        del valid_ins, clean_params, clean_buffs, clean_vals

                    try:
                        del inputs, targets, output_tensor, predicted, pbar, real_t_list
                    except NameError:
                        pass

                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                avg_acc = sum(acc_list) / len(acc_list)
                _nl_key = np.mean(list(noise_level.values())).round(3) if isinstance(noise_level, dict) else noise_level
                noise_acc_spec[_nl_key] = acc_list
                log.warning("Average test acc over {} trials is {}".format(trials, avg_acc))
            noise_acc_spec_all[offset_eps_ if offset_eps_ is not None else "Johnson"] = noise_acc_spec
        if not args.ablation_single_case:
            log.warning("-------- Final Result ODEBlock with t_end: {}, real_t_end: {} --------".format(
                t_end, real_t_end))
            log.warning("-------- Model name: {} --------".format(args.model_name))
            if args.ode_wrapper is not None:
                log.warning("wrapper params: {}".format(wrapper_params))
            for _eps, _noise_acc_spec in noise_acc_spec_all.items():
                for _nl, _acc in _noise_acc_spec.items():
                    log.warning("t_end: {}, real_t_end: {}, Acc:{:.2f}±{:.2f}%".format(
                        t_end, real_t_end, sum(_acc) / len(_acc), np.std(_acc)))

        acc_dict[real_t_end] = {"noise_acc_spec": noise_acc_spec_all, "t": (t_end, real_t_end, min_real_t, max_real_t)}

    if args.ablation_single_case:
        return

    # save noise acc spec to a pkl
    if args.ode_wrapper is None:
        spec_path = os.path.join(
            "logs/ode_noisy_acc", "TEnd{}_{}_{}_{}_{}NoiseLevel.pkl".format(
                str(round(t_end_list[0], 2)).replace(".", "p"),
                str(round(t_end_list[-1], 2)).replace(".", "p"),
                args.method, args.model_name, len(noise_level_list_)))
    else:
        spec_path = os.path.join(
            "logs/ode_noisy_acc", "TEnd{}_{}_{}Cap_{}_{}_{}{}b_{}NL.pkl".format(
                str(round(t_end_list[0], 2)).replace(".", "p"),
                str(round(t_end_list[-1], 2)).replace(".", "p"),
                args.C, args.method, args.model_name[:170], args.ode_wrapper, args.w_bits,
                len(noise_level_list_)))
    with open(spec_path, "wb") as fp:
        pickle.dump(acc_dict, fp)
    log.warning("-------- ODEBlock Noisy experiment finished, spec saved to {} --------".format(spec_path))


if __name__ == "__main__":
    run_ode_inference()

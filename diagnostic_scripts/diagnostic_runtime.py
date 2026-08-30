"""Build one runtime trial using the up-to-date 45-corner configuration."""

from contextlib import contextmanager
from dataclasses import dataclass
import os
import random
import sys

import numpy as np
import torch

from diagnostic_config import (
    C_FARAD,
    COUPLER_SOURCE,
    REPO_ROOT,
    R_OHM,
    V_DD,
    WEIGHT_QUANT_FACTOR_BITS,
    model_paths,
)

from data_utils import MC45CornerData
from inference_utils import get_test_data, load_and_prepare_model
from measured_pooling import configure_measured_pooling
from ode_pc import ODEBLOCK_CLASSES, ODEWrapper_CLASSES
from pc_model import PCNet, PC_CONV_CLASS
from validation import Validator
from scripts import run_toggle_nonideality_ablation as ablation
import ode_inference


DEFAULT_CORNER = "TT_V1_T1"
DEFAULT_SEED = 20260723


@contextmanager
def _temporary_argv(arguments):
    original = sys.argv
    sys.argv = [original[0]] + list(arguments)
    try:
        yield
    finally:
        sys.argv = original


def _parse_ablation_args(model_name, model_root, batch_size, trial_index, seed,
                         nonideality_profile="all_on"):
    if nonideality_profile not in {"all_on", "clean"}:
        raise ValueError("nonideality_profile must be all_on or clean")
    enable_nonidealities = str(nonideality_profile == "all_on").lower()
    paths = model_paths(model_name=model_name, model_root=model_root)
    arguments = [
        "--full_45_corner_test", "true",
        "--n_trials", str(max(5, trial_index + 1)),
        "--base_seed", str(seed),
        "--model_name", paths["model_name"],
        "--model_dir", str(paths["model_root"]),
        "--ckpt", "best",
        "--test_bs", str(batch_size),
        "--toggle_level", "3",
        "--odexinit_scaling_mode", "direct",
        "--weight_quant_factor_bits", str(WEIGHT_QUANT_FACTOR_BITS),
        "--enob", "none",
        "--R", str(R_OHM),
        "--R_max", "none",
        "--C", str(C_FARAD),
        "--full_45_corner_C", str(C_FARAD),
        "--v_dd", str(V_DD),
        "--mc_coupler_nonlinear_variation_source",
        COUPLER_SOURCE.name,
        "--mc_coupler_nonlinear_variation_quantity", "conductance",
        "--mc_coupler_nominal_R", str(R_OHM),
        "--nonlinear_R_curve_sharing", "per_coupler",
        "--nonlinear_R_curve_sampling", "empirical_with_replacement",
        "--full_45_corner_enable_spin_variation", enable_nonidealities,
        "--full_45_corner_enable_measured_activation", enable_nonidealities,
        "--full_45_corner_enable_measured_pooling", enable_nonidealities,
        "--full_45_corner_enable_nonlinear_R", enable_nonidealities,
        "--full_45_corner_enable_diff_mismatch", "false",
        "--full_45_corner_enable_summing_current_noise", enable_nonidealities,
        "--full_45_corner_enable_coupler_noise", enable_nonidealities,
        "--full_45_corner_enable_dtc_nonideality", enable_nonidealities,
        "--activation_curve_sharing", "per_spin",
        "--full_45_corner_summing_current_p", "0.6e-12",
        "--coupler_noise_p", "0.6e-12",
        "--expanded_w_dir", str(REPO_ROOT / "expanded_weights"),
        "--output_dir", str(REPO_ROOT / "results" / "diagnostic_runtime"),
    ]
    with _temporary_argv(arguments):
        return ablation.parse_args()


def reference_corner_command(model_name=None, model_root=None,
                             corner=DEFAULT_CORNER, batch_size=128, trial_index=0,
                             seed=DEFAULT_SEED,
                             nonideality_profile="all_on"):
    """Return the exact ode_inference arguments for one reference trial."""
    ablation_args = _parse_ablation_args(
        model_name, model_root, batch_size, trial_index, seed,
        nonideality_profile=nonideality_profile)
    catalog = MC45CornerData(
        ablation_args.mc_45_corner_dir,
        spin_variation_source=ablation_args.mc_spin_variation_source,
        dtc_pulse_width_variation_source=(
            ablation_args.mc_dtc_pulse_width_variation_source),
        relu_monte_carlo_source=ablation_args.mc_relu_monte_carlo_source,
        coupler_nonlinear_variation_source=(
            ablation_args.mc_coupler_nonlinear_variation_source),
        coupler_nonlinear_variation_quantity=(
            ablation_args.mc_coupler_nonlinear_variation_quantity),
        coupler_nominal_R=ablation_args.mc_coupler_nominal_R)
    corner = str(corner).strip().upper()
    corner_index = next(
        index for index, entry in enumerate(catalog.corners)
        if entry["id"] == corner)
    command, trial_config = ablation.build_corner_command(
        ablation_args, catalog.corners[corner_index], corner_index)
    with _temporary_argv(command[3:]):
        inference_args = ode_inference.parse_args()
    return inference_args, trial_config, command


def _set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _model_parameters(args, trial_index):
    t_end = ode_inference.get_t_end(args)
    num_layers = int(args.model_name.split("Layers")[0].split("_")[-1])
    return_init = [bool(int(value)) for value in args.return_init.split(",")]
    return_init += [False] * (num_layers - len(return_init))

    ode_params = {
        "ode_block": ODEBLOCK_CLASSES[args.ode_block],
        "t_end": t_end,
        "method": args.method,
        "tol": args.tol,
        "ts_scale": args.ts_scale,
        "n_steps": args.n_steps,
        "return_init": return_init,
        "switch_period": args.switch_period,
        "n_iters": args.switch_iter,
        "i_leak": args.i_leak,
        "sde_noise_type": args.sde_noise_type,
        "mismatch_type": args.mismatch_type,
        "patch_node": args.patch_node,
        "patch_stride": args.patch_stride,
        "patch_cycle": args.patch_cycle,
        "patch_pad": args.patch_pad,
        "fold_scalar": args.fold_scalar,
        "toggle_n_cycles": args.toggle_n_cycles,
        "toggle_time_split": args.toggle_time_split,
        "toggle_fast_path": args.toggle_fast_path,
        "odexinit_scaling_mode": args.odexinit_scaling_mode,
        "enable_spin_variation": args.enable_spin_variation,
        "sigma_spin": args.sigma_spin,
        "spin_variation_mean": args.spin_variation_mean,
        "spin_variation_seed": args.spin_variation_seed + trial_index,
        "enable_summing_current_noise": args.enable_summing_current_noise,
        "summing_current_p": args.summing_current_p,
        "summing_noise_seed": args.summing_noise_seed + trial_index,
        "enable_coupler_noise": args.enable_coupler_noise,
        "coupler_noise_p": args.coupler_noise_p,
        "coupler_noise_seed": args.coupler_noise_seed + trial_index,
        "enable_dtc_nonideality": args.enable_dtc_nonideality,
        "dtc_leading_edge_variation_std": (
            args.dtc_leading_edge_variation_std),
        "dtc_width_variation_mean": args.dtc_width_variation_mean,
        "dtc_width_variation_std": args.dtc_width_variation_std,
        "dtc_leading_edge_jitter_std": args.dtc_leading_edge_jitter_std,
        "dtc_falling_edge_jitter_std": args.dtc_falling_edge_jitter_std,
        "dtc_timing_seed": args.dtc_timing_seed + trial_index,
    }
    curve_indices = ode_inference._parse_index_list(
        args.nonlinear_R_curve_bank_indices)
    activation_indices = ode_inference._parse_index_list(
        args.activation_mc_curve_indices)
    wrapper_params = {
        "ode_wrapper": ODEWrapper_CLASSES[args.ode_wrapper],
        "calib_path": args.state_calib,
        "R": args.R,
        "R_max": args.R_max,
        "C": args.C,
        "k": args.k,
        "v_dd": args.v_dd,
        "w_bits": args.w_bits,
        "weight_quant_factor_bits": args.weight_quant_factor_bits,
        "enob": args.enob,
        "tie_cap": args.tie_cap,
        "one_over_q": args.one_over_q,
        "nonlinear_R": args.nonlinear_R,
        "nonlinear_R_table": args.nonlinear_R_table,
        "nonlinear_R_mc_curve_index": None,
        "nonlinear_R_mc_quantity": args.nonlinear_R_mc_quantity,
        "nonlinear_R_curve_sharing": args.nonlinear_R_curve_sharing,
        "nonlinear_R_curve_sampling": args.nonlinear_R_curve_sampling,
        "nonlinear_R_curve_bank_indices": curve_indices,
        "nonlinear_R_curve_seed": args.nonlinear_R_curve_seed + trial_index,
        "nonlinear_R_curve_edge_chunk_size": (
            args.nonlinear_R_curve_edge_chunk_size),
        "nonlinear_R_train_mode": args.nonlinear_R_train_mode,
        "nonlinear_R_corner_range": args.nonlinear_R_corner_range,
        "mul_mismatch_mode": args.mul_mismatch_mode,
        "enable_measured_activation": args.enable_measured_activation,
        "activation_curve_path": args.activation_curve_path,
        "activation_corner": (
            "MC{}".format(activation_indices[trial_index] + 1)
            if activation_indices is not None else args.activation_corner),
        "activation_curve_sharing": args.activation_curve_sharing,
        "activation_curve_seed": args.activation_curve_seed + trial_index,
        "activation_interpolation": args.activation_interpolation,
        "activation_spline_parameters": args.activation_spline_parameters,
        "activation_fit_constraint": args.activation_fit_constraint,
        "activation_normalize_positive_endpoint": (
            args.activation_normalize_positive_endpoint),
        "compile_measured_activation": args.compile_measured_activation,
        "w_quant_mode": args.w_quant_mode,
        "thermal_noise": args.thermal_noise,
        "w_perc": args.w_perc,
        "offset_eps": None,
    }
    return ode_params, wrapper_params, curve_indices


@dataclass
class RuntimeTrial:
    model: torch.nn.Module
    dataloader: object
    device: torch.device
    args: object
    wrappers: list
    corner: str
    trial_index: int
    command: list

    def first_batch(self):
        self.reset_data_rng()
        inputs, targets = next(iter(self.dataloader))
        return inputs.to(self.device), targets.to(self.device)

    def reset_data_rng(self):
        _set_all_seeds(self.args.data_seed)


def build_runtime_trial(model_name=None, model_root=None,
                        corner=DEFAULT_CORNER, batch_size=128, trial_index=0,
                        seed=DEFAULT_SEED, device="auto",
                        nonideality_profile="all_on"):
    """Build one expanded Level-3 clean or all-on runtime trial."""
    args, _, command = reference_corner_command(
        model_name=model_name, model_root=model_root,
        corner=corner, batch_size=batch_size,
        trial_index=trial_index, seed=seed,
        nonideality_profile=nonideality_profile)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    trial_seed = args.hardware_seed + trial_index
    _set_all_seeds(trial_seed)

    dataloader = get_test_data(
        test_bs=args.test_bs, img_type=args.img_type, task=args.task)
    checkpoint = os.path.join(
        args.model_dir, args.model_name,
        args.model_name + "_{}_ckpt.pth".format(args.ckpt))
    ode_params, wrapper_params, curve_indices = _model_parameters(
        args, trial_index)
    wrappers = {}
    with torch.no_grad():
        model = load_and_prepare_model(
            model_path=checkpoint,
            device=device,
            model_struct=PCNet,
            pc_conv_layer=PC_CONV_CLASS[args.pc_conv],
            data_parallel=False,
            noise_to_bn=True,
            noise_to_linear=True,
            fuse_bn=False,
            conv_only=args.conv_only,
            ode_params=ode_params,
            ode_wrapper_params=wrapper_params,
            wrappers=wrappers,
            noise_level=0.0,
            weight=None)
        configure_measured_pooling(
            model, wrappers["wrappers"],
            enable_nonideality=args.enable_measured_pooling,
            curve_path=args.nonlinear_R_table,
            quantity=args.nonlinear_R_mc_quantity,
            curve_indices=curve_indices,
            nominal_R=args.R,
            seed=trial_seed)
        validator = Validator(
            model=model,
            expanded_weight_dir=os.path.join(
                args.expanded_w_dir, args.model_name, args.ode_wrapper,
                "{}b".format(args.w_bits)),
            device=device,
            test_dataloader=dataloader,
            result_path=str(REPO_ROOT / "results" / "diagnostic_runtime"),
            wrapper=wrappers["wrappers"])
        model = validator.model
        model.noise_level = 0.0
        for block in model.PcConvs:
            block.noise_level = 0.0
        model.add_noise(noise_to_bn=True, noise_to_linear=True)
        model.eval()
    return RuntimeTrial(
        model=model,
        dataloader=dataloader,
        device=device,
        args=args,
        wrappers=wrappers["wrappers"],
        corner=str(corner).upper(),
        trial_index=trial_index,
        command=command)


def set_current_noise(model, enabled):
    """Change only the two Brownian current-noise enable flags."""
    for block in model.PcConvs:
        block.enable_summing_current_noise = bool(enabled)
        block.enable_coupler_noise = bool(enabled)


def snapshot_rng_state(model):
    custom = []
    pulse_caches = []
    seen = set()
    for module in model.modules():
        if hasattr(module, "_pulse_on_values"):
            pulse_caches.append((module, {
                key: value.detach().clone()
                for key, value in module._pulse_on_values.items()}))
        for value in vars(module).values():
            generators = []
            if isinstance(value, torch.Generator):
                generators = [value]
            elif isinstance(value, dict):
                generators = [item for item in value.values()
                              if isinstance(item, torch.Generator)]
            for generator in generators:
                if id(generator) not in seen:
                    seen.add(id(generator))
                    custom.append((generator, generator.get_state().clone()))
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state().clone(),
        "cuda": [state.clone() for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available() else None,
        "custom": custom,
        "pulse_caches": pulse_caches,
    }


def restore_rng_state(snapshot):
    random.setstate(snapshot["python"])
    np.random.set_state(snapshot["numpy"])
    torch.set_rng_state(snapshot["torch"])
    if snapshot["cuda"] is not None:
        torch.cuda.set_rng_state_all(snapshot["cuda"])
    for generator, state in snapshot["custom"]:
        generator.set_state(state)
    for module, cache in snapshot["pulse_caches"]:
        module._pulse_on_values = {
            key: value.detach().clone() for key, value in cache.items()}

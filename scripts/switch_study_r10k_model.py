"""Original 16L96C checkpoint recipe; R=10k, R_max=150k, ENOB=None."""
import json
import math
import os
import sys

import torch

ROOT = "/home/rongzeng/_workspce_old/repos/pcn/collaboration/ScAN-PCN"
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from TorchDiffEqPack.odesolver import odesolve
from inference_utils import get_test_data, load_and_prepare_model
from ode_pc import ODEWrapper_CLASSES
from pc_conv import PCConvReLU6Noisy
from pc_model import PCNet
from switch import (
    ODEXInitFFFBPixelSwitch,
    ODEXInitFFFBPixelSwitchExplicit,
    ODEXInitFFFBPixelSwitchStrang,
)

MODEL = ("TIMMQAT5b8aNT0p25mulTIMMPCNetNoBatchNorm_PCConvReLU6_0.0eps_"
         "ODEXInitFFFB_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_"
         "C100_3K1S96C_0.25Dropout_16Layers4l5l4_2Pool_srrlDistill_a0p3_t2p0_"
         "scanGFI_1REP")
CKPT = os.path.join(ROOT, "saved_ckpt", MODEL, MODEL + "_best_ckpt.pth")


def build(block_cls, n_iters=1):
    saved = {}
    ode = dict(
        ode_block=block_cls, t_end=1.75, method="dopri5", tol=1e-6,
        ts_scale=1, n_steps=5.0, return_init=[False] * 16,
        switch_period=None, n_iters=n_iters, i_leak=None,
        sde_noise_type="mul", mismatch_type="mul",
        patch_node=8, patch_stride=8, patch_cycle=1, patch_pad=0,
        fold_scalar=1, toggle_n_cycles=None, toggle_time_split=.5,
        toggle_fast_path=True, odexinit_scaling_mode="approx",
        toggle_timing_mode="derived", toggle_y_time=5e-9,
        z_over_y_time=3., toggle_timing_R=20e3, toggle_timing_C=49e-15,
        enable_spin_variation=False, sigma_spin=.1, spin_variation_mean=1.,
        spin_variation_seed=None, enable_summing_current_noise=False,
        summing_current_p=1.85e-11, summing_noise_seed=None,
        enable_coupler_noise=False, coupler_noise_p=.6e-12,
        coupler_noise_seed=None, enable_slow_summing_current=False,
        slow_summing_current=2.47e-9, enable_slow_coupler_noise=False,
        slow_coupler_noise=2.47e-9, enable_dtc_nonideality=False,
        dtc_leading_edge_variation_std=0., dtc_width_variation_mean=0.,
        dtc_width_variation_std=.018, dtc_leading_edge_jitter_std=.005,
        dtc_falling_edge_jitter_std=.005, dtc_timing_seed=None)
    wrapper = dict(
        ode_wrapper=ODEWrapper_CLASSES["QATTester1State"], calib_path=None,
        R=10e3, R_max=150e3, C=49e-15, k=1e3, v_dd=.1, w_bits=5,
        weight_quant_factor_bits=None, enob=None, w_quant_mode="min_max",
        tie_cap=False, one_over_q=1., thermal_noise=True,
        nonlinear_R=False, nonlinear_R_table=None,
        nonlinear_R_mc_curve_index=None, nonlinear_R_mc_quantity="conductance",
        nonlinear_R_curve_sharing="shared",
        nonlinear_R_curve_sampling="empirical_with_replacement",
        nonlinear_R_curve_bank_indices=None, nonlinear_R_curve_seed=None,
        nonlinear_R_curve_edge_chunk_size=65536,
        nonlinear_R_train_mode="none", nonlinear_R_corner_range="all",
        mul_mismatch_mode="scale_mismatch", enable_measured_activation=False,
        activation_curve_path=os.path.join(ROOT, "hardware_data/relu_current_0p2uA_finer.csv"),
        activation_corner="TT", activation_curve_sharing="per_model",
        activation_curve_seed=None, activation_interpolation="piecewise_linear",
        activation_spline_parameters=10, activation_fit_constraint="auto",
        activation_normalize_positive_endpoint=False, adapt_relu_offset=True,
        compile_measured_activation=False, offset_eps=None, w_perc=.99999)
    net = load_and_prepare_model(
        model_path=CKPT, device=torch.device("cuda"), model_struct=PCNet,
        pc_conv_layer=PCConvReLU6Noisy, data_parallel=False,
        noise_to_bn=True, noise_to_linear=True, fuse_bn=False, conv_only=True,
        ode_params=ode, ode_wrapper_params=wrapper, wrappers=saved,
        noise_level=0.)
    net.eval()
    wraps = saved["wrappers"]
    assert len(wraps) == 16 and not net.training
    assert all(w.R == 10000 and w.R_max == 150000 and w.enob is None for w in wraps)
    return net, wraps


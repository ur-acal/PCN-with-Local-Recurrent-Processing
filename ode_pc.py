import copy
import math
import os
import time
import random
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as P
import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Any
from functools import wraps

from pc_model import PCNet
from pc_conv import PCConv, PCConvNoisy, PCConvHardTanhLimit, PCConvHardTanhLimitNoisy, PCConvHardTanhNoisy, PCConvHardTanh
from pc_conv import ReLUX, HardTanhByX
from utils import expand_weights_to_matrix
from torchdiffeq import odeint
from TorchDiffEqPack.odesolver import odesolve as aca_ode_solve
# from TorchDiffEqPack.odesolver_mem import odesolve_adjoint as aca_ode_solve

import logging
log = logging.getLogger(__name__)


def is_adaptive(method):
    adaptive_sols = ['dopri8', 'dopri5', 'bosh3', 'fehlberg2', 'adaptive_heun']
    fixed_grid_sols = ['euler', 'midpoint', 'heun2', 'heun3', 'rk4', 'explicit_adams',
                       'implicit_adams', 'fixed_adams']  # 'scipy_solver'
    return method in adaptive_sols


class _TAddedModule(nn.Module):
    def __init__(self, base_module: nn.Module):
        super(_TAddedModule, self).__init__()
        self.base_module = base_module

    def forward(self, t, x):
        # t is an unused added input argument to fit the torchdiffeq module
        return self.base_module(x)


class ODEBlockPC(nn.Module):

    def __init__(self, pc_conv: Union[PCConvNoisy, PCConv], noise_level=0.0, method="dopri5", t_end=None, t_step=None,
                 tol=1e-3, return_mid=False, init_b=False, sde_noise_type="mul", **kwargs):
        super(ODEBlockPC, self).__init__()
        self.noise_level = noise_level
        self.tie_weights = pc_conv.tie_weights
        self.tie_bp = pc_conv.tie_bp
        self.layer_idx = pc_conv.layer_idx

        self.FFconv = pc_conv.FFconv
        self.FBconv = pc_conv.FBconv
        self.bypass = pc_conv.bypass
        self.act_fn = pc_conv.relu
        # Used in ODESelfCoupleInitY
        self.b0 = pc_conv.b0
        if init_b:
            with torch.no_grad():
                logging.warning("Initialize b0 as the summation of weights")
                ff_weight = self.FFconv.weight.data
                self.b0 = nn.ParameterList([ff_weight.view(ff_weight.shape[0], -1).sum(-1).view(1, -1, 1, 1)])
        self._transfer_reg_buff(pc_conv)

        # cache clean parameters
        self.clean_params = {
            "FFconv": nn.Parameter(self.FFconv.weight.clone()),
            "FBconv": nn.Parameter(self.FBconv.weight.clone()),
            "b0": nn.Parameter(self.b0[0].clone()),
        }

        if self.noise_level is not None and self.noise_level > 0:
            self.add_noise()

        if t_end is not None:
            self.integration_time = torch.tensor([0, t_end]).float()
        elif t_end is not None and t_step is not None and return_mid:
            self.integration_time = torch.arange(0, t_end + t_step, t_step).float()
        else:
            self.integration_time = torch.arange(0, (pc_conv.cls + 1) * pc_conv.lr, pc_conv.lr).float()
        self.integration_time = self.integration_time.to(self.FFconv.weight.device)

        if is_adaptive(method):
            # automatically pick the initial step size for adaptive methods
            t_step = None

        self.tol = tol
        self.method = method

        # noise type used in sde simulation; only useful when option["eps"] is set.
        self.sde_noise_type = sde_noise_type

        self.option_aca = {"t0": self.integration_time[0], "t1": self.integration_time[-1],
                           "t_eval": self.integration_time.tolist(), "rtol": self.tol, "atol": self.tol,
                           "h": t_step, "method": self.method, "noise_type": self.sde_noise_type}

    def _transfer_reg_buff(self, pc_conv):
        for _name, _val in pc_conv.named_buffers():
            self.register_buffer(_name, _val)

    def _make_ode_fn(self, x):
        def ode_func(t, y):
            return self.FFconv(self.act_fn(x - self.FBconv(y)))
        return ode_func

    def init_y(self, x):
        return self.act_fn(self.FFconv(x))

    def forward(self, x, layer_idx=None):
        y0 = self.init_y(x)
        self.integration_time = self.integration_time.type_as(x)

        out = aca_ode_solve(self._make_ode_fn(x), y0, self.option_aca)
        # out = odeint(ode_func, y0, self.integration_time, rtol=self.tol, atol=self.tol, method=self.method)
        out = out[-1]

        if self.bypass is not None:
            out = self.bypass(out) + out
        return out

    def _apply_noise(self, p):
        if getattr(p, "is_sparse_csr", False):
            v_ = p.values()
            noise_ = torch.randn_like(v_, device=p.device, requires_grad=False) * self.noise_level
            v_.mul_(1 + noise_)
        else:
            noise_ = torch.randn_like(p, device=p.device, requires_grad=False) * self.noise_level
            p.mul_(1 + noise_)

    def add_noise(self):
        logging.warning("Adding noise to FFconv in ODEBlockPC")
        self._apply_noise(self.FFconv.weight)
        if not self.tie_weights:
            logging.warning("Adding noise to FBconv in ODEBlockPC")
            self._apply_noise(self.FBconv.weight)
        if not self.tie_bp and self.bypass is not None:
            logging.warning("Adding noise to BPconv in ODEBlockPC")
            self._apply_noise(self.bypass.weight)
        if not torch.allclose(self.b0[0], torch.zeros_like(self.b0[0])):
            self._apply_noise(self.b0[0])
            logging.warning("Adding noise to self.b0[0] in ODEBlockPC")

    def recover_params(self):
        self.FFconv.weight = self.clean_params["FFconv"]
        self.FBconv.weight = self.clean_params["FBconv"]
        self.b0[0] = self.clean_params["b0"]

    @property
    def nfe(self):
        return self.ode_func.nfe

    @nfe.setter
    def nfe(self, value):
        self.ode_func.nfe = value


class ODEBlockXInit(ODEBlockPC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.in_chan = self.FFconv.in_channels
        self.out_chan = self.FFconv.out_channels
        self.chan_diff = self.out_chan - self.in_chan
        if self.chan_diff == 0:
            self.chan_pad_a, self.chan_pad_b = 0, 0
        elif self.in_chan * 2 == self.out_chan:
            self.chan_pad_a = self.chan_diff // 2
            self.chan_pad_b = self.chan_pad_a
        else:
            self.chan_pad_a = self.chan_diff // 2
            self.chan_pad_b = self.chan_diff - self.chan_pad_a

    def init_y(self, x):
        if self.chan_diff == 0:
            return x
        elif self.in_chan * 2 == self.out_chan:
            return torch.cat([x, x], dim=1)
        else:
            return F.pad(torch.cat([x for _ in range(self.out_chan // self.in_chan)], dim=1),
                         (0, 0, 0, 0, 0, self.out_chan % self.in_chan), "constant", 0)
####################################################################################
####################################################################################
class ODEWrapperRC(nn.Module):
    def __init__(self, ode_block: ODEBlockPC, state_bound=50.0, R=1e5, C=49e-15, v_dd=1.0, patch=True, **kwargs):
        super().__init__()
        self.ode_block = ode_block

        self.R = R
        self.C = C
        self.v_dd = v_dd

        # the bound to scale the states, can be either the maximum absolute value or percentile (99.99)
        self.state_bound = state_bound
        self.q = v_dd / state_bound

        self.original_make_fn = self.ode_block._make_ode_fn
        self.original_forward = self.ode_block.forward
        self.original_init_y = self.ode_block.init_y
        if patch:
            self._patch()
        # logging.warning("self.q: {}, self.R: {}, self.C: {}, self.act_fn: {}".format(self.q, self.R, self.C, self.ode_block.act_fn))

    def get_time_scaler(self):
        return self.R * self.C

    def transform(self, inner_fn):
        @wraps(inner_fn)
        def scaled(*f_args, **f_kwargs):
            return inner_fn(*f_args, **f_kwargs) / self.time_scaler
        return scaled

    def _scale_act_fn(self):
        if "relu6" in self.ode_block.act_fn.__class__.__name__.lower():
            self.ode_block.act_fn = ReLUX(6 * self.q)
        elif "hardtanh" in self.ode_block.act_fn.__class__.__name__.lower():
            self.ode_block.act_fn = nn.Hardtanh(min_val=-self.q, max_val=self.q)

    def _scale_time(self):
        integral_option = self.ode_block.option_aca
        integration_time = self.ode_block.integration_time

        # scale and set value
        integration_time = integration_time * self.time_scaler
        integral_option["t0"], integral_option["t1"] = integration_time[0], integration_time[-1]
        integral_option["t_eval"] = integration_time.tolist()
        integral_option["h"] = integral_option["h"] * self.time_scaler if integral_option["h"] is not None else None

        self.ode_block.option_aca = integral_option
        self.ode_block.integration_time = integration_time
        logging.info("Scaled end time: {} s".format(self.ode_block.option_aca["t1"]))

    def _patch_make_fn(self):
        # scale the ode_func
        orig = self.original_make_fn
        @wraps(orig)
        def patched_make_fn(*args, **kwargs):
            inner = orig(*args, **kwargs)
            return self.transform(inner)
        self.ode_block._make_ode_fn = patched_make_fn

    def _patch_forward(self):
        # scale the forward method
        orig_call = self.original_forward
        @wraps(orig_call)
        def patched_forward(x, *args, **kwargs):
            return orig_call(self.q * x, *args, **kwargs) / self.q
        self.ode_block.forward = patched_forward

    def _patch_init_y(self):
        pass

    def _patch(self):
        self.time_scaler = self.get_time_scaler()
        # scale integration time
        self._scale_time()
        self._scale_act_fn()
        self._patch_make_fn()
        self._patch_forward()
        self._patch_init_y()

    def forward(self, x, layer_idx=None):
        return self.ode_block(self.q * x, layer_idx=layer_idx) / self.q

    def get_ode_block(self):
        return self.ode_block


class WrapQuantizeW(ODEWrapperRC):
    def __init__(self, w_bits=8, w_quant_mode="min_max", perc=None, quantize=True, **kwargs):
        patch = kwargs.get("patch", True)
        kwargs.update({"patch": False})
        super().__init__(**kwargs)
        self.w_bits = w_bits
        self.q_hi = (1 << (w_bits - 1)) - 1
        self.w_quant_mode = w_quant_mode
        self.perc = perc

        self.quantize = quantize
        if self.quantize:
            # Quantize the trained floating point weights
            self.get_quantize_factor()
        else:
            # Load quantized parameters directly
            self.set_quantized_params()
        self.alpha = self.s_ff * self.s_fb
        self.beta = self.q * self.s_fb

        # patch for the child method
        if patch:
            self._patch()

    def set_quantized_params(self):
        # loaded weights are already quantized
        # the ode block must have registered s_ff/fb buffer
        self.register_buffer("s_ff", self.ode_block.s_ff if hasattr(self.ode_block, "s_ff") else torch.tensor(1.0))
        self.register_buffer("s_fb", self.ode_block.s_fb if hasattr(self.ode_block, "s_fb") else torch.tensor(1.0))
        if self.ode_block.noise_level is not None and self.ode_block.noise_level > 0:
            self.ode_block.add_noise()

    @staticmethod
    def cal_quant_factor_and_set(q_hi, p: nn.Parameter):
        with torch.no_grad():
            abs_max = p.data.abs().max()
            # -2 ** n is not used for symmetry
            s = q_hi / abs_max
            torch.clamp((s * p).round(), min=-q_hi, max=q_hi, out=p)
            p.div_(q_hi)
            return s / q_hi

    def get_quantize_factor(self):
        # calculate quantization coefficient
        # the clean_params of the ode_block is set to the quantized weight in-place
        self.register_buffer(
            "s_ff", self.cal_quant_factor_and_set(self.q_hi, self.ode_block.clean_params["FFconv"]))
        self.register_buffer(
            "s_fb", self.cal_quant_factor_and_set(self.q_hi, self.ode_block.clean_params["FBconv"]))
        if not torch.allclose(torch.zeros_like(self.ode_block.b0[0]), self.ode_block.clean_params["b0"]):
            with torch.no_grad():
                ff_weight = self.ode_block.clean_params["FFconv"].data
                self.ode_block.clean_params["b0"] = nn.Parameter(
                    ff_weight.view(ff_weight.shape[0], -1).sum(-1).view(1, -1, 1, 1))

        # Recovered using clean_params, which has being quantized before
        # Then add noise
        self.ode_block.recover_params()
        if self.ode_block.noise_level is not None and self.ode_block.noise_level > 0:
            self.ode_block.add_noise()

    def _scale_time(self):
        integral_option = self.ode_block.option_aca
        integration_time = self.ode_block.integration_time

        # scale and set value
        integration_time = integration_time * self.time_scaler / self.alpha
        integral_option["t0"], integral_option["t1"] = integration_time[0], integration_time[-1]
        integral_option["t_eval"] = integration_time.tolist()
        integral_option["h"] = integral_option["h"] * self.time_scaler / self.alpha if integral_option["h"] is not None else None

        self.ode_block.option_aca = integral_option
        self.ode_block.integration_time = integration_time
        logging.info("Scaled end time: {} s".format(self.ode_block.option_aca["t1"]))

    def _scale_act_fn(self):
        if "relu6" in self.ode_block.act_fn.__class__.__name__.lower():
            self.ode_block.act_fn = ReLUX(6 * self.beta)
        elif "hardtanh" in self.ode_block.act_fn.__class__.__name__.lower():
            self.ode_block.act_fn = nn.Hardtanh(min_val=-self.beta, max_val=self.beta)

    def _patch_forward(self):
        # scale the forward method
        orig_call = self.original_forward
        @wraps(orig_call)
        def patched_forward(x, *args, **kwargs):
            return orig_call(self.beta * x, *args, **kwargs) / self.q
        self.ode_block.forward = patched_forward

    def _patch_init_y(self):
        orig_init_y = self.original_init_y
        @wraps(orig_init_y)
        def patched_init_y(x, *args, **kwargs):
            ode_blk_cls = self.ode_block.__class__.__name__.lower()
            if "yasx" in ode_blk_cls or "xinit" in ode_blk_cls:
                return orig_init_y(x) / self.s_fb
            else:
                return orig_init_y(x / self.s_ff) / self.s_fb
        self.ode_block.init_y = patched_init_y


def make_ode_block(pc_net: PCNet, ode_block=ODEBlockPC, noise_level=0.0, method=None, t_end=None, tol=1e-3, ts_scale=1,
                   n_steps=None, **kwargs):
    for i in range(pc_net.num_layers):
        cls = pc_net.PcConvs[i].cls if n_steps is None else n_steps
        t_step = pc_net.PcConvs[i].lr
        if t_end is None:
            t_end = cls * t_step
        else:
            t_step = t_end / cls if cls != 0 else 1.0
        t_step = t_step / ts_scale
        pc_net.PcConvs[i] = ode_block(
            pc_conv=pc_net.PcConvs[i], noise_level=noise_level, method=method, t_end=t_end, t_step=t_step, tol=tol,
            **kwargs)
    return pc_net


def wrap_ode_block(pc_net: PCNet, ode_wrapper=ODEWrapperRC, calib_path=None, R=1e5, C=49e-15, v_dd=1.0, one_over_q=10,
                   **kwargs):
    calib_res = [one_over_q for _ in range(pc_net.num_layers)]
    wrappers = []
    # Todo: Perform calibration for intermediate states if we are going to quantize them and the weights
    if calib_path is None:
        pass
    else:
        pass
        # calib_res = np.load(calib_path)
    for i in range(pc_net.num_layers):
        if i == 0:
            ode_wrapper_ins = ode_wrapper(ode_block=pc_net.PcConvs[i], state_bound=calib_res[i],
                                          R=R, C=C, v_dd=v_dd, is_first=True, is_last=False, **kwargs)
        elif i == pc_net.num_layers - 1:
            ode_wrapper_ins = ode_wrapper(ode_block=pc_net.PcConvs[i], state_bound=calib_res[i],
                                          R=R, C=C, v_dd=v_dd, is_first=False, is_last=True, **kwargs)
        else:
            ode_wrapper_ins = ode_wrapper(ode_block=pc_net.PcConvs[i], state_bound=calib_res[i],
                                          R=R, C=C, v_dd=v_dd, is_first=False, is_last=False, **kwargs)
        ode_wrapper_ins.to(pc_net.device)
        pc_net.PcConvs[i] = ode_wrapper_ins.get_ode_block()
        wrappers.append(ode_wrapper_ins)
    return pc_net, wrappers


ODEBLOCK_CLASSES = {
    "ODEBlockPC": ODEBlockPC,
    "ODEBlockXInit": ODEBlockXInit,
}

ODEWrapper_CLASSES = {
    "ODEWrapperRC": ODEWrapperRC,
    "WrapQuantizeW": WrapQuantizeW,
}

QUANTIZER_CLASSES = {None: None}

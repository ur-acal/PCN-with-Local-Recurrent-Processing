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
from typing import Union, Any, Tuple
from functools import wraps

from pc_model import PCNet
from pc_conv import PCConv, PCConvNoisy, PCConvHardTanhLimit, PCConvHardTanhLimitNoisy, PCConvHardTanhNoisy, PCConvHardTanh
from pc_conv import ReLUX, HardTanhByX
from utils import (expand_weights_to_matrix, interpolate_R_eff,
                   load_mc_res_curve_bank, load_mc_res_curve_gaussian,
                   load_mc_res_training_curve_bank, load_res_vs_vin)
from torchdiffeq import odeint
from TorchDiffEqPack.odesolver import odesolve as aca_ode_solve
from measured_activation import (
    CubicBSplineActivation, MEASURED_ACTIVATION_TYPES,
    PiecewiseLinearActivation)
# from TorchDiffEqPack.odesolver_mem import odesolve_adjoint as aca_ode_solve

import logging
log = logging.getLogger(__name__)


def _symmetric_qat_weight_scale(layer_weight):
    """Return the detached s_w used by SymQuantizeWeight.compute_s()."""
    with torch.no_grad():
        return 1 / layer_weight.detach().abs().max()


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
                 tol=1e-3, return_mid=False, init_b=False, sde_noise_type="mul", mismatch_type="mul",
                 return_init=False, **kwargs):
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

        self.mismatch_type = mismatch_type
        assert self.mismatch_type in {"mul", "add"}
        if isinstance(self.noise_level, dict) or (self.noise_level is not None and self.noise_level > 0):
            self.add_noise()

        self.t_end_sf = kwargs.get("t_end_sf", 1.0)
        if t_end is not None:
            if np.allclose(self.t_end_sf, 1.0):
                self.integration_time = torch.tensor([0, t_end]).float()
            else:
                self.integration_time = torch.tensor([0, t_end / self.t_end_sf, t_end]).float()
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

        # SDE noise related members
        self.offset_eps = 0.002
        self.eps_scale = None

        # Directly return the init result
        self.return_init = return_init

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
        if self.return_init:
            return y0
        self.integration_time = self.integration_time.type_as(x)

        out = aca_ode_solve(self._make_ode_fn(x), y0, self.option_aca)
        # out = odeint(ode_func, y0, self.integration_time, rtol=self.tol, atol=self.tol, method=self.method)
        out = out[-1]

        if self.bypass is not None:
            out = self.bypass(out) + out
        return out

    def forward_full_steps(self, x, layer_idx=None):
        # Returning the full trajectory of the solver.
        y0 = self.init_y(x)
        self.integration_time = self.integration_time.type_as(x)

        out = aca_ode_solve(self._make_ode_fn(x), y0, self.option_aca, full_traj=True)
        # out = odeint(ode_func, y0, self.integration_time, rtol=self.tol, atol=self.tol, method=self.method)

        if self.bypass is not None:
            out = self.bypass(out) + out
        return out

    def _get_quant_magnitude_levels(self, ref_vals):
        k = torch.arange(self.q_hi + 1, device=ref_vals.device, dtype=ref_vals.dtype)

        if self.weight_scale == 1.0:
            # Uniform levels: {0, 1/q_hi, 2/q_hi, ..., 1}
            levels = k / self.q_hi
        else:
            # Non-uniform remap:
            # k = 0 -> 0
            # k >= 1 -> first + (k-1)*delta
            first = self.weight_scale / self.q_hi
            delta = (1.0 - first) / (self.q_hi - 1)

            levels = torch.zeros_like(k)
            mask = k > 0
            levels[mask] = first + (k[mask] - 1.0) * delta

        return levels

    def _values_to_level_idx(self, values):
        # For self.q_hi, there are self.q_hi + 1 levels in total.
        # Map values to one of these levels [0, 1, ..., self.q_hi].
        levels = self._get_quant_magnitude_levels(values) # [q_hi + 1]
        v_abs = values.abs().reshape(-1, 1) # [nnz, 1]
        dist = (v_abs - levels.reshape(1, -1)).abs() # [nnz, q_hi + 1]
        level_idx = dist.argmin(dim=1).to(torch.long) # [nnz]
        return level_idx.reshape_as(values)

    def _get_sparse_sigma_tensor(self, values):
        if not isinstance(self.noise_level, dict):
            return self.noise_level

        level_idx = self._values_to_level_idx(values)

        # Missing keys default to the maximum sigma
        sigma_lut = torch.full((self.q_hi + 1,), max(self.noise_level.values()),
                               device=values.device, dtype=values.dtype)
        for _k, _sigma in self.noise_level.items():
            sigma_lut[_k] = _sigma

        sigma = sigma_lut[level_idx]
        return sigma

    def _get_dense_sigma_tensor(self, p):
        if not isinstance(self.noise_level, dict):
            return self.noise_level

        level_idx = self._values_to_level_idx(p)

        # Missing keys default to 0.0 sigma
        sigma_lut = torch.full((self.q_hi + 1,), max(self.noise_level.values()), device=p.device, dtype=p.dtype)
        for _k, _sigma in self.noise_level.items():
            sigma_lut[_k] = _sigma

        sigma = sigma_lut[level_idx]
        return sigma

    def _apply_noise(self, p):
        if getattr(p, "is_sparse_csr", False):
            v_ = p.values()
            sigma_ = self._get_sparse_sigma_tensor(v_)
            if self.mismatch_type == "mul":
                noise_ = torch.randn_like(v_, device=p.device, requires_grad=False) * sigma_
                v_.mul_(1 + noise_)  # This will change values of CSR matrix in-place
            else:
                max_abs = v_.abs().max()
                noise_ = torch.randn_like(v_, device=p.device, requires_grad=False) * (sigma_ * max_abs)
                v_.add_(noise_)
        else:
            sigma_ = self._get_dense_sigma_tensor(p)
            if self.mismatch_type == "mul":
                noise_ = torch.randn_like(p, device=p.device, requires_grad=False) * sigma_
                p.mul_(1 + noise_)
            else:
                max_abs = p.abs().max()
                noise_ = torch.randn_like(p, device=p.device, requires_grad=False) * (sigma_ * max_abs)
                p.add_(noise_)

    def add_noise(self):
        # If the conv layers in the model have been replaced to MVMConv, call the add_noise
        # method of the MVMConv. In this case, we should initialize the model with 0.0
        # noise_level first, and then wrap with validator, and then set mismatch, and
        # finally call the add_noise method.
        if hasattr(self.FFconv, "add_noise"):
            self.FFconv.add_noise(
                noise_level=self.noise_level,
                mismatch_type=self.mismatch_type,
                q_hi=getattr(self, "q_hi", None),
                weight_scale=getattr(self, "weight_scale", 1.0)
            )
        else:
            self._apply_noise(self.FFconv.weight)

        if not self.tie_weights:
            if hasattr(self.FBconv, "add_noise"):
                self.FBconv.add_noise(
                    noise_level=self.noise_level,
                    mismatch_type=self.mismatch_type,
                    q_hi=getattr(self, "q_hi", None),
                    weight_scale=getattr(self, "weight_scale", 1.0)
                )
            else:
                self._apply_noise(self.FBconv.weight)

        if not self.tie_bp and self.bypass is not None:
            if hasattr(self.bypass, "add_noise"):
                self.bypass.add_noise(
                    noise_level=self.noise_level,
                    mismatch_type=self.mismatch_type,
                    q_hi=getattr(self, "q_hi", None),
                    weight_scale=getattr(self, "weight_scale", 1.0)
                )
            else:
                self._apply_noise(self.bypass.weight)

        if not torch.allclose(self.b0[0], torch.zeros_like(self.b0[0])):
            self._apply_noise(self.b0[0])

    @torch.no_grad()
    def recover_params(self):
        self.FFconv.weight.copy_(self.clean_params["FFconv"])
        self.FBconv.weight.copy_(self.clean_params["FBconv"])
        self.b0[0].copy_(self.clean_params["b0"])

    @property
    def nfe(self):
        return self.ode_func.nfe

    @nfe.setter
    def nfe(self, value):
        self.ode_func.nfe = value


class ODEBlockFFFB(ODEBlockPC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _make_ode_fn(self, x, noisy_cu=None):
        def ode_func(t, y):
            return self.FFconv(self.act_fn(self.FBconv(y)))
        return ode_func


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

        # SDE noise related members
        self.offset_eps = 0.002
        self.eps_scale = None

    def init_y(self, x):
        if self.chan_diff == 0:
            return x
        elif self.in_chan > self.out_chan:
            return x[:, :self.out_chan, :, :]
        elif self.in_chan * 2 == self.out_chan:
            return torch.cat([x, x], dim=1)
        else:
            return F.pad(torch.cat([x for _ in range(self.out_chan // self.in_chan)], dim=1),
                         (0, 0, 0, 0, 0, self.out_chan % self.in_chan), "constant", 0)


class ODEBlockPCLimitDyn(ODEBlockPC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _make_ode_fn(self, x):
        def ode_func(t, y):
            return self.act_fn(self.FFconv(self.act_fn(x - self.FBconv(y))))
        return ode_func


class ODEBlkActInp(ODEBlockPC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _make_ode_fn(self, x):
        def ode_func(t, y):
            y = self.act_fn(y)
            weight_sum = self.FFconv.weight.view(y.shape[1], -1).sum(-1)
            offset = weight_sum.view(1, -1, 1, 1) * y
            # Todo: Can we add t as the min/max value in hardTanh?
            #  So that when t=0, the initializer is the same as before.
            #  Then the min/max value gradually increases so that the dynamics is the same as before
            return self.FFconv(self.act_fn(x - self.FBconv(y))) - offset
        return ode_func

    def forward(self, x, layer_idx=None):
        y0 = torch.zeros((x.shape[0], self.FFconv.weight.shape[0], x.shape[2], x.shape[3]), device=x.device)
        if self.return_init:
            return y0
        out = aca_ode_solve(self._make_ode_fn(x), y0, self.option_aca)
        out = out[-1]

        if self.bypass is not None:
            out = self.bypass(out) + out
        return out

class ODEBlkActDyn(ODEBlkActInp):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _make_ode_fn(self, x):
        def ode_func(t, y):
            weight_sum = self.FFconv.weight.view(y.shape[1], -1).sum(-1)
            offset = weight_sum.view(1, -1, 1, 1) * y
            # Todo: Can we add t as the min/max value in hardTanh?
            #  So that when t=0, the initializer is the same as before.
            #  Then the min/max value gradually increases so that the dynamics is the same as before
            return self.act_fn(self.FFconv(x - self.act_fn(self.FBconv(y)))) - offset
        return ode_func

class ODEActInpNoMinus(ODEBlkActInp):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _make_ode_fn(self, x):
        def ode_func(t, y):
            return self.FFconv(self.act_fn(x - self.FBconv(self.act_fn(y))))
        return ode_func

class ODEActDynNoMinus(ODEBlkActInp):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _make_ode_fn(self, x):
        def ode_func(t, y):
            return self.act_fn(self.FFconv(x - self.act_fn(self.FBconv(y))))
        return ode_func

class ODEBlkProj(ODEBlockPC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _make_ode_fn(self, x):
        def ode_func(t, y):
            weight_sum = self.FFconv.weight.view(y.shape[1], -1).sum(-1)
            offset = weight_sum.view(1, -1, 1, 1) * y
            return self.FFconv(self.act_fn(x - self.FBconv(y))) - offset
        return ode_func

    def forward(self, x, layer_idx=None):
        y0 = torch.zeros((x.shape[0], self.FFconv.weight.shape[0], x.shape[2], x.shape[3]), device=x.device)
        if self.return_init:
            return y0
        out = aca_ode_solve(self._make_ode_fn(x), y0, self.option_aca, proj_fn=self.act_fn)
        out = out[-1]

        if self.bypass is not None:
            out = self.bypass(out) + out
        return out

class ODEBlkProjInitY(ODEBlkProj):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x, layer_idx=None):
        y0 = self.act_fn(self.FFconv(x))
        if self.return_init:
            return y0
        out = aca_ode_solve(self._make_ode_fn(x), y0, self.option_aca, proj_fn=self.act_fn)
        out = out[-1]

        if self.bypass is not None:
            out = self.bypass(out) + out
        return out

class ODEBlkProjActDyn(ODEBlkProj):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _make_ode_fn(self, x):
        # Todo: Check the dynamics and where to apply the self.act_fn
        def ode_func(t, y):
            weight_sum = self.FFconv.weight.view(y.shape[1], -1).sum(-1)
            offset = weight_sum.view(1, -1, 1, 1) * y
            return self.act_fn(self.FFconv(x - self.act_fn(self.FBconv(y)))) - offset
        return ode_func

class ODEBlk0Init(ODEBlkActInp):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _make_ode_fn(self, x):
        def ode_func(t, y):
            weight_sum = self.FFconv.weight.view(y.shape[1], -1).sum(-1)
            offset = weight_sum.view(1, -1, 1, 1) * y
            return self.FFconv(self.act_fn(x - self.FBconv(y))) - offset
        return ode_func

class ODESelfCoupleInitY(ODEBlkActInp):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Todo: optional, initialize self.b0 according to the summation of the weights of FFconv

    def _make_ode_fn(self, x):
        def ode_func(t, y):
            weight_sum = self.FFconv.weight.view(y.shape[1], -1).sum(-1)
            offset = (weight_sum.view(1, -1, 1, 1) - self.b0[0]) * y
            return self.FFconv(self.act_fn(x - self.FBconv(y))) - offset
        return ode_func

    def forward(self, x, layer_idx=None):
        y0 = self.init_y(x)
        if self.return_init:
            return y0
        out = aca_ode_solve(self._make_ode_fn(x), y0, self.option_aca)
        out = out[-1]

        if self.bypass is not None:
            out = self.bypass(out) + out
        return out

class ODEActDynInitY(ODESelfCoupleInitY):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _make_ode_fn(self, x):
        def ode_func(t, y):
            return self.act_fn(self.FFconv(x - self.act_fn(self.FBconv(y))))
        return ode_func

class ODESumAsBInitY(ODESelfCoupleInitY):
    def __init__(self, **kwargs):
        kwargs.update({"init_b": True})
        super().__init__(**kwargs)

class ODESumAsBInitYFFFB(ODESumAsBInitY):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _make_ode_fn(self, x):
        def ode_func(t, y):
            weight_sum = self.FFconv.weight.view(y.shape[1], -1).sum(-1)
            offset = (weight_sum.view(1, -1, 1, 1) - self.b0[0]) * y
            return self.FFconv(self.act_fn(self.FBconv(y))) - offset
        return ode_func

class ODENoisyOffset(ODESelfCoupleInitY):
    """
    Only used in training.
    During inference, use ODESumAsBInitY and add noise to all the weights including self.b0[0].
    """
    def __init__(self, offset_eps=0.2, **kwargs):
        super().__init__(**kwargs)
        self.offset_eps = offset_eps

    def _make_ode_fn(self, x, weight_sum=None):
        def ode_func(t, y):
            noisy_cu = weight_sum * torch.randn_like(weight_sum, requires_grad=False, device=y.device) * self.offset_eps
            return self.FFconv(self.act_fn(x - self.FBconv(y))) - noisy_cu * y
        return ode_func

    def forward(self, x, layer_idx=None):
        y0 = self.act_fn(self.FFconv(x))
        if self.return_init:
            return y0
        with torch.no_grad():
            weight_sum = self.FFconv.weight.data.view(y0.shape[1], -1).sum(-1).view(1, -1, 1, 1)
        out = aca_ode_solve(self._make_ode_fn(x, weight_sum), y0, self.option_aca)
        out = out[-1]

        if self.bypass is not None:
            out = self.bypass(out) + out
        return out

class ODEFixNoiseOffset(ODESelfCoupleInitY):
    """
    Only used in training.
    During inference, use ODESumAsBInitY and add noise to all the weights including self.b0[0].
    """
    def __init__(self, offset_eps=0.2, **kwargs):
        super().__init__(**kwargs)
        self.offset_eps = offset_eps

    def _make_ode_fn(self, x, noisy_cu=None):
        def ode_func(t, y):
            return self.FFconv(self.act_fn(x - self.FBconv(y))) - noisy_cu * y
        return ode_func

    def init_y(self, x):
        return self.act_fn(self.FFconv(x))

    def forward(self, x, layer_idx=None):
        y0 = self.init_y(x)
        if self.return_init:
            return y0
        with torch.no_grad():
            weight_sum = self.FFconv.weight.data.view(y0.shape[1], -1).sum(-1).view(1, -1, 1, 1)
            if self.FFconv.training:
                noisy_cu = weight_sum * torch.randn_like(weight_sum, requires_grad=False, device=y0.device) * self.offset_eps
            else:
                noisy_cu = torch.zeros_like(weight_sum, requires_grad=False, device=y0.device)
        out = aca_ode_solve(self._make_ode_fn(x, noisy_cu), y0, self.option_aca)
        out = out[-1]

        if self.bypass is not None:
            out = self.bypass(out) + out
        return out

class ODEFixNoiseFFFB(ODEFixNoiseOffset):
    def __init__(self,  **kwargs):
        super().__init__(**kwargs)

    def _make_ode_fn(self, x, noisy_cu=None):
        def ode_func(t, y):
            return self.FFconv(self.act_fn(self.FBconv(y))) - noisy_cu * y
        return ode_func

class ODEFixNoise0Init(ODEFixNoiseOffset):
    def __init__(self,  **kwargs):
        super().__init__(**kwargs)

    def init_y(self, x):
        return torch.zeros((x.shape[0], self.FFconv.weight.shape[0], x.shape[2], x.shape[3]), device=x.device)

class ODEXInitFFFB(ODEBlockXInit):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _make_ode_fn(self, x, noisy_cu=None):
        def ode_func(t, y):
            return self.FFconv(self.act_fn(self.FBconv(y)))
        return ode_func


class ToggleBaseFFFB(ODEXInitFFFB):
    """
    Base for staged toggle hardware dynamics:
    y is the output spin state, z is the pre-nonlinearity spin state,
    h = f(z), and the topology is W_FB -> f -> W_FF.
    """
    reset_z = None

    def __init__(self, toggle_n_cycles=None, toggle_time_split=0.5,
                 toggle_fast_path=True, odexinit_scaling_mode="approx", **kwargs):
        super().__init__(**kwargs)
        self.toggle_n_cycles = toggle_n_cycles
        self.toggle_time_split = toggle_time_split
        self.toggle_fast_path = toggle_fast_path
        self.odexinit_scaling_mode = str(odexinit_scaling_mode).lower()

        if self.reset_z is None:
            raise ValueError("ToggleBaseFFFB subclasses must set reset_z.")
        if self.toggle_n_cycles is not None and self.toggle_n_cycles <= 0:
            raise ValueError("toggle_n_cycles must be positive when provided.")
        if not 0.0 < self.toggle_time_split < 1.0:
            raise ValueError("toggle_time_split must be between 0 and 1.")
        if self.odexinit_scaling_mode not in {"approx", "direct"}:
            raise ValueError(
                "odexinit_scaling_mode must be 'approx' or 'direct'.")

    def _integration_end_like(self, ref):
        return self.integration_time[-1].to(device=ref.device, dtype=ref.dtype)

    def init_z(self, y):
        return torch.zeros_like(self.FBconv(y), device=y.device)

    def _finish_forward(self, y):
        if self.bypass is not None:
            y = self.bypass(y) + y
        return y

    def _run_cycles(self, y, z, cycle_params):
        n_cycles = 1 if self.toggle_n_cycles is None else int(self.toggle_n_cycles)
        for _ in range(n_cycles):
            if self.reset_z:
                z = torch.zeros_like(z)
            y, z = self.run_one_cycle(y, z, cycle_params)
        self._last_toggle_z = z
        return y

    def forward(self, x, layer_idx=None):
        y = self.init_y(x)
        if self.return_init:
            return y
        z = self.init_z(y)
        cycle_params = self.resolve_cycle_params(y)
        y = self._run_cycles(y, z, cycle_params)
        return self._finish_forward(y)


class ToggleAveragedPhysicalFFFB(ToggleBaseFFFB):
    supports_post_quant_mismatch = True

    def __init__(self, enable_spin_variation=False, sigma_spin=0.10,
                 spin_variation_mean=1.0, spin_variation_seed=None,
                 enable_summing_current_noise=False, summing_current_p=12.73e-12,
                 summing_noise_seed=None, enable_coupler_noise=False,
                 coupler_noise_p=0.6e-12, coupler_noise_seed=None, **kwargs):
        super().__init__(**kwargs)
        self.enable_spin_variation = bool(enable_spin_variation)
        self.sigma_spin = float(sigma_spin)
        self.spin_variation_mean = float(spin_variation_mean)
        self.spin_variation_seed = spin_variation_seed
        self.enable_summing_current_noise = bool(enable_summing_current_noise)
        self.summing_current_p = float(summing_current_p)
        self.summing_noise_seed = summing_noise_seed
        self.enable_coupler_noise = bool(enable_coupler_noise)
        self.coupler_noise_p = float(coupler_noise_p)
        self.coupler_noise_seed = coupler_noise_seed
        self.register_buffer("_spin_factor_y", None, persistent=False)
        self.register_buffer("_spin_factor_z", None, persistent=False)
        self._spin_variation_generators = {}
        self._summing_noise_generators = {}
        self._coupler_noise_generators = {}
        self._active_coupler_count_cache = {}
        self._training_pulse_mismatch = None
        self._nonlinear_R_training_generators = {}
        self._nonlinear_R_training_curve_indices = {}

    def begin_training_pulse_mismatch(self, noise_level, mismatch_type):
        """Sample one post-quantization coupler-amplitude mismatch for this forward."""
        noise_level = float(noise_level)
        mismatch = {"type": mismatch_type}
        for key, module in (("FFconv", self.FFconv), ("FBconv", self.FBconv)):
            if key == "FBconv" and self.tie_weights:
                mismatch[key] = mismatch["FFconv"]
                continue
            noise = torch.randn_like(module.weight, requires_grad=False) * noise_level
            mismatch[key] = 1.0 + noise if mismatch_type == "mul" else noise
        self._training_pulse_mismatch = mismatch

    def end_training_pulse_mismatch(self):
        self._training_pulse_mismatch = None

    def _apply_averaged_module(self, module, source):
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            source = self._averaged_nonlinear_R_source(module, source)

        mismatch = self._training_pulse_mismatch
        if mismatch is None:
            return module(source)

        key = "FBconv" if module is self.FBconv else "FFconv"
        weight = module.weight
        if mismatch["type"] == "mul":
            weight = weight * mismatch[key]
        else:
            weight = weight + weight.abs() * mismatch[key]

        if isinstance(module, nn.Conv2d):
            return F.conv2d(
                source, weight, module.bias, module.stride, module.padding,
                module.dilation, module.groups)
        if isinstance(module, nn.ConvTranspose2d):
            return F.conv_transpose2d(
                source, weight, module.bias, module.stride, module.padding,
                module.output_padding, module.groups, module.dilation)
        raise TypeError(
            "Level-2 post-quantization mismatch supports Conv2d and ConvTranspose2d modules.")

    def reset_spin_variation(self):
        self._spin_factor_y = None
        self._spin_factor_z = None

    def reset_nonlinear_R_variation(self):
        self._nonlinear_R_training_curve_indices.clear()

    def forward(self, x, layer_idx=None):
        # Training samples one hardware realization per forward. Evaluation
        # keeps the lazily sampled factors fixed for this model/trial.
        if self.training and self.enable_spin_variation:
            self.reset_spin_variation()
        if self.training and hasattr(self, "_nonlinear_R_training_pkg"):
            self.reset_nonlinear_R_variation()
        return super().forward(x, layer_idx=layer_idx)

    def _layer_seed_offset(self):
        try:
            return 1009 * int(self.layer_idx)
        except (TypeError, ValueError):
            return 0


    def _nonlinear_R_training_module_key(self, module):
        if module is self.FBconv:
            return "FBconv"
        if module is self.FFconv:
            return "FFconv"
        raise ValueError("Nonlinear-R training received an unknown convolution module.")

    def _nonlinear_R_training_generator(self, ref):
        package = getattr(self, "_nonlinear_R_training_pkg", None)
        if package is None or package.get("nonlinear_R_curve_seed") is None:
            return None
        key = str(ref.device)
        generator = self._nonlinear_R_training_generators.get(key)
        if generator is None:
            generator = torch.Generator(device=ref.device)
            generator.manual_seed(
                int(package["nonlinear_R_curve_seed"]) +
                self._layer_seed_offset())
            self._nonlinear_R_training_generators[key] = generator
        return generator

    def _sample_nonlinear_R_training_curve(self, ref, module_key):
        package = self._nonlinear_R_training_pkg
        if not self._nonlinear_R_training_curve_indices:
            n_curves = int(package["v_grid"].shape[0])
            if n_curves < 2:
                raise ValueError(
                    "At least two nonlinear-R curves are required to sample "
                    "different FF and FB curves without replacement.")
            pair = torch.randperm(
                n_curves, device=ref.device,
                generator=self._nonlinear_R_training_generator(ref))[:2].tolist()
            self._nonlinear_R_training_curve_indices.update(
                FFconv=int(pair[0]), FBconv=int(pair[1]))
        return self._nonlinear_R_training_curve_indices[module_key]

    def _averaged_nonlinear_R_source(self, module, source):
        package = getattr(self, "_nonlinear_R_training_pkg", None)
        if package is None:
            return source

        module_key = self._nonlinear_R_training_module_key(module)
        curve_index = self._sample_nonlinear_R_training_curve(
            source, module_key)
        length = int(package["lengths"][curve_index].item())
        grid = package["v_grid"][curve_index, :length].to(
            device=source.device, dtype=source.dtype)
        left = package["R_left"][curve_index, :length - 1].to(
            device=source.device, dtype=source.dtype)
        slope = package["R_slope"][curve_index, :length - 1].to(
            device=source.device, dtype=source.dtype)

        query = source
        proj_fn = package.get("proj_fn")
        if proj_fn is not None:
            query = proj_fn(query)
        query = query.clamp(min=grid[0], max=grid[-1])
        interval = torch.bucketize(query.contiguous(), grid.contiguous()) - 1
        interval = interval.clamp(min=0, max=length - 2)
        R_eff = (
            left[interval] +
            slope[interval] * (query - grid[interval]))
        return source * source.new_tensor(package["R"]) / R_eff

    def _spin_variation_generator(self, ref, stage):
        if self.spin_variation_seed is None:
            return None
        key = (str(ref.device), stage)
        generator = self._spin_variation_generators.get(key)
        if generator is None:
            generator = torch.Generator(device=ref.device)
            stage_offset = 0 if stage == "z" else 1
            generator.manual_seed(
                int(self.spin_variation_seed) + self._layer_seed_offset() + stage_offset)
            self._spin_variation_generators[key] = generator
        return generator

    def _spin_factor(self, stage, summed_rhs):
        if not self.enable_spin_variation:
            return None
        name = "_spin_factor_{}".format(stage)
        factor = getattr(self, name)
        expected_shape = (1,) + tuple(summed_rhs.shape[1:])
        if factor is None:
            factor = self.spin_variation_mean + self.sigma_spin * torch.randn(
                expected_shape, device=summed_rhs.device, dtype=summed_rhs.dtype,
                generator=self._spin_variation_generator(summed_rhs, stage))
            setattr(self, name, factor)
        elif tuple(factor.shape) != expected_shape:
            raise RuntimeError(
                "Cached {}-spin factor has shape {}, but this hardware run requested {}.".format(
                    stage, tuple(factor.shape), expected_shape))
        return factor.to(device=summed_rhs.device, dtype=summed_rhs.dtype)

    def _apply_spin_variation(self, stage, summed_rhs):
        factor = self._spin_factor(stage, summed_rhs)
        return summed_rhs if factor is None else factor * summed_rhs

    def _stage_capacitance(self, stage):
        return self.C_fb if stage == "z" else self.C_ff

    def _stage_eps(self, stage):
        return self.summing_current_p / self._stage_capacitance(stage)

    def _summing_noise_generator(self, state):
        if self.summing_noise_seed is None:
            return None
        key = str(state.device)
        generator = self._summing_noise_generators.get(key)
        if generator is None:
            generator = torch.Generator(device=state.device)
            generator.manual_seed(int(self.summing_noise_seed) + self._layer_seed_offset())
            self._summing_noise_generators[key] = generator
        return generator

    def _coupler_noise_generator(self, state):
        if self.coupler_noise_seed is None:
            return None
        key = str(state.device)
        generator = self._coupler_noise_generators.get(key)
        if generator is None:
            generator = torch.Generator(device=state.device)
            generator.manual_seed(int(self.coupler_noise_seed) + self._layer_seed_offset())
            self._coupler_noise_generators[key] = generator
        return generator

    def _brownian_increment(self, state, duration, stage):
        normal = torch.randn(
            state.shape, device=state.device, dtype=state.dtype,
            generator=self._summing_noise_generator(state))
        duration = torch.as_tensor(duration, device=state.device, dtype=state.dtype)
        return self._stage_eps(stage) * duration.sqrt() * normal

    def _averaged_coupler_level_sum(self, module, source, state):
        """Sum quantized pulse levels feeding each destination spin."""
        if not getattr(self, "physical", False):
            raise ValueError(
                "Physical coupler noise requires a hardware wrapper with stage capacitances.")
        if not hasattr(self, "q_hi") or self.q_hi <= 0:
            raise ValueError("Level-2 coupler noise requires a positive quantization maximum.")

        with torch.no_grad():
            levels = self._values_to_level_idx(module.weight.detach()).to(
                device=state.device, dtype=state.dtype)
            source_ones = source.new_ones((1,) + tuple(source.shape[1:]))
            if isinstance(module, nn.Conv2d):
                level_sum = F.conv2d(
                    source_ones, levels, None, module.stride, module.padding,
                    module.dilation, module.groups)
            elif isinstance(module, nn.ConvTranspose2d):
                level_sum = F.conv_transpose2d(
                    source_ones, levels, None, module.stride, module.padding,
                    module.output_padding, module.groups, module.dilation)
            else:
                raise TypeError(
                    "Level-2 coupler noise supports Conv2d and ConvTranspose2d modules.")

            if tuple(level_sum.shape[1:]) != tuple(state.shape[1:]):
                raise RuntimeError(
                    "Coupler-level sum shape {} does not match destination state {}.".format(
                        tuple(level_sum.shape), tuple(state.shape)))
            return level_sum

    def _averaged_coupler_brownian_increment(
            self, state, duration, stage, coupler_level_sum):
        # A level-l coupler is active for l of q_hi equal pulse slices.
        # Summing their independent variances gives:
        # (p/C)^2 * duration/q_hi * sum_c(level_c).
        normal = torch.randn(
            state.shape, device=state.device, dtype=state.dtype,
            generator=self._coupler_noise_generator(state))
        duration = torch.as_tensor(duration, device=state.device, dtype=state.dtype)
        level_sum = coupler_level_sum.to(device=state.device, dtype=state.dtype)
        scale = self.coupler_noise_p / self._stage_capacitance(stage)
        return scale * (duration * level_sum / float(self.q_hi)).sqrt() * normal

    def _averaged_stage_update(
            self, state, summed_rhs, duration, stage, coupler_level_sum=None):
        summed_rhs = self._apply_spin_variation(stage, summed_rhs)

        if not getattr(self, "physical", False):
            if self.enable_summing_current_noise or self.enable_coupler_noise:
                raise ValueError(
                    "Physical current noise requires a hardware wrapper with stage capacitances.")
            return state + duration * summed_rhs

        updated = state + (duration / (self.R * self._stage_capacitance(stage))) * summed_rhs
        if self.enable_summing_current_noise:
            updated = updated + self._brownian_increment(state, duration, stage)
        if self.enable_coupler_noise:
            if coupler_level_sum is None:
                raise ValueError(
                    "Level-2 coupler noise requires the summed incoming pulse levels.")
            updated = updated + self._averaged_coupler_brownian_increment(
                state, duration, stage, coupler_level_sum)
        return updated

    def project_state(self, state):
        if not hasattr(self, "v_dd"):
            raise ValueError("Physical toggle blocks require wrapper-provided v_dd.")
        v_dd = self.v_dd
        return state.clamp(min=-v_dd, max=v_dd)

    def resolve_cycle_params(self, ref):
        if not getattr(self, "physical", False):
            # Unitless mode.
            n_cycles = 1 if self.toggle_n_cycles is None else int(self.toggle_n_cycles)
            cycle_t = self._integration_end_like(ref) / float(n_cycles)
            t_z = cycle_t * float(self.toggle_time_split)
            t_y = cycle_t * float(1.0 - self.toggle_time_split)
            if t_z.detach().cpu().item() <= 0 or t_y.detach().cpu().item() <= 0:
                raise ValueError("Unitless toggle stage durations must be positive.")
            return t_z, t_y

        missing = [name for name in ("R", "C", "C_fb", "C_ff", "alpha", "w_bits") if not hasattr(self, name)]
        if missing:
            raise ValueError("ToggleAveragedPhysicalFFFB requires wrapper-provided " + ", ".join(missing))
        n_cycles = 1 if self.toggle_n_cycles is None else int(self.toggle_n_cycles)
        cycle_t = self._integration_end_like(ref) / float(n_cycles)
        t_z = cycle_t * float(self.toggle_time_split)
        t_y = cycle_t * float(1.0 - self.toggle_time_split)
        time_scale = self.R * self.C / self.alpha
        T_z = t_z * time_scale
        T_y = t_y * time_scale
        if T_z <= 0 or T_y <= 0:
            raise ValueError("Physical toggle stage durations must be positive.")
        return T_z, T_y

    def z_stage_update(self, z, y_hold, T_z):
        level_sum = None
        if self.enable_coupler_noise:
            level_sum = self._averaged_coupler_level_sum(self.FBconv, y_hold, z)
        return self._averaged_stage_update(
            z, self._apply_averaged_module(self.FBconv, y_hold), T_z, "z",
            coupler_level_sum=level_sum)

    def y_stage_update(self, y, h_hold, T_y):
        level_sum = None
        if self.enable_coupler_noise:
            level_sum = self._averaged_coupler_level_sum(self.FFconv, h_hold, y)
        return self._averaged_stage_update(
            y, self._apply_averaged_module(self.FFconv, h_hold), T_y, "y",
            coupler_level_sum=level_sum)

    def run_one_cycle(self, y, z, cycle_params):
        # cycle_params is generated by resolve_cycle_params
        # if in physical model: scale the time accordingly.
        # otherwise in unitless mode.
        T_z, T_y = cycle_params
        physical = getattr(self, "physical", False)
        z = self.z_stage_update(z, y, T_z)
        if physical:
            z = self.project_state(z)
        h = self.act_fn(z)
        y = self.y_stage_update(y, h, T_y)
        if physical:
            y = self.project_state(y)
        return y, z


class ToggleResetZ(ToggleAveragedPhysicalFFFB):
    reset_z = True


class ToggleKeepZ(ToggleAveragedPhysicalFFFB):
    reset_z = False


class ToggleODEXInitFFFB(ToggleAveragedPhysicalFFFB):
    reset_z = True

    def __init__(self, unitless_measured_pullback_mode=None,
                 enable_unitless_measured_pullback=None,
                 unitless_pullback_q=None, unitless_pullback_k=1e3,
                 unitless_pullback_R=10e3, **kwargs):
        super().__init__(**kwargs)
        if unitless_measured_pullback_mode is None:
            unitless_measured_pullback_mode = (
                "approx" if bool(enable_unitless_measured_pullback) else "none")
        self.unitless_measured_pullback_mode = str(
            unitless_measured_pullback_mode).lower()
        if self.unitless_measured_pullback_mode not in {
                "approx", "direct", "none"}:
            raise ValueError(
                "unitless_measured_pullback_mode must be "
                "'approx', 'direct', or 'none'.")
        self.enable_unitless_measured_pullback = (
            self.unitless_measured_pullback_mode != "none")
        self.unitless_pullback_q = (
            None if unitless_pullback_q is None else float(unitless_pullback_q))
        self.unitless_pullback_k = float(unitless_pullback_k)
        self.unitless_pullback_R = float(unitless_pullback_R)
        self._active_unitless_pullback_beta_c = None

        if self.enable_unitless_measured_pullback:
            if self.unitless_pullback_q is None or self.unitless_pullback_q <= 0:
                raise ValueError("unitless_pullback_q must be positive when pullback is enabled.")
            if (self.unitless_measured_pullback_mode == "approx" and
                    (self.unitless_pullback_k <= 0 or self.unitless_pullback_R <= 0)):
                raise ValueError(
                    "unitless pullback k and R must be positive in approx mode.")

    def prospective_unitless_pullback_scales(self):
        """Return detached prospective (s_fb, beta, beta_c) for current FB weights."""
        s_fb = _symmetric_qat_weight_scale(self.FBconv.weight)
        beta = s_fb * self.unitless_pullback_q
        beta_c = beta * self.unitless_pullback_k / self.unitless_pullback_R
        return s_fb, beta, beta_c

    def forward(self, x, layer_idx=None):
        if (not self.enable_unitless_measured_pullback or
                getattr(self, "physical", False)):
            return super().forward(x, layer_idx=layer_idx)
        if not isinstance(self.act_fn, MEASURED_ACTIVATION_TYPES):
            raise TypeError(
                "Unitless measured pullback requires a measured activation.")

        _, beta, beta_c = self.prospective_unitless_pullback_scales()
        pullback_scale = (
            beta_c if self.unitless_measured_pullback_mode == "approx"
            else beta.new_tensor(self.unitless_pullback_q))
        previous_scale = self.act_fn._coordinate_pullback_scale
        self._active_unitless_pullback_beta_c = pullback_scale
        self.act_fn.set_coordinate_pullback_scale(pullback_scale)
        try:
            return super().forward(x, layer_idx=layer_idx)
        finally:
            self.act_fn.set_coordinate_pullback_scale(previous_scale)
            self._active_unitless_pullback_beta_c = None

    def resolve_cycle_params(self, ref):
        n_cycles = 1 if self.toggle_n_cycles is None else int(self.toggle_n_cycles)
        cycle_t = self._integration_end_like(ref) / float(n_cycles)
        if not getattr(self, "physical", False):
            return torch.ones_like(cycle_t), cycle_t

        required = ["R", "C", "C_fb", "w_bits"]
        if self.odexinit_scaling_mode == "direct":
            required.extend(["s_ff", "s_fb"])
        else:
            required.extend(["k", "alpha_1state"])
        missing = [name for name in required if not hasattr(self, name)]
        if missing:
            raise ValueError("ToggleODEXInitFFFB requires wrapper-provided " + ", ".join(missing))
        R = self.R
        C = self.C
        C_fb = self.C_fb
        if self.odexinit_scaling_mode == "direct":
            T_z = R * C_fb / self.s_fb
            T_y = cycle_t * R * C / self.s_ff
        else:
            k = self.k
            alpha_1state = self.alpha_1state
            T_z = k * C_fb
            T_y = cycle_t * R * R * C / (k * alpha_1state)
        if T_z <= 0 or T_y <= 0:
            raise ValueError("ODEXInit toggle stage durations must be positive.")
        return T_z, T_y

    def _stage_capacitance(self, stage):
        return self.C_fb if stage == "z" else self.C


class ToggleODEXInitKeep(ToggleODEXInitFFFB):
    reset_z = False


class TogglePulseFFFB(ToggleAveragedPhysicalFFFB):
    supports_post_quant_mismatch = False

    def __init__(self, enable_dtc_nonideality=False,
                 dtc_leading_edge_variation_std=0.0,
                 dtc_width_variation_mean=0.0,
                 dtc_width_variation_std=0.018,
                 dtc_leading_edge_jitter_std=0.005,
                 dtc_falling_edge_jitter_std=0.005,
                 dtc_timing_seed=None, **kwargs):
        super().__init__(**kwargs)
        self.enable_dtc_nonideality = bool(enable_dtc_nonideality)
        self.dtc_leading_edge_variation_std = float(dtc_leading_edge_variation_std)
        self.dtc_width_variation_mean = float(dtc_width_variation_mean)
        self.dtc_width_variation_std = float(dtc_width_variation_std)
        self.dtc_leading_edge_jitter_std = float(dtc_leading_edge_jitter_std)
        self.dtc_falling_edge_jitter_std = float(dtc_falling_edge_jitter_std)
        self.dtc_timing_seed = dtc_timing_seed
        for name in (
                "dtc_leading_edge_variation_std", "dtc_width_variation_std",
                "dtc_leading_edge_jitter_std", "dtc_falling_edge_jitter_std"):
            if getattr(self, name) < 0.0:
                raise ValueError("{} must be nonnegative.".format(name))
        self._dtc_fixed_variation = {}
        self._dtc_timing_generators = {}
        self._last_dtc_window = {}

    @torch.no_grad()
    def add_noise(self):
        self._pulse_on_values = {}
        sigma = self.noise_level
        if isinstance(sigma, dict):
            sigma = sigma.get(self.q_hi, max(sigma.values()))
        sigma = 0.0 if sigma is None else float(sigma)

        def add_coupler_noise(module):
            if hasattr(module, "add_noise"):
                module.add_noise(
                    noise_level=sigma, mismatch_type=self.mismatch_type,
                    q_hi=self.q_hi, weight_scale=self.weight_scale)
            elif sigma > 0.0:
                mismatch = torch.randn_like(module.weight) * sigma
                if self.mismatch_type == "mul":
                    module.weight.mul_(1 + mismatch)
                else:
                    module.weight.add_(mismatch * module.weight.abs().max())

        add_coupler_noise(self.FFconv)
        if not self.tie_weights:
            add_coupler_noise(self.FBconv)

        if not self.tie_bp and self.bypass is not None:
            if hasattr(self.bypass, "add_noise"):
                self.bypass.add_noise(
                    noise_level=self.noise_level,
                    mismatch_type=self.mismatch_type,
                    q_hi=self.q_hi, weight_scale=self.weight_scale)
            else:
                self._apply_noise(self.bypass.weight)

        if not torch.allclose(self.b0[0], torch.zeros_like(self.b0[0])):
            self._apply_noise(self.b0[0])

    def _num_slices(self, stage):
        assert self.q_hi > 0
        return self.q_hi

    def reset_dtc_variation(self):
        self._dtc_fixed_variation.clear()

    def _dtc_timing_generator(self, ref, stage):
        if self.dtc_timing_seed is None:
            return None
        key = (str(ref.device), stage)
        generator = self._dtc_timing_generators.get(key)
        if generator is None:
            generator = torch.Generator(device=ref.device)
            stage_offset = 0 if stage == "z" else 104729
            generator.manual_seed(
                int(self.dtc_timing_seed) + self._layer_seed_offset() + stage_offset)
            self._dtc_timing_generators[key] = generator
        return generator

    @staticmethod
    def _dtc_num_outputs(module):
        if hasattr(module, "mat"):
            return (int(module.meta["out_chan"]) * int(module.meta["inp_chan"]) *
                    int(module.meta["ker_h"]) * int(module.meta["ker_w"]))
        return module.weight.numel()

    def _dtc_output_index(self, module, ref):
        if not hasattr(module, "mat"):
            return torch.arange(ref.numel(), device=ref.device, dtype=torch.int64).reshape_as(ref)
        if not hasattr(module, "dtc_block_ids") or not hasattr(module, "dtc_output_ids"):
            raise ValueError("Expanded pulse modules require Validator DTC metadata.")
        outputs_per_block = int(module.meta["ker_h"]) * int(module.meta["ker_w"])
        return module.dtc_block_ids * outputs_per_block + module.dtc_output_ids

    def _sample_dtc_window(self, module, stage):
        # sample the variation and jitter of the DTC pulses
        # each individual param from dense kernel have its own variation and jitter
        # The values in unrolled matrix from the same param in the dense kernel shares
        # the same variation and jitter.
        if not self.enable_dtc_nonideality:
            return None
        _, clean_values = self._clean_weight(module)
        # num parameters in this module = num of independent variation and jitter to sample.
        num_outputs = self._dtc_num_outputs(module)
        fixed = self._dtc_fixed_variation.get(stage)
        if fixed is None:
            generator = self._dtc_timing_generator(clean_values, stage)
            fixed = {
                "leading_edge": self.dtc_leading_edge_variation_std * torch.randn(
                    num_outputs, device=clean_values.device, dtype=clean_values.dtype,
                    generator=generator),
                "width": (
                    self.dtc_width_variation_mean +
                    self.dtc_width_variation_std * torch.randn(
                        num_outputs, device=clean_values.device,
                        dtype=clean_values.dtype, generator=generator)),
            }
            self._dtc_fixed_variation[stage] = fixed
        generator = self._dtc_timing_generator(clean_values, stage)
        leading_jitter = self.dtc_leading_edge_jitter_std * torch.randn(
            num_outputs, device=clean_values.device, dtype=clean_values.dtype,
            generator=generator)
        falling_jitter = self.dtc_falling_edge_jitter_std * torch.randn(
            num_outputs, device=clean_values.device, dtype=clean_values.dtype,
            generator=generator)
        window = {
            "start_offset": fixed["leading_edge"] + leading_jitter,
            "end_offset": fixed["leading_edge"] + fixed["width"] + falling_jitter,
        }
        self._last_dtc_window[stage] = window
        return window

    def _dtc_slice_duty(self, module, clean_values, slice_idx, dtc_window):
        # Calculates the overlap between the true pulse window (with variation and jitter)
        # and the current slice.
        if dtc_window is None:
            return None
        if getattr(self, "pulse_weight_encoding", None) == "integer_level":
            pulse_count = clean_values.abs()
        else:
            pulse_count = torch.round(clean_values.abs() * self.q_hi)
        pulse_count = pulse_count.clamp(min=0, max=self.q_hi)
        output_index = self._dtc_output_index(module, clean_values)
        # Firstly get unitless pulse width. The smallest is 1, then 2, ..., self.q_hi.
        start = dtc_window["start_offset"][output_index]
        end = pulse_count + dtc_window["end_offset"][output_index]
        slice_start = clean_values.new_tensor(float(slice_idx))
        slice_end = slice_start + 1.0
        overlap = torch.minimum(end, slice_end) - torch.maximum(start, slice_start)
        duty = overlap.clamp(min=0.0, max=1.0)
        return duty * (pulse_count > 0).to(dtype=clean_values.dtype)

    def _timed_pulse_values(self, clean_values, on_values, slice_idx, duty):
        return on_values * duty

    def _pulse_values_for_slice(
            self, module, clean_values, on_values, slice_idx, dtc_window):
        # on_values = sign(W) * W'/W
        if dtc_window is None:
            # No DTC variation and jitter, perfect pulses
            return self._pulse_values(clean_values, on_values, slice_idx)
        # With DTC variation and jitter, now each slice is not perfect 1 or 0.
        # We need to calculate the "duty" within each slice, which is the
        # overlap between the real pulse width (with variation and jitter)
        # and the slice width.
        duty = self._dtc_slice_duty(module, clean_values, slice_idx, dtc_window)
        return self._timed_pulse_values(
            clean_values, on_values, slice_idx, duty)

    def _clean_weight(self, module):
        # get clean weights based on if we are in test_expanded mode
        key = "FBconv" if module is self.FBconv else "FFconv"
        clean_weight = self.clean_params[key]
        if not hasattr(module, "mat"):
            return key, clean_weight

        if not hasattr(module, "clean_mat_values"):
            raise ValueError("Expanded pulse modules require Validator's clean MVM value snapshot.")
        return key, module.clean_mat_values

    def _on_values(self, clean_values, noisy_values, key):
        # To get mismatch and sign
        if not hasattr(self, "_pulse_on_values"):
            self._pulse_on_values = {}
        cached = self._pulse_on_values.get(key)
        if cached is not None and cached.shape == clean_values.shape:
            return cached

        if self.mismatch_type == "mul":
            scale = torch.ones_like(clean_values)
            nonzero = clean_values != 0
            scale[nonzero] = noisy_values[nonzero] / clean_values[nonzero]
            on_values = clean_values.sign() * scale
        else:
            on_values = clean_values.sign() + noisy_values - clean_values
        self._pulse_on_values[key] = on_values
        return on_values

    def _pulse_values(self, clean_values, on_values, slice_idx):
        # From the clean_values in {-1, -14/15, ..., -1/15, 0, 1/15, ..., 14/15, 1}, calculate
        # which quantization level each clean_values are in: {1, 2, ..., 15}.
        # on_values determine the sign and merging mismatch in (if any).
        # We have split a T_z/y into number of quant levels slices.
        # If the quantization level >= current pulse slice_idx, then this quantization level is on.
        # Todo: Currently we are generating perfect pulse width based on clean weights.
        #  The mismatch is added via on_values. We need to implement non-ideal width pulse.
        pulse_count = torch.round(clean_values.abs() * self.q_hi)
        pulse_count = pulse_count.clamp(min=0, max=self.q_hi)
        active = pulse_count > slice_idx
        return on_values * active.to(dtype=clean_values.dtype)

    def get_pulse_matrix(self, module, slice_idx, num_slices, dtc_window=None):
        assert num_slices == self.q_hi
        key, clean_weight = self._clean_weight(module)
        if hasattr(module, "mat"):
            clean_values = clean_weight
            noisy_values = getattr(module, "pulse_noisy_values", None)
            noisy_values = module.mat.values() if noisy_values is None else noisy_values
            # clean_values are to get quantization levels.
            # clean_values + slice_idx determines if the coupler is on for this slice.
            # _on_values merges the sign + mismatch
            # Perfect pulse: values = mask_if_on * (sign(W) * W'/W) = mask_if_on * on_values
            # DTC non-ideality: duty * weight_non_zero * (sign(W) * W'/W)
            values = self._pulse_values_for_slice(
                module, clean_values,
                self._on_values(clean_values, noisy_values, key), slice_idx,
                dtc_window)
            assert getattr(module.mat, "is_sparse_csr", False)
            # Reconstruct the sparse tensor based on current slice_idx and quant level of each weight.
            return torch.sparse_csr_tensor(
                module.mat.crow_indices(), module.mat.col_indices(), values,
                size=module.mat.shape, dtype=module.mat.dtype,
                device=module.mat.device)
        return self._pulse_values_for_slice(
            module, clean_weight,
            self._on_values(clean_weight, module.weight, key), slice_idx,
            dtc_window)

    def _apply_sparse_pulse_module(self, module, x, pulse_weight):
        if hasattr(module, "forward_pulse"):
            return module.forward_pulse(x, pulse_weight, nominal_R=self.R)
        batch_size, _, input_h, input_w = x.shape
        padding = int(module.meta["padding"])
        stride = int(module.meta["stride"])
        ker_h = int(module.meta["ker_h"])
        ker_w = int(module.meta["ker_w"])
        output_h = (input_h + 2 * padding - ker_h) // stride + 1
        output_w = (input_w + 2 * padding - ker_w) // stride + 1
        x_flat = x.reshape(batch_size, -1).t()
        y = torch.sparse.mm(pulse_weight, x_flat)
        return y.t().reshape(batch_size, module.meta["out_chan"], output_h, output_w)

    def _pulse_nonlinear_source(self, x):
        package = getattr(self, "_nonlinear_R_pkg", None)
        if package is None:
            return x
        nominal_idx = (package["R_codes"] - self.R).abs().argmin()
        R_eff = interpolate_R_eff(
            x, package["v_grid"], package["R_codes"], package["R_left"],
            package["R_slope"], nominal_idx, proj_fn=package.get("proj_fn"))
        return x * self.R / R_eff

    def _apply_pulse_module(self, module, x, pulse_weight):
        # Calculate MVM. Incorporate nonlinear effect of R changing with input x if any.
        # The nonlinear interpolated effective R is merged into x.
        # Calculating the MVM between x and pulse_weight.
        if isinstance(module, nn.Conv2d):
            x = self._pulse_nonlinear_source(x)
            return F.conv2d(x, pulse_weight, None, module.stride, module.padding, module.dilation, module.groups)
        if isinstance(module, nn.ConvTranspose2d):
            x = self._pulse_nonlinear_source(x)
            return F.conv_transpose2d(x, pulse_weight, None, module.stride, module.padding,
                                      module.output_padding, module.groups, module.dilation)
        if hasattr(module, "mat") and (getattr(pulse_weight, "is_sparse", False) or
                                       getattr(pulse_weight, "is_sparse_csr", False)):
            return self._apply_sparse_pulse_module(module, x, pulse_weight)
        raise TypeError("Pulse-level toggle supports Conv2d, ConvTranspose2d, and MVMConv modules.")

    def z_stage_rhs(self, t, z, y_hold, pulse_weight):
        summed_rhs = self._apply_pulse_module(self.FBconv, y_hold, pulse_weight)
        summed_rhs = self._apply_spin_variation("z", summed_rhs)
        return summed_rhs / (self.R * self.C_fb)

    def y_stage_rhs(self, t, y, h_hold, pulse_weight):
        summed_rhs = self._apply_pulse_module(self.FFconv, h_hold, pulse_weight)
        summed_rhs = self._apply_spin_variation("y", summed_rhs)
        return summed_rhs / (self.R * self.C_ff)

    def _active_pulse_mask(self, module, slice_idx):
        _, clean_weight = self._clean_weight(module)
        clean_magnitude = clean_weight.detach().abs()
        if getattr(self, "pulse_weight_encoding", None) == "integer_level":
            pulse_count = clean_magnitude
        else:
            pulse_count = torch.round(clean_magnitude * self.q_hi)
        return pulse_count > int(slice_idx)

    def _active_coupler_count(self, module, source, active_mask, state, slice_idx=None):
        # Calculate the MVM between on-mask and all-one tensor to tell how many couplers
        # are on in a single row. This is to calculate the overall noise spectral density
        # for a single spin.
        cache = getattr(self, "_active_coupler_count_cache", None)
        cache_key = None
        # With DTC non-idealities, the cache look up is skipped. Because the
        # active coupler count will change even for the same slice_idx and param.
        if (cache is not None and slice_idx is not None and
                not getattr(self, "supports_pulse_training", False)):
            cache_key = (id(module), int(slice_idx), tuple(source.shape[1:]),
                         str(state.device), state.dtype)
            cached = cache.get(cache_key)
            if cached is not None:
                return cached
        with torch.no_grad():
            active = active_mask.to(device=state.device, dtype=state.dtype)
            source_ones = source.new_ones((1,) + tuple(source.shape[1:]))
            if isinstance(module, nn.Conv2d):
                count = F.conv2d(
                    source_ones, active, None, module.stride, module.padding,
                    module.dilation, module.groups)
            elif isinstance(module, nn.ConvTranspose2d):
                count = F.conv_transpose2d(
                    source_ones, active, None, module.stride, module.padding,
                    module.output_padding, module.groups, module.dilation)
            elif hasattr(module, "mat"):
                mat = module.mat
                active_mat = torch.sparse_csr_tensor(
                    mat.crow_indices(), mat.col_indices(), active, size=mat.shape,
                    dtype=state.dtype, device=state.device)
                source_ones_flat = state.new_ones((mat.shape[1], 1))
                count = torch.sparse.mm(active_mat, source_ones_flat).t().reshape(
                    (1,) + tuple(state.shape[1:]))
            else:
                raise TypeError(
                    "Coupler noise supports Conv2d, ConvTranspose2d, and MVMConv modules.")
            if tuple(count.shape[1:]) != tuple(state.shape[1:]):
                raise RuntimeError(
                    "Active-coupler count shape {} does not match destination state {}.".format(
                        tuple(count.shape), tuple(state.shape)))
            if cache_key is not None:
                cache[cache_key] = count
            return count

    def _coupler_stage_eps(self, stage, active_coupler_count):
        # Fractional DTC duty factors are nonnegative, but convolution can
        # produce tiny negative roundoff at locations whose exact count is zero.
        active_coupler_count = active_coupler_count.clamp_min(0.0)
        return (self.coupler_noise_p / self._stage_capacitance(stage)) * (
            active_coupler_count.sqrt())

    def _coupler_brownian_increment(self, state, duration, stage, active_coupler_count):
        normal = torch.randn(
            state.shape, device=state.device, dtype=state.dtype,
            generator=self._coupler_noise_generator(state))
        duration = torch.as_tensor(duration, device=state.device, dtype=state.dtype)
        return (self._coupler_stage_eps(stage, active_coupler_count) *
                duration.sqrt() * normal)

    def _make_slice_options(self, duration, state, stage, active_coupler_count=None):
        opts = deepcopy(self.option_aca)
        opts["t0"] = state.new_tensor(0.0)
        opts["t1"] = torch.as_tensor(duration, device=state.device, dtype=state.dtype)
        opts["t_eval"] = [opts["t0"], opts["t1"]]
        opts["h"] = opts["t1"] if opts.get("h", None) is not None else None
        if self.enable_summing_current_noise:
            opts["eps"] = self._stage_eps(stage)
            opts["noise_type"] = "add"
            opts["noise_generator"] = self._summing_noise_generator(state)
        if self.enable_coupler_noise:
            if active_coupler_count is None:
                raise ValueError("Coupler noise requires an active-coupler count for each pulse slice.")
            coupler_eps = self._coupler_stage_eps(stage, active_coupler_count)
            if opts.get("eps") is None:
                opts["eps"] = coupler_eps
            else:
                summing_eps = torch.as_tensor(
                    opts["eps"], device=state.device, dtype=state.dtype)
                opts["eps"] = (summing_eps.square() + coupler_eps.square()).sqrt()
            opts["noise_type"] = "add"
            opts["noise_generator"] = self._coupler_noise_generator(state)
        return opts

    def integrate_pulse_slice(self, state, duration, rhs_fn, stage, slice_idx, constant_rhs=None,
                              active_coupler_count=None):
        if self.toggle_fast_path and constant_rhs is not None:
            updated = state + duration * constant_rhs
            if self.enable_summing_current_noise:
                updated = updated + self._brownian_increment(state, duration, stage)
            if self.enable_coupler_noise:
                updated = updated + self._coupler_brownian_increment(
                    state, duration, stage, active_coupler_count)
        else:
            updated = aca_ode_solve(
                lambda t, y: rhs_fn(t, y), state,
                self._make_slice_options(
                    duration, state, stage, active_coupler_count=active_coupler_count))[-1]
        return self.project_state(updated)

    def run_z_stage(self, y_hold, z, T_z):
        # get the number of quantization levels
        # split the full pulse width into number of quant levels
        num_slices = self._num_slices("z")
        dt = T_z / float(num_slices)
        # sample possible pulse non-idealities.
        dtc_window = self._sample_dtc_window(self.FBconv, "z")
        for slice_idx in range(num_slices):
            # Perfect pulse: mask_if_on * (sign(W) * W'/W)
            # DTC non-ideality: duty * weight_non_zero * (sign(W) * W'/W)
            pulse_weight = self.get_pulse_matrix(
                self.FBconv, slice_idx, num_slices, dtc_window=dtc_window)
            active_coupler_count = None
            if self.enable_coupler_noise:
                if dtc_window is None:
                    # Perfect pulse, determined by quant level and slice_idx
                    active_mask = self._active_pulse_mask(self.FBconv, slice_idx)
                else:
                    _, clean_weight = self._clean_weight(self.FBconv)
                    # Calculates the overlap of the real pulse window with the current slice
                    active_mask = self._dtc_slice_duty(
                        self.FBconv, clean_weight, slice_idx, dtc_window)
                # Calculate how many couplers are on in a single row via MVM.
                active_coupler_count = self._active_coupler_count(
                    self.FBconv, y_hold, active_mask, z,
                    slice_idx=slice_idx if dtc_window is None else None)
            # Calculate the core MVM between the weight and the input.
            # Multiply the whole RHS but spin/current summing variation if any.
            constant_rhs = self.z_stage_rhs(None, z, y_hold, pulse_weight) if self.toggle_fast_path else None
            rhs_fn = lambda t, cur_z, pw=pulse_weight: self.z_stage_rhs(t, cur_z, y_hold, pw)
            # Perform spin update, add brownian noise if any with correct scale.
            z = self.integrate_pulse_slice(
                z, dt, rhs_fn, "z", slice_idx, constant_rhs=constant_rhs,
                active_coupler_count=active_coupler_count)
        return z

    def run_y_stage(self, y, h_hold, T_y):
        num_slices = self._num_slices("y")
        dt = T_y / float(num_slices)
        dtc_window = self._sample_dtc_window(self.FFconv, "y")
        for slice_idx in range(num_slices):
            pulse_weight = self.get_pulse_matrix(
                self.FFconv, slice_idx, num_slices, dtc_window=dtc_window)
            active_coupler_count = None
            if self.enable_coupler_noise:
                if dtc_window is None:
                    active_mask = self._active_pulse_mask(self.FFconv, slice_idx)
                else:
                    _, clean_weight = self._clean_weight(self.FFconv)
                    active_mask = self._dtc_slice_duty(
                        self.FFconv, clean_weight, slice_idx, dtc_window)
                active_coupler_count = self._active_coupler_count(
                    self.FFconv, h_hold, active_mask, y,
                    slice_idx=slice_idx if dtc_window is None else None)
            constant_rhs = self.y_stage_rhs(None, y, h_hold, pulse_weight) if self.toggle_fast_path else None
            rhs_fn = lambda t, cur_y, pw=pulse_weight: self.y_stage_rhs(t, cur_y, h_hold, pw)
            y = self.integrate_pulse_slice(
                y, dt, rhs_fn, "y", slice_idx, constant_rhs=constant_rhs,
                active_coupler_count=active_coupler_count)
        return y

    def run_one_cycle(self, y, z, cycle_params):
        # cycle_params are from self.resolve_cycle_params, which will generate t_z, t_y
        # and perform scaling to get physical time from unitless time.
        T_z, T_y = cycle_params
        z = self.run_z_stage(y, z, T_z)
        h = self.act_fn(z)
        y = self.run_y_stage(y, h, T_y)
        return y, z


class TogglePulseResetZ(TogglePulseFFFB):
    reset_z = True


class TogglePulseKeepZ(TogglePulseFFFB):
    reset_z = False


class TogglePulseODEXInitFFFB(TogglePulseFFFB):
    reset_z = True

    def resolve_cycle_params(self, ref):
        required = ["R", "C", "C_fb", "w_bits"]
        if self.odexinit_scaling_mode == "direct":
            required.extend(["s_ff", "s_fb"])
        else:
            required.extend(["k", "alpha_1state"])
        missing = [name for name in required if not hasattr(self, name)]
        if missing:
            raise ValueError("TogglePulseODEXInitFFFB requires wrapper-provided " + ", ".join(missing))
        n_cycles = 1 if self.toggle_n_cycles is None else int(self.toggle_n_cycles)
        cycle_t = self._integration_end_like(ref) / float(n_cycles)
        R = self.R
        C = self.C
        C_fb = self.C_fb
        if self.odexinit_scaling_mode == "direct":
            T_z = R * C_fb / self.s_fb
            T_y = cycle_t * R * C / self.s_ff
        else:
            k = self.k
            alpha_1state = self.alpha_1state
            T_z = k * C_fb
            T_y = cycle_t * R * R * C / (k * alpha_1state)
        if T_z <= 0 or T_y <= 0:
            raise ValueError("ODEXInit pulse stage durations must be positive.")
        return T_z, T_y

    def y_stage_rhs(self, t, y, h_hold, pulse_weight):
        summed_rhs = self._apply_pulse_module(self.FFconv, h_hold, pulse_weight)
        summed_rhs = self._apply_spin_variation("y", summed_rhs)
        return summed_rhs / (self.R * self.C)

    def _stage_capacitance(self, stage):
        return self.C_fb if stage == "z" else self.C


class TogglePulseODEXInitKeep(TogglePulseODEXInitFFFB):
    reset_z = False


class TogglePulseBlk(TogglePulseFFFB):
    """Trainable dense/expanded pulse block using signed integer pulse levels."""
    reset_z = True
    supports_pulse_training = True
    pulse_weight_encoding = "integer_level"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._training_pulse_mismatch = None

    def begin_training_pulse_mismatch(self, noise_level, mismatch_type):
        """Sample one post-quantization pulse-amplitude mismatch for this forward."""
        noise_level = float(noise_level)
        mismatch = {"type": mismatch_type}
        for key, module in (("FFconv", self.FFconv), ("FBconv", self.FBconv)):
            if key == "FBconv" and self.tie_weights:
                mismatch[key] = mismatch["FFconv"]
                continue
            ref = module.weight
            noise = torch.randn_like(ref, requires_grad=False) * noise_level
            mismatch[key] = 1.0 + noise if mismatch_type == "mul" else noise
        self._training_pulse_mismatch = mismatch

    def end_training_pulse_mismatch(self):
        self._training_pulse_mismatch = None

    def _clean_weight(self, module):
        key = "FBconv" if module is self.FBconv else "FFconv"
        if not hasattr(module, "mat") and P.is_parametrized(module, "weight"):
            # Pulse QAT exposes the live signed integer level through module.weight.
            return key, module.weight
        return super()._clean_weight(module)

    def _on_values(self, clean_values, noisy_values, key):
        mismatch = self._training_pulse_mismatch
        if mismatch is not None:
            if mismatch["type"] == "mul":
                return clean_values.sign() * mismatch[key]
            return clean_values.sign() + mismatch[key]

        if clean_values is noisy_values:
            return clean_values.sign()
        if self.mismatch_type == "mul":
            scale = torch.ones_like(clean_values)
            nonzero = clean_values != 0
            scale[nonzero] = noisy_values[nonzero] / clean_values[nonzero]
            return clean_values.sign() * scale
        return clean_values.sign() + noisy_values - clean_values

    def _pulse_values(self, clean_values, on_values, slice_idx):
        signed_mask = PulseLevelMaskImpl.apply(
            clean_values, int(slice_idx), int(self.q_hi))
        if self.mismatch_type == "mul":
            clean_sign = clean_values.sign()
            amplitude_scale = torch.ones_like(clean_values)
            nonzero = clean_sign != 0
            amplitude_scale[nonzero] = on_values[nonzero] / clean_sign[nonzero]
            return signed_mask * amplitude_scale

        active = (signed_mask != 0).to(dtype=clean_values.dtype)
        return signed_mask + (on_values - clean_values.sign()) * active

    def _timed_pulse_values(self, clean_values, on_values, slice_idx, duty):
        signed_mask = PulseLevelMaskImpl.apply(
            clean_values, int(slice_idx), int(self.q_hi))
        timed_signed = clean_values.sign() * duty
        # Preserve the existing pulse-mask surrogate gradient while replacing
        # its forward value with the DTC duty-weighted pulse.
        timed_signed = signed_mask + (timed_signed - signed_mask).detach()
        if self.mismatch_type == "mul":
            clean_sign = clean_values.sign()
            amplitude_scale = torch.ones_like(clean_values)
            nonzero = clean_sign != 0
            amplitude_scale[nonzero] = on_values[nonzero] / clean_sign[nonzero]
            return timed_signed * amplitude_scale
        return timed_signed + (on_values - clean_values.sign()) * duty


class TogglePulseBlkXInitFFFB(TogglePulseBlk):
    """Integer-level pulse block with the ODEXInitFFFB physical timing."""

    def resolve_cycle_params(self, ref):
        required = ["R", "C", "C_fb", "w_bits"]
        if self.odexinit_scaling_mode == "direct":
            required.extend(["s_ff", "s_fb"])
        else:
            required.extend(["k", "alpha_1state"])
        missing = [name for name in required if not hasattr(self, name)]
        if missing:
            raise ValueError(
                "TogglePulseBlkXInitFFFB requires wrapper-provided " + ", ".join(missing))
        n_cycles = 1 if self.toggle_n_cycles is None else int(self.toggle_n_cycles)
        cycle_t = self._integration_end_like(ref) / float(n_cycles)
        if self.odexinit_scaling_mode == "direct":
            T_z = self.R * self.C_fb / self.s_fb
            T_y = cycle_t * self.R * self.C / self.s_ff
        else:
            T_z = self.k * self.C_fb
            T_y = cycle_t * self.R * self.R * self.C / (self.k * self.alpha_1state)
        if T_z <= 0 or T_y <= 0:
            raise ValueError("ODEXInit pulse stage durations must be positive.")
        return T_z, T_y

    def y_stage_rhs(self, t, y, h_hold, pulse_weight):
        summed_rhs = self._apply_pulse_module(self.FFconv, h_hold, pulse_weight)
        summed_rhs = self._apply_spin_variation("y", summed_rhs)
        return summed_rhs / (self.R * self.C)

    def _stage_capacitance(self, stage):
        return self.C_fb if stage == "z" else self.C


class ODEFixNoiseXInitFFFB(ODEFixNoiseOffset):
    """
    Used in training. During testing, use SumAsBInitYAsXFFFB.
    """
    def __init__(self,  **kwargs):
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

    def _make_ode_fn(self, x, noisy_cu=None):
        def ode_func(t, y):
            return self.FFconv(self.act_fn(self.FBconv(y))) - noisy_cu * y
        return ode_func

    def forward(self, x, layer_idx=None):
        """
        Different from before, the gradient of the summation of weights is calculated.
        """
        y0 = self.init_y(x)
        if self.return_init:
            return y0
        weight_sum = self.FFconv.weight.view(y0.shape[1], -1).sum(-1).view(1, -1, 1, 1).expand_as(y0)
        if self.FFconv.training:
            noisy_cu = weight_sum * torch.randn_like(weight_sum, requires_grad=False, device=y0.device) * self.offset_eps
        else:
            noisy_cu = torch.zeros_like(weight_sum, requires_grad=False, device=y0.device)
        out = aca_ode_solve(self._make_ode_fn(x, noisy_cu), y0, self.option_aca)
        out = out[-1]

        if self.bypass is not None:
            out = self.bypass(out) + out
        return out

class ODEFixNoiseXInit(ODEFixNoiseXInitFFFB):
    """
    Used in training. During testing, use ODESumAsBInitYAsX.
    """
    def __init__(self,  **kwargs):
        super().__init__(**kwargs)

    def _make_ode_fn(self, x, noisy_cu=None):
        def ode_func(t, y):
            return self.FFconv(self.act_fn(x - self.FBconv(y))) - noisy_cu * y
        return ode_func

class ODEFixNoise0InitFFFB(ODEFixNoiseXInitFFFB):
    """
    Used in training. During testing, use SumAsBInitYAsXFFFB.
    """
    def __init__(self,  **kwargs):
        super().__init__(**kwargs)

    def init_y(self, x):
        return torch.zeros((x.shape[0], self.FFconv.weight.shape[0], x.shape[2], x.shape[3]), device=x.device)

class ODEFixNoise0InitExpand(ODEFixNoiseXInit):
    """
    Used in training. During testing, use ODESumAsBInitYAsX.
    """
    def __init__(self,  **kwargs):
        super().__init__(**kwargs)

    def init_y(self, x):
        return torch.zeros((x.shape[0], self.FFconv.weight.shape[0], x.shape[2], x.shape[3]), device=x.device)

    def _make_ode_fn(self, x, noisy_cu=None):
        def ode_func(t, y):
            return self.FFconv(self.act_fn(x - self.FBconv(y))) - noisy_cu * y
        return ode_func

class FixNoiseXInitFFFBNoExpand(ODEFixNoiseXInitFFFB):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x, layer_idx=None):
        """
        Different with its parent class ODEFixNoiseXInitFFFB, the gradient of the summation of weights is NOT calculated.
        The weight sum is NOT expanded.
        """
        y0 = self.init_y(x)
        if self.return_init:
            return y0
        with torch.no_grad():
            weight_sum = self.FFconv.weight.data.view(y0.shape[1], -1).sum(-1).view(1, -1, 1, 1)
            if self.FFconv.training:
                noisy_cu = weight_sum * torch.randn_like(weight_sum, requires_grad=False, device=y0.device) * self.offset_eps
            else:
                noisy_cu = torch.zeros_like(weight_sum, requires_grad=False, device=y0.device)
        out = aca_ode_solve(self._make_ode_fn(x, noisy_cu), y0, self.option_aca)
        out = out[-1]

        if self.bypass is not None:
            out = self.bypass(out) + out
        return out

####################################################################################
# Inference blocks for noisy self-coupling training
####################################################################################
class ODESumAsBInitYAsX(ODESumAsBInitY):
    """
    Used in test.
    """
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

class SumAsBInitYAsXFFFB(ODESumAsBInitYAsX):
    """
    Used in test.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _make_ode_fn(self, x, noisy_cu=None):
        def ode_func(t, y):
            weight_sum = self.FFconv.weight.view(y.shape[1], -1).sum(-1)
            offset = (weight_sum.view(1, -1, 1, 1) - self.b0[0]) * y
            return self.FFconv(self.act_fn(self.FBconv(y))) - offset
        return ode_func

class ODESumAsBInitYAs0(ODESumAsBInitYAsX):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def init_y(self, x):
        return torch.zeros((x.shape[0], self.FFconv.weight.shape[0], x.shape[2], x.shape[3]), device=x.device)

class SumAsBInitYAs0FFFB(SumAsBInitYAsXFFFB):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def init_y(self, x):
        return torch.zeros((x.shape[0], self.FFconv.weight.shape[0], x.shape[2], x.shape[3]), device=x.device)
####################################################################################
####################################################################################

####################################################################################
# Abs Summation
####################################################################################
class SelfCUAbsSumFFFB(ODEBlockXInit):
    """
    Initialize y0 = x or concat([x,x]).
    self.b0 is trainable, meaning that the active self-coupling units are not compensate exactly
    the summation of the abs of weights times voltage term.
    Active.
    Todo: How do we add noise to the active self-coupling terms?
    Note: Initialize b with zero failed to train.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _make_ode_fn(self, x):
        def ode_func(t, y):
            weight_sum = self.FFconv.weight.view(y.shape[1], -1).abs().sum(-1)
            offset = (weight_sum.view(1, -1, 1, 1) - self.b0[0]) * y
            # logging.warning("weight sum mean: {}, median: {}, max: {}, min: {}".format(
            #     weight_sum.mean(), weight_sum.median(), weight_sum.max(), weight_sum.min()))
            # logging.warning("b0 sum mean: {}, median: {}, max: {}, min: {}".format(
            #     self.b0[0].mean(), self.b0[0].median(), self.b0[0].max(), self.b0[0].min()))
            return self.FFconv(self.act_fn(self.FBconv(y))) - offset
        return ode_func

class SelfCUAbsSumFFFBInitB(SelfCUAbsSumFFFB):
    """
    When inference, use SelfCUAbsSumFFFB to avoid overwriting the loaded trained b0 with
    the summation of the abs of weights.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        with torch.no_grad():
            logging.warning("Initialize b0 as the summation of the absolute value of weights")
            ff_weight = self.FFconv.weight.data
            self.b0 = nn.ParameterList([ff_weight.view(ff_weight.shape[0], -1).abs().sum(-1).view(1, -1, 1, 1)])
        self.clean_params["b0"] = nn.Parameter(self.b0[0].clone())

class SelfCUAbsSumFFFBFixNoise(ODEFixNoiseXInitFFFB):
    """
    Assuming the active self-coupling units are exactly compensating the summation of the abs of weights times
    the voltage term.
    The noisy_cu can be interpreted as the noise of the active self-coupling units.
    Todo: Check the type of noise in the active self-coupling terms (or current mirror) in order to model
        the behavior of the real circuit. If the self-coupling terms are active, then using the old
        multiplicative mismatch added to self.b0 may not be appropriate.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _make_ode_fn(self, x, noisy_i=None):
        def ode_func(t, y):
            return self.FFconv(self.act_fn(self.FBconv(y))) - noisy_i
        return ode_func

    def forward(self, x, layer_idx=None):
        """
        Different from parent class:
        1. Using the summation of absolute value of weights.
        2. Expand the last two dimensions of y0 only. Does NOT expand the batch dimension.
        Todo: If the self-coupling terms are now active (or current mirrors), then the noisy_cu should be
            directly added to the dynamics as a whole, mimicking the current instead of having noisy_cu * y
            in the dynamics.
        """
        y0 = self.init_y(x)
        if self.return_init:
            return y0
        weight_sum = self.FFconv.weight.view(y0.shape[1], -1).abs().sum(-1).view(1, -1, 1, 1).expand(1, -1, y0.shape[2], y0.shape[3])
        # logging.warning("weight sum mean: {}, median: {}, max: {}, min: {}".format(
        #     weight_sum.mean(), weight_sum.median(), weight_sum.max(), weight_sum.min()))
        if self.FFconv.training:
            noisy_i = y0 * weight_sum * torch.randn_like(weight_sum, requires_grad=False, device=y0.device) * self.offset_eps
        else:
            noisy_i = torch.zeros_like(weight_sum, requires_grad=False, device=y0.device)
        out = aca_ode_solve(self._make_ode_fn(x, noisy_i), y0, self.option_aca)
        out = out[-1]

        if self.bypass is not None:
            out = self.bypass(out) + out
        return out

class SelfCUAbsSumFFFBNoisy(SelfCUAbsSumFFFBFixNoise):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # sqrt((k_B * T) * R * df * 4) = sqrt(4.16e-21 * 1e5 * 10e9 * 4)
        self.offset_eps = 0.002

    def _make_ode_fn(self, x, noisy_i=None):
        def ode_func(t, y):
            _noisy_v = y.abs().max() * self.offset_eps * torch.randn_like(y, requires_grad=False, device=y.device)
            return self.FFconv(self.act_fn(x - self.FBconv(y))) - _noisy_v
        return ode_func

    def forward(self, x, layer_idx=None):
        y0 = self.init_y(x)
        if self.return_init:
            return y0
        out = aca_ode_solve(self._make_ode_fn(x, None), y0, self.option_aca)
        out = out[-1]

        if self.bypass is not None:
            out = self.bypass(out) + out
        return out
####################################################################################
####################################################################################


class ODEBlockPCMinusY(ODEBlockPC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _make_ode_fn(self, x):
        def ode_func(t, y):
            weight_sum = self.FFconv.weight.view(y.shape[1], -1).sum(-1)
            # weight_sum = self.FFconv.weight.data.view(y.shape[1], -1).mean(-1)
            offset = weight_sum.view(1, -1, 1, 1) * y
            return self.FFconv(self.act_fn(x - self.FBconv(y))) - offset
        return ode_func

    def forward(self, x, layer_idx=None):
        y0 = self.act_fn(self.FFconv(x))
        if self.return_init:
            return y0

        # class OdeFuncClass(nn.Module):
        #     def __init__(self, ff_conv, act_fn, fb_conv):
        #         super().__init__()
        #         self.ff_conv = ff_conv
        #         self.act_fn = act_fn
        #         self.fb_conv = fb_conv
        #     def forward(self, t, y):
        #         # weight_sum = self.ff_conv.weight.data.view(y.shape[1], -1).sum(-1)
        #         weight_sum = self.ff_conv.weight.data.view(y.shape[1], -1).mean(-1)
        #         offset = weight_sum.view(1, -1, 1, 1) * y
        #         return self.ff_conv(self.act_fn(x - self.fb_conv(y))) - offset
        # ode_func = OdeFuncClass(self.FFconv, self.act_fn, self.FBconv)
        #
        # self.integration_time = self.integration_time.type_as(x)

        out = aca_ode_solve(self._make_ode_fn(x), y0, self.option_aca)
        out = out[-1]

        if self.bypass is not None:
            out = self.bypass(out) + out
        return out

class ODEFFFBConv(ODEBlockPC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _make_ode_fn(self, x):
        def ode_func(t, y):
            return self.act_fn(self.FFconv(self.act_fn(self.FBconv(y))))
        return ode_func

class ODEFFConv(ODEBlockPC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        chan_diff = self.FFconv.out_channels - self.FFconv.in_channels
        self.pad_a = chan_diff // 2
        self.pad_b = self.pad_a
        if chan_diff % 2 != 0:
            self.pad_b += 1

    def init_y(self, x):
        return torch.zeros((x.shape[0], self.FFconv.weight.shape[0], x.shape[2], x.shape[3]), device=x.device)

    def _make_ode_fn(self, x):
        def ode_func(t, y):
            return self.act_fn(self.FBconv(self.act_fn(self.FFconv(x))))
        return ode_func

    def forward(self, x, layer_idx=None):
        y0 = self.init_y(x)
        if self.return_init:
            return y0
        self.integration_time = self.integration_time.type_as(x)

        out = aca_ode_solve(self._make_ode_fn(x), y0, self.option_aca)
        # out = odeint(ode_func, y0, self.integration_time, rtol=self.tol, atol=self.tol, method=self.method)

        # Have to add residual connection, otherwise the plain CNN is not trainable
        out = out[-1] + F.pad(x, (0, 0, 0, 0, self.pad_a, self.pad_b), "constant", 0)

        # Todo: Verify the Jacobian of aca_ode_solve and plain feedforward operation
        # with torch.no_grad():
        #     out_ff = self.option_aca["t1"] * self.act_fn(self.FBconv(self.act_fn(self.FFconv(x))))
        #     out_diff = (out_ff - out).abs()
        #     logging.warning("out and ff diff max: {}, mean: {}".format(out_diff.max(), out_diff.mean()))

        if self.bypass is not None:
            out = self.bypass(out) + out
        return out


####################################################################################
# 2 state ode blocks
# dy/dt = f(z); dz/dt = g(y)
####################################################################################
class _FuncWrapper(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func
    def forward(self, t, y):
        return self.func(t, y)

class ODEState2FFFB(ODEBlockXInit):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.y_init_with = "x"
        self.z_init_with = "conv"

        # SDE noise related members
        self.offset_eps = 0.002
        self.eps_scale = None
        self.option_aca["noise_type"] = self.sde_noise_type

    def _set_ode_option(self, option_dict, time_ratio, t1_all, t_step):
        # Note: Avoid calling this function multiple times with the same copy.
        option_dict["t1"] = t1_all * time_ratio
        option_dict["t_eval"] = [_ts * time_ratio for _ts in self.integration_time.tolist()]
        option_dict["h"] = t_step * time_ratio if t_step is not None else t_step
        return option_dict

    def _set_eps(self, x):
        if self.eps_scale is not None:
            with torch.no_grad():
                if isinstance(self.FFconv, nn.Conv2d):
                    ff_ws = self.FFconv.weight.data.view(
                        self.out_chan, -1).abs().sum(-1).sqrt().view(1, -1, 1, 1)
                else:
                    # Assuming unrolled
                    ff_ws = self.FFconv.inp_param_sum(x)
                if isinstance(self.FBconv, nn.ConvTranspose2d):
                    fb_ws = self.FBconv.weight.data.view(
                        self.in_chan, -1).abs().sum(-1).sqrt().view(1, -1, 1, 1)
                elif isinstance(self.FBconv, nn.Conv2d):
                    fb_ws = self.FBconv.weight.data.flip([2,3]).permute([1,0,2,3]).view(
                        self.in_chan, -1).abs().sum(-1).sqrt().view(1, -1, 1, 1)
                else:
                    # Assuming unrolled
                    fb_ws = self.FBconv.inp_param_sum(x)
                weight_sum = (ff_ws, fb_ws)
            # directly use the eps_scale
            if hasattr(self, "option_init"):
                self.option_init["eps"] = self.eps_scale * self.offset_eps * weight_sum[1]
            self.option_aca["eps"] = tuple(self.eps_scale * self.offset_eps * _ws for _ws in weight_sum)
        else:
            eps_scale = 1 if self.sde_noise_type == "mul" else x.abs().max()
            if hasattr(self, "option_init"):
                self.option_init["eps"] = eps_scale * self.offset_eps
            self.option_aca["eps"] = eps_scale * self.offset_eps

    def init_y(self, x):
        if self.y_init_with == "x":
            y0 = super().init_y(x)
        elif self.y_init_with == "0":
            y0 = torch.zeros((x.shape[0], self.out_chan, x.shape[2], x.shape[3]), device=x.device)
        elif self.y_init_with == "conv":
            y0 = self.act_fn(self.FFconv(x))
        else:
            raise NotImplementedError

        if self.z_init_with == "x":
            z0 = x
        elif self.z_init_with == "0":
            z0 = torch.zeros((x.shape[0], self.in_chan, x.shape[2], x.shape[3]), device=x.device)
        elif self.z_init_with == "conv":
            z0 = self.FBconv(y0)
        else:
            raise NotImplementedError
        return y0, z0

    def _make_ode_fn(self, x):
        def ode_func(t, y):
            y_, z_ = y
            dy = self.FFconv(self.act_fn(z_))
            dz = self.FBconv(y_) - z_
            return dy, dz
        return _FuncWrapper(ode_func)

    def forward(self, x, layer_idx=None):
        yz = self.init_y(x)
        if self.return_init:
            return yz[0]
        self.integration_time = self.integration_time.type_as(x)

        out = aca_ode_solve(self._make_ode_fn(x), yz, self.option_aca)
        # out = odeint(ode_func, y0, self.integration_time, rtol=self.tol, atol=self.tol, method=self.method)
        out = out[0][-1]

        if self.bypass is not None:
            out = self.bypass(out) + out
        return out

    def forward_full_steps(self, x, layer_idx=None):
        # Returning the full trajectory of the solver.
        yz = self.init_y(x)
        self.integration_time = self.integration_time.type_as(x)

        out = aca_ode_solve(self._make_ode_fn(x), yz, self.option_aca, full_traj=True)
        # out = odeint(ode_func, y0, self.integration_time, rtol=self.tol, atol=self.tol, method=self.method)
        out = out

        return out

class State2InitYZ(ODEState2FFFB):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.y_init_with = "conv"
        self.z_init_with = "conv"

class State2InitYAsXZAs0(ODEState2FFFB):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.y_init_with = "x"
        self.z_init_with = "0"

class State2InitYAs0ZAsX(ODEState2FFFB):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.y_init_with = "0"
        self.z_init_with = "x"

class State2InitYAsXZAsX(ODEState2FFFB):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.y_init_with = "x"
        self.z_init_with = "x"


class State2NoMinusZ(ODEState2FFFB):
    """
    dy/dt = W_FF (f(z))
    dz/dt = W_FB (y)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _make_ode_fn(self, x):
        def ode_func(t, y):
            y_, z_ = y
            dy = self.FFconv(self.act_fn(z_))
            dz = self.FBconv(y_)
            return dy, dz
        return _FuncWrapper(ode_func)

class State2NoMinusZYAsXZAs0(State2NoMinusZ):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.y_init_with = "x"
        self.z_init_with = "0"

class State2NoMinusZYAs0ZAsX(State2NoMinusZ):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.y_init_with = "0"
        self.z_init_with = "x"

class State2NoMinusZYAsXZAsX(State2NoMinusZ):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.y_init_with = "x"
        self.z_init_with = "x"

class S2NoMinusZChargeZ(State2NoMinusZ):
    def __init__(self, time_split=0.5, **kwargs):
        super().__init__(**kwargs)
        self.time_split = time_split
        self.option_init = copy.deepcopy(self.option_aca)
        t1_all = self.option_aca["t1"]
        t_step = self.option_aca["h"]

        # Note: Now self.integration_time can not be used.
        self.option_init = self._set_ode_option(self.option_init, time_split, t1_all, t_step)
        self.option_aca = self._set_ode_option(self.option_aca, 1 - time_split, t1_all, t_step)

    def _make_z_ode_fn(self, y):
        def ode_func(t, z):
            return self.FBconv(y)
        return ode_func

    def init_y(self, x):
        # set this to avoid the conv op in the parent's init_y
        self.z_init_with = "0"

        # init y and z
        y0 = super().init_y(x)[0]
        z0 = torch.zeros_like(x, device=x.device) # Todo: add option for z0 = x
        z0 = aca_ode_solve(self._make_z_ode_fn(y0), z0, self.option_init)[-1]
        return y0, z0

class S2NoMinusZChargeZMinus(S2NoMinusZChargeZ):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _make_z_ode_fn(self, y):
        def ode_func(t, z):
            return self.FBconv(y) - z
        return ode_func

class S2NoMinusZChgZNoisyI(S2NoMinusZChargeZ):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # sqrt((k_B * T) * R * df * 4) = sqrt(4.16e-21 * 1e5 * 10e9 * 4)
        # Todo: use voltage or current?
        self.option_init["noise_type"] = self.sde_noise_type

    def init_y(self, x):
        z0 = torch.zeros_like(x, device=x.device)  # Todo: add option for z0 = x
        if self.FFconv.stride == 2 or self.FFconv.stride == (2, 2):
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
        # init y with x
        if self.chan_diff == 0:
            y0 = x
        elif self.in_chan * 2 == self.out_chan:
            y0 = torch.cat([x, x], dim=1)
        else:
            y0 = F.pad(torch.cat([x for _ in range(self.out_chan // self.in_chan)], dim=1),
                         (0, 0, 0, 0, 0, self.out_chan % self.in_chan), "constant", 0)

        # init z
        z0 = aca_ode_solve(self._make_z_ode_fn(y0), z0, self.option_init)[-1]
        return y0, z0

    def forward(self, x, layer_idx=None):
        self._set_eps(x)

        yz = self.init_y(x)
        if self.return_init:
            return yz[0]
        self.integration_time = self.integration_time.type_as(x)

        out = aca_ode_solve(self._make_ode_fn(x), yz, self.option_aca)
        # out = odeint(ode_func, y0, self.integration_time, rtol=self.tol, atol=self.tol, method=self.method)
        out = out[0][-1]

        if self.bypass is not None:
            out = self.bypass(out) + out
        return out

class S2NoMinusZChgZMinusNoisyI(S2NoMinusZChgZNoisyI):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _make_z_ode_fn(self, y):
        def ode_func(t, z):
            return self.FBconv(y) - z
        return ode_func


class S2NoisyIYAsXZAsX(State2NoMinusZ):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.y_init_with = "x"
        self.z_init_with = "x"

    def forward(self, x, layer_idx=None):
        self._set_eps(x)
        return super().forward(x, layer_idx)

class S2NoisyIYAs0ZAsX(State2NoMinusZ):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.y_init_with = "0"
        self.z_init_with = "x"

    def forward(self, x, layer_idx=None):
        self._set_eps(x)
        return super().forward(x, layer_idx)

class S2NoisyIYAsXZAs0(State2NoMinusZ):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.y_init_with = "x"
        self.z_init_with = "0"

    def forward(self, x, layer_idx=None):
        self._set_eps(x)
        return super().forward(x, layer_idx)


class S2Circ(State2NoMinusZ):
    def __init__(self, patch_node=8, patch_stride=4, patch_cycle=5, patch_pad=0, fold_scalar=None,
                 time_patch_ratio=0.6, **kwargs):
        super().__init__(**kwargs)
        self.patch_node = patch_node
        self.patch_stride = patch_stride
        self.patch_cycle = patch_cycle
        self.patch_pad = patch_pad

        # self.option_aca_raw = deepcopy(self.option_aca)
        self.option_patch = deepcopy(self.option_aca)
        t1_all = self.option_aca["t1"]
        t_step = self.option_aca["h"]
        self.option_patch = self._set_ode_option(self.option_patch, time_patch_ratio, t1_all, t_step)
        self.fold_scalar = patch_node if fold_scalar is None else fold_scalar

        self.patch_overlap = patch_node != patch_stride
        if not self.patch_overlap:
            self.fold_scalar = 1

        # Each patch's compute time = [t_end * (1 - time_split)] / patch_cycle

    def _make_ode_fn(self, x):
        def ode_func(t, yz):
            y_, z_ = yz
            dy = self.FFconv(self.act_fn(z_))
            dz = self.FBconv(y_)
            return dy, dz
        return _FuncWrapper(ode_func)

    def forward(self, x, layer_idx=None):
        self._set_eps(x)
        yz = self.init_y(x)
        if self.return_init:
            return yz[0]

        x_h, x_w = x.shape[2], x.shape[3]

        # Always true
        if x_h > self.patch_node and x_w > self.patch_node:
            for i in range(self.patch_cycle):
                # logging.warning("Cycle: {}, yz shape: {}, yz mean: {}".format(i, [_s.shape for _s in yz], [_s.mean() for _s in yz]))
                if self.patch_overlap:
                    yz = self._run_one_cycle(x, yz)
                else:
                    yz = self._run_one_cycle(x, yz, "raw")
                    yz = self._run_one_cycle(x, yz, "h")
                    yz = self._run_one_cycle(x, yz, "raw")
                    yz = self._run_one_cycle(x, yz, "v")
            yz = yz[0]
        else:
            # cur_t_aca, cur_h = self.option_aca_raw["t1"], self.option_aca_raw["h"]
            # self.integration_time[-1] = cur_t_aca
            # self.option_aca = self._set_ode_option(
            #     self.option_aca, 3.5, cur_t_aca, cur_h)
            yz = aca_ode_solve(self._make_ode_fn(x), yz, self.option_aca)
            yz = yz[0][-1]
            # logging.warning(
            #     "Cycle: {}, yz shape: {}, yz mean: {}".format("One Time solve", yz.shape, yz.mean()))

        if self.bypass is not None:
            yz = self.bypass(yz) + yz
        return yz

    def _run_one_cycle(self, x, yz, mode="raw"):
        if not isinstance(yz, Tuple):
            yz = (yz,)

        bs = x.shape[0]
        yz_c = tuple(_s.shape[1] for _s in yz)
        yz_h = tuple(_s.shape[2] for _s in yz)
        yz_w = tuple(_s.shape[3] for _s in yz)

        # If patch_node is bigger than the input feature size, don't split into multiple patches.
        yz_psz = tuple((min(self.patch_node, _h), min(self.patch_node, _w)) for _h, _w in zip(yz_h, yz_w))
        yz_stride = tuple((self.patch_stride if self.patch_node < _h else 1,
                           self.patch_stride if self.patch_node < _w else 1)
                          for _h, _w in zip(yz_h, yz_w))

        # Keep the padding
        pad = int(getattr(self, "patch_pad", 0))
        p_h, p_w = yz_psz[0]
        s_h, s_w = yz_stride[0]
        H, W = yz_h[0], yz_w[0]
        H_pad, W_pad = H + 2 * pad, W + 2 * pad

        # half-patch shifts
        sh_h = p_h // 2
        sh_w = p_w // 2

        # roll helper
        def maybe_roll(t):
            if mode == "h":
                return torch.roll(t, shifts=(0, -sh_w), dims=(-2, -1))
            elif mode == "v":
                return torch.roll(t, shifts=(-sh_h, 0), dims=(-2, -1))
            else:
                return t

        # unfold grid sizes on the padded canvas
        n_h = (H_pad - p_h) // s_h + 1
        n_w = (W_pad - p_w) // s_w + 1

        # which columns to KEEP (full windows only after shift)
        if mode == "raw":
            keep_i_max = n_h
            keep_j_max = n_w
        elif mode == "h":
            keep_i_max = n_h
            keep_j_max = max(0, (W_pad - p_w - sh_w) // s_w + 1)
        elif mode == "v":
            keep_i_max = max(0, (H_pad - p_h - sh_h) // s_h + 1)
            keep_j_max = n_w
        else:
            raise ValueError(f"unknown mode: {mode}")

        keep_cols = [i * n_w + j
                     for i in range(keep_i_max)
                     for j in range(keep_j_max)]

        # Unfold after roll
        yz_src = tuple(maybe_roll(_s) for _s in yz)

        yz_cols_full = tuple(F.unfold(_s, kernel_size=_p_sz, stride=_p_stride, padding=pad)
                             for _s, _p_sz, _p_stride in zip(yz_src, yz_psz, yz_stride))

        # (bs, C*k*k, n_h*n_w) -> keep cols -> (bs, N_kept, C*k*k)
        yz = tuple(_cols.index_select(2, _cols.new_tensor(keep_cols, dtype=torch.long)).transpose(1, 2)
                   for _cols in yz_cols_full)

        yz_patch_num = tuple(_s.shape[1] for _s in yz)

        # Reshape to (BS*N_kept, C, p_h, p_w)
        yz = tuple(_s.reshape(bs * _np, _nc, p_h, p_w)
                   for _s, _np, _nc in zip(yz, yz_patch_num, yz_c))

        self.integration_time = self.integration_time.type_as(x)

        # Run node on each patch
        yz = aca_ode_solve(self._make_ode_fn(x), yz, self.option_patch)
        yz = tuple(_s[-1] for _s in yz)

        # Place kept patch to full feature map, then fold back
        yz_cols_kept = tuple(_s.reshape(bs, _np, _nc * p_h * p_w).transpose(1, 2).contiguous()
                             for _s, _np, _nc in zip(yz, yz_patch_num, yz_c))

        # logging.warning("yz_cols_kept shape: {}".format([_.shape for _ in yz_cols_kept]))

        yz_cols_scatter = []
        for _cols_full, _cols_kept in zip(yz_cols_full, yz_cols_kept):
            _cols_out = _cols_full.clone()
            _cols_out.index_copy_(2, _cols_full.new_tensor(keep_cols, dtype=torch.long), _cols_kept)
            yz_cols_scatter.append(_cols_out)

        yz = tuple(F.fold(_cols_sc, output_size=(_h, _w),
                          kernel_size=_p_sz, stride=_p_stride, padding=pad)
                   for _cols_sc, _h, _w, _p_sz, _p_stride in
                   zip(yz_cols_scatter, yz_h, yz_w, yz_psz, yz_stride))

        # reverse the roll so outputs align with the canonical grid
        if mode == "h":
            yz = tuple(torch.roll(_s, shifts=(0, +sh_w), dims=(-2, -1)) for _s in yz)
        elif mode == "v":
            yz = tuple(torch.roll(_s, shifts=(+sh_h, 0), dims=(-2, -1)) for _s in yz)

        yz = tuple(_s / self.fold_scalar for _s in yz)

        return yz


class S2CircYAsXZasX(S2Circ):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.y_init_with = "x"
        self.z_init_with = "x"
        # self.fold_scalar = self.patch_node

class S2CircYAs0ZasX(S2Circ):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.y_init_with = "0"
        self.z_init_with = "x"
        # self.fold_scalar = int(self.patch_node ** 0.8)
        # self.fold_scalar = int(self.patch_node ** 0.8) * (self.patch_node * (self.patch_node - self.patch_stride) / 32)

class S2CircYAsXZas0(S2Circ):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.y_init_with = "x"
        self.z_init_with = "0"
        # self.fold_scalar = int(self.patch_node ** 0.8)
        # self.fold_scalar = int((self.patch_node * (self.patch_node * (self.patch_node - self.patch_stride) / 32)) ** 1.2)
        # logging.warning("Fold scaler: {}".format(self.fold_scalar))
####################################################################################
####################################################################################
class OutputQuantImpl(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, v_dd, enob):
        n_interval = 2 ** enob - 1
        v_dd = torch.tensor(v_dd, device=x.device, dtype=x.dtype)
        if n_interval <= 0:
            x_clipped = x.clamp(min=-v_dd, max=v_dd)
            ctx.save_for_backward(x, v_dd)
            return x_clipped

        ctx.save_for_backward(x, v_dd)

        x = x.clamp(min=-v_dd, max=v_dd)

        delta = (2.0 * v_dd) / n_interval
        x_q = torch.round((x + v_dd) / delta) * delta - v_dd

        x_q = x_q.clamp(min=-v_dd, max=v_dd)
        return x_q

    @staticmethod
    def backward(ctx, grad_output):
        x_in, v = ctx.saved_tensors

        mask = (x_in >= -v) & (x_in <= v)
        grad_x = grad_output * mask.to(dtype=grad_output.dtype)

        return grad_x, None, None

class PulseLevelMaskImpl(torch.autograd.Function):
    """Exact integer-level pulse mask with an explicit boundary subgradient."""

    @staticmethod
    def forward(ctx, level, slice_idx, q_hi):
        slice_level = level.new_tensor(float(slice_idx))
        positive = (level - slice_level).clamp(min=0.0, max=1.0)
        negative = (-level - slice_level).clamp(min=0.0, max=1.0)
        ctx.save_for_backward(level)
        ctx.slice_idx = int(slice_idx)
        ctx.q_hi = int(q_hi)
        return positive - negative

    @staticmethod
    def backward(ctx, grad_output):
        (level,) = ctx.saved_tensors
        slice_level = level.new_tensor(float(ctx.slice_idx))
        u_positive = level - slice_level
        u_negative = -level - slice_level

        def clamp_grad(u):
            inside = ((u > 0.0) & (u < 1.0)).to(dtype=level.dtype)
            boundary = ((u == 0.0) | (u == 1.0)).to(dtype=level.dtype)
            return inside + 0.5 * boundary

        grad_level = clamp_grad(u_positive) + clamp_grad(u_negative)
        if ctx.slice_idx == ctx.q_hi - 1:
            at_positive_limit = level == float(ctx.q_hi)
            at_negative_limit = level == -float(ctx.q_hi)
            grad_level = torch.where(
                at_positive_limit | at_negative_limit,
                torch.ones_like(grad_level), grad_level)
        return grad_output * grad_level, None, None


class QuantizationImpl(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, s, q_min, q_max, w_scalar=1.0):
        q_weight = weight * s * q_max
        q_mask = (q_weight >= q_min) & (q_weight <= q_max)

        # Quantize to {0, 1, ..., q_max}
        q_weight = torch.clamp(q_weight.round(), min=q_min, max=q_max)

        if w_scalar == 1.0:
            ctx.save_for_backward(q_mask, s, q_weight.new_tensor(1.0))
            return q_weight / q_max

        else:
            # Remap quantize the range (w_scalar / q_max, 1) into (q_max - 1) girds
            _sign = q_weight.sign()
            _k = q_weight.abs()  # integer levels in [0, q_max]

            _first = float(w_scalar) / float(q_max)                # first nonzero level
            _delta = (1.0 - _first) / float(q_max - 1)            # spacing for the remaining levels

            # q_weight in [-1,1] after remap, with the first quantization step being w_scalar / q_max
            # and steps after being _delta
            q_weight = (_first + (_k.to(dtype=weight.dtype) - 1.0) * _delta) * _sign
            q_weight.masked_fill_(_k == 0, 0.0)
            q_weight = torch.clamp(q_weight, -1.0, 1.0)  # Guard

            # Calculate d q_weight_remapped / d q_weight (q_weight here is {0, 1/q_max, 2/q_max, ..., 1}, not integer)
            # Two options:
            # 1. for k==0: gradient = 1
            # 2. for k == 0: gradient = 0 (Seems to have higher acc and worse robustness)
            masked_scale = q_weight.new_full(q_weight.shape, _delta * q_max)
            masked_scale.masked_fill_(_k == 0, 0.0)
            # masked_scale.masked_fill_(_k == 1, w_scalar)

            ctx.save_for_backward(q_mask, s, masked_scale)
            return q_weight

    @staticmethod
    def backward(ctx, grad_output):
        q_mask, s, masked_scale = ctx.saved_tensors
        return s * grad_output * q_mask * masked_scale, None, None, None, None


class SymQuantizeWeight(nn.Module):
    def __init__(self, w_bits=8, w_scalar=1.0, **kwargs):
        super().__init__()
        self.register_buffer("w_bits", torch.tensor(w_bits))
        self.register_buffer("upper", torch.tensor((1 << (w_bits - 1)) - 1))
        self.register_buffer("lower", -self.upper)
        self.register_buffer("s_w", torch.tensor(1.0))
        self.register_buffer("w_scalar", torch.tensor(w_scalar))

    def forward(self, layer_weight: nn.Parameter):
        return QuantizationImpl.apply(layer_weight, self.s_w, self.lower, self.upper, self.w_scalar)

    def compute_s(self, layer_weight: nn.Parameter):
        self.s_w.copy_(_symmetric_qat_weight_scale(layer_weight))


class PulseQuantizationImpl(torch.autograd.Function):
    """Symmetric integer-level quantization with a straight-through backward."""

    @staticmethod
    def forward(ctx, weight, s, q_min, q_max):
        scaled = weight * s * q_max
        q_mask = (scaled >= q_min) & (scaled <= q_max)
        level = scaled.round().clamp(min=q_min, max=q_max)
        ctx.save_for_backward(q_mask, s, q_max)
        return level

    @staticmethod
    def backward(ctx, grad_output):
        q_mask, s, q_max = ctx.saved_tensors
        return s * q_max * grad_output * q_mask, None, None, None


class PulseSymQuantizeWeight(SymQuantizeWeight):
    """Default symmetric QAT quantizer returning signed pulse levels."""

    def __init__(self, w_scalar=1.0, **kwargs):
        if not np.allclose(float(w_scalar), 1.0):
            raise ValueError("Pulse QAT requires uniform integer levels and w_scalar=1.")
        super().__init__(w_scalar=w_scalar, **kwargs)

    def forward(self, layer_weight: nn.Parameter):
        return PulseQuantizationImpl.apply(
            layer_weight, self.s_w, self.lower, self.upper)


class LSQImpl(torch.autograd.Function):
    """
    Todo: Add noise inject training related code here.
    """
    @staticmethod
    def forward(ctx, weight, s, q_min, q_max, s_g_scale, w_scalar=1.0):
        q_weight = weight * s
        q_mask = (q_weight >= q_min) & (q_weight <= q_max)
        q_weight = q_weight.round()
        ctx.save_for_backward(q_mask, s, q_weight, q_max, s_g_scale)
        q_weight = torch.clamp(q_weight, min=q_min, max=q_max)
        q_weight = q_weight / q_max

        # scale_mask = q_weight.abs() < 1.0 # positions to scale
        # masked_scale = torch.ones_like(q_weight)
        # masked_scale[scale_mask] = w_scalar
        # q_weight = q_weight * masked_scale
        # q_weight = torch.clamp(q_weight, -1.0, 1.0)  # Guard
        #
        # ctx.save_for_backward(q_mask, s, masked_scale)

        return q_weight

    @staticmethod
    def backward(ctx, grad_out):
        q_mask, s, q_weight, q_max, s_g_scale = ctx.saved_tensors
        return s * grad_out * q_mask / q_max, s_g_scale * (q_weight * grad_out * q_mask).sum() / q_max, None, None, None, None


class LSQWeight(SymQuantizeWeight):
    def __init__(self, layer_weight=torch.tensor(1.0), **kwargs):
        super().__init__(**kwargs)
        with torch.no_grad():
            self.s_w_Param = nn.Parameter( torch.sqrt(self.upper.cpu()) / (2 * layer_weight.abs().mean().cpu()) )
            self.register_buffer("s_g_scale", 1 / torch.sqrt(layer_weight.numel() * self.upper))

    def forward(self, layer_weight: nn.Parameter):
        return LSQImpl.apply(layer_weight, self.s_w_Param, self.lower, self.upper, self.s_g_scale, self.w_scalar)

    def compute_s(self, layer_weight: nn.Parameter):
        # Set the parameter to registered buffer for saving purpose
        with torch.no_grad():
            self.s_w.copy_(self.s_w_Param.data.div(self.upper))


class ODEWrapperRC(nn.Module):
    def __init__(self, ode_block: ODEBlockPC, state_bound=50.0, R=1e5, C=49e-15, v_dd=1.0, patch=True, **kwargs):
        super().__init__()
        self.ode_block = ode_block

        self.R = R
        self.C = C
        self.v_dd = v_dd

        setattr(self.ode_block, "R", R)
        setattr(self.ode_block, "C", C)
        setattr(self.ode_block, "v_dd", v_dd)

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

    def wrap_input(self, x):
        return self.q * x

    def unwrap_output(self, out):
        if isinstance(out, (tuple, list)):
            return type(out)(self.unwrap_output(o) for o in out)
        return out / self.q

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

        self.nonlinear_R = kwargs.pop("nonlinear_R", False)
        self.nonlinear_R_table = kwargs.pop("nonlinear_R_table", None)
        self.nonlinear_R_mc_curve_index = kwargs.pop(
            "nonlinear_R_mc_curve_index", None)
        self.nonlinear_R_mc_quantity = kwargs.pop(
            "nonlinear_R_mc_quantity", "conductance")
        self.nonlinear_R_curve_sharing = kwargs.pop(
            "nonlinear_R_curve_sharing", "shared")
        self.nonlinear_R_curve_bank_indices = kwargs.pop(
            "nonlinear_R_curve_bank_indices", None)
        self.nonlinear_R_curve_seed = kwargs.pop(
            "nonlinear_R_curve_seed", None)
        self.nonlinear_R_curve_edge_chunk_size = kwargs.pop(
            "nonlinear_R_curve_edge_chunk_size", 65536)
        self.nonlinear_R_curve_sampling = kwargs.pop(
            "nonlinear_R_curve_sampling", "empirical_with_replacement")
        self.nonlinear_R_train_mode = str(kwargs.pop(
            "nonlinear_R_train_mode", "none")).lower()
        self.nonlinear_R_corner_range = kwargs.pop(
            "nonlinear_R_corner_range", "all")
        if self.nonlinear_R_train_mode not in {"none", "exact_curve", "mean"}:
            raise ValueError("Unknown nonlinear_R_train_mode.")
        self.nonlinear_R_curve_bank = None
        self.nonlinear_R_curve_gaussian = None
        self.R_code_round_base = kwargs.pop("R_code_round_base", 1)
        self.mul_mismatch_mode = kwargs.pop("mul_mismatch_mode", "scale_mismatch")
        assert self.mul_mismatch_mode in {"scale_mismatch", "static_mismatch"}

        self.enable_measured_activation = kwargs.pop("enable_measured_activation", False)
        self.activation_curve_path = kwargs.pop("activation_curve_path", None)
        self.activation_corner = kwargs.pop("activation_corner", "TT")
        self.activation_interpolation = str(kwargs.pop(
            "activation_interpolation", "piecewise_linear")).lower()
        if self.activation_interpolation not in {"cubic_bspline", "piecewise_linear"}:
            raise ValueError(
                "activation_interpolation must be cubic_bspline or piecewise_linear.")
        self.activation_spline_parameters = kwargs.pop("activation_spline_parameters", 10)
        self.activation_fit_constraint = kwargs.pop("activation_fit_constraint", "auto")
        self.activation_normalize_positive_endpoint = kwargs.pop(
            "activation_normalize_positive_endpoint", False)
        self.compile_measured_activation = kwargs.pop("compile_measured_activation", False)

        self.v_grid, self.R_codes, self.R_table = None, None, None
        self.R_left, self.R_slope = None, None # Use piecewise-linear function as interpolant

        super().__init__(**kwargs)
        self.w_bits = w_bits
        self.q_hi = (1 << (w_bits - 1)) - 1
        self.w_quant_mode = w_quant_mode
        self.perc = perc
        self.R_max = kwargs.get("R_max", self.R * self.q_hi)

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

    def _configure_measured_activation(self):
        if not self.enable_measured_activation:
            return
        activation_cls = (
            CubicBSplineActivation
            if self.activation_interpolation == "cubic_bspline"
            else PiecewiseLinearActivation)
        if isinstance(self.ode_block.act_fn, activation_cls):
            return
        curve_path = self.activation_curve_path
        if curve_path is None:
            curve_path = os.path.join(
                os.path.dirname(__file__), "hardware_data", "relu_current_0p2uA_finer.csv")
        activation_kwargs = dict(
            curve_path=curve_path, v_dd=self.v_dd, corner=self.activation_corner,
            normalize_positive_endpoint=self.activation_normalize_positive_endpoint)
        if activation_cls is CubicBSplineActivation:
            activation_kwargs.update(
                num_parameters=self.activation_spline_parameters,
                compile_evaluator=self.compile_measured_activation,
                fit_constraint=self.activation_fit_constraint)
        activation = activation_cls(**activation_kwargs)
        device = self.ode_block.FFconv.weight.device
        self.ode_block.act_fn = activation.to(device=device)

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

    def wrap_input(self, x):
        return self.beta * x

    def unwrap_output(self, out):
        if isinstance(out, (tuple, list)):
            return type(out)(self.unwrap_output(o) for o in out)
        return out / self.q

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
            return orig_init_y(x / self.s_ff) / self.s_fb
        self.ode_block.init_y = patched_init_y

    def _round_R(self, x):
        return torch.round(x / self.R_code_round_base) * self.R_code_round_base

    def _csv_prepare_table(self):
        # If R varies with the input voltage, load in pre-simulated R vs Vin data for interpolation
        # v_grid: voltage range; (N,)
        # R_codes: different ground truth programming resistance levels; (M,)
        # R_table: Actual resistance value at different voltage; (N, M)
        if not self.nonlinear_R:
            return False

        if (self.nonlinear_R_curve_sharing != "shared" or
                self.nonlinear_R_curve_bank_indices is not None):
            if self.nonlinear_R_table is None:
                raise ValueError(
                    "Sampled nonlinear-R curves require nonlinear_R_table.")
            if self.nonlinear_R_curve_sampling == "multivariate_gaussian":
                self.nonlinear_R_curve_gaussian = load_mc_res_curve_gaussian(
                    self.nonlinear_R_table,
                    quantity=self.nonlinear_R_mc_quantity,
                    curve_indices=self.nonlinear_R_curve_bank_indices,
                    device=self.ode_block.FFconv.weight.device)
            else:
                self.nonlinear_R_curve_bank = load_mc_res_curve_bank(
                    self.nonlinear_R_table,
                    quantity=self.nonlinear_R_mc_quantity,
                    curve_indices=self.nonlinear_R_curve_bank_indices,
                    device=self.ode_block.FFconv.weight.device)
            shared_curve_index = (
                0 if self.nonlinear_R_curve_bank_indices is None else
                int(self.nonlinear_R_curve_bank_indices[0]))
        else:
            shared_curve_index = self.nonlinear_R_mc_curve_index

        self.v_grid, self.R_codes, self.R_table = load_res_vs_vin(
            R=self.R, R_max=self.R_max, device=self.ode_block.FFconv.weight.device,
            nonlinear_R_table=self.nonlinear_R_table,
            nonlinear_R_mc_curve_index=shared_curve_index,
            nonlinear_R_mc_quantity=self.nonlinear_R_mc_quantity)

        v_sort_idx = torch.argsort(self.v_grid)
        self.v_grid = self.v_grid[v_sort_idx]
        self.R_table = self.R_table[v_sort_idx, :]

        r_sort_idx = torch.argsort(self.R_codes)
        self.R_codes = self.R_codes[r_sort_idx]
        self.R_table = self.R_table[:, r_sort_idx]

        return True

    def _csv_build_interpolant(self):
        # Build the piecewise-linear interpolant
        # R_hat = R_left + R_slope * (v - v_left)
        # When interpolating, find the correct R_left, R_slope and v_left value based on the gt resistance
        # of the current quantized weight
        dv = (self.v_grid[1:] - self.v_grid[:-1]).clamp(min=1e-12) # (N-1,)
        self.R_left = self.R_table[:-1, :]  # (N-1, M)
        self.R_slope = (self.R_table[1:, :] - self.R_left) / dv[:, None] # (N-1,M)

    def _ship_nonlinear_R_pkg(self):
        if not self.nonlinear_R:
            if self.nonlinear_R_train_mode != "none":
                raise ValueError(
                    "nonlinear_R_train_mode requires nonlinear_R=True.")
            return

        if self.nonlinear_R_train_mode != "none":
            if not isinstance(self.ode_block, ToggleAveragedPhysicalFFFB):
                raise TypeError(
                    "Level-2 nonlinear-R training requires "
                    "ToggleAveragedPhysicalFFFB or a derived Level-2 block.")
            if isinstance(self.ode_block, TogglePulseFFFB):
                raise TypeError(
                    "nonlinear_R_train_mode is for non-expanded Level-2 blocks.")
            if self.nonlinear_R_curve_sharing != "shared":
                raise ValueError(
                    "Level-2 nonlinear-R training supports shared curves only.")
            if self.nonlinear_R_table is None:
                raise ValueError(
                    "nonlinear_R_table must point to the coupler MC source.")
            bank = load_mc_res_training_curve_bank(
                self.nonlinear_R_table,
                mode=self.nonlinear_R_train_mode,
                corner_range=self.nonlinear_R_corner_range,
                quantity=self.nonlinear_R_mc_quantity,
                dtype=self.ode_block.FFconv.weight.dtype,
                device=self.ode_block.FFconv.weight.device)
            self.ode_block._nonlinear_R_training_pkg = {
                **bank,
                "R": self.R,
                "proj_fn": getattr(self, "proj_fn", None),
                "nonlinear_R_train_mode": self.nonlinear_R_train_mode,
                "nonlinear_R_corner_range": self.nonlinear_R_corner_range,
                "nonlinear_R_curve_seed": self.nonlinear_R_curve_seed,
            }
            return

        self._csv_prepare_table()
        self._csv_build_interpolant()

        # Set the data needed for interpolating R based on current v values.
        # Those will be set to MVMConv after every conv layer in the model is replaced in the Validator.
        # Code written in this way because we call the validator after the model is wrapped by the wrapper.
        self.ode_block._nonlinear_R_pkg = {
            # This should match exactly the input args of enable_csv in MVMConv
            "v_grid": self.v_grid,
            "R_codes": self.R_codes,
            "R": self.R,
            "R_left": self.R_left,
            "R_slope": self.R_slope,
            "proj_fn": getattr(self, "proj_fn", None),
            "mul_mismatch_mode": self.mul_mismatch_mode,
            "curve_bank": self.nonlinear_R_curve_bank,
            "curve_gaussian": self.nonlinear_R_curve_gaussian,
            "nonlinear_R_curve_sampling": self.nonlinear_R_curve_sampling,
            "nonlinear_R_curve_sharing": self.nonlinear_R_curve_sharing,
            "nonlinear_R_curve_seed": self.nonlinear_R_curve_seed,
            "nonlinear_R_curve_edge_chunk_size": (
                self.nonlinear_R_curve_edge_chunk_size),
        }


class WrapQuantizeWXInit(WrapQuantizeW):
    def _patch_init_y(self):
        """
        The _patch_forward has already made the input to beta*x,
        thus in the transformed domain:
        Y_0 = q y_0 = (q/beta) * (beta*x)
        """
        orig_init_y = self.original_init_y
        @wraps(orig_init_y)
        def patched_init_y(x, *args, **kwargs):
            return orig_init_y(x / self.beta) * self.q
        self.ode_block.init_y = patched_init_y


class ODEWrapper2State(WrapQuantizeW):
    def __init__(self, is_first=False, is_last=False, thermal_noise=True, offset_eps=None, tie_cap=False,
                 R_max=180e3, enob=None, **kwargs):
        patch = kwargs.get("patch", True)
        quantize = kwargs.get("quantize", True)
        kwargs.update({"patch": False})
        super().__init__(**kwargs)
        self.R_max = R_max
        self.enob = enob
        self.s_R, self.weight_scale = None, 1.0
        R_max_ideal = self.R * self.q_hi
        if R_max is not None and R_max_ideal > R_max:
            assert R_max > self.R
            self.weight_scale = R_max_ideal / R_max # Must be greater than 1 and smaller than q_hi
            assert self.weight_scale <= self.q_hi
            if quantize:
                # Only do this after the weights are quantized
                # logging.warning("Pre-scaling weight mean: {}, min: {}, max: {}".format(
                #     self.ode_block.FFconv.weight.mean(), self.ode_block.FFconv.weight.min(),
                #     self.ode_block.FFconv.weight.max()))
                self.ode_block.recover_params()
                self.scale_weight_below_one(self.ode_block.FFconv.weight, self.weight_scale, self.q_hi)
                self.scale_weight_below_one(self.ode_block.FBconv.weight, self.weight_scale, self.q_hi)
                if self.ode_block.noise_level is not None and self.ode_block.noise_level > 0:
                    self.ode_block.add_noise()
                # logging.warning("Post-scaling weight mean: {}, min: {}, max: {}".format(
                #     self.ode_block.FFconv.weight.mean(), self.ode_block.FFconv.weight.min(),
                #     self.ode_block.FFconv.weight.max()))
            # Note: This applies only to dynamics like dy/dt = Wf(z) or Wf(y) or Wy, where W is applied to the
            # output of non-linearity.
            _delta = (1 - self.weight_scale / self.q_hi) / (self.q_hi - 1)
            # self.s_R = self.R * (_delta * self.q_hi)
            self.s_R = None
            # self.s_R = (1 - 1 / self.q_hi) / (1 / self.R - 1 / self.R_max)
            # logging.warning("Scaled weight with s_R: {}".format(self.s_R))

        self.ode_block.q_hi = self.q_hi
        self.ode_block.weight_scale = self.weight_scale

        # Todo: Right now using the same cap value seems to be fine. Need more experiment.
        self.tie_cap = tie_cap
        self.cap_scale = self._round(self.s_ff / self.s_fb, 1) if not self.tie_cap else 1
        self.C_fb = self.C
        self.C_ff = self.C_fb * self.cap_scale

        self.alpha = self.s_fb
        self.beta = self.q        # Assuming q is the same for all layers
        self.is_first = is_first
        self.is_last = is_last
        self.inp_scale = self.beta if self.is_first else 1
        # Todo: Right don't scaling back the last layer's output seems to be fine (when q >= 0.1 is not very small)
        self.out_scale = self.beta if self.is_last else 1
        # self.out_scale = 1

        self._has_init_ode = hasattr(self.ode_block, "_make_z_ode_fn")
        self._patch_conv = hasattr(self.ode_block, "option_patch")
        if self._has_init_ode:
            self.original_make_z_fn = self.ode_block._make_z_ode_fn
        self.proj_fn = nn.Hardtanh(min_val=-self.v_dd, max_val=self.v_dd)
        # Todo: What's the right eps_scale?
        # Pick sqrt(4 * k_B * T / R) / C
        if thermal_noise:
            if offset_eps is None:
                self.ode_block.eps_scale = (1 / self.ode_block.offset_eps) * ((4.16e-21 * 4 / self.R) ** 0.5 / self.C_fb)
            else:
                self.ode_block.eps_scale = None
                self.ode_block.offset_eps = offset_eps / ((self.R * self.C_fb) ** 0.5)
        else:
            self.ode_block.eps_scale, self.ode_block.offset_eps = None, 0.0

        if patch:
            self._patch()
        if self._has_init_ode:
            self.ode_block.option_init["proj_fn"] = self.proj_fn
        if self._patch_conv:
            self.ode_block.option_patch["proj_fn"] = self.proj_fn
        self.ode_block.option_aca["proj_fn"] = self.proj_fn

    def _quantize_output(self, x):
        if self.enob is None:
            return x

        _is_qat = ("QATWrapper" in self.__class__.__name__)

        if _is_qat:
            return OutputQuantImpl.apply(x, self.v_dd, self.enob)

        x = self.proj_fn(x)
        n_interval = 2 ** self.enob - 1
        if n_interval <= 0:
            return x

        _delta = 2 * self.v_dd / n_interval
        x_q = torch.round((x + self.v_dd) / _delta) * _delta - self.v_dd
        x_q = self.proj_fn(x_q)

        return x_q

    @staticmethod
    def scale_weight_below_one(w, scalar, q_hi):
        with torch.no_grad():
            sign = w.sign()
            a = w.abs().clamp(0.0, 1.0)

            # Get quantization levels, {0, 1, ..., q_hi}
            k = torch.round(a * q_hi).clamp_(0, q_hi)

            first = float(scalar) / q_hi
            delta = (1.0 - first) / (q_hi - 1)

            # Map: k=0 -> 0; k>=1 -> first + (k-1)*delta
            a_new = torch.where(
                k <= 0.0,
                torch.zeros_like(a),
                a.new_tensor(first) + (k - 1.0) * a.new_tensor(delta)
            )

            w.copy_((sign * a_new).clamp(-1.0, 1.0))

    def get_time_scaler(self):
        _R = self.s_R if self.s_R is not None else self.R
        return _R * self.C_ff, _R * self.C_fb

    @staticmethod
    def _scale_time_impl(integral_option, end_time_scaler):
        # scale and set value
        integral_option["t0"] = integral_option["t0"] * end_time_scaler
        integral_option["t1"] = integral_option["t1"] * end_time_scaler
        integral_option["t_eval"] = [_ts * end_time_scaler for _ts in integral_option["t_eval"]]
        integral_option["h"] = integral_option["h"] * end_time_scaler if integral_option["h"] is not None else None

        return integral_option

    def _scale_time(self):
        # scale integration time based on s_fb
        _R = self.s_R if self.s_R is not None else self.R
        end_time_scaler = _R * self.C / self.alpha
        self.ode_block.integration_time = self.ode_block.integration_time * end_time_scaler

        if self._has_init_ode:
            self.ode_block.option_init = self._scale_time_impl(self.ode_block.option_init, end_time_scaler)
            logging.info("Scaled init end time: {} s".format(self.ode_block.option_init["t1"]))
        if self._patch_conv:
            self.ode_block.option_patch = self._scale_time_impl(self.ode_block.option_patch, end_time_scaler)
            logging.info("Scaled patch end time: {} s".format(self.ode_block.option_patch["t1"]))
        self.ode_block.option_aca = self._scale_time_impl(self.ode_block.option_aca, end_time_scaler)
        logging.info("Scaled compute end time: {} s".format(self.ode_block.option_aca["t1"]))

    def _scale_act_fn(self):
        if "relu6" in self.ode_block.act_fn.__class__.__name__.lower():
            self.ode_block.act_fn = ReLUX(min(6 * self.beta, self.v_dd))
        elif "hardtanh" in self.ode_block.act_fn.__class__.__name__.lower():
            self.ode_block.act_fn = nn.Hardtanh(min_val=-min(self.beta, self.v_dd), max_val=min(self.beta, self.v_dd))

    def transform(self, inner_fn):
        @wraps(inner_fn)
        def scaled(*f_args, **f_kwargs):
            y_, z_ = inner_fn(*f_args, **f_kwargs)
            return y_ / self.C_ff / self.R, z_ / self.C_fb / self.R
        return _FuncWrapper(scaled)

    def wrap_input(self, x):
        x = self.inp_scale * x
        return self.proj_fn(x) if getattr(self, "proj_fn", None) is not None else x

    def unwrap_output(self, out):
        if isinstance(out, (tuple, list)):
            return type(out)(self.unwrap_output(o) for o in out)
        return out / self.out_scale

    def _patch_forward(self):
        # scale the forward method
        orig_call = self.original_forward
        @wraps(orig_call)
        def patched_forward(x, *args, **kwargs):
            return self._quantize_output(orig_call(self.proj_fn(self.inp_scale * x), *args, **kwargs)) / self.out_scale
        self.ode_block.forward = patched_forward

    def _patch_init_y(self):
        """
        No modification on the init_y of the ode_block.
        """
        orig_init_y = self.original_init_y
        @wraps(orig_init_y)
        def patched_init_y(x, *args, **kwargs):
            return orig_init_y(x)
        self.ode_block.init_y = patched_init_y

    def transform_z(self, inner_fn):
        @wraps(inner_fn)
        def scaled(*f_args, **f_kwargs):
            return inner_fn(*f_args, **f_kwargs) / self.C_fb / self.R
        return scaled

    def _patch_make_z_fn(self):
        # scale the ode_func for charging z
        orig = self.original_make_z_fn
        @wraps(orig)
        def patched_make_z_fn(*args, **kwargs):
            inner = orig(*args, **kwargs)
            return self.transform_z(inner)
        self.ode_block._make_z_ode_fn = patched_make_z_fn

    def _patch(self):
        """
        1. Scale the integration time for option_init and option_aca
        2. Scale the act_fn
        3. Scale the dynamics of y and z
        4. Scale the input/output of the layer for the first/last layer
        """
        self.time_scaler = self.get_time_scaler()
        # scale integration time
        self._scale_time()
        self._scale_act_fn()
        self._patch_make_fn()
        self._patch_forward()
        self._patch_init_y()
        if self._has_init_ode:
            self._patch_make_z_fn()
        self._ship_nonlinear_R_pkg()

    @staticmethod
    def _round(x, n):
        if x.abs().item() >= 1:
            return torch.round(x, decimals=n)
        elif not torch.allclose(x, torch.zeros_like(x)):
            return torch.round(x, decimals=-int(math.floor(math.log10(x.abs().item()))) + n - 1)
        else:
            return 0


class QATTester2State(ODEWrapper2State):
    """
    Same as the parent class except for:
    1. Loading already quantized weights with s_ff/fb registered as buffers in ode_block.
    2. Change the activation functon to align with the QATWrapper2State.

    For 2 state ODEBlocks without QAT and want to test without quantization, use this class.
    Do NOT use ODEWrapperRC in this case, because it did not wrap the make_fn as a nn.Module.
    """
    def __init__(self, **kwargs):
        kwargs.update({"patch": True, "quantize": False})
        super().__init__(**kwargs)
        self.out_scale = self.beta if self.is_last else 1
        # self.out_scale = 1

    def _scale_act_fn(self):
        # Replace ReLU6 with clamp(x, 0, v_dd)
        if "relu6" in self.ode_block.act_fn.__class__.__name__.lower():
            self.ode_block.act_fn = ReLUX(min(6 * self.beta, self.v_dd))
        elif "hardtanh" in self.ode_block.act_fn.__class__.__name__.lower():
            self.ode_block.act_fn = nn.Hardtanh(min_val=-min(self.beta, self.v_dd), max_val=min(self.beta, self.v_dd))


class QATWrapper2State(ODEWrapper2State):
    """
    Same as the parent class except for:
    1. Changed the activation function.
    2. Dynamically update the scaling parameters before each forward pass based on the weights.
    During test, use QATTester2State.
    """
    def __init__(self, qat_cls=SymQuantizeWeight, **kwargs):
        kwargs.update({"patch": False, "quantize": False})
        super().__init__(**kwargs)
        self.w_bits = kwargs.get("w_bits", 8)
        self.FF_quantizer = qat_cls(w_bits=self.w_bits, w_scalar=self.weight_scale,
                                    layer_weight=self.ode_block.FFconv.weight).to(self.ode_block.FFconv.weight.device)
        self.FB_quantizer = qat_cls(w_bits=self.w_bits, w_scalar=self.weight_scale,
                                    layer_weight=self.ode_block.FBconv.weight).to(self.ode_block.FBconv.weight.device)
        self.FF_quantizer.compute_s(self.ode_block.FFconv.weight)
        self.FB_quantizer.compute_s(self.ode_block.FBconv.weight)
        P.register_parametrization(self.ode_block.FFconv, "weight", self.FF_quantizer)
        P.register_parametrization(self.ode_block.FBconv, "weight", self.FB_quantizer)
        self.ode_block.register_buffer("s_ff", self.FF_quantizer.s_w)
        self.ode_block.register_buffer("s_fb", self.FB_quantizer.s_w)
        self.s_ff, self.s_fb = None, None

        # Original copy of integration time
        self.orig_integration_time = self.ode_block.integration_time.clone()
        if self._has_init_ode:
            self.orig_option_init = deepcopy(self.ode_block.option_init)
        if self._patch_conv:
            self.orig_option_patch = deepcopy(self.ode_block.option_patch)
        self.orig_option_aca = deepcopy(self.ode_block.option_aca)

        # Register hook
        self.update_hook = self.ode_block.register_forward_pre_hook(self._update_vals)
        self._patch()

    def _set_quantize_s(self, module):
        # Set the quantization step size before each patched forward call
        # the quantization step size s is set in compute_s
        self.FF_quantizer.compute_s(module.FFconv.parametrizations.weight.original)
        self.FB_quantizer.compute_s(module.FBconv.parametrizations.weight.original)
        # For saving and loading purpose
        # module should be exactly self.ode_block
        module.s_ff.copy_(self.FF_quantizer.s_w)
        module.s_fb.copy_(self.FB_quantizer.s_w)
        # For calculation in the wrapper
        self.s_ff, self.s_fb = self.FF_quantizer.s_w, self.FB_quantizer.s_w

    def _update_vals(self, module, inputs):
        # Compute quantization step size first
        self._set_quantize_s(module)

        # Reset values based on new s_ff and s_fb before forward
        self.cap_scale = self._round(self.s_ff / self.s_fb, 1) if not self.tie_cap else 1
        self.C_fb = self.C
        self.C_ff = self.C_fb * self.cap_scale

        self.alpha = self.s_fb
        self.beta = self.q
        self.inp_scale = self.beta if self.is_first else 1

        # The original patch
        self.time_scaler = self.get_time_scaler()
        self._scale_time_dynamically(module)

        # Todo: During training, we have to scaling back the last activation, but during test,
        #  this seems to be removable.
        self.out_scale = self.beta if self.is_last else 1

        return None

    def _scale_time_dynamically(self, module):
        # scale integration time based on s_fb
        _R = self.s_R if self.s_R is not None else self.R
        end_time_scaler = _R * self.C / self.alpha
        module.integration_time = self.orig_integration_time * end_time_scaler

        if self._has_init_ode:
            module.option_init = self._scale_time_impl(deepcopy(self.orig_option_init), end_time_scaler)
        if self._patch_conv:
            module.option_patch = self._scale_time_impl(deepcopy(self.orig_option_patch), end_time_scaler)
        module.option_aca = self._scale_time_impl(deepcopy(self.orig_option_aca), end_time_scaler)

    def _scale_act_fn(self):
        # Replace ReLU6 with clamp(x, 0, v_dd)
        if "relu6" in self.ode_block.act_fn.__class__.__name__.lower():
            self.ode_block.act_fn = ReLUX(min(6 * self.beta, self.v_dd))
        elif "hardtanh" in self.ode_block.act_fn.__class__.__name__.lower():
            self.ode_block.act_fn = nn.Hardtanh(min_val=-min(self.beta, self.v_dd), max_val=min(self.beta, self.v_dd))


class ODEWrapper1State(ODEWrapper2State):
    """
    Works for dy/dt = W_F f(W_B y). ODEXInitFFFB.
    May not work for other dynamics like f(W_F f(W_B y)) or W_F f(x - W_B y)
    """
    def __init__(self, k=1e3, **kwargs):
        patch = kwargs.get("patch", True)
        quantize = kwargs.get("quantize", True)
        kwargs.update({"patch": False})
        super().__init__(**kwargs)
        self.alpha = self.s_fb * self.s_ff
        self.beta = self.q * self.s_fb
        self.k = k
        self.beta_c = self.beta * self.k / self.R
        # logging.warning("6 * beta_c = {}, self.s_fb = {}".format(6 * self.beta_c, self.s_fb))
        self.inp_scale = self.q if self.is_first else 1
        self.out_scale = self.q if self.is_last else 1

        if patch:
            self._patch()

    def get_time_scaler(self):
        _R = self.s_R if self.s_R is not None else self.R
        return _R * self.C

    def _scale_time(self):
        # T * R^2 * C / (k * alpha)
        _R = self.s_R if self.s_R is not None else self.R
        end_time_scaler = _R * _R * self.C / (self.k * self.alpha)
        self.ode_block.integration_time = self.ode_block.integration_time * end_time_scaler

        if self._has_init_ode:
            self.ode_block.option_init = self._scale_time_impl(self.ode_block.option_init, end_time_scaler)
            logging.info("Scaled init end time: {} s".format(self.ode_block.option_init["t1"]))
        if self._patch_conv:
            self.ode_block.option_patch = self._scale_time_impl(self.ode_block.option_patch, end_time_scaler)
            logging.info("Scaled patch end time: {} s".format(self.ode_block.option_patch["t1"]))
        self.ode_block.option_aca = self._scale_time_impl(self.ode_block.option_aca, end_time_scaler)
        logging.info("Scaled compute end time: {} s".format(self.ode_block.option_aca["t1"]))

    def _scale_act_fn(self):
        # ReLU6*beta_c
        if "relu6" in self.ode_block.act_fn.__class__.__name__.lower():
            # self.ode_block.act_fn = ReLUX(min(6 * self.beta_c, self.v_dd))
            self.ode_block.act_fn = ReLUX(self.v_dd)
        elif "hardtanh" in self.ode_block.act_fn.__class__.__name__.lower():
            self.ode_block.act_fn = nn.Hardtanh(min_val=-min(self.beta_c, self.v_dd),
                                                max_val=min(self.beta_c, self.v_dd))

    def transform(self, inner_fn):
        @wraps(inner_fn)
        def scaled(t, y, *f_args, **f_kwargs):
            y_ = inner_fn(t, y * self.k / self.R, *f_args, **f_kwargs)
            return y_ / self.C / self.R
        return scaled


class QATTester1State(ODEWrapper1State):
    """
    Same as the parent class except for:
    1. Loading already quantized weights with s_ff/fb registered as buffers in ode_block.
    """
    def __init__(self, **kwargs):
        kwargs.update({"patch": True, "quantize": False})
        super().__init__(**kwargs)


class QATWrapper1State(ODEWrapper1State):
    """
    Same as the parent class except for:
    1. Dynamically update the scaling parameters before each forward pass based on the weights.
    During test, use QATTester1State.
    """
    def __init__(self, qat_cls=SymQuantizeWeight, **kwargs):
        kwargs.update({"patch": False, "quantize": False})
        super().__init__(**kwargs)
        self.w_bits = kwargs.get("w_bits", 8)
        self.FF_quantizer = qat_cls(w_bits=self.w_bits, w_scalar=self.weight_scale,
                                    layer_weight=self.ode_block.FFconv.weight).to(self.ode_block.FFconv.weight.device)
        self.FB_quantizer = qat_cls(w_bits=self.w_bits, w_scalar=self.weight_scale,
                                    layer_weight=self.ode_block.FBconv.weight).to(self.ode_block.FBconv.weight.device)
        self.FF_quantizer.compute_s(self.ode_block.FFconv.weight)
        self.FB_quantizer.compute_s(self.ode_block.FBconv.weight)
        P.register_parametrization(self.ode_block.FFconv, "weight", self.FF_quantizer)
        P.register_parametrization(self.ode_block.FBconv, "weight", self.FB_quantizer)
        self.ode_block.register_buffer("s_ff", self.FF_quantizer.s_w)
        self.ode_block.register_buffer("s_fb", self.FB_quantizer.s_w)
        self.s_ff, self.s_fb = None, None

        # Original copy of integration time
        self.orig_integration_time = self.ode_block.integration_time.clone()
        if self._has_init_ode:
            self.orig_option_init = deepcopy(self.ode_block.option_init)
        if self._patch_conv:
            self.orig_option_patch = deepcopy(self.ode_block.option_patch)
        self.orig_option_aca = deepcopy(self.ode_block.option_aca)

        # Register hook
        self.update_hook = self.ode_block.register_forward_pre_hook(self._update_vals)
        self._patch()

    def _set_quantize_s(self, module):
        # Set the quantization step size before each patched forward call
        # the quantization step size s is set in compute_s
        self.FF_quantizer.compute_s(module.FFconv.parametrizations.weight.original)
        self.FB_quantizer.compute_s(module.FBconv.parametrizations.weight.original)
        # For saving and loading purpose
        # module should be exactly self.ode_block
        module.s_ff.copy_(self.FF_quantizer.s_w)
        module.s_fb.copy_(self.FB_quantizer.s_w)
        # For calculation in the wrapper
        self.s_ff, self.s_fb = self.FF_quantizer.s_w, self.FB_quantizer.s_w

    def _update_vals(self, module, inputs):
        # Compute quantization step size first
        self._set_quantize_s(module)

        # Reset values based on new s_ff and s_fb before forward
        self.alpha = self.s_fb * self.s_ff
        self.beta = self.q * self.s_fb
        self.beta_c = self.beta * self.k / self.R
        self.inp_scale = self.q if self.is_first else 1

        # The original patch
        self.time_scaler = self.get_time_scaler()
        self._scale_time_dynamically(module)
        self._scale_act_fn_dynamically()

        # Todo: During training, we have to scale back the last activation, but during test,
        #  this seems to be removable.
        self.out_scale = self.q if self.is_last else 1

        return None

    def _scale_time_dynamically(self, module):
        # scale integration time based on s_fb
        _R = self.s_R if self.s_R is not None else self.R
        end_time_scaler = _R * _R * self.C / (self.k * self.alpha)
        module.integration_time = self.orig_integration_time * end_time_scaler

        if self._has_init_ode:
            module.option_init = self._scale_time_impl(deepcopy(self.orig_option_init), end_time_scaler)
        if self._patch_conv:
            module.option_patch = self._scale_time_impl(deepcopy(self.orig_option_patch), end_time_scaler)
        module.option_aca = self._scale_time_impl(deepcopy(self.orig_option_aca), end_time_scaler)

    def _scale_act_fn_dynamically(self):
        # ReLU6*beta_c
        act_fn_cls = self.ode_block.act_fn.__class__.__name__.lower()
        if "relu6" in act_fn_cls or "relux" in act_fn_cls:
            # self.ode_block.act_fn.set_scale(min(6 * self.beta_c, self.v_dd))
            self.ode_block.act_fn.set_scale(self.v_dd)
        elif "hardtanh" in act_fn_cls:
            # Ignore hardtanh branch for now
            self.ode_block.act_fn = nn.Hardtanh(min_val=-min(self.beta_c, self.v_dd),
                                                max_val=min(self.beta_c, self.v_dd))


class ToggleWrapper1State(ODEWrapper1State):
    """
    Quantizing wrapper for ToggleAveragedPhysical* and TogglePulse* blocks.
    It keeps the existing weight/input/output quantization path, but does not
    scale ODE time or wrap _make_ode_fn because toggle blocks convert unitless
    stage times into physical durations directly.
    """
    def _set_toggle_scale_values(self):
        if self.s_ff is None or self.s_fb is None:
            self.s_ff, self.s_fb = self.FF_quantizer.s_w, self.FB_quantizer.s_w
        self.alpha_1state = self.s_fb * self.s_ff
        self.alpha = self.s_fb
        self.cap_scale = self._round(self.s_ff / self.s_fb, 1) if not self.tie_cap else 1
        self.C_fb = self.C
        self.C_ff = self.C_fb * self.cap_scale

    def _ship_toggle_params(self, module=None):
        module = self.ode_block if module is None else module
        setattr(module, "physical", True)
        setattr(module, "R", self.R)
        setattr(module, "C", self.C)
        setattr(module, "C_fb", self.C_fb)
        setattr(module, "C_ff", self.C_ff)
        setattr(module, "k", self.k)
        setattr(module, "alpha", self.alpha)
        setattr(module, "alpha_1state", self.alpha_1state)
        setattr(module, "s_fb", self.s_fb)
        setattr(module, "s_ff", self.s_ff)
        setattr(module, "cap_scale", self.cap_scale)
        setattr(module, "tie_cap", self.tie_cap)
        setattr(module, "w_bits", self.w_bits)
        setattr(module, "q_hi", self.q_hi)
        setattr(module, "beta_c", self.beta_c)

    def _scale_act_fn(self):
        if self.enable_measured_activation:
            self._configure_measured_activation()
            return
        if "relu6" in self.ode_block.act_fn.__class__.__name__.lower():
            if isinstance(self.ode_block, (ToggleODEXInitFFFB, TogglePulseODEXInitFFFB, TogglePulseBlkXInitFFFB)):
                if self.ode_block.odexinit_scaling_mode == "direct":
                    upper = min(6 * self.q, self.v_dd)
                else:
                    upper = self.v_dd
            else:
                upper = min(6 * self.q, self.v_dd)
            self.ode_block.act_fn = ReLUX(upper)
        elif "hardtanh" in self.ode_block.act_fn.__class__.__name__.lower():
            bound = min(self.q, self.v_dd)
            self.ode_block.act_fn = nn.Hardtanh(min_val=-bound, max_val=bound)

    def _scale_act_fn_dynamically(self):
        if self.enable_measured_activation:
            return
        act_fn_cls = self.ode_block.act_fn.__class__.__name__.lower()
        if "relu6" in act_fn_cls or "relux" in act_fn_cls:
            if isinstance(self.ode_block, (ToggleODEXInitFFFB, TogglePulseODEXInitFFFB, TogglePulseBlkXInitFFFB)):
                if self.ode_block.odexinit_scaling_mode == "direct":
                    upper = min(6 * self.q, self.v_dd)
                else:
                    upper = self.v_dd
            else:
                upper = min(6 * self.q, self.v_dd)
            self.ode_block.act_fn.set_scale(upper)
        elif "hardtanh" in act_fn_cls:
            bound = min(self.q, self.v_dd)
            self.ode_block.act_fn = nn.Hardtanh(min_val=-bound, max_val=bound)

    def _patch(self):
        self._set_toggle_scale_values()
        self.time_scaler = self.R * self.C / self.alpha
        self._ship_toggle_params()
        self._scale_act_fn()
        self._patch_forward()
        self._patch_init_y()
        self._ship_nonlinear_R_pkg()


class ToggleQATTester1State(ToggleWrapper1State):
    def __init__(self, **kwargs):
        kwargs.update({"patch": True, "quantize": False})
        super().__init__(**kwargs)


class ToggleQATWrapper1State(QATWrapper1State):
    """
    QAT wrapper for trainable averaged toggle hardware models. It updates fake
    quantization scales before each forward pass, but does not rescale the
    integration time because the toggle block owns physical timing.
    """
    def _set_toggle_scale_values(self):
        if self.s_ff is None or self.s_fb is None:
            self.s_ff, self.s_fb = self.FF_quantizer.s_w, self.FB_quantizer.s_w
        self.alpha_1state = self.s_fb * self.s_ff
        self.alpha = self.s_fb
        self.cap_scale = self._round(self.s_ff / self.s_fb, 1) if not self.tie_cap else 1
        self.C_fb = self.C
        self.C_ff = self.C_fb * self.cap_scale

    def _ship_toggle_params(self, module=None):
        module = self.ode_block if module is None else module
        setattr(module, "physical", True)
        setattr(module, "R", self.R)
        setattr(module, "C", self.C)
        setattr(module, "C_fb", self.C_fb)
        setattr(module, "C_ff", self.C_ff)
        setattr(module, "k", self.k)
        setattr(module, "alpha", self.alpha)
        setattr(module, "alpha_1state", self.alpha_1state)
        setattr(module, "s_fb", self.s_fb)
        setattr(module, "s_ff", self.s_ff)
        setattr(module, "cap_scale", self.cap_scale)
        setattr(module, "tie_cap", self.tie_cap)
        setattr(module, "w_bits", self.w_bits)
        setattr(module, "q_hi", self.q_hi)
        setattr(module, "beta_c", self.beta_c)

    def _scale_act_fn(self):
        if self.enable_measured_activation:
            self._configure_measured_activation()
            return
        if "relu6" in self.ode_block.act_fn.__class__.__name__.lower():
            if isinstance(self.ode_block, (ToggleODEXInitFFFB, TogglePulseODEXInitFFFB, TogglePulseBlkXInitFFFB)):
                if self.ode_block.odexinit_scaling_mode == "direct":
                    upper = min(6 * self.q, self.v_dd)
                else:
                    upper = self.v_dd
            else:
                upper = min(6 * self.q, self.v_dd)
            self.ode_block.act_fn = ReLUX(upper)
        elif "hardtanh" in self.ode_block.act_fn.__class__.__name__.lower():
            bound = min(self.q, self.v_dd)
            self.ode_block.act_fn = nn.Hardtanh(min_val=-bound, max_val=bound)

    def _scale_act_fn_dynamically(self):
        if self.enable_measured_activation:
            return
        act_fn_cls = self.ode_block.act_fn.__class__.__name__.lower()
        if "relu6" in act_fn_cls or "relux" in act_fn_cls:
            if isinstance(self.ode_block, (ToggleODEXInitFFFB, TogglePulseODEXInitFFFB, TogglePulseBlkXInitFFFB)):
                if self.ode_block.odexinit_scaling_mode == "direct":
                    upper = min(6 * self.q, self.v_dd)
                else:
                    upper = self.v_dd
            else:
                upper = min(6 * self.q, self.v_dd)
            self.ode_block.act_fn.set_scale(upper)
        elif "hardtanh" in act_fn_cls:
            bound = min(self.q, self.v_dd)
            self.ode_block.act_fn = nn.Hardtanh(min_val=-bound, max_val=bound)

    def _patch(self):
        self._set_toggle_scale_values()
        self.time_scaler = self.R * self.C / self.alpha
        self._ship_toggle_params()
        self._scale_act_fn()
        self._patch_forward()
        self._patch_init_y()
        self._ship_nonlinear_R_pkg()

    def _update_vals(self, module, inputs):
        self._set_quantize_s(module)

        self._set_toggle_scale_values()
        self.beta = self.q * self.s_fb
        self.beta_c = self.beta * self.k / self.R
        self.inp_scale = self.q if self.is_first else 1
        self.out_scale = self.q if self.is_last else 1

        self.time_scaler = self.R * self.C / self.alpha
        self._ship_toggle_params(module)
        self._scale_act_fn_dynamically()
        return None


class TogglePulseWrapper1State(ToggleWrapper1State):
    """Physical wrapper for integer-level pulse weights."""

    def __init__(self, **kwargs):
        q_hi = (1 << (int(kwargs.get("w_bits", 8)) - 1)) - 1
        kwargs["R_max"] = float(kwargs.get("R", 1e5)) * q_hi
        super().__init__(**kwargs)

    @staticmethod
    def cal_quant_factor_and_set(q_hi, p: nn.Parameter):
        with torch.no_grad():
            abs_max = p.data.abs().max()
            if torch.allclose(abs_max, torch.zeros_like(abs_max)):
                return p.new_tensor(1.0)
            s = 1.0 / abs_max
            torch.clamp((s * q_hi * p).round(), min=-q_hi, max=q_hi, out=p)
            return s


class TogglePulseQATTester1State(TogglePulseWrapper1State):
    def __init__(self, **kwargs):
        kwargs.update({"patch": True, "quantize": False})
        super().__init__(**kwargs)


class TogglePulseQATWrapper1State(ToggleQATWrapper1State):
    """QAT wrapper exposing live signed integer levels to TogglePulseBlk."""

    def __init__(self, qat_cls=PulseSymQuantizeWeight, **kwargs):
        q_hi = (1 << (int(kwargs.get("w_bits", 8)) - 1)) - 1
        kwargs["R_max"] = float(kwargs.get("R", 1e5)) * q_hi
        super().__init__(qat_cls=PulseSymQuantizeWeight, **kwargs)


class ODEWrapper1StateWithX(ODEWrapper1State):
    def _patch_forward(self):
        # scale the forward method
        orig_call = self.original_forward

        @wraps(orig_call)
        def patched_forward(x, *args, **kwargs):
            # For the first layer, x is x;
            # For layers after, x is z_T, in the scaled domain.
            return self._quantize_output(orig_call(self.proj_fn(
                self.beta_c * x / (self.q / self.inp_scale)
            ), *args, **kwargs)) / self.out_scale

        self.ode_block.forward = patched_forward

    def _patch_init_y(self):
        """
        Now according to _patch_forward, input x to this function is beta_c * x.
        For the first layer, z_0 is q * x;
        For layers after, z_0 is z_T = q * x.
        """
        orig_init_y = self.original_init_y

        @wraps(orig_init_y)
        def patched_init_y(x, *args, **kwargs):
            return orig_init_y(self.q * x / self.beta_c)

        self.ode_block.init_y = patched_init_y

    def _scale_act_fn(self):
        # ReLU6*beta_c
        if "relu6" in self.ode_block.act_fn.__class__.__name__.lower():
            self.ode_block.act_fn = ReLUX(min(6 * self.beta_c, self.v_dd))
            # self.ode_block.act_fn = ReLUX(self.v_dd)
        elif "hardtanh" in self.ode_block.act_fn.__class__.__name__.lower():
            self.ode_block.act_fn = nn.Hardtanh(min_val=-min(self.beta_c, self.v_dd),
                                                max_val=min(self.beta_c, self.v_dd))

    def wrap_input(self, x):
        x = self.beta_c * x / (self.q / self.inp_scale)
        return self.proj_fn(x) if getattr(self, "proj_fn", None) is not None else x

    def unwrap_output(self, out):
        if isinstance(out, (tuple, list)):
            return type(out)(self.unwrap_output(o) for o in out)
        return out / self.out_scale


class QATTester1StateWithX(ODEWrapper1StateWithX):
    def __init__(self, **kwargs):
        kwargs.update({"patch": True, "quantize": False})
        super().__init__(**kwargs)


class QATWrapper1StateWithX(ODEWrapper1StateWithX):
    """
    Same as the parent class except for:
    1. Dynamically update the scaling parameters before each forward pass based on the weights.
    During test, use QATTester1State.
    """
    def __init__(self, qat_cls=SymQuantizeWeight, **kwargs):
        kwargs.update({"patch": False, "quantize": False})
        super().__init__(**kwargs)
        self.w_bits = kwargs.get("w_bits", 8)
        self.FF_quantizer = qat_cls(w_bits=self.w_bits, w_scalar=self.weight_scale,
                                    layer_weight=self.ode_block.FFconv.weight).to(self.ode_block.FFconv.weight.device)
        self.FB_quantizer = qat_cls(w_bits=self.w_bits, w_scalar=self.weight_scale,
                                    layer_weight=self.ode_block.FBconv.weight).to(self.ode_block.FBconv.weight.device)
        self.FF_quantizer.compute_s(self.ode_block.FFconv.weight)
        self.FB_quantizer.compute_s(self.ode_block.FBconv.weight)
        P.register_parametrization(self.ode_block.FFconv, "weight", self.FF_quantizer)
        P.register_parametrization(self.ode_block.FBconv, "weight", self.FB_quantizer)
        self.ode_block.register_buffer("s_ff", self.FF_quantizer.s_w)
        self.ode_block.register_buffer("s_fb", self.FB_quantizer.s_w)
        self.s_ff, self.s_fb = None, None

        # Original copy of integration time
        self.orig_integration_time = self.ode_block.integration_time.clone()
        if self._has_init_ode:
            self.orig_option_init = deepcopy(self.ode_block.option_init)
        if self._patch_conv:
            self.orig_option_patch = deepcopy(self.ode_block.option_patch)
        self.orig_option_aca = deepcopy(self.ode_block.option_aca)

        # Register hook
        self.update_hook = self.ode_block.register_forward_pre_hook(self._update_vals)
        self._patch()

    def _set_quantize_s(self, module):
        # Set the quantization step size before each patched forward call
        # the quantization step size s is set in compute_s
        self.FF_quantizer.compute_s(module.FFconv.parametrizations.weight.original)
        self.FB_quantizer.compute_s(module.FBconv.parametrizations.weight.original)
        # For saving and loading purpose
        # module should be exactly self.ode_block
        module.s_ff.copy_(self.FF_quantizer.s_w)
        module.s_fb.copy_(self.FB_quantizer.s_w)
        # For calculation in the wrapper
        self.s_ff, self.s_fb = self.FF_quantizer.s_w, self.FB_quantizer.s_w

    def _update_vals(self, module, inputs):
        # Compute quantization step size first
        self._set_quantize_s(module)

        # Reset values based on new s_ff and s_fb before forward
        self.alpha = self.s_fb * self.s_ff
        self.beta = self.q * self.s_fb
        self.beta_c = self.beta * self.k / self.R
        self.inp_scale = self.q if self.is_first else 1

        # The original patch
        self.time_scaler = self.get_time_scaler()
        self._scale_time_dynamically(module)
        self._scale_act_fn_dynamically()

        # Todo: During training, we have to scale back the last activation, but during test,
        #  this seems to be removable.
        self.out_scale = self.q if self.is_last else 1

        return None

    def _scale_time_dynamically(self, module):
        # scale integration time based on s_fb
        _R = self.s_R if self.s_R is not None else self.R
        end_time_scaler = _R * _R * self.C / (self.k * self.alpha)
        module.integration_time = self.orig_integration_time * end_time_scaler

        if self._has_init_ode:
            module.option_init = self._scale_time_impl(deepcopy(self.orig_option_init), end_time_scaler)
        if self._patch_conv:
            module.option_patch = self._scale_time_impl(deepcopy(self.orig_option_patch), end_time_scaler)
        module.option_aca = self._scale_time_impl(deepcopy(self.orig_option_aca), end_time_scaler)

    def _scale_act_fn_dynamically(self):
        # ReLU6*beta_c
        act_fn_cls = self.ode_block.act_fn.__class__.__name__.lower()
        if "relu6" in act_fn_cls or "relux" in act_fn_cls:
            self.ode_block.act_fn.set_scale(min(6 * self.beta_c, self.v_dd))
            # self.ode_block.act_fn.set_scale(self.v_dd)
        elif "hardtanh" in act_fn_cls:
            # Ignore hardtanh branch for now
            self.ode_block.act_fn = nn.Hardtanh(min_val=-min(self.beta_c, self.v_dd),
                                                max_val=min(self.beta_c, self.v_dd))


def make_ode_block(pc_net: PCNet, ode_block=ODEBlockPC, noise_level=0.0, method=None, t_end=None, tol=1e-3, ts_scale=1,
                   return_init=False, n_steps=None, **kwargs):
    for i in range(pc_net.num_layers):
        cls = pc_net.PcConvs[i].cls if n_steps is None else n_steps
        t_step = pc_net.PcConvs[i].lr
        if t_end is None:
            t_end = cls * t_step
        else:
            t_step = t_end / cls if cls != 0 else 1.0
        t_step = t_step / ts_scale
        ode_kwargs = dict(kwargs)
        if isinstance(ode_block, type) and issubclass(ode_block, ToggleBaseFFFB):
            ode_kwargs.setdefault("toggle_n_cycles", int(cls))
        pc_net.PcConvs[i] = ode_block(
            pc_conv=pc_net.PcConvs[i], noise_level=noise_level, method=method, t_end=t_end, t_step=t_step, tol=tol,
            return_init=return_init[i] if isinstance(return_init, list) else False,
            **ode_kwargs)
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
    toggle_wrappers = (ToggleWrapper1State, ToggleQATTester1State, ToggleQATWrapper1State,
                       TogglePulseWrapper1State, TogglePulseQATTester1State,
                       TogglePulseQATWrapper1State)
    for i in range(pc_net.num_layers):
        if isinstance(pc_net.PcConvs[i], (ToggleAveragedPhysicalFFFB, TogglePulseFFFB)) and \
                not issubclass(ode_wrapper, toggle_wrappers):
            raise ValueError("Use ToggleWrapper1State/ToggleQATTester1State/ToggleQATWrapper1State with physical or pulse toggle blocks to avoid legacy RC/time double scaling.")
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
    "ODEBlockFFFB": ODEBlockFFFB,
    "ODEBlockXInit": ODEBlockXInit,
    "ODEBlockPCLimitDyn": ODEBlockPCLimitDyn,
    "ODEBlockPCMinusY": ODEBlockPCMinusY,
    "ODEBlk0Init": ODEBlk0Init,
    "ODEBlkActInp": ODEBlkActInp,
    "ODEBlkActDyn": ODEBlkActDyn,
    "ODEActInpNoMinus": ODEActInpNoMinus,
    "ODEActDynNoMinus": ODEActDynNoMinus,
    "ODEBlkProj": ODEBlkProj,
    "ODEBlkProjInitY": ODEBlkProjInitY,
    "ODESelfCoupleInitY": ODESelfCoupleInitY,
    "ODESumAsBInitY": ODESumAsBInitY,
    "ODESumAsBInitYFFFB": ODESumAsBInitYFFFB,
    "ODENoisyOffset": ODENoisyOffset,
    "ODEFixNoiseOffset": ODEFixNoiseOffset,
    "ODEFixNoiseFFFB": ODEFixNoiseFFFB,
    "ODEFixNoise0Init": ODEFixNoise0Init,
    "ODEActDynInitY": ODEActDynInitY,
    "ODEFixNoiseXInit": ODEFixNoiseXInit,
    "ODEFFFBConv": ODEFFFBConv,
    "ODEFFConv": ODEFFConv,
    "ODESumAsBInitYAsX": ODESumAsBInitYAsX,
    "ODEFixNoiseXInitFFFB": ODEFixNoiseXInitFFFB,
    "SumAsBInitYAsXFFFB": SumAsBInitYAsXFFFB,
    "FixNoiseXInitFFFBNoExpand": FixNoiseXInitFFFBNoExpand,
    "ODEXInitFFFB": ODEXInitFFFB,
    "ToggleResetZ": ToggleResetZ,
    "ToggleKeepZ": ToggleKeepZ,
    "ToggleODEXInitFFFB": ToggleODEXInitFFFB,
    "ToggleODEXInitKeep": ToggleODEXInitKeep,
    "TogglePulseResetZ": TogglePulseResetZ,
    "TogglePulseKeepZ": TogglePulseKeepZ,
    "TogglePulseODEXInitFFFB": TogglePulseODEXInitFFFB,
    "TogglePulseODEXInitKeep": TogglePulseODEXInitKeep,
    "TogglePulseBlk": TogglePulseBlk,
    "TogglePulseBlkXInitFFFB": TogglePulseBlkXInitFFFB,
    "ODEFixNoise0InitExpand": ODEFixNoise0InitExpand,
    "ODEFixNoise0InitFFFB": ODEFixNoise0InitFFFB,
    "ODESumAsBInitYAs0": ODESumAsBInitYAs0,
    "SumAsBInitYAs0FFFB": SumAsBInitYAs0FFFB,
    # Using 2 states
    "ODEState2FFFB": ODEState2FFFB,
    "State2InitYZ": State2InitYZ,
    "State2InitYAsXZAs0": State2InitYAsXZAs0,
    "State2InitYAs0ZAsX": State2InitYAs0ZAsX,
    "State2InitYAsXZAsX": State2InitYAsXZAsX,
    "State2NoMinusZ": State2NoMinusZ,
    "State2NoMinusZYAsXZAs0": State2NoMinusZYAsXZAs0,
    "State2NoMinusZYAs0ZAsX": State2NoMinusZYAs0ZAsX,
    "State2NoMinusZYAsXZAsX": State2NoMinusZYAsXZAsX,
    "S2NoMinusZChargeZ": S2NoMinusZChargeZ,
    "S2NoMinusZChargeZMinus": S2NoMinusZChargeZMinus,
    "S2NoMinusZChgZNoisyI": S2NoMinusZChgZNoisyI,
    "S2NoMinusZChgZMinusNoisyI": S2NoMinusZChgZMinusNoisyI,
    "S2NoisyIYAsXZAsX": S2NoisyIYAsXZAsX,
    "S2NoisyIYAs0ZAsX": S2NoisyIYAs0ZAsX,
    "S2NoisyIYAsXZAs0": S2NoisyIYAsXZAs0,
    "S2Circ": S2Circ,
    "S2CircYAsXZasX": S2CircYAsXZasX,
    "S2CircYAs0ZasX": S2CircYAs0ZasX,
    "S2CircYAsXZas0": S2CircYAsXZas0,
    # Using summation of abs value
    "SelfCUAbsSumFFFB": SelfCUAbsSumFFFB,
    "SelfCUAbsSumFFFBInitB": SelfCUAbsSumFFFBInitB,
    "SelfCUAbsSumFFFBFixNoise": SelfCUAbsSumFFFBFixNoise,
    "SelfCUAbsSumFFFBNoisy": SelfCUAbsSumFFFBNoisy,
}

ODEWrapper_CLASSES = {
    "ODEWrapperRC": ODEWrapperRC,
    "WrapQuantizeW": WrapQuantizeW,
    "WrapQuantizeWXInit": WrapQuantizeWXInit,
    "ODEWrapper2State": ODEWrapper2State,
    "QATTester2State": QATTester2State,
    "QATWrapper2State": QATWrapper2State,
    "ODEWrapper1State": ODEWrapper1State,
    "QATTester1State": QATTester1State,
    "QATWrapper1State": QATWrapper1State,
    "ToggleWrapper1State": ToggleWrapper1State,
    "ToggleQATTester1State": ToggleQATTester1State,
    "ToggleQATWrapper1State": ToggleQATWrapper1State,
    "TogglePulseWrapper1State": TogglePulseWrapper1State,
    "TogglePulseQATTester1State": TogglePulseQATTester1State,
    "TogglePulseQATWrapper1State": TogglePulseQATWrapper1State,
    "ODEWrapper1StateWithX": ODEWrapper1StateWithX,
    "QATTester1StateWithX": QATTester1StateWithX,
    "QATWrapper1StateWithX": QATWrapper1StateWithX,
}

QUANTIZER_CLASSES = {
    "SymQuantizeWeight": SymQuantizeWeight,
    "PulseSymQuantizeWeight": PulseSymQuantizeWeight,
    "LSQWeight": LSQWeight,
    None: SymQuantizeWeight,
}

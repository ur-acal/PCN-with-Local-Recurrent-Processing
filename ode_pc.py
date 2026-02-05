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
from utils import expand_weights_to_matrix, load_res_vs_vin
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

    def forward_full_steps(self, x, layer_idx=None):
        # Returning the full trajectory of the solver.
        y0 = self.init_y(x)
        self.integration_time = self.integration_time.type_as(x)

        out = aca_ode_solve(self._make_ode_fn(x), y0, self.option_aca, full_traj=True)
        # out = odeint(ode_func, y0, self.integration_time, rtol=self.tol, atol=self.tol, method=self.method)

        if self.bypass is not None:
            out = self.bypass(out) + out
        return out

    def _apply_noise(self, p):
        if getattr(p, "is_sparse_csr", False):
            v_ = p.values()
            noise_ = torch.randn_like(v_, device=p.device, requires_grad=False) * self.noise_level
            v_.mul_(1 + noise_) # This will change values of CSR matrix in-place
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
        with torch.no_grad():
            s_w = 1 / layer_weight.data.abs().max()
            self.s_w.copy_(s_w)


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
        self.R_code_round_base = kwargs.pop("R_code_round_base", 1)

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

        self.v_grid, self.R_codes, self.R_table = load_res_vs_vin(
            R=self.R, R_max=self.R_max, device=self.ode_block.FFconv.weight.device)

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
            "proj_fn": getattr(self, "proj_fn", None)
        }


class ODEWrapper2State(WrapQuantizeW):
    def __init__(self, is_first=False, is_last=False, thermal_noise=True, offset_eps=None, tie_cap=False,
                 R_max=180e3, **kwargs):
        patch = kwargs.get("patch", True)
        quantize = kwargs.get("quantize", True)
        kwargs.update({"patch": False})
        super().__init__(**kwargs)
        self.R_max = R_max
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
            return orig_call(self.proj_fn(self.inp_scale * x), *args, **kwargs) / self.out_scale
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
        logging.warning("6 * beta_c = {}, self.s_fb = {}".format(6 * self.beta_c, self.s_fb))
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
            self.ode_block.act_fn = ReLUX(min(6 * self.beta_c, self.v_dd))
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
            self.ode_block.act_fn.set_scale(min(6 * self.beta_c, self.v_dd))
        elif "hardtanh" in act_fn_cls:
            # Ignore hardtanh branch for now
            self.ode_block.act_fn = nn.Hardtanh(min_val=-min(self.beta_c, self.v_dd),
                                                max_val=min(self.beta_c, self.v_dd))


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
    "ODEWrapper2State": ODEWrapper2State,
    "QATTester2State": QATTester2State,
    "QATWrapper2State": QATWrapper2State,
    "ODEWrapper1State": ODEWrapper1State,
    "QATTester1State": QATTester1State,
    "QATWrapper1State": QATWrapper1State,
}

QUANTIZER_CLASSES = {
    "SymQuantizeWeight": SymQuantizeWeight,
    "LSQWeight": LSQWeight,
    None: SymQuantizeWeight,
}

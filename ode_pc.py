import os
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as P
import numpy as np
import matplotlib.pyplot as plt
from typing import Union

from pc_model import PCNet
from pc_conv import PCConv, PCConvNoisy, PCConvHardTanhLimit, PCConvHardTanhLimitNoisy, PCConvHardTanhNoisy, PCConvHardTanh
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
                 tol=1e-3, return_mid=False, init_b=False):
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

        if self.noise_level is not None and self.noise_level > 0:
            self.add_noise()

        if t_end is not None:
            self.integration_time = torch.tensor([0, t_end]).float()
        elif t_end is not None and t_step is not None and return_mid:
            self.integration_time = torch.arange(0, t_end + t_step, t_step).float()
        else:
            self.integration_time = torch.arange(0, (pc_conv.cls + 1) * pc_conv.lr, pc_conv.lr).float()

        if is_adaptive(method):
            # automatically pick the initial step size for adaptive methods
            t_step = None

        self.tol = tol
        self.method = method

        self.option_aca = {"t0": self.integration_time[0], "t1": self.integration_time[-1],
                           "t_eval": self.integration_time.tolist(), "rtol": self.tol, "atol": self.tol,
                           "h": t_step, "method": self.method}

    def forward(self, x, layer_idx=None):
        y0 = self.act_fn(self.FFconv(x))
        def ode_func(t, y):
            return self.FFconv(self.act_fn(x - self.FBconv(y)))

        self.integration_time = self.integration_time.type_as(x)

        out = aca_ode_solve(ode_func, y0, self.option_aca)
        # out = odeint(ode_func, y0, self.integration_time, rtol=self.tol, atol=self.tol, method=self.method)
        out = out[-1]

        if self.bypass is not None:
            out = self.bypass(out) + out
        return out

    def _apply_noise(self, p):
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

    @property
    def nfe(self):
        return self.ode_func.nfe

    @nfe.setter
    def nfe(self, value):
        self.ode_func.nfe = value


class ODEBlockPCLimitDyn(ODEBlockPC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x, layer_idx=None):
        y0 = self.act_fn(self.FFconv(x))
        def ode_func(t, y):
            return self.act_fn(self.FFconv(self.act_fn(x - self.FBconv(y))))

        self.integration_time = self.integration_time.type_as(x)
        out = aca_ode_solve(ode_func, y0, self.option_aca)
        # out = odeint(ode_func, y0, self.integration_time, rtol=self.tol, atol=self.tol, method=self.method)
        out = out[-1]

        if self.bypass is not None:
            out = self.bypass(out) + out
        return out


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
        y0 = self.act_fn(self.FFconv(x))
        out = aca_ode_solve(self._make_ode_fn(x), y0, self.option_aca)
        out = out[-1]

        if self.bypass is not None:
            out = self.bypass(out) + out
        return out

class ODESumAsBInitY(ODESelfCoupleInitY):
    def __init__(self, **kwargs):
        kwargs.update({"init_b": True})
        super().__init__(**kwargs)

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

    def forward(self, x, layer_idx=None):
        y0 = self.act_fn(self.FFconv(x))
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

class ODEBlockPCMinusY(ODEBlockPC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x, layer_idx=None):
        y0 = self.act_fn(self.FFconv(x))
        def ode_func(t, y):
            weight_sum = self.FFconv.weight.view(y.shape[1], -1).sum(-1)
            # weight_sum = self.FFconv.weight.data.view(y.shape[1], -1).mean(-1)
            offset = weight_sum.view(1, -1, 1, 1) * y
            return self.FFconv(self.act_fn(x - self.FBconv(y))) - offset

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

        out = aca_ode_solve(ode_func, y0, self.option_aca)
        out = out[-1]

        if self.bypass is not None:
            out = self.bypass(out) + out
        return out


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


ODEBLOCK_CLASSES = {
    "ODEBlockPC": ODEBlockPC,
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
    "ODENoisyOffset": ODENoisyOffset,
    "ODEFixNoiseOffset": ODEFixNoiseOffset,
}

import os
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as P
import numpy as np
import matplotlib.pyplot as plt

from pc_model import PCNet
from pc_conv import PCConv, PCConvNoisy, PCConvHardTanhLimit, PCConvHardTanhLimitNoisy, PCConvHardTanhNoisy, PCConvHardTanh
from torchdiffeq import odeint

import logging
log = logging.getLogger(__name__)


def is_adaptive(method):
    adaptive_sols = ['dopri8', 'dopri5', 'bosh3', 'fehlberg2', 'adaptive_heun']
    fixed_grid_sols = ['euler', 'midpoint', 'heun2', 'heun3', 'rk4', 'explicit_adams',
                       'implicit_adams', 'fixed_adams']  # 'scipy_solver'
    return method in adaptive_sols


class ODEBlockPC(nn.Module):

    def __init__(self, pc_conv: PCConvNoisy, noise_level=0.0, method="dopri5", t_end=None, t_step=None, tol=1e-3):
        super(ODEBlockPC, self).__init__()
        self.noise_level = noise_level
        self.tie_weights = pc_conv.tie_weights
        self.tie_bp = pc_conv.tie_bp
        self.layer_idx = pc_conv.layer_idx

        self.FFconv = pc_conv.FFconv
        self.FBconv = pc_conv.FBconv
        self.bypass = pc_conv.bypass
        self.act_fn = pc_conv.relu

        if self.noise_level is not None and self.noise_level > 0:
            self.add_noise()

        if is_adaptive(method):
            self.integration_time = torch.tensor([0, t_end]).float()
        elif t_end is None or t_step is None:
            self.integration_time = torch.arange(0, (pc_conv.cls + 1) * pc_conv.lr, pc_conv.lr).float()
        else:
            self.integration_time = torch.arange(0, t_end + t_step, t_step).float()

        self.tol = tol
        self.method = method

    def forward(self, x, layer_idx=None):
        y0 = self.act_fn(self.FFconv(x))
        def ode_func(t, y):
            return self.FFconv(self.act_fn(x - self.FBconv(y)))

        self.integration_time = self.integration_time.type_as(x)
        out = odeint(ode_func, y0, self.integration_time, rtol=self.tol, atol=self.tol, method=self.method)
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

    @property
    def nfe(self):
        return self.ode_func.nfe

    @nfe.setter
    def nfe(self, value):
        self.ode_func.nfe = value


def make_ode_block(pc_net: PCNet, ode_block=ODEBlockPC, noise_level=0.0, method=None, t_end=None, tol=1e-3, ts_scale=1):
    for i in range(pc_net.num_layers):
        cls, t_step = pc_net.PcConvs[i].cls, pc_net.PcConvs[i].lr
        if t_end is None:
            t_end = cls * t_step
        else:
            t_step = t_end / cls
        t_step = t_step / ts_scale
        pc_net.PcConvs[i] = ode_block(
            pc_net.PcConvs[i], noise_level=noise_level, method=method, t_end=t_end, t_step=t_step, tol=tol)
    return pc_net

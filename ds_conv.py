import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DSConv(nn.Module):
    """
    Implementing forward pass of normal convolutional layer using gradient descent.
    """
    def __init__(self, conv_layer: nn.Conv2d, n_block=1, **kwargs):
        super().__init__()
        pass

    def forward(self, x):
        pass


class BRIMSolver(nn.Module):
    def __init__(self, brim_r=300e3, brim_c=49e-15, t_step=2.2e-11, t_stop=2.2e-5, scale_start=2.2, scale_end=0.6,
                 clip_spin=True, J=None, h=None, n_var=20):
        super().__init__()
        self.brim_r = brim_r
        self.brim_c = brim_c
        self.t_step = t_step
        self.t_stop = t_stop
        self.scale_start = scale_start
        self.scale_end = scale_end
        self.clip_spin = clip_spin

        self.noise_std = np.sqrt(t_step) * np.sqrt(1 / (brim_r * brim_c))

        self.annealing_steps = t_stop / t_step
        self.scale_step = (scale_start - scale_end) / self.annealing_steps

        self.J = J / brim_r if J is not None else torch.randn(n_var, n_var)
        self.h = h if h is not None else torch.zeros(self.J.shape[0], 1)

    def forward(self, x=None):
        spin = self.init_spin(x)
        spin = self.solve(spin)
        return spin

    def solve(self, spin):
        current_scale = self.scale_start
        for i in range(int(self.annealing_steps)):
            noise_ = current_scale * torch.randn_like(spin) * self.noise_std
            spin = spin + (self.t_step / self.brim_c) * (self.J @ spin + self.h) + noise_
            if self.clip_spin:
                spin = torch.clip(spin, -1, 1)
            else:
                # Todo: Add self coupling to make J definite?
                # spin = spin + (self.t_step / self.brim_c) * ((self.J + torch.diag(self.h)) @ spin) + noise_
                pass
            current_scale += self.scale_step
        return spin

    def init_spin(self, x):
        if x is None:
            return 2 * torch.rand(self.J.shape[0], 1) - 1
        return x


class DSConvBlock(BRIMSolver):
    def __init__(self, conv_layer: nn.Conv2d, n_blocks=1, pvt_noise_level=0, pvt_noise=None, **kwargs):
        super().__init__(**kwargs)
        # Todo: Do annealing or not in DSConvBlock
        self.weight = conv_layer.weight.data
        self.device = self.weight.device
        self.C_in = self.weight.shape[1]
        self.C_out = self.weight.shape[0]
        self.kernel_size = self.weight.shape[-1]
        self.weight = self.weight.view(self.C_out, -1)

        self.y_len = self.C_out
        self.x_len = self.C_in * self.kernel_size * self.kernel_size
        self.n_blocks = n_blocks

        # noise related
        self.pvt_noise_level = pvt_noise_level
        self.pvt_noise = pvt_noise

        self._constr_cu()

    def solve(self, spin):
        current_scale = self.scale_start
        for i in range(int(self.annealing_steps)):
            # calculate noise and gradient
            noise_ = current_scale * torch.randn_like(spin) * self.noise_std
            grad_ = (self.t_step / self.brim_c) * torch.matmul(self.J, spin)

            # clamp x; Assume there are no thermal noise added to the clamped x
            grad_[:, self.y_len:, :] = 0.0
            noise_[:, self.y_len:, :] = 0.0

            # update spin
            spin = spin + grad_ + noise_
            if self.clip_spin:
                spin = torch.clamp(spin, -1, 1)
            current_scale += self.scale_step
        return spin

    def _constr_cu(self):
        # Need to divide generated CU matrix, i.e. J, by self.brim_r
        # J = [[-I,  W]]
        #     [[W.T, 0]]
        upper_ = torch.cat(
            [-1 * torch.eye(self.y_len).to(self.device), self.weight], dim=1)
        lower_ = torch.cat(
            [self.weight.t(), torch.zeros(self.x_len, self.x_len).to(self.device)], dim=1)
        self.J = torch.cat([upper_, lower_], dim=0) / self.brim_r

        self.J_list = []
        if self.pvt_noise is not None:
            # Todo: Right now we assume that we can do HIL training for hardware with only one compute block
            #       for each layer. If we have multiple compute blocks, how should we do HIL training?
            # pvt_noise has the same shape as J
            self.J_list.append(self.J * (1 + self.pvt_noise.to(self.device)))
        elif self.pvt_noise_level is not None:
            for i in range(self.n_blocks):
                pvt_noise_ = torch.randn_like(self.J) * self.pvt_noise_level
                self.J_list.append(self.J * (1 + pvt_noise_.to(self.device)))
        else:
            # We consider n_blocks > 1 only when we are using pvt_noise_level to generate noise.
            # If there is no mismatch, one compute block is enough for simulation.
            self.J_list.append(self.J)


    def init_spin(self, x):
        # x is expected to have shape (BS, C_in * Ker * Ker, H * W)
        y_ = 2 * torch.rand(x.shape[0], self.y_len, x.shape[-1], device=self.device) - 1
        return torch.cat([y_, x], dim=1)


class DSPcRecurrentBlock(DSConvBlock):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._constr_cu()
        # Todo: 1. Use fully expanded weights of fb_flipped and flatten x and y
        #       2. Try to use block method to optimize the linear least squares problem

    def _constr_cu(self):
        # Need to divide generated CU matrix, i.e. J, by self.brim_r
        # J = [[-W.T @ W,   W]]
        #     [[W.T,        0]]
        upper_ = torch.cat(
            [torch.eye(self.y_len).to(self.device), self.weight], dim=1)
        lower_ = torch.cat(
            [self.weight.t(), torch.zeros(self.x_len, self.x_len).to(self.device)], dim=1)
        self.J = torch.cat([upper_, lower_], dim=0) / self.brim_r

    def init_spin(self, x):
        # x is expected to be flattened
        y_ = 2 * torch.rand(self.y_len, 1) - 1
        return torch.cat([y_.to(self.device), x], dim=0)


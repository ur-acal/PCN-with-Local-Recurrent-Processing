import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from pc_conv import PCConvNoisy

import logging
log = logging.getLogger(__name__)

class PCConvDS(PCConvNoisy):
    """
    Implementing forward pass of PC convolutional layer using LD based on MVM method.
    """
    def __init__(self, n_blocks=1, pvt_noise=None, **kwargs):
        super().__init__(**kwargs)
        self.pvt_noise = pvt_noise
        self.n_blocks = n_blocks

        if kwargs.get("use_pc", False):
            self.FBconv_inv = nn.Conv2d(kwargs["out_chan"], kwargs["inp_chan"],
                                        kwargs["kernel_size"], kwargs["stride"],
                                        kwargs["padding"], bias=kwargs["bias"])

    def _init_ds_conv_block(self):
        """
        *** Must call this after the weights are loaded. ***
        PVT Noise or mismatch will be added when initialize the DSConvBlock, whose J_list will have noisy matrices.
        """
        self.noisy_fb, self.noisy_ff = None, None # not used, compatible with PCConvNoisy
        self.tie_weights_impl()
        self.FFconvDS = DSConvBlock(self.FFconv, self.n_blocks, pvt_noise=self.pvt_noise,
                                    pvt_noise_level=self.noise_level)
        if self.use_pc:
            self.FBconv_inv.weight.data = self.FBconv.weight.data.permute([1,0,2,3]).flip([2, 3])
            self.FBconvDS = DSConvBlock(self.FBconv_inv, self.n_blocks, pvt_noise=self.pvt_noise,
                                        pvt_noise_level=self.noise_level)
        if self.bypass is not None:
            self.BPconvDS = DSConvBlock(self.bypass, self.n_blocks, pvt_noise=self.pvt_noise,
                                        pvt_noise_level=self.noise_level)

    def forward(self, x, layer_idx=None, w_type_used=None, use_relu=True):
        log.info("--- Forward in DS PC layer: {} ---".format(self.layer_idx))
        y = self.relu(self.FFconvDS(x))

        # PC recurrent
        if self.use_pc:
            log.info("USE PC")
            # injected noise inside find_optimal_r
            y = self.find_optimal_r(x, y, self.layer_idx, w_type_used, use_relu)

        # Bypass convolution
        if self.bypass is not None and not self.relu_bp:
            log.info("DO NOT USE ReLU after BP DS")
            y = y + self.BPconvDS(x)
        if self.bypass is not None and self.relu_bp:
            log.info("USE ReLU after BP DS")
            y = y + self.relu(self.BPconvDS(x))
        return y

    def find_optimal_r(self, x, y, layer_idx=None, w_type_used=None, use_relu=None):
        for _ in range(self.cls):
            if self.relu_between:
                log.info("USE ReLU between FF/FB DS")
                error = self.relu(x - self.FBconvDS(y))
            else:
                log.info("DO NOT USE ReLU between FF/FB DS")
                error = x - self.FBconvDS(y)
            y += self.lr * self.FFconvDS(error)
        return y


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
    def __init__(self, conv_layer: nn.Conv2d, n_blocks=1, pvt_noise_level=None, pvt_noise=None, **kwargs):
        super().__init__(**kwargs)
        # Todo: Do annealing or not in DSConvBlock
        self.weight = conv_layer.weight.data
        self.device = self.weight.device
        self.C_in = self.weight.shape[1]
        self.C_out = self.weight.shape[0]
        self.kernel_size = self.weight.shape[-1]
        self.padding = conv_layer.padding
        self.weight = self.weight.view(self.C_out, -1)

        self.y_len = self.C_out
        self.x_len = self.C_in * self.kernel_size * self.kernel_size

        # noise related
        self.pvt_noise_level = pvt_noise_level
        self.pvt_noise = pvt_noise

        # Construct J; If there are multiple compute blocks and need to initialize noise, set n_blocks also
        self._constr_cu(n_blocks)

    def forward(self, x=None):
        _bs, _, _h, _w = x.shape
        x = F.unfold(x, self.kernel_size, padding=self.padding)
        spin = self.init_spin(x)
        spin = self.solve(spin)
        return spin[:, :self.y_len, :].view(_bs, self.y_len, _h, _w)

    def solve(self, spin):
        if self.n_blocks == 1:
            spin = self._solve_one_block(self.J_list[0], spin)
        else:
            workload = self._dist_workload(spin)
            spin_res = []
            # Todo: How to get rid of the for loop
            for i in range(self.n_blocks):
                _start_idx, _end_idx = workload[i]
                # Each compute block computes the same position on every batch
                cur_spin = self._solve_one_block(self.J_list[i], spin[:, :, _start_idx:_end_idx])
                spin_res.append(cur_spin)
            spin = torch.cat(spin_res, dim=-1)
        return spin

    def _solve_one_block(self, J, spin):
        current_scale = self.scale_start
        for i in range(int(self.annealing_steps)):
            # calculate noise and gradient
            noise_ = current_scale * torch.randn_like(spin) * self.noise_std
            grad_ = (self.t_step / self.brim_c) * torch.matmul(J, spin)

            # clamp x; Assume there are no thermal noise added to the clamped x
            grad_[:, self.y_len:, :] = 0.0
            noise_[:, self.y_len:, :] = 0.0

            # update spin
            spin = spin + grad_ + noise_
            if self.clip_spin:
                spin = torch.clamp(spin, -1, 1)
            current_scale += self.scale_step
        return spin

    def _dist_workload(self, spin):
        annealing_round = spin.shape[-1]
        compute_per_block = annealing_round // self.n_blocks
        rem_compute = annealing_round % self.n_blocks
        workload = []
        start_idx = 0
        for i in range(self.n_blocks):
            # distribute rem_compute to the first rem_compute blocks
            extra_work = 1 if i < rem_compute else 0
            workload.append((start_idx, start_idx + compute_per_block + extra_work))
            start_idx += compute_per_block + extra_work
        return workload

    def _constr_cu(self, n_blocks):
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
        else:
            pvt_noise_level = self.pvt_noise_level if self.pvt_noise_level is not None else 0.0
            for i in range(n_blocks):
                pvt_noise_ = torch.randn_like(self.J) * pvt_noise_level
                self.J_list.append(self.J * (1 + pvt_noise_.to(self.device)))
        self.n_blocks = len(self.J_list)


    def init_spin(self, x):
        # x is expected to have shape (BS, C_in * Ker * Ker, H * W)
        y_ = 2 * torch.rand(x.shape[0], self.y_len, x.shape[-1], device=self.device) - 1
        return torch.cat([y_, x], dim=1)


class DSPcRecurrentBlock(DSConvBlock):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._constr_cu(kwargs.get("n_blocks", 1))
        # Todo: 1. Use fully expanded weights of fb_flipped and flatten x and y
        #       2. Try to use block method to optimize the linear least squares problem

    def _constr_cu(self, n_blocks):
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


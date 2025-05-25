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
    def __init__(self, n_blocks=1, pvt_noise=None, fast=True, **kwargs):
        super().__init__(**kwargs)
        self.pvt_noise = pvt_noise
        self.n_blocks = n_blocks
        self.kwargs = kwargs

        self.ds_conv_blk = DSConvBlockFast if fast else DSConvBlock
        self.FFconvDS, self.FBconvDS, self.BPconvDS = None, None, None

    def init_ds_conv_block(self):
        """
        *** Must call this after the weights are loaded. ***
        PVT Noise or mismatch will be added when initialize the DSConvBlock, whose J_list will have noisy matrices.
        """
        self.tie_weights_impl()
        self.FFconvDS = self.ds_conv_blk(conv_layer=self.FFconv, n_blocks=self.n_blocks,
                                         pvt_noise=self.pvt_noise, pvt_noise_level=self.noise_level)
        if self.use_pc:
            fb_conv_inv = nn.Conv2d(self.kwargs["out_chan"], self.kwargs["inp_chan"],
                                    self.kwargs["kernel_size"], self.kwargs["stride"],
                                    self.kwargs["padding"], bias=self.kwargs["bias"])
            fb_conv_inv.weight.data = self.FBconv.weight.data.permute([1, 0, 2, 3]).flip([2, 3])
            self.FBconvDS = self.ds_conv_blk(conv_layer=fb_conv_inv, n_blocks=self.n_blocks,
                                             pvt_noise=self.pvt_noise, pvt_noise_level=self.noise_level)
        if self.bypass is not None:
            self.BPconvDS = self.ds_conv_blk(conv_layer=self.bypass, n_blocks=self.n_blocks,
                                             pvt_noise=self.pvt_noise, pvt_noise_level=self.noise_level)

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
    def __init__(self, brim_r=300e3, brim_c=49e-15, t_step=1.5e-11, t_stop=1e-7, scale_start=0.02, scale_end=0.02,
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
    def __init__(self, conv_layer: nn.Conv2d, n_blocks=1, pvt_noise_level=None, pvt_noise=None, x_noise=True, **kwargs):
        super().__init__(**kwargs)
        # Todo: Do annealing or not in DSConvBlock
        self.weight = conv_layer.weight.data
        self.device = self.weight.device
        self.C_in = self.weight.shape[1]
        self.C_out = self.weight.shape[0]
        self.kernel_size = self.weight.shape[-1]
        self.padding = conv_layer.padding
        self.weight = self.weight.reshape(self.C_out, -1)

        self.y_len = self.C_out
        self.x_len = self.C_in * self.kernel_size * self.kernel_size

        # noise related
        self.pvt_noise_level = pvt_noise_level
        self.pvt_noise = pvt_noise
        self.x_noise = x_noise

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
            if i - 777 == 0:
                st = time.time()
            noise_ = current_scale * torch.randn_like(spin) * self.noise_std
            if i - 777 == 0:
                log.info("Random sample using: {} s".format(time.time() - st))
                st = time.time()
            grad_ = (self.t_step / self.brim_c) * torch.matmul(J, spin)
            if i - 777 == 0:
                log.info("Calculate gradient using: {} s".format(time.time() - st))
                st = time.time()

            # clamp x; Assume there are no thermal noise added to the clamped x
            grad_[:, self.y_len:, :] = 0.0
            if not self.x_noise:
                noise_[:, self.y_len:, :] = 0.0
            if i - 777 == 0:
                log.info("Zero out gradient using: {} s".format(time.time() - st))
                st = time.time()

            # update spin
            spin = spin + grad_ + noise_
            if i - 777 == 0:
                log.info("Update spin using: {} s".format(time.time() - st))
                st = time.time()
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


class DSConvBlockFast(DSConvBlock):
    def __init__(self, J_list=None, **kwargs):
        super().__init__(**kwargs)
        if J_list is not None:
            # Use the passed in J_list instead of the reinitialized J_list
            self.J_list = J_list
        # Construct I and W list
        self._constr_cu_fast()

    def forward(self, x=None):
        _bs, _, _h, _w = x.shape
        x = F.unfold(x, self.kernel_size, padding=self.padding)
        spin_y, spin_x = self.init_spin(x)
        spin_y = self.solve_fast(spin_y, spin_x)
        return spin_y.view(_bs, self.y_len, _h, _w)

    def solve_fast(self, spin_y, spin_x):
        if self.n_blocks == 1:
            spin_y = self._solve_one_block_fast(self.neg_I_list[0], self.W_list[0], spin_y, spin_x)
        else:
            workload = self._dist_workload(spin_y)
            spin_res = []
            # Todo: How to get rid of the for loop
            for i in range(self.n_blocks):
                _start_idx, _end_idx = workload[i]
                # Each compute block computes the same position on every batch
                cur_spin = self._solve_one_block_fast(self.neg_I_list[0], self.W_list[0],
                                                      spin_y[:, :, _start_idx:_end_idx],
                                                      spin_x[:, :, _start_idx:_end_idx])
                spin_res.append(cur_spin)
            spin_y = torch.cat(spin_res, dim=-1)
        return spin_y

    def _solve_one_block_fast(self, neg_I, W, spin_y, spin_x):
        current_scale = self.scale_start
        for i in range(int(self.annealing_steps)):
            # calculate noise and gradient
            if i - 777 == 0:
                st = time.time()
            y_noise_ = current_scale * torch.randn_like(spin_y) * self.noise_std
            if i - 777 == 0:
                log.info("Random sample using: {} s".format(time.time() - st))
                st = time.time()
            y_grad_ = torch.matmul(neg_I, spin_y) + torch.matmul(W, spin_x)
            if i - 777 == 0:
                log.info("Calculate gradient using: {} s".format(time.time() - st))
                st = time.time()

            # update spin
            spin_y = spin_y + y_grad_ + y_noise_
            if self.x_noise:
                x_noise_ = current_scale * torch.randn_like(spin_x) * self.noise_std
                spin_x = spin_x + x_noise_
            if i - 777 == 0:
                log.info("Update spin using: {} s".format(time.time() - st))
                st = time.time()
            if self.clip_spin:
                spin_y = torch.clamp(spin_y, -1, 1)
                spin_x = torch.clamp(spin_x, -1, 1)
            current_scale += self.scale_step
        return spin_y

    def _constr_cu_fast(self):
        # _constr_cu of parent class called before
        tc_const = self.t_step / self.brim_c
        self.neg_I_list, self.W_list = [], []
        for J in self.J_list:
            self.neg_I_list.append(tc_const * J[:self.y_len, :self.y_len])
            self.W_list.append(tc_const * (J[:self.y_len, self.y_len:] + J[self.y_len:, :self.y_len].t()) / 2)

    def init_spin(self, x):
        # x is expected to have shape (BS, C_in * Ker * Ker, H * W)
        y_ = 2 * torch.rand(x.shape[0], self.y_len, x.shape[-1], device=self.device) - 1
        return y_, x


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


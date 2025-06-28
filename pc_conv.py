'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function
import os
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as P
import numpy as np
import matplotlib.pyplot as plt

import logging
log = logging.getLogger(__name__)


class PCConv(nn.Module):
    def __init__(self, inp_chan, out_chan, kernel_size=3, stride=1, padding=1, cls=5, bias=False, lr=1e-2,
                 tie_weights=False, tie_bp=False, relu_between=True, bypass=True, layer_idx=None,
                 relu_bp=False, use_pc=True, zero_init=False):
        super().__init__()
        self.FFconv = nn.Conv2d(inp_chan, out_chan, kernel_size, stride, padding, bias=bias)
        self.FBconv = None
        self.b0 = nn.ParameterList([nn.Parameter(torch.zeros(1, out_chan, 1, 1))])
        self.relu = nn.ReLU(inplace=True)
        self.cls = cls
        self.lr = lr
        self.bypass = None
        self.relu_between = relu_between
        self.relu_bp = relu_bp
        self.use_pc = use_pc

        if use_pc:
            log.info("Use PC, initialize FBconv")
            self.FBconv = nn.ConvTranspose2d(out_chan, inp_chan, kernel_size, stride, padding, bias=bias)
            if zero_init:
                log.info("Initialize FBconv with zero weight")
                nn.init.constant_(self.FBconv.weight, 0)
        if tie_weights and use_pc:
            log.info("Tie the weights of FF and FB")
            self.FFconv.weight = self.FBconv.weight
            self.FFconv.bias = self.FBconv.bias
        if not tie_bp and bypass:
            log.info("With independent Bypass convolution")
            self.bypass = nn.Conv2d(inp_chan, out_chan, kernel_size=1, stride=1, bias=False)
        elif tie_bp and bypass:
            log.info("Tie the weights of Bypass and FF")
            self.bypass = self.FFconv

        self.layer_idx = layer_idx

    def forward(self, x, layer_idx=None):
        log.info("--- Forward in PC layer: {} ---".format(self.layer_idx))
        # Initializer of recurrent
        y = self.relu(self.FFconv(x))

        # PC recurrent
        if self.use_pc:
            log.info("USE PC")
            y = self.find_optimal_r(x, y, layer_idx)

        # Bypass convolution
        if self.bypass is not None:
            if self.relu_bp:
                log.info("USE {} as Non-linearity after BP".format(self.relu))
                y = y + self.relu(self.bypass(x))
            else:
                log.info("DO NOT USE Non-linearity after BP")
                y = y + self.bypass(x)
        return y

    def find_optimal_r(self, x, y, layer_idx=None):
        for _ in range(self.cls):
            if self.relu_between:
                log.info("USE {} as Non-linearity between FF/FB".format(self.relu))
                y = self.lr * self.FFconv(self.relu(x - self.FBconv(y))) + y
            else:
                log.info("DO NOT USE Non-linearity between FF/FB")
                y = self.lr * self.FFconv(x - self.FBconv(y)) + y
        return y


class PCConvNoisy(nn.Module):
    def __init__(self, inp_chan, out_chan, kernel_size=3, stride=1, padding=1, cls=5, bias=False, lr=1e-2,
                 tie_weights=False, tie_bp=False, relu_between=True, bypass=True, layer_idx=None,
                 relu_bp=False, use_pc=True, zero_init=False, # below are parameters in Noisy PCConv only
                 noise_level=None, weight=None, plot_path=None, w_type="fb_flip",
                 noise_to_ff=True, noise_to_bp=True, tie_noise=False, tie_noise_bp=False, diff_noise=False,
                 call_pc=True):
        super().__init__()
        log.warning("Initializing PC layer {} with noise level: {}, cycles: {}, LR PC: {}".format(
            layer_idx, noise_level, cls, lr))
        self.noise_level = noise_level
        self.padding = padding
        self.stride = stride
        self.kernel_size = kernel_size
        self.C_in = inp_chan
        self.C_out = out_chan

        self.FFconv = nn.Conv2d(inp_chan, out_chan, self.kernel_size, self.stride, self.padding, bias=bias)
        self.FBconv = None
        self.b0 = nn.ParameterList([nn.Parameter(torch.zeros(1, out_chan, 1, 1))])
        self.bypass = None
        self.relu_between = relu_between
        self.relu_bp = relu_bp
        self.use_pc = use_pc
        self.call_pc = call_pc

        if use_pc:
            log.info("Use PC, initialize FBconv")
            self.FBconv = nn.ConvTranspose2d(out_chan, inp_chan, kernel_size, stride, padding, bias=bias)
            if zero_init:
                log.info("Initialize FBconv with zero weight")
                nn.init.constant_(self.FBconv.weight, 0)
        self.tie_weights = tie_weights
        if tie_weights and use_pc:
            log.info("Tie the weights of FF and FB")
            self.FFconv.weight = self.FBconv.weight
            self.FFconv.bias = self.FBconv.bias
        self.tie_bp = tie_bp
        if not tie_bp and bypass:
            log.info("With independent Bypass convolution")
            self.bypass = nn.Conv2d(inp_chan, out_chan, kernel_size=1, stride=1, bias=False)
        elif tie_bp and bypass:
            log.info("Tie the weights of Bypass and FF")
            self.bypass = self.FFconv

        # noise related
        self.diff_noise = diff_noise
        self.tie_noise = tie_noise
        self.tie_noise_bp = tie_noise_bp
        self.noise_ff_matrix, self.noise_fb_matrix, self.noise_bp_matrix = None, None, None
        self.noisy_ff, self.noisy_fb, self.noisy_bp = None, None, None
        self._init_noise(noise_level)

        self.relu = nn.ReLU(inplace=True)
        self.cls = cls
        self.lr = lr

        self.noise_to_ff = noise_to_ff
        self.noise_to_bp = noise_to_bp
        self.w_type = w_type
        self.plot_path = plot_path
        self.weight = weight
        self.layer_idx = layer_idx

    def forward(self, x, layer_idx=None, w_type_used=None, use_relu=True):
        log.info("--- Forward in PC layer: {} ---".format(self.layer_idx))
        if self.noise_to_ff:
            log.info("USE noisy_ff. noisy_ff equals ideal weight: {}".format(
                torch.allclose(self.noisy_ff, self.FFconv.weight.data)))
            assert self.noise_level == 0.0 or not torch.allclose(self.noisy_ff, self.FFconv.weight.data)
            y = self.relu(torch.conv2d(x, self.noisy_ff, padding=self.FFconv.padding))
        else:
            log.info("USE non-noisy FF")
            y = self.relu(self.FFconv(x))

        # PC recurrent
        if self.use_pc and self.call_pc:
            log.info("USE PC")
            # injected noise inside find_optimal_r
            y = self.find_optimal_r(x, y, self.layer_idx, w_type_used, use_relu)

        # Bypass convolution
        if self.bypass is not None and not self.relu_bp:
            if self.noise_to_bp:
                log.info("DO NOT USE ReLU after BP and use noisy_bp. noisy_bp equals ideal weight: {}".format(
                    torch.allclose(self.noisy_bp, self.bypass.weight.data)))
                assert self.noise_level == 0.0 or not torch.allclose(self.noisy_bp, self.bypass.weight.data)
                y = y + torch.conv2d(x, self.noisy_bp, padding=self.bypass.padding)
            else:
                log.info("DO NOT USE ReLU after BP and use non-noisy BP")
                y = y + self.bypass(x)
        if self.bypass is not None and self.relu_bp:
            if self.noise_to_bp:
                log.info("USE ReLU after BP and use noisy_bp. noisy_bp equals ideal weight: {}".format(
                    torch.allclose(self.noisy_bp, self.bypass.weight.data)))
                assert self.noise_level == 0.0 or not torch.allclose(self.noisy_bp, self.bypass.weight.data)
                y = y + self.relu(torch.conv2d(x, self.noisy_bp, padding=self.bypass.padding))
            else:
                log.info("USE ReLU after BP and use non-noisy BP")
                y = y + self.relu(self.bypass(x))
        return y

    def _gen_noisy_weight(self, p):
        noise_ = torch.randn_like(p, device=p.device,
                                  requires_grad=False) * (0.0 if self.noise_level is None else self.noise_level)
        return p * (1 + noise_)

    def find_optimal_r(self, x, y, layer_idx=None, w_type_used=None, use_relu=None):
        # if weights are tied, must call add_noise or tie_weights_impl after loading the weights
        # of the model and before calling forward
        for _ in range(self.cls):
            if self.diff_noise:
                log.info("Set different noise at each cycle")
                self.noisy_fb = self._gen_noisy_weight(self.FBconv.weight)
                self.noisy_ff = self._gen_noisy_weight(self.FFconv.weight)
            log.info("noisy_fb, noisy_ff equals ideal weight: {}, {}".format(
                torch.allclose(self.noisy_fb, self.FBconv.weight.data),
                     torch.allclose(self.noisy_ff, self.FFconv.weight.data)))
            if self.relu_between:
                log.info("USE Non-linearity: {} between FF/FB".format(self.relu))
                assert self.noise_level == 0.0 or not torch.allclose(self.noisy_fb, self.FBconv.weight.data)
                error = self.relu(x - torch.conv_transpose2d(y, self.noisy_fb, padding=self.FBconv.padding))
            else:
                log.info("DO NOT USE ReLU between FF/FB")
                assert self.noise_level == 0.0 or not torch.allclose(self.noisy_fb, self.FBconv.weight.data)
                error = x - torch.conv_transpose2d(y, self.noisy_fb, padding=self.FBconv.padding)
            assert self.noise_level == 0.0 or not torch.allclose(self.noisy_ff, self.FFconv.weight.data)
            y += self.lr * torch.conv2d(error, self.noisy_ff, padding=self.FFconv.padding)
        return y

    def _init_noise(self, noise_level):
        # tie weights processed in self.add_noise()
        self.noise_ff_matrix = torch.randn_like(self.FFconv.weight) * (0.0 if noise_level is None else noise_level)
        if self.use_pc:
            self.noise_fb_matrix = torch.randn_like(self.FBconv.weight) * (0.0 if noise_level is None else noise_level)
            self.noise_fb_matrix = self.noise_ff_matrix if self.tie_noise else self.noise_fb_matrix
        if self.bypass is not None:
            self.noise_bp_matrix = torch.randn_like(self.bypass.weight) * (0.0 if noise_level is None else noise_level)
            self.noise_bp_matrix = self.noise_ff_matrix[:,:,:1,:1] if self.tie_noise_bp else self.noise_bp_matrix

    def tie_weights_impl(self):
        """
        Call this or add_noise after the weight is loaded
        """
        if self.tie_weights and self.use_pc:
            log.info("After noise added, tie the weights of FF/FB")
            self.noisy_ff = self.noisy_fb
            self.FFconv.weight = self.FBconv.weight
            self.FFconv.bias = self.FBconv.bias

        if self.tie_bp and self.bypass is not None:
            log.info("After noise added, Tie the weights of Bypass and FF")
            self.noisy_bp = self.noisy_ff
            self.bypass.weight = self.FFconv.weight
            self.bypass.bias = self.FFconv.bias

    def add_noise(self):
        """
        Call this or tie_weights_impl after the weight is loaded.
        :return: None
        """
        self._init_noise(self.noise_level)
        log.info("Add noise to FF/FB")
        self.noise_ff_matrix = self.noise_ff_matrix.to(device=self.FFconv.weight.device)
        self.noisy_ff = (self.noise_ff_matrix + 1) * self.FFconv.weight

        if self.use_pc:
            log.info("Add noise to FB")
            self.noise_fb_matrix = self.noise_fb_matrix.to(device=self.FBconv.weight.device)
            self.noisy_fb = (self.noise_fb_matrix + 1) * self.FBconv.weight

        if self.bypass is not None:
            log.info("Add noise to Bypass")
            self.noise_bp_matrix = self.noise_bp_matrix.to(device=self.bypass.weight.device)
            self.noisy_bp = (self.noise_bp_matrix + 1) * self.bypass.weight

        self.tie_weights_impl()

class PCConvHardTanh(PCConv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.relu = nn.Hardtanh()

class PCConvHardTanhNoisy(PCConvNoisy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.relu = nn.Hardtanh()

class PCConvReLU6(PCConv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.relu = nn.ReLU6(inplace=True)

class PCConvReLU6Noisy(PCConvNoisy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.relu = nn.ReLU6(inplace=True)

class PCConvHardTanhLimit(PCConvHardTanh):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x, layer_idx=None):
        y = super().forward(x, layer_idx)
        return self.relu(y)

    def find_optimal_r(self, x, y, layer_idx=None):
        for _ in range(self.cls):
            if self.relu_between:
                log.info("USE {} as Non-linearity between FF/FB".format(self.relu))
                y = self.lr * self.FFconv(self.relu(x - self.FBconv(y))) + y
            else:
                log.info("DO NOT USE Non-linearity between FF/FB")
                y = self.lr * self.FFconv(x - self.FBconv(y)) + y
            y = self.relu(y)
        return y

class PCConvHardTanhLimitNoisy(PCConvHardTanhNoisy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x, layer_idx=None, w_type_used=None, use_relu=True):
        y = super().forward(x, layer_idx, w_type_used, use_relu)
        return self.relu(y)

    def find_optimal_r(self, x, y, layer_idx=None, w_type_used=None, use_relu=None):
        for _ in range(self.cls):
            if self.diff_noise:
                log.info("Set different noise at each cycle")
                self.noisy_fb = self._gen_noisy_weight(self.FBconv.weight)
                self.noisy_ff = self._gen_noisy_weight(self.FFconv.weight)
            log.info("noisy_fb, noisy_ff equals ideal weight: {}, {}".format(
                torch.allclose(self.noisy_fb, self.FBconv.weight.data),
                     torch.allclose(self.noisy_ff, self.FFconv.weight.data)))
            if self.relu_between:
                log.info("USE Non-linearity: {} between FF/FB".format(self.relu))
                assert self.noise_level == 0.0 or not torch.allclose(self.noisy_fb, self.FBconv.weight.data)
                error = self.relu(x - torch.conv_transpose2d(y, self.noisy_fb, padding=self.FBconv.padding))
            else:
                log.info("DO NOT USE Non-linearity between FF/FB")
                assert self.noise_level == 0.0 or not torch.allclose(self.noisy_fb, self.FBconv.weight.data)
                error = x - torch.conv_transpose2d(y, self.noisy_fb, padding=self.FBconv.padding)
            assert self.noise_level == 0.0 or not torch.allclose(self.noisy_ff, self.FFconv.weight.data)
            y += self.lr * torch.conv2d(error, self.noisy_ff, padding=self.FFconv.padding)
            y = self.relu(y)
        return y

class PCConvReLU6Limit(PCConvHardTanhLimit):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.relu = nn.ReLU6(inplace=True)

class PCConvReLU6LimitNoisy(PCConvHardTanhLimitNoisy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.relu = nn.ReLU6(inplace=True)



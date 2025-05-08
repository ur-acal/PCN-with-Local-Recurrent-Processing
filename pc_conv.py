'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import expand_weights_to_matrix
import matplotlib.pyplot as plt


class PCConv(nn.Module):
    def __init__(self, inp_chan, out_chan, kernel_size=3, stride=1, padding=1, cls=5, bias=False, lr=1e-2,
                 tie_weights=False, tie_bp=False, relu_between=True, bypass=True, layer_idx=None):
        super().__init__()
        self.FFconv = nn.Conv2d(inp_chan, out_chan, kernel_size, stride, padding, bias=bias)
        self.FBconv = nn.ConvTranspose2d(out_chan, inp_chan, kernel_size, stride, padding, bias=bias)
        self.b0 = nn.ParameterList([nn.Parameter(torch.zeros(1, out_chan, 1, 1))])
        self.relu = nn.ReLU(inplace=True)
        self.cls = cls
        self.lr = lr
        self.bypass = None
        self.relu_between = relu_between

        if tie_weights:
            self.FFconv.weight = self.FBconv.weight
            self.FFconv.bias = self.FBconv.bias
        if not tie_bp and bypass:
            self.bypass = nn.Conv2d(inp_chan, out_chan, kernel_size=1, stride=1, bias=False)
        elif tie_bp and bypass:
            self.bypass = self.FFconv

    def forward(self, x, layer_idx=None):
        y = self.relu(self.FFconv(x))
        for _ in range(self.cls):
            if self.relu_between:
                y = self.lr * self.FFconv(self.relu(x - self.FBconv(y))) + y
            else:
                y = self.lr * self.FFconv(x - self.FBconv(y)) + y
        if self.bypass is not None:
            y = y + self.bypass(x)
        return y


class PCConvNoisy(nn.Module):
    def __init__(self, inp_chan, out_chan, kernel_size=3, stride=1, padding=1, cls=5, bias=False, lr=1e-2,
                 tie_weights=False, tie_bp=False, relu_between=True, bypass=True, layer_idx=None,
                 noise_level=None, weight=None, plot_path=None, w_type="fb_flip",
                 noise_to_ff=True, noise_to_bp=True):
        super().__init__()
        print("Initializing PC layer {} with noise level: {}".format(layer_idx, noise_level))
        self.noise_level = noise_level
        self.padding = padding
        self.stride = stride
        self.kernel_size = kernel_size
        self.C_in = inp_chan
        self.C_out = out_chan

        self.FFconv = nn.Conv2d(inp_chan, out_chan, self.kernel_size, self.stride, self.padding, bias=bias)
        self.FBconv = nn.ConvTranspose2d(out_chan, inp_chan, self.kernel_size, self.stride, self.padding, bias=bias)
        self.b0 = nn.ParameterList([nn.Parameter(torch.zeros(1, out_chan, 1, 1))])
        self.bypass = None
        self.relu_between = relu_between

        self.tie_weights = tie_weights
        if tie_weights:
            self.FFconv.weight = self.FBconv.weight
            self.FFconv.bias = self.FBconv.bias
        self.tie_bp = tie_bp
        if not tie_bp and bypass:
            self.bypass = nn.Conv2d(inp_chan, out_chan, kernel_size=1, stride=1, bias=False)
        elif tie_bp and bypass:
            self.bypass = self.FFconv

        # noise related
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
        if isinstance(weight, str):
            self._load_expanded_weights(noise_level)
        elif weight is not None:
            pass

    def forward(self, x, layer_idx=None, w_type_used=None, use_relu=True):
        if self.noise_to_ff:
            y = self.relu(torch.conv2d(x, self.noisy_ff, padding=self.FFconv.padding))
        else:
            y = self.relu(self.FFconv(x))
        # injected noise inside find_optimal_r
        y = self.find_optimal_r(x, y, self.layer_idx, w_type_used, use_relu)
        if self.bypass is not None:
            if self.noise_to_bp:
                y = y + torch.conv2d(x, self.noisy_bp, padding=self.bypass.padding)
            else:
                y = y + self.bypass(x)
        return y

    def find_optimal_r(self, x, y, layer_idx=None, w_type_used=None, use_relu=None):
        # if weights are tied, must call add_noise or tie_weights_impl after loading the weights
        # of the model and before calling forward
        for _ in range(self.cls):
            if self.relu_between:
                error = self.relu(x - torch.conv_transpose2d(y, self.noisy_fb, padding=self.FBconv.padding))
            else:
                error = x - torch.conv_transpose2d(y, self.noisy_fb, padding=self.FBconv.padding)
            y += self.lr * torch.conv2d(error, self.noisy_ff, padding=self.FFconv.padding)
        return y

    def _init_noise(self, noise_level):
        # tie weights processed in self.add_noise()
        self.noise_ff_matrix = torch.randn_like(self.FFconv.weight) * (0.0 if noise_level is None else noise_level)
        self.noise_fb_matrix = torch.randn_like(self.FBconv.weight) * (0.0 if noise_level is None else noise_level)
        if self.bypass is not None:
            self.noise_bp_matrix = torch.randn_like(self.bypass.weight) * (0.0 if noise_level is None else noise_level)

    def tie_weights_impl(self):
        """
        Call this or add_noise after the weight is loaded
        """
        if self.tie_weights:
            self.noisy_ff = self.noisy_fb
            self.FFconv.weight = self.FBconv.weight
            self.FFconv.bias = self.FBconv.bias

        if self.tie_bp and self.bypass is not None:
            self.noisy_bp = self.noisy_fb
            self.bypass.weight = self.FFconv.weight
            self.bypass.bias = self.FFconv.bias

    def add_noise(self):
        """
        Call this or tie_weights_impl after the weight is loaded.
        :return: None
        """
        self.noise_ff_matrix = self.noise_ff_matrix.to(device=self.FFconv.weight.device)
        self.noise_fb_matrix = self.noise_fb_matrix.to(device=self.FBconv.weight.device)
        self.noise_bp_matrix = self.noise_bp_matrix.to(device=self.bypass.weight.device)

        self.noisy_ff = (self.noise_ff_matrix + 1) * self.FFconv.weight
        self.noisy_fb = (self.noise_fb_matrix + 1) * self.FBconv.weight
        self.noisy_bp = (self.noise_bp_matrix + 1) * self.bypass.weight

        self.tie_weights_impl()

    def _load_expanded_weights(self, noise_level):
        if self.w_type == "fb":
            expanded_weights = torch.load(
                os.path.join(self.weight, 'expanded_weights_layer_fb_{}.pt'.format(self.layer_idx + 1)),
                weights_only=True)
        elif self.w_type == "fb_flip":
            expanded_weights = torch.load(
                os.path.join(self.weight, 'expanded_weights_layer_fb_{}_flip.pt'.format(self.layer_idx + 1)),
                weights_only=True)
        elif self.w_type == "ff":
            expanded_weights = torch.load(
                os.path.join(self.weight, 'expanded_weights_layer_ff_{}.pt'.format(self.layer_idx + 1)),
                weights_only=True)
        elif self.w_type == "bp":
            expanded_weights = torch.load(
                os.path.join(self.weight, 'expanded_weights_layer_bp_{}.pt'.format(self.layer_idx + 1)),
                weights_only=True)
        else:
            expanded_weights = torch.load(
                os.path.join(self.weight, 'expanded_weights_layer_fb_{}.pt'.format(self.layer_idx + 1)),
                weights_only=True)

        if noise_level is not None:
            noise_ = torch.randn(expanded_weights.shape) * noise_level
            expanded_weights = expanded_weights * (1 + noise_)
        self.expanded_weights = {self.w_type: expanded_weights}

        # if self.solver == "LD":
        if self.w_type != "fb":
            self.expanded_weights.update({"fb": torch.load(
                os.path.join(self.weight, 'expanded_weights_layer_fb_{}.pt'.format(self.layer_idx + 1)),
                weights_only=True)})

    def plot_and_save(self, energy_list, plot_save_path, plot_title=None):
        plt.rcParams['font.family'] = 'Times New Roman'

        # Create the plot
        fig, ax = plt.subplots()
        ax.plot(range(len(energy_list)), energy_list)

        # Labeling
        ax.set_xlabel('Iteration', fontname='Times New Roman')
        ax.set_ylabel('PCN loss', fontname='Times New Roman')
        if plot_title:
            ax.set_title(plot_title, fontname='Times New Roman')

        # Improve layout and save
        plt.tight_layout()
        fig.savefig(plot_save_path, format='pdf')
        plt.close(fig)

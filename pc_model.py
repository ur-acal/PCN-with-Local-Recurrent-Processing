import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from pc_conv import PCConv, PCConvNoisy
from utils import expand_weights_to_matrix

import logging
log = logging.getLogger(__name__)


class PCNet(nn.Module):
    def __init__(self, inp_channels, out_channels, max_pool, num_classes=10, pc_conv_layer=PCConv, **kwargs):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.init_args = self._get_init_args(inp_channels, out_channels, max_pool, num_classes, pc_conv_layer, **kwargs)

        self.ics = inp_channels # input channels
        self.ocs = out_channels # output channels
        self.max_pool = max_pool # downsample flag
        self.num_layers = len(self.ics)

        # PC recurrent layers
        self.PcConvs = nn.ModuleList(
            [pc_conv_layer(inp_chan=self.ics[i], out_chan=self.ocs[i], layer_idx=i, **kwargs) for i in range(self.num_layers)])
        self.BNs = nn.ModuleList([nn.BatchNorm2d(self.ics[i]) for i in range(self.num_layers)])
        # Linear layer
        self.linear = nn.Linear(self.ocs[-1], num_classes)
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU(inplace=True)

        self.noise_level = kwargs.get("noise_level", 0.0)
        self.noise_level = 0.0 if self.noise_level is None else self.noise_level

    def forward(self, x):
        for i in range(self.num_layers):
            if i == 0:
                x_bn = self.BNs[i](x)
                log.info("Before the first PcConv layer, mean: {} std: {} before BNs[0]; mean: {} std: {} after BNs[0]".format(
                    x.mean(), x.std(), x_bn.mean(), x_bn.std()
                ))
            x = self.PcConvs[i](x, i)  # ReLU + Conv
            if self.max_pool[i]:
                x = self.max_pool2d(x)

        # classifier
        out = F.avg_pool2d(self.relu(x), x.size(-1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def save_expanded_weights(self, sample_imgs, save_to):
        x_ = sample_imgs.clone()
        for layer_idx, pc_conv in enumerate(self.PcConvs):
            y_ = pc_conv.relu(pc_conv.FFconv(x_))
            # fb weights
            if pc_conv.FBconv is not None:
                weights_ = pc_conv.FBconv.weight.data.cpu()
                expanded_weights_ = expand_weights_to_matrix(y_.shape[1:], weights_.permute(1, 0, 2, 3),
                                                             stride=pc_conv.stride,
                                                             padding=pc_conv.padding, flip_weight=False)
                torch.save(expanded_weights_,
                           os.path.join(save_to, 'expanded_weights_layer_fb_{}.pt'.format(layer_idx + 1)))
                expanded_weights_flip_ = expand_weights_to_matrix(y_.shape[1:], weights_.permute(1, 0, 2, 3).flip([2, 3]),
                                                                  stride=pc_conv.stride,
                                                                  padding=pc_conv.padding, flip_weight=False)
                torch.save(expanded_weights_flip_,
                           os.path.join(save_to, 'expanded_weights_layer_fb_{}_flip.pt'.format(layer_idx + 1)))

            # ff weights
            expanded_weights_ = expand_weights_to_matrix(x_.shape[1:], pc_conv.FFconv.weight.data.cpu(),
                                                         stride=pc_conv.stride,
                                                         padding=pc_conv.padding, flip_weight=False)
            torch.save(expanded_weights_,
                       os.path.join(save_to, 'expanded_weights_layer_ff_{}.pt'.format(layer_idx + 1)))

            # bypass weights
            if pc_conv.bypass is not None:
                expanded_weights_ = expand_weights_to_matrix(x_.shape[1:], pc_conv.bypass.weight.data.cpu(),
                                                             stride=pc_conv.stride,
                                                             padding=pc_conv.padding, flip_weight=False)
                torch.save(expanded_weights_,
                           os.path.join(save_to, 'expanded_weights_layer_bp_{}.pt'.format(layer_idx + 1)))
            if self.max_pool[layer_idx]:
                y_ = self.max_pool2d(y_)
            x_ = y_

    def _apply_noise(self, p):
        noise_ = torch.randn_like(p, device=self.device, requires_grad=False) * self.noise_level
        p.mul_(1 + noise_)

    def add_noise(self, noise_to_bn=False, noise_to_linear=False):
        for pc_conv in self.PcConvs:
            if hasattr(pc_conv, "init_ds_conv_block"):
                pc_conv.init_ds_conv_block()
            elif hasattr(pc_conv, "add_noise"):
                pc_conv.add_noise()
        # Todo: Add noise for BN and linear
        with torch.no_grad():
            for _name, _p in self.named_parameters():
                if "conv" in _name.lower() and "pc" not in _name.lower():
                    self._apply_noise(_p)
                if noise_to_bn and "bn" in _name.lower() and "pc" not in _name.lower():
                    log.info("Adding noise to batch norm")
                    self._apply_noise(_p)
                elif noise_to_linear and "linear" in _name.lower() and "pc" not in _name.lower():
                    log.info("Adding noise to linear layer")
                    self._apply_noise(_p)

            if noise_to_bn:
                # adding noise to running mean and variance of batch norm
                for _name, _buf in self.named_buffers():
                    if _name.endswith(('running_mean', 'running_var')):
                        log.info("Adding noise to running mean and variance")
                        self._apply_noise(_buf)

    @staticmethod
    def _get_init_args(inp_channels, out_channels, max_pool, num_classes, pc_conv_layer, **kwargs):
        init_args = {
            "model_args": {
                "inp_channels": inp_channels,
                "out_channels": out_channels,
                "max_pool": max_pool,
                "num_classes": num_classes,
                "pc_conv_layer": pc_conv_layer,
            },
            "kwargs": kwargs
        }
        return init_args


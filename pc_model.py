import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from pc_conv import PCConv, PCConvNoisy
from utils import expand_weights_to_matrix


class PCNet(nn.Module):
    def __init__(self, inp_channels, out_channels, max_pool, num_classes=10, pc_conv_layer=PCConv, **kwargs):
        super().__init__()
        self.init_args = self._get_init_args(inp_channels, out_channels, max_pool, num_classes, **kwargs)

        self.ics = inp_channels # input channels
        self.ocs = out_channels # output channels
        self.max_pool = max_pool # downsample flag
        self.num_layers = len(self.ics)

        # PC recurrent layers
        self.PcConvs = nn.ModuleList(
            [pc_conv_layer(inp_chan=self.ics[i], out_chan=self.ocs[i], **kwargs) for i in range(self.num_layers)])
        self.BNs = nn.ModuleList([nn.BatchNorm2d(self.ics[i]) for i in range(self.num_layers)])
        # Linear layer
        self.linear = nn.Linear(self.ocs[-1], num_classes)
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU(inplace=True)
        self.BNend = nn.BatchNorm2d(self.ocs[-1])


    def forward(self, x):
        for i in range(self.num_layers):
            x = self.BNs[i](x)
            x = self.PcConvs[i](x, i)  # ReLU + Conv
            if self.max_pool[i]:
                x = self.max_pool2d(x)

        # classifier
        out = F.avg_pool2d(self.relu(self.BNend(x)), x.size(-1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def save_expanded_weights(self, sample_imgs, save_to):
        x_ = sample_imgs.clone()
        for layer_idx, pc_conv in enumerate(self.PcConvs):
            y_ = pc_conv.relu(pc_conv.FFconv(x_))
            weights_ = pc_conv.FBconv.weight.data.cpu()
            # fb weights
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
            if pc_conv.BPconv is not None:
                expanded_weights_ = expand_weights_to_matrix(x_.shape[1:], pc_conv.BPconv.weight.data.cpu(),
                                                             stride=pc_conv.stride,
                                                             padding=pc_conv.padding, flip_weight=False)
                torch.save(expanded_weights_,
                           os.path.join(save_to, 'expanded_weights_layer_bp_{}.pt'.format(layer_idx + 1)))
            if self.max_pool[layer_idx]:
                y_ = self.max_pool2d(y_)
            x_ = y_

    @staticmethod
    def _get_init_args(inp_channels, out_channels, max_pool, num_classes, **kwargs):
        init_args = {
            "model_args": {
                "inp_channels": inp_channels,
                "out_channels": out_channels,
                "max_pool": max_pool,
                "num_classes": num_classes,
            },
            "kwargs": kwargs
        }
        return init_args


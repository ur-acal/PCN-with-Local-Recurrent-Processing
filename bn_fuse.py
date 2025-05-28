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
from pc_conv import PCConvNoisy

import logging
log = logging.getLogger(__name__)


class FirstLastFuseConv(nn.Module):
    def __init__(self,
                 conv,
                 bn_start:   nn.BatchNorm2d,
                 bn_end:     nn.BatchNorm2d,
                 total_calls: int):
        super().__init__()
        self.conv_orig        = conv
        self.conv_fused_start = fuse_single_conv_bn_pair(bn_start, conv)
        self.conv_fused_end   = fuse_single_conv_bn_pair(bn_end, conv)
        self.total_calls      = total_calls
        self._call_count      = 0

    def forward(self, x):
        self._call_count += 1
        if self._call_count == 1:
            return self.conv_fused_start(x)
        elif self._call_count == self.total_calls:
            return self.conv_fused_end(x)
        else:
            return self.conv_orig(x)


class IdentityConv(nn.Conv2d):
    def __init__(self, num_channels: int):
        super().__init__(
            in_channels  = num_channels,
            out_channels = num_channels,
            kernel_size  = 1,
            stride       = 1,
            padding      = 0,
            bias         = True
        )
        with torch.no_grad():
            self.weight.zero_()
            for i in range(num_channels):
                self.weight[i, i, 0, 0] = 1.0
            self.bias.zero_()


def fuse_single_conv_bn_pair(block1, block2, weight_only=False):
    if isinstance(block1, nn.BatchNorm2d) and isinstance(block2, nn.Conv2d):
        if not weight_only:
            # fuse both the weight and the bias
            return nn.utils.fuse_conv_bn_eval(block2, block1)
        # fuse the weight only
        bn_st_dict = block1.state_dict()
        conv_st_dict = block2.state_dict()

        eps = block1.eps
        var = bn_st_dict['running_var']
        gamma = bn_st_dict['weight']

        W = conv_st_dict['weight']

        denom = torch.sqrt(var + eps)
        A = gamma.div(denom)

        A = A.expand_as(W.transpose(0, -1)).transpose(0, -1)
        W.mul_(A)

        block2.weight.data.copy_(W)
        return block2
    else:
        return False


def fuse_bn_recursively(model: PCNet, total_last_calls: int) -> nn.Module:
    # 1) beginning‐BN fusion for every PcConv
    if hasattr(model, 'BNs') and hasattr(model, 'PcConvs'):
        for i, (bn_layer, pc_layer) in enumerate(zip(model.BNs, model.PcConvs)):
            fused = fuse_single_conv_bn_pair(bn_layer, pc_layer.FFconv)
            if fused:
                # non‐last layers: fuse only the first call with bn_layer
                model.PcConvs[i].FFconv = FirstLastFuseConv(
                    conv=pc_layer.FFconv, bn_start=bn_layer, bn_end=bn_layer, total_calls=1)
                model.BNs[i] = nn.Identity()

    # 2) final‐BN fusion into both FFconv and skip for last PcConv
    if hasattr(model, 'BNs') and hasattr(model, 'PcConvs') and hasattr(model, 'BNend'):
        last_idx = len(model.PcConvs) - 1
        pc_layer = model.PcConvs[last_idx]
        bn_start = model.BNs[last_idx]
        bn_end   = model.BNend

        # 2a) wrap FFconv so first uses bn_start, last uses bn_end
        pc_layer.FFconv = FirstLastFuseConv(
            conv=pc_layer.FFconv, bn_start=bn_start, bn_end=bn_end, total_calls=total_last_calls)

        # 2b) replace the skip‐through branch with an IdentityConv fused with bn_end
        num_ch    = bn_end.num_features
        idc       = IdentityConv(num_ch)
        fused_idc = fuse_single_conv_bn_pair(bn_end, idc)
        if fused_idc:
            # assume `pc_layer` has attribute `bypass` for skip; otherwise attach new
            pc_layer.bypass = fused_idc

        # remove the standalone BN modules
        model.BNs[last_idx] = nn.Identity()
        model.BNend          = nn.Identity()

    # 3) generic recursive fusion for any other Conv2d→BatchNorm2d pairs
    prev_name = None
    for name, child in list(model._modules.items()):
        if prev_name is None:
            prev_name = name

        fused = fuse_single_conv_bn_pair(
            model._modules[prev_name],
            model._modules[name]
        )
        if fused:
            model._modules[prev_name] = fused
            model._modules[name]      = nn.Identity()

        if len(child._modules) > 0:
            fuse_bn_recursively(child, total_last_calls)

        prev_name = name

    return model
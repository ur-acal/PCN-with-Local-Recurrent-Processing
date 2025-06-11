import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from pc_conv import PCConv, PCConvNoisy, PlainFFFBConv, PlainFFFBConvNoisy
from pc_conv import PlainFFFBConvRes, PlainFFFBConvResFixedX
from pc_conv import PlainFFFBConvResNoisy, PlainFFFBConvResFixedXNoisy
from ds_conv import PCConvDS
from utils import expand_weights_to_matrix

import logging
log = logging.getLogger(__name__)


class PCNet(nn.Module):
    def __init__(self, inp_channels, out_channels, max_pool, num_classes=10, pc_conv_layer=PCConv,
                 first_bn=True, dropout=0.0, **kwargs):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.init_args = self._get_init_args(
            inp_channels, out_channels, max_pool, num_classes, pc_conv_layer, first_bn, **kwargs)

        self.ics = inp_channels # input channels
        self.ocs = out_channels # output channels
        self.max_pool = max_pool # downsample flag
        self.num_layers = len(self.ics)
        self.dropout = dropout

        # PC recurrent layers
        self.PcConvs = nn.ModuleList(
            [pc_conv_layer(inp_chan=self.ics[i], out_chan=self.ocs[i], layer_idx=i, **kwargs) for i in range(self.num_layers)])
        self.BNs = nn.ModuleList([nn.BatchNorm2d(self.ics[i]) for i in range(self.num_layers)])
        if not first_bn:
            logging.warning("Drop the first BN layer")
            self.BNs[0] = nn.Identity()
        # Linear layer
        self.linear = nn.Linear(self.ocs[-1], num_classes)
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU(inplace=True)
        self.BNend = nn.BatchNorm2d(self.ocs[-1])

        self.noise_level = kwargs.get("noise_level", 0.0)
        self.noise_level = 0.0 if self.noise_level is None else self.noise_level

    def forward(self, x, clamp=False):
        for i in range(self.num_layers):
            x = self.BNs[i](x)
            x = self.PcConvs[i](x, i)  # ReLU + Conv
            if self.max_pool[i]:
                x = self.max_pool2d(x)
            if clamp:
                x = torch.clamp(x, -1, 1)

        # classifier
        if self.dropout > 0.0:
            log.info("Calling dropout with p = {} when training = {}".format(self.dropout, self.training))
            x = F.dropout(input=x, p=self.dropout, training=self.training)
        out = F.avg_pool2d(self.relu(self.BNend(x)), x.size(-1))
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
                    log.info("Adding noise to conv layer: {}".format(_name))
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
    def _get_init_args(inp_channels, out_channels, max_pool, num_classes, pc_conv_layer, first_bn, **kwargs):
        init_args = {
            "model_args": {
                "inp_channels": inp_channels,
                "out_channels": out_channels,
                "max_pool": max_pool,
                "num_classes": num_classes,
                "pc_conv_layer": pc_conv_layer,
                "first_bn": first_bn,
            },
            "kwargs": kwargs
        }
        return init_args

class PCNetNoBatchNorm(PCNet):
    """
    A No-BatchNorm version of PCNet.
    """
    def __init__(self, **kwargs):
        kwargs.update({"zero_init": False})
        super().__init__(**kwargs)
        self.BNs, self.BNend = None, None

    def get_max_hidden_val(self, x):
        max_abs = torch.tensor(0.0, device=self.device)
        for i in range(self.num_layers):
            x = self.PcConvs[i](x, i)  # ReLU + Conv
            if self.max_pool[i]:
                x = self.max_pool2d(x)
            cur_max = torch.max(torch.abs(x.max()), torch.abs(x.min()))
            max_abs = torch.max(max_abs, cur_max)
        return max_abs

    def forward(self, x, clamp=False):
        for i in range(self.num_layers):
            x = self.PcConvs[i](x, i)  # ReLU + Conv
            if self.max_pool[i]:
                x = self.max_pool2d(x)
            if clamp:
                x = torch.clamp(x, -1, 1)
            log.info("For intermediate x in layer: {}, Mean={}; Median={}; Min={}; Max={}".format(
                i, x.mean(), x.median(), x.min(), x.max()))

        # classifier
        if self.dropout > 0.0:
            log.info("Calling dropout with p = {} when training = {}".format(self.dropout, self.training))
            x = F.dropout(input=x, p=self.dropout, training=self.training)
        out = F.avg_pool2d(F.relu(x), x.size(-1)) # Here inplace ReLU can't be used. Will throw error.
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class PCNetWithMiddleConv(PCNet):
    def __init__(self, mid_kernel=3, **kwargs):
        super().__init__(**kwargs)
        self.mid_convs = nn.ModuleList([
            nn.Conv2d(self.ics[i], self.ics[i], kernel_size=mid_kernel,
                      stride=1, padding=(mid_kernel-1)//2, bias=False) for i in range(self.num_layers)
        ])
        self.mid_convs.append(
            nn.Conv2d(self.ocs[-1], self.ocs[-1], kernel_size=mid_kernel,
                      stride=1, padding=(mid_kernel-1)//2, bias=False))

    def forward(self, x, clamp=False):
        for i in range(self.num_layers):
            log.info("layer {} shape: {}".format(i, x.shape))
            x = self.mid_convs[i](x)
            x = self.BNs[i](x)
            x = self.PcConvs[i](x, i)  # ReLU + Conv
            if self.max_pool[i]:
                x = self.max_pool2d(x)
            if clamp:
                x = torch.clamp(x, -1, 1)

        # classifier
        if self.dropout > 0.0:
            log.info("Calling dropout with p = {} when training = {}".format(self.dropout, self.training))
            x = F.dropout(input=x, p=self.dropout, training=self.training)
        out = F.avg_pool2d(self.relu(self.BNend(self.mid_convs[-1](x))), x.size(-1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


PCN_CLASSES = {
    "PCNet": PCNet,
    "PCNetWithMiddleConv": PCNetWithMiddleConv,
    "PCNetNoBatchNorm": PCNetNoBatchNorm,
    None: PCNet,
}

PC_CONV_CLASS = {
    "PCConv": PCConv,
    "PlainFFFBConv": PlainFFFBConv,
    "PlainFFFBConvRes": PlainFFFBConvRes,
    "PlainFFFBConvResFixedX": PlainFFFBConvResFixedX,
    # noisy pc conv
    "PCConvNoisy": PCConvNoisy,
    "PlainFFFBConvNoisy": PlainFFFBConvNoisy,
    "PlainFFFBConvResNoisy": PlainFFFBConvResNoisy,
    "PlainFFFBConvResFixedXNoisy": PlainFFFBConvResFixedXNoisy,
    "PCConvDS": PCConvDS,
}
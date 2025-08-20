import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from collections.abc import Iterable
import numpy as np

from pc_conv import PCConv, PCConvNoisy, PlainFFFBConv, PlainFFFBConvNoisy
from pc_conv import PlainFFFBConvRes, PlainFFFBConvResFixedX
from pc_conv import PlainFFFBConvResNoisy, PlainFFFBConvResFixedXNoisy
from pc_conv import PCConvScaled, PCConvScaledNoisy, PCConvFFReLU6, PCConvFFReLU6Noisy
from pc_conv import PCConvSigmoid, PCConvSigmoidNoisy, PCConvReLU6, PCConvReLU6Noisy
from pc_conv import PCConvScaledReLU6, PCConvScaledReLU6Noisy, PCConvReLU6Sep
from pc_conv import PCConvHardTanh10, PCConvHardTanh10Noisy, PCConvReLU20, PCConvReLU20Noisy
from pc_conv import PCConvHardTanh, PCConvHardTanhNoisy, PCConvHardTanhDyn, PCConvHardTanhDynNoisy
from pc_conv import PCConvHardTanh2, PCConvHardTanh2Noisy, PCConvHardTanh2Dyn, PCConvHardTanh2DynNoisy
from pc_conv import PCConvHardTanhWSFF, PCConvHardTanhWSFFNoisy, PCConvHardTanhWSFFFB, PCConvHardTanhWSFFFBNoisy
from pc_conv import PCConvHardTanhLimit, PCConvHardTanhLimitNoisy, PCConvReLU6Limit, PCConvReLU6LimitNoisy
from ds_conv import PCConvDS
from utils import expand_weights_to_matrix

import logging
log = logging.getLogger(__name__)


class PCNet(nn.Module):
    def __init__(self, inp_channels, out_channels, max_pool, num_classes=10, pc_conv_layer=PCConv,
                 first_bn=True, dropout=0.0, separable=None, avg_pooling=False, stride=1, kernel_size=3, **kwargs):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.ics = inp_channels # input channels
        self.ocs = out_channels # output channels
        self.max_pool = max_pool # downsample flag
        self.stride = stride if isinstance(stride, Iterable) else [stride for _ in inp_channels]
        self.kernel_size = kernel_size if isinstance(kernel_size, Iterable) else [kernel_size for _ in inp_channels]
        self.sep = separable if isinstance(separable, Iterable) else [False for _ in inp_channels]
        self.num_layers = len(self.ics)
        self.dropout = dropout

        self.init_args = self._get_init_args(
            inp_channels, out_channels, max_pool, num_classes, pc_conv_layer, first_bn, avg_pooling,
            self.stride, self.kernel_size, **kwargs)

        # PC recurrent layers
        self.PcConvs = nn.ModuleList(
            [pc_conv_layer(inp_chan=self.ics[i], out_chan=self.ocs[i], stride=self.stride[i],
                           kernel_size=self.kernel_size[i], layer_idx=i, separable=self.sep[i], **kwargs)
             for i in range(self.num_layers)])
        self.BNs = nn.ModuleList([nn.BatchNorm2d(self.ics[i]) for i in range(self.num_layers)])
        if not first_bn:
            logging.warning("Drop the first BN layer")
            self.BNs[0] = nn.Identity()
        # Linear layer
        self.linear = nn.Linear(self.ocs[-1], num_classes)
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2) if not avg_pooling else nn.AvgPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU(inplace=True)
        self.BNend = nn.BatchNorm2d(self.ocs[-1])

        self.clean_params = {}
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
                    self.clean_params[_name] = _p.clone()
                    self._apply_noise(_p)
                if noise_to_bn and "bn" in _name.lower() and "pc" not in _name.lower():
                    log.info("Adding noise to batch norm")
                    self.clean_params[_name] = _p.clone()
                    self._apply_noise(_p)
                elif noise_to_linear and "linear" in _name.lower() and "pc" not in _name.lower():
                    log.info("Adding noise to linear layer")
                    self.clean_params[_name] = _p.clone()
                    self._apply_noise(_p)

            if noise_to_bn:
                # adding noise to running mean and variance of batch norm
                for _name, _buf in self.named_buffers():
                    if _name.endswith(('running_mean', 'running_var')):
                        log.info("Adding noise to running mean and variance")
                        self.clean_params[_name] = _p.clone()
                        self._apply_noise(_buf)
                    elif _name.endswith('conv_beta_init'):
                        log.info("Adding noise to conv beta init")
                        self.clean_params[_name] = _p.clone()
                        self._apply_noise(_buf)

    def recover_params(self):
        with torch.no_grad():
            for _pc_conv in self.PcConvs:
                if hasattr(_pc_conv, "recover_params"):
                    _pc_conv.recover_params()

            for _name, _p in self.named_parameters():
                if _name in self.clean_params:
                    _p.copy_(self.clean_params[_name])

            for _name, _buf in self.named_buffers():
                if _name in self.clean_params:
                    _buf.copy_(self.clean_params[_name])

    @staticmethod
    def _get_init_args(inp_channels, out_channels, max_pool, num_classes, pc_conv_layer, first_bn, avg_pooling=False,
                       stride=None, kernel_size=None, **kwargs):
        init_args = {
            "model_args": {
                "inp_channels": inp_channels,
                "out_channels": out_channels,
                "max_pool": max_pool,
                "num_classes": num_classes,
                "pc_conv_layer": pc_conv_layer,
                "first_bn": first_bn,
                "avg_pooling": avg_pooling,
                "stride": stride,
                "kernel_size": kernel_size,
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
            log.info("For intermediate x in layer: {}, Mean={}; Median={}; Min={}; Max={}; std={}".format(
                i, x.mean(), x.median(), x.min(), x.max(), x.std()))

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


class PCNetSeparable(PCNetNoBatchNorm):
    def __init__(self, patch_dim=4, **kwargs):
        self.inp_chan = kwargs["inp_channels"][0]
        self.chan = kwargs["out_channels"][0]
        self.patch_dim = patch_dim
        kwargs.update({
            "inp_channels": kwargs["inp_channels"][1:],
            "out_channels": kwargs["out_channels"][1:]
        })
        super().__init__(**kwargs)
        if patch_dim is not None:
            self.first_conv = nn.Conv2d(self.inp_chan, self.chan, kernel_size=patch_dim, stride=patch_dim)
        self.init_args = self._get_init_args(**kwargs)

    def forward(self, x, clamp=False):
        x = F.relu(self.first_conv(x))
        out = super().forward(x, clamp)
        return out

    def _get_init_args(self, inp_channels, out_channels, max_pool, num_classes, pc_conv_layer, first_bn, avg_pooling=None,
                       stride=None, kernel_size=None, **kwargs):
        init_args = super()._get_init_args(inp_channels, out_channels, max_pool, num_classes, pc_conv_layer, first_bn,
                                           avg_pooling, stride=self.stride, kernel_size=self.kernel_size, **kwargs)
        init_args["model_args"]["inp_channels"] = [self.inp_chan] + init_args["model_args"]["inp_channels"]
        init_args["model_args"]["out_channels"] = [self.chan] + init_args["model_args"]["out_channels"]
        init_args["model_args"]["patch_dim"] = self.patch_dim
        return init_args


class PCNetWith1stConv(PCNetSeparable):
    def __init__(self, first_ksz=5, first_stride=1, first_pad="valid", **kwargs):
        kwargs.update({"patch_dim": None})
        self.first_ksz = first_ksz
        self.first_stride = first_stride
        self.first_pad = first_pad
        super().__init__(**kwargs)
        self.first_conv = nn.Conv2d(
            self.inp_chan, self.chan, kernel_size=first_ksz, stride=first_stride, padding=first_pad)
        self.init_args["model_args"].update({
            "first_ksz": self.first_ksz, "first_stride": self.first_stride, "first_pad": self.first_pad})


class PCNetSepBN(PCNetSeparable):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.BN_start = nn.BatchNorm2d(self.chan)
        self.BNs = nn.ModuleList([nn.BatchNorm2d(self.ocs[i]) for i in range(self.num_layers)])

    def forward(self, x, clamp=False):
        x = F.relu(self.first_conv(x))
        for i in range(self.num_layers):
            x = self.PcConvs[i](x, i)  # ReLU + Conv
            if self.max_pool[i]:
                x = self.max_pool2d(x)
            log.info("For intermediate x in layer: {}, Mean={}; Median={}; Min={}; Max={}; std={}".format(
                i, x.mean(), x.median(), x.min(), x.max(), x.std()))
            x = self.BNs[i](x)

        # classifier
        if self.dropout > 0.0:
            x = F.dropout(input=x, p=self.dropout, training=self.training)
        out = F.avg_pool2d(F.relu(x), x.size(-1)) # Here inplace ReLU can't be used. Will throw error.
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class PCNetSepBNRes(PCNetSepBN):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x, clamp=False):
        x = F.relu(self.first_conv(x))
        for i in range(self.num_layers):
            inp = x.clone()
            x = self.PcConvs[i](x, i)  # ReLU + Conv
            if self.max_pool[i]:
                x = self.max_pool2d(x)
            log.info("For intermediate x in layer: {}, Mean={}; Median={}; Min={}; Max={}; std={}".format(
                i, x.mean(), x.median(), x.min(), x.max(), x.std()))
            x = self.BNs[i](x) + inp

        # classifier
        if self.dropout > 0.0:
            x = F.dropout(input=x, p=self.dropout, training=self.training)
        out = F.avg_pool2d(F.relu(x), x.size(-1)) # Here inplace ReLU can't be used. Will throw error.
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


PCN_CLASSES = {
    "PCNet": PCNet,
    "PCNetWithMiddleConv": PCNetWithMiddleConv,
    "PCNetNoBatchNorm": PCNetNoBatchNorm,
    "PCNetSeparable": PCNetSeparable,
    "PCNetSepBN": PCNetSepBN,
    "PCNetSepBNRes": PCNetSepBNRes,
    "PCNetWith1stConv": PCNetWith1stConv,
    None: PCNet,
}

PC_CONV_CLASS = {
    "PCConv": PCConv,
    "PCConvSigmoid": PCConvSigmoid,
    "PCConvHardTanh": PCConvHardTanh,
    "PCConvHardTanh2": PCConvHardTanh2,
    "PCConvHardTanh10": PCConvHardTanh10,
    "PCConvHardTanhWSFF": PCConvHardTanhWSFF,
    "PCConvHardTanhWSFFFB": PCConvHardTanhWSFFFB,
    "PCConvHardTanhDyn": PCConvHardTanhDyn,
    "PCConvHardTanh2Dyn": PCConvHardTanh2Dyn,
    "PCConvHardTanhLimit": PCConvHardTanhLimit,
    "PCConvReLU6": PCConvReLU6,
    "PCConvReLU6Sep": PCConvReLU6Sep,
    "PCConvReLU20": PCConvReLU20,
    "PCConvReLU6Limit": PCConvReLU6Limit,
    "PCConvScaled": PCConvScaled,
    "PCConvScaledReLU6": PCConvScaledReLU6,
    "PlainFFFBConv": PlainFFFBConv,
    "PlainFFFBConvRes": PlainFFFBConvRes,
    "PlainFFFBConvResFixedX": PlainFFFBConvResFixedX,
    "PCConvFFReLU6": PCConvFFReLU6,
    # noisy pc conv
    "PCConvNoisy": PCConvNoisy,
    "PCConvSigmoidNoisy": PCConvSigmoidNoisy,
    "PCConvHardTanhNoisy": PCConvHardTanhNoisy,
    "PCConvHardTanh2Noisy": PCConvHardTanh2Noisy,
    "PCConvHardTanh10Noisy": PCConvHardTanh10Noisy,
    "PCConvHardTanhWSFFNoisy": PCConvHardTanhWSFFNoisy,
    "PCConvHardTanhWSFFFBNoisy": PCConvHardTanhWSFFFBNoisy,
    "PCConvHardTanhDynNoisy": PCConvHardTanhDynNoisy,
    "PCConvHardTanh2DynNoisy": PCConvHardTanh2DynNoisy,
    "PCConvHardTanhLimitNoisy": PCConvHardTanhLimitNoisy,
    "PCConvReLU6Noisy": PCConvReLU6Noisy,
    "PCConvReLU20Noisy": PCConvReLU20Noisy,
    "PCConvReLU6LimitNoisy": PCConvReLU6LimitNoisy,
    "PCConvScaledNoisy": PCConvScaledNoisy,
    "PCConvScaledReLU6Noisy": PCConvScaledReLU6Noisy,
    "PlainFFFBConvNoisy": PlainFFFBConvNoisy,
    "PlainFFFBConvResNoisy": PlainFFFBConvResNoisy,
    "PlainFFFBConvResFixedXNoisy": PlainFFFBConvResFixedXNoisy,
    "PCConvFFReLU6Noisy": PCConvFFReLU6Noisy,
    "PCConvDS": PCConvDS,
}
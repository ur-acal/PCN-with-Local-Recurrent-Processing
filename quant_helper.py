import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from copy import deepcopy


class QuantImpl(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, s, q_min, q_max):
        x_q = x * s
        q_mask = (x_q >= q_min) & (x_q <= q_max)
        ctx.save_for_backward(q_mask, s)
        x_q = torch.clamp(x_q.round(), min=q_min, max=q_max)
        return x_q

    @staticmethod
    def backward(ctx, grad_output):
        q_mask, s = ctx.saved_tensors
        return s * grad_output * q_mask, None, None, None


class QuantHelper(nn.Module):
    def __init__(self, quant_type="per_tensor", sym_quant=False, n_bits=8, static=True, channel_dim=1, **kwargs):
        super().__init__()
        assert quant_type in {"per_tensor", "per_channel"}
        self.static = static
        self.quant_type = quant_type
        self.sym_quant = sym_quant
        self.n_bits = n_bits
        self.channel_dim = channel_dim
        self.q_min, self.q_max = self.calc_quant_range()
        self.eps_ = 1e-8

    def calc_quant_range(self):
        if self.sym_quant:
            q_max = 2 ** (self.n_bits - 1) - 1
            q_min = -q_max
        else:
            q_max = 2 ** self.n_bits - 1
            q_min = 0
        return q_min, q_max

    def get_act_max(self, x, reduced_dim=None):
        if reduced_dim is None:
            return x.abs().max()
        else:
            return x.abs().amax(dim=reduced_dim)

    @torch.no_grad()
    def _per_tensor_sym(self, x, return_s=False):
        if not hasattr(self, "s_q"):
            x_max = self.get_act_max(x)
            self.register_buffer("s_q", self.q_max / (x_max + self.eps_))
        elif not self.static:
            x_max = self.get_act_max(x)
            self.s_q = self.q_max / (x_max + self.eps_)
        if return_s:
            return self.s_q
        return torch.clamp((x * self.s_q).round(), min=self.q_min, max=self.q_max)

    def _per_tenser_non_sym(self, x):
        pass

    @torch.no_grad()
    def _per_channel_sym(self, x: torch.Tensor, return_s=False):
        reduced_dim = [d for d in range(x.ndim) if d != self.channel_dim]
        s_shape = [1] * x.ndim
        s_shape[self.channel_dim] = x.shape[self.channel_dim]

        if not hasattr(self, "s_q"):
            x_max = self.get_act_max(x, reduced_dim)
            self.register_buffer(
                "s_q", (self.q_max / (x_max + self.eps_)).view(*s_shape))
        elif not self.static:
            x_max = self.get_act_max(x, reduced_dim)
            self.s_q = (self.q_max / (x_max + self.eps_)).view(*s_shape)

        if return_s:
            return self.s_q
        return torch.clamp((x * self.s_q).round(), min=self.q_min, max=self.q_max)

    def _per_channel_non_sym(self, x):
        pass

    def forward(self, x):
        if self.quant_type == "per_tensor" and self.sym_quant:
            return self._per_tensor_sym(x)
        elif self.quant_type == "per_tensor" and not self.sym_quant:
            return self._per_tensor_sym(x)
        elif self.quant_type == "per_channel" and self.sym_quant:
            return self._per_channel_sym(x)
        elif self.quant_type == "per_channel" and not self.sym_quant:
            return self._per_channel_sym(x)


class PercQuantHelper(QuantHelper):
    def __init__(self, calib_perc=0.9999, **kwargs):
        self.calib_perc = calib_perc
        super().__init__(**kwargs)

    def get_act_max(self, x, reduced_dim=None):
        if reduced_dim is None:
            return x.abs().quantile(self.calib_perc)
        else:
            return x.abs().transpose(0, self.channel_dim).contiguous()\
                    .reshape(x.shape[self.channel_dim], -1).quantile(self.calib_perc, dim=1)


class QATHelper(QuantHelper):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x):
        if self.quant_type == "per_tensor" and self.sym_quant:
            s_q = self._per_tensor_sym(x, True)
        elif self.quant_type == "per_tensor" and not self.sym_quant:
            s_q = self._per_tensor_sym(x, True)
        elif self.quant_type == "per_channel" and self.sym_quant:
            s_q = self._per_channel_sym(x, True)
        elif self.quant_type == "per_channel" and not self.sym_quant:
            s_q = self._per_channel_sym(x, True)
        else:
            raise NotImplementedError
        return QuantImpl.apply(x, s_q, self.q_min, self.q_max)


class PercQATHelper(PercQuantHelper):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x):
        if self.quant_type == "per_tensor" and self.sym_quant:
            s_q = self._per_tensor_sym(x, True)
        elif self.quant_type == "per_tensor" and not self.sym_quant:
            s_q = self._per_tensor_sym(x, True)
        elif self.quant_type == "per_channel" and self.sym_quant:
            s_q = self._per_channel_sym(x, True)
        elif self.quant_type == "per_channel" and not self.sym_quant:
            s_q = self._per_channel_sym(x, True)
        else:
            raise NotImplementedError
        return QuantImpl.apply(x, s_q, self.q_min, self.q_max)


class EntropyQuantHelper(QuantHelper):
    """
    Entropy-based (KL) calibration:
    - Build a histogram of |x|
    - For a range of thresholds, simulate clipping + quantization
    - Pick the threshold that minimizes KL(P || Q)
    """
    def __init__(self, calib_bins: int = 2048, min_threshold_ratio: float = 0.6, **kwargs):
        self.calib_bins = calib_bins
        self.min_threshold_ratio = min_threshold_ratio
        super().__init__(**kwargs)
        # number of positive quantization bins; for symmetric we use q_max
        if self.sym_quant:
            self.n_quant_bins = int(self.q_max)
        else:
            self.n_quant_bins = int(self.q_max + 1)

    def _find_kl_clip_1d(self, x_1d: torch.Tensor) -> torch.Tensor:
        # x_1d: 1D tensor of activations
        x_abs = x_1d.abs()
        max_val = x_abs.max()
        if max_val <= self.eps_:
            return torch.tensor(0.0, device=x_1d.device, dtype=x_1d.dtype)

        # histogram on [0, max_val]
        hist = torch.histc(x_abs, bins=self.calib_bins, min=0.0, max=float(max_val))
        hist_sum = hist.sum()
        if hist_sum <= 0:
            return torch.tensor(float(max_val), device=x_1d.device, dtype=x_1d.dtype)

        pdf = hist / hist_sum
        n_bins = self.calib_bins
        n_quant = min(self.n_quant_bins, n_bins)

        # search over candidate thresholds (in histogram-bin space)
        start_bin = max(int(self.min_threshold_ratio * n_bins), n_quant)
        best_kl = None
        best_t = n_bins  # number of bins to keep

        for t in range(start_bin, n_bins + 1):
            # P: original distribution, clipped at bin t-1
            P = pdf.clone()
            if t < n_bins:
                tail = P[t:].sum()
                P[t - 1] = P[t - 1] + tail
            P = P[:t]

            # Q: distribution reconstructed after quantizing P into n_quant bins
            idx = torch.arange(t, device=x_1d.device)
            quant_idx = (idx * n_quant) // t  # map bins -> quant bins [0, n_quant-1]

            Q = torch.zeros_like(P)
            for qi in range(n_quant):
                mask = (quant_idx == qi)
                if mask.any():
                    mass = P[mask].sum()
                    # distribute mass uniformly over merged bins
                    Q[mask] = mass / mask.sum()

            # KL(P || Q)
            P_ = P + self.eps_
            Q_ = Q + self.eps_
            kl = (P_ * (P_.log() - Q_.log())).sum()

            if best_kl is None or kl < best_kl:
                best_kl = kl
                best_t = t

        # convert best_t (number of bins) back to a threshold value
        threshold = max_val * (best_t / n_bins)
        return threshold.to(device=x_1d.device, dtype=x_1d.dtype)

    def get_act_max(self, x: torch.Tensor, reduced_dim=None) -> torch.Tensor:
        # per-tensor case
        if reduced_dim is None:
            x_flat = x.reshape(-1)
            return self._find_kl_clip_1d(x_flat)

        # per-channel case: move channel_dim to front, then process each channel
        x_abs = x.abs().movedim(self.channel_dim, 0).contiguous()  # (C, ...)
        C = x_abs.shape[0]
        x_flat = x_abs.view(C, -1)

        thresholds = []
        for c in range(C):
            thresholds.append(self._find_kl_clip_1d(x_flat[c]))
        return torch.stack(thresholds, dim=0)


class QuantConv2d(nn.Module):
    def __init__(self, conv: nn.Conv2d, w_quant, act_quant,
                 w_quant_cls=QuantHelper, act_quant_cls=QuantHelper, adc_quant_cls=QuantHelper,
                 agg_bits=32, sigma_lsb=0.6, pvt_level=None, max_inp=None):
        super().__init__()
        self.conv = conv
        self.w_q_helper = w_quant_cls(**w_quant)
        self.act_q_helper = act_quant_cls(**act_quant)
        self.agg_bits = agg_bits
        self.sigma_lsb = sigma_lsb
        self.adc = adc_quant_cls(quant_type="per_tensor", sym_quant=True, n_bits=self.agg_bits,
                                 static=True, channel_dim=1)
        self.pvt_level = pvt_level
        self.max_inp = max_inp

    @torch.no_grad()
    def _apply_noise(self, p, p_type="weight"):
        # One time set; for mismatch aware training, don't use this method
        if self.pvt_level is not None and self.pvt_level > 0.0:
            err_type = "{}_err".format(p_type)
            if not hasattr(self, err_type):
                setattr(self, err_type, torch.randn_like(p, device=p.device, requires_grad=False) * p * self.pvt_level)
            return p + getattr(self, err_type)
        return p

    def _fwd_impl(self, x_q, w_q):
        x_q = F.conv2d(x_q, w_q, bias=None, stride=self.conv.stride, padding=self.conv.padding,
                       groups=self.conv.groups, dilation=self.conv.dilation)
        if hasattr(self.adc, "s_q") and self.sigma_lsb is not None and self.sigma_lsb > 0.0:
            x_q += torch.randn_like(x_q) * self.sigma_lsb / self.adc.s_q
        # if not, assume we do calibration with zero-noise; if calibrated before, we do quantization after adding noise
        x_q = self.adc(x_q) / self.adc.s_q

        q_scale = self.w_q_helper.s_q * self.act_q_helper.s_q
        if q_scale.ndim > 0:
            q_scale = q_scale.view(1, -1, 1, 1)
        x_q = x_q / q_scale
        return x_q

    def forward(self, x):
        x_q = self.act_q_helper(x)
        w_q = self.w_q_helper(self.conv.weight)
        w_q = self._apply_noise(w_q, "weight")

        inp_num = w_q.shape.numel() / w_q.shape[0]
        if self.max_inp is None or self.max_inp > inp_num:
            if self.conv.bias is None:
                return self._fwd_impl(x_q, w_q)
            return self._fwd_impl(x_q, w_q) + self._apply_noise(self.conv.bias.view(1, -1, 1, 1), "bias")
        else:
            C_out, C_in, k_h, k_w = w_q.shape
            max_c_in = max(1, int(self.max_inp) // (k_h * k_w))
            max_c_in = min(max_c_in, C_in)

            out = None
            for start in range(0, C_in, max_c_in):
                end = min(start + max_c_in, C_in)
                x_slice = x_q[:, start:end, :, :]
                w_slice = w_q[:, start:end, :, :]

                y_part = self._fwd_impl(x_slice, w_slice)
                out = y_part if out is None else out + y_part
            if self.conv.bias is None:
                return out
            return out + self._apply_noise(self.conv.bias.view(1, -1, 1, 1), "bias")


class QuantConvTranspose2d(nn.Module):
    def __init__(self, conv_transpose: nn.ConvTranspose2d, w_quant, act_quant,
                 w_quant_cls=QuantHelper, act_quant_cls=QuantHelper, adc_quant_cls=QuantHelper,
                 agg_bits=32, sigma_lsb=0.6, pvt_level=None, max_inp=None):
        super().__init__()
        self.conv = conv_transpose
        self.w_q_helper = w_quant_cls(**w_quant)
        self.act_q_helper = act_quant_cls(**act_quant)
        self.agg_bits = agg_bits
        self.sigma_lsb = sigma_lsb
        self.adc = adc_quant_cls(quant_type="per_tensor", sym_quant=True, n_bits=self.agg_bits,
                                 static=True, channel_dim=1)
        self.pvt_level = pvt_level
        self.max_inp = max_inp

    @torch.no_grad()
    def _apply_noise(self, p, p_type="weight"):
        # One time set; for mismatch aware training, don't use this method
        if self.pvt_level is not None and self.pvt_level > 0.0:
            err_type = "{}_err".format(p_type)
            if not hasattr(self, err_type):
                setattr(self, err_type, torch.randn_like(p, device=p.device, requires_grad=False) * p * self.pvt_level)
            return p + getattr(self, err_type)
        return p

    def _fwd_impl(self, x_q, w_q):
        x_q = F.conv_transpose2d(x_q, w_q, bias=None, stride=self.conv.stride,
                                 padding=self.conv.padding, output_padding=self.conv.output_padding,
                                 groups=self.conv.groups, dilation=self.conv.dilation)
        if hasattr(self.adc, "s_q") and self.sigma_lsb is not None and self.sigma_lsb > 0.0:
            x_q += torch.randn_like(x_q) * self.sigma_lsb / self.adc.s_q
        # if not, assume we do calibration with zero-noise; if calibrated before, we do quantization after adding noise
        x_q = self.adc(x_q) / self.adc.s_q

        q_scale = self.w_q_helper.s_q * self.act_q_helper.s_q
        if q_scale.ndim > 0:
            q_scale = q_scale.view(1, -1, 1, 1)
        x_q = x_q / q_scale
        return x_q

    def forward(self, x):
        x_q = self.act_q_helper(x)
        w_q = self.w_q_helper(self.conv.weight)
        w_q = self._apply_noise(w_q, "weight")

        inp_num = w_q.shape.numel() / w_q.shape[1]
        if self.max_inp is None or self.max_inp > inp_num:
            if self.conv.bias is None:
                return self._fwd_impl(x_q, w_q)
            return self._fwd_impl(x_q, w_q) + self._apply_noise(self.conv.bias.view(1, -1, 1, 1), "bias")
        else:
            C_in, C_out_per_g, k_h, k_w = w_q.shape
            max_c_in = max(1, int(self.max_inp) // (k_h * k_w))
            max_c_in = min(max_c_in, C_in)

            out = None
            for start in range(0, C_in, max_c_in):
                end = min(start + max_c_in, C_in)
                x_slice = x_q[:, start:end, :, :]
                w_slice = w_q[start:end, :, :, :]

                y_part = self._fwd_impl(x_slice, w_slice)
                out = y_part if out is None else out + y_part
            if self.conv.bias is None:
                return out
            return out + self._apply_noise(self.conv.bias.view(1, -1, 1, 1), "bias")


class QuantLinear(nn.Module):
    def __init__(self, linear: nn.Linear, w_quant, act_quant,
                 w_quant_cls=QuantHelper, act_quant_cls=QuantHelper, adc_quant_cls=QuantHelper,
                 agg_bits=32, sigma_lsb=0.6, pvt_level=None, max_inp=None):
        super().__init__()
        self.linear = linear
        self.w_q_helper = w_quant_cls(**w_quant)
        self.act_q_helper = act_quant_cls(**act_quant)
        self.agg_bits = agg_bits
        self.sigma_lsb = sigma_lsb
        self.adc = adc_quant_cls(quant_type="per_tensor", sym_quant=True, n_bits=self.agg_bits,
                                 static=True, channel_dim=1)
        self.pvt_level = pvt_level
        self.max_inp = max_inp

    @torch.no_grad()
    def _apply_noise(self, p, p_type="weight"):
        # One time set; for mismatch aware training, don't use this method
        if self.pvt_level is not None and self.pvt_level > 0.0:
            err_type = "{}_err".format(p_type)
            if not hasattr(self, err_type):
                setattr(self, err_type, torch.randn_like(p, device=p.device, requires_grad=False) * p * self.pvt_level)
            return p + getattr(self, err_type)
        return p

    def _fwd_impl(self, x_q, w_q):
        x_q = F.linear(x_q, w_q, bias=None)
        if hasattr(self.adc, "s_q") and self.sigma_lsb is not None and self.sigma_lsb > 0.0:
            x_q += torch.randn_like(x_q) * self.sigma_lsb / self.adc.s_q
        # if not, assume we do calibration with zero-noise; if calibrated before, we do quantization after adding noise
        x_q = self.adc(x_q) / self.adc.s_q

        q_scale = self.w_q_helper.s_q * self.act_q_helper.s_q
        if q_scale.ndim > 0:
            q_scale = q_scale.view(1, -1)
        return x_q / q_scale

    def forward(self, x):
        x_q = self.act_q_helper(x)
        w_q = self.w_q_helper(self.linear.weight)
        w_q = self._apply_noise(w_q, "weight")

        inp_num = w_q.shape[-1]
        if self.max_inp is None or self.max_inp > inp_num:
            if self.linear.bias is None:
                return self._fwd_impl(x_q, w_q)
            return self._fwd_impl(x_q, w_q) + self._apply_noise(self.linear.bias, "bias")
        else:
            in_features = w_q.shape[1]
            out = None
            for start in range(0, in_features, int(self.max_inp)):
                end = min(start + int(self.max_inp), in_features)
                x_slice = x_q[..., start:end]
                w_slice = w_q[:, start:end]

                y_part = self._fwd_impl(x_slice, w_slice)
                out = y_part if out is None else out + y_part
            if self.linear.bias is None:
                return out
            return out + self._apply_noise(self.linear.bias, "bias")


def replace_with_quant_layers(model: nn.Module, w_conv_quant, act_conv_quant,
                              w_conv_trans_quant, act_conv_trans_quant,
                              w_linear_quant, act_linear_quant,
                              w_quant_cls=QuantHelper, act_quant_cls=QuantHelper, adc_quant_cls=QuantHelper,
                              agg_bits=32, sigma_lsb=0.6, pvt_level=None, max_inp=None):
    # Todo: How do we set this correctly for possibly first stem conv?
    for _name, _child in model.named_children():
        if isinstance(_child, (QuantConv2d, QuantConvTranspose2d, QuantLinear)):
            continue

        if isinstance(_child, nn.Conv2d):
            setattr(model, _name, QuantConv2d(_child, w_quant=w_conv_quant, act_quant=act_conv_quant,
                                              w_quant_cls=w_quant_cls, act_quant_cls=act_quant_cls,
                                              adc_quant_cls=adc_quant_cls, agg_bits=agg_bits,
                                              sigma_lsb=sigma_lsb, pvt_level=pvt_level, max_inp=max_inp))

        elif isinstance(_child, nn.ConvTranspose2d):
            setattr(model, _name, QuantConvTranspose2d(_child, w_quant=w_conv_trans_quant, act_quant=act_conv_trans_quant,
                                              w_quant_cls=w_quant_cls, act_quant_cls=act_quant_cls,
                                              adc_quant_cls=adc_quant_cls, agg_bits=agg_bits,
                                              sigma_lsb=sigma_lsb, pvt_level=pvt_level, max_inp=max_inp))

        elif isinstance(_child, nn.Linear):
            setattr(model, _name, QuantLinear(_child, w_quant=w_linear_quant, act_quant=act_linear_quant,
                                              w_quant_cls=w_quant_cls, act_quant_cls=act_quant_cls,
                                              adc_quant_cls=adc_quant_cls, agg_bits=agg_bits,
                                              sigma_lsb=sigma_lsb, pvt_level=pvt_level, max_inp=max_inp))

        replace_with_quant_layers(_child, w_conv_quant, act_conv_quant,
                                  w_conv_trans_quant, act_conv_trans_quant,
                                  w_linear_quant, act_linear_quant,
                                  w_quant_cls, act_quant_cls, adc_quant_cls,
                                  agg_bits, sigma_lsb, pvt_level, max_inp)


QUANT_HELPER_CLS = {
    # Quantization helper
    "QuantHelper": QuantHelper,
    "PercQuantHelper": PercQuantHelper,
    "EntropyQuantHelper": EntropyQuantHelper,
    # QAT related
    "QATHelper": QATHelper,
    "PercQATHelper": PercQATHelper,
}

QUANT_SCHEME_PC = {
    "default": {
        # y0 = init with x; y += lr * FFConv(ReLU(FBConv(y)))
        "w_conv": {"sym_quant": True, "channel_dim": 0}, "act_conv": {"sym_quant": False, "channel_dim": 0},
        "w_conv_trans": {"sym_quant": True, "channel_dim": 1}, "act_conv_trans": {"sym_quant": True, "channel_dim": 1},
        # By default, there is a relu before linear
        "w_linear": {"sym_quant": True, "channel_dim": 0}, "act_linear": {"sym_quant": False, "channel_dim": 0}
    }
}
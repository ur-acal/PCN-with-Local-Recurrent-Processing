import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from copy import deepcopy
from types import MethodType

from TorchDiffEqPack.odesolver import odesolve as aca_ode_solve
from ode_pc import ODEXInitFFFB


def _clone_conv2d_like(conv: nn.Conv2d) -> nn.Conv2d:
    new_conv = nn.Conv2d(
        in_channels=conv.in_channels,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=(conv.bias is not None),
        padding_mode=conv.padding_mode,
        device=conv.weight.device,
        dtype=conv.weight.dtype,
    )
    new_conv.load_state_dict(deepcopy(conv.state_dict()))
    return new_conv


def _make_single_pixel_mask(h, w, active_h, active_w, device, dtype):
    mask = torch.zeros((1, 1, h, w), device=device, dtype=dtype)
    mask[0, 0, active_h, active_w] = 1.0
    return mask


class ODEXInitFFFBPixelSwitch(ODEXInitFFFB):
    """
    Pixel-block switched dynamics for dy/dt = W_FF f(W_FB y).

    Within one cycle T_s, visit all spatial pixels once.
    At each time interval, only one output pixel block is updated.

    Need to set conv_only=True when loading the model.

    Current implementation assumptions:
    stride == 1
    odd kernel size
    inner/outer operators are Conv2d
    only inference / validation-side usage is intended
    """

    def __init__(self, switch_period=None, **kwargs):
        super().__init__(**kwargs)

        assert isinstance(self.FFconv, nn.Conv2d), \
            "ODEXInitFFFBPixelSwitch currently expects FFconv to be nn.Conv2d."
        assert isinstance(self.FBconv, nn.Conv2d), \
            "ODEXInitFFFBPixelSwitch currently expects FBconv to be nn.Conv2d."

        ff_stride = self.FFconv.stride[0] if isinstance(self.FFconv.stride, tuple) else self.FFconv.stride
        fb_stride = self.FBconv.stride[0] if isinstance(self.FBconv.stride, tuple) else self.FBconv.stride
        assert ff_stride == 1 and fb_stride == 1, \
            "ODEXInitFFFBPixelSwitch currently assumes stride=1."

        ff_kh, ff_kw = self.FFconv.kernel_size if isinstance(self.FFconv.kernel_size, tuple) else (
            self.FFconv.kernel_size, self.FFconv.kernel_size
        )
        fb_kh, fb_kw = self.FBconv.kernel_size if isinstance(self.FBconv.kernel_size, tuple) else (
            self.FBconv.kernel_size, self.FBconv.kernel_size
        )
        assert ff_kh % 2 == 1 and ff_kw % 2 == 1 and fb_kh % 2 == 1 and fb_kw % 2 == 1, \
            "ODEXInitFFFBPixelSwitch currently assumes odd kernel sizes."

        self.ff_kh, self.ff_kw = ff_kh, ff_kw
        self.fb_kh, self.fb_kw = fb_kh, fb_kw
        self.ff_pad_h = self.FFconv.padding[0] if isinstance(self.FFconv.padding, tuple) else self.FFconv.padding
        self.ff_pad_w = self.FFconv.padding[1] if isinstance(self.FFconv.padding, tuple) else self.FFconv.padding
        self.fb_pad_h = self.FBconv.padding[0] if isinstance(self.FBconv.padding, tuple) else self.FBconv.padding
        self.fb_pad_w = self.FBconv.padding[1] if isinstance(self.FBconv.padding, tuple) else self.FBconv.padding

        self.switch_period = switch_period

        # Nine physical compact W_FB copies, each gets its own fixed mismatch later.
        self.FBconv_copies = nn.ModuleList(
            [_clone_conv2d_like(self.FBconv) for _ in range(self.ff_kh * self.ff_kw)]
        )
        self._sync_fb_copies_from_base()

    def _sync_fb_copies_from_base(self):
        for conv in self.FBconv_copies:
            conv.weight.data.copy_(self.FBconv.weight.detach())
            if conv.bias is not None and self.FBconv.bias is not None:
                conv.bias.data.copy_(self.FBconv.bias.detach())

    def _get_switch_period(self, t):
        if self.switch_period is not None:
            return float(self.switch_period)

        if hasattr(self, "option_aca"):
            t0 = self.option_aca["t0"]
            t1 = self.option_aca["t1"]
            t0 = float(t0.item()) if torch.is_tensor(t0) else float(t0)
            t1 = float(t1.item()) if torch.is_tensor(t1) else float(t1)
            return t1 - t0

        return float(self.integration_time[-1].item() - self.integration_time[0].item())

    def _time_to_active_pixel(self, t, h, w):
        n_pix = h * w
        T_s = self._get_switch_period(t)
        if T_s <= 0:
            return 0, 0

        t_val = float(t.item()) if torch.is_tensor(t) else float(t)

        if hasattr(self, "option_aca"):
            t0 = self.option_aca["t0"]
            t0 = float(t0.item()) if torch.is_tensor(t0) else float(t0)
        else:
            t0 = float(self.integration_time[0].item())

        tau = (t_val - t0) % T_s
        delta = T_s / n_pix
        pix_idx = int(tau / delta)
        pix_idx = min(max(pix_idx, 0), n_pix - 1)

        active_h = pix_idx // w
        active_w = pix_idx % w
        return active_h, active_w

    def _extract_site_patch(self, y_pad, center_h, center_w):
        """
        Extract one local FB receptive-field patch centered at (center_h, center_w)
        in the original unpadded coordinates.
        Returns shape: [B, Cin, fb_kh, fb_kw]
        """
        h0 = center_h
        w0 = center_w
        return y_pad[:, :, h0:h0 + self.fb_kh, w0:w0 + self.fb_kw]

    def _eval_fb_site(self, conv, patch):
        """
        conv: one FB conv copy
        patch: [B, Cin, fb_kh, fb_kw]
        return: [B, Cout]
        """
        bsz = patch.shape[0]
        patch_flat = patch.reshape(bsz, -1)
        weight_flat = conv.weight.reshape(conv.out_channels, -1)
        out = torch.einsum("bi,oi->bo", patch_flat, weight_flat)
        if conv.bias is not None:
            out = out + conv.bias.view(1, -1)
        return out

    def _eval_ff_center(self, z_local):
        """
        z_local: [B, Cmid, ff_kh, ff_kw]
        return: [B, Cout]
        """
        bsz = z_local.shape[0]
        z_flat = z_local.reshape(bsz, -1)
        weight_flat = self.FFconv.weight.reshape(self.FFconv.out_channels, -1)
        out = torch.einsum("bi,oi->bo", z_flat, weight_flat)
        if self.FFconv.bias is not None:
            out = out + self.FFconv.bias.view(1, -1)
        return out

    def _extract_all_site_patches(self, y_pad, active_h, active_w, h, w):
        """
        Returns:
            patches: [B, S, Cin, fb_kh, fb_kw]
            valid_mask: [S]
        where S = ff_kh * ff_kw.
        The site order matches the original nested loop over (dh, dw).
        """
        bsz, cin, _, _ = y_pad.shape
        site_patches = []
        valid_list = []

        for dh in range(-(self.ff_kh // 2), self.ff_kh // 2 + 1):
            for dw in range(-(self.ff_kw // 2), self.ff_kw // 2 + 1):
                inter_h = active_h + dh
                inter_w = active_w + dw

                valid = (0 <= inter_h < h) and (0 <= inter_w < w)
                valid_list.append(valid)

                if valid:
                    patch = self._extract_site_patch(y_pad, inter_h, inter_w)
                else:
                    patch = torch.zeros(
                        (bsz, cin, self.fb_kh, self.fb_kw),
                        device=y_pad.device,
                        dtype=y_pad.dtype,
                    )

                site_patches.append(patch)

        patches = torch.stack(site_patches, dim=1)  # [B, S, Cin, fb_kh, fb_kw]
        valid_mask = torch.tensor(valid_list, device=y_pad.device, dtype=y_pad.dtype)  # [S]
        return patches, valid_mask

    def _eval_fb_sites_batched(self, patches, valid_mask):
        """
        patches: [B, S, Cin, fb_kh, fb_kw]
        valid_mask: [S]
        return:
            z_local: [B, Cout, ff_kh, ff_kw]
        """
        bsz, n_sites, _, _, _ = patches.shape

        patches_flat = patches.reshape(bsz, n_sites, -1)  # [B, S, I]

        weight_flat = torch.stack(
            [conv.weight.reshape(conv.out_channels, -1) for conv in self.FBconv_copies],
            dim=0
        )  # [S, O, I]

        z_sites = torch.einsum("bsi,soi->bso", patches_flat, weight_flat)  # [B, S, O]

        if self.FBconv_copies[0].bias is not None:
            bias = torch.stack([conv.bias for conv in self.FBconv_copies], dim=0)  # [S, O]
            z_sites = z_sites + bias.unsqueeze(0)  # [B, S, O]

        z_sites = z_sites * valid_mask.view(1, n_sites, 1)

        z_local = z_sites.permute(0, 2, 1).reshape(
            bsz, self.FBconv_copies[0].out_channels, self.ff_kh, self.ff_kw
        )
        return z_local

    def _make_ode_fn(self, x, noisy_cu=None):
        def ode_func(t, y):
            bsz, _, h, w = y.shape
            active_h, active_w = self._time_to_active_pixel(t, h, w)
            n_pix = h * w

            y_pad = F.pad(y, (self.fb_pad_w, self.fb_pad_w, self.fb_pad_h, self.fb_pad_h))

            patches, valid_mask = self._extract_all_site_patches(y_pad, active_h, active_w, h, w)
            z_local = self._eval_fb_sites_batched(patches, valid_mask)

            z_local = self.act_fn(z_local)
            out_site = self._eval_ff_center(z_local)

            out = torch.zeros(
                (bsz, self.FFconv.out_channels, h, w),
                device=y.device,
                dtype=y.dtype,
            )
            out[:, :, active_h, active_w] = out_site
            return n_pix * out

        return ode_func

    def add_noise(self):
        # If this gets called early from parent __init__ before FBconv_copies exists,
        # fall back to original behavior.
        if not hasattr(self, "FBconv_copies"):
            self._apply_noise(self.FFconv.weight)
            if not self.tie_weights:
                self._apply_noise(self.FBconv.weight)
            if not self.tie_bp and self.bypass is not None:
                self._apply_noise(self.bypass.weight)
            if not torch.allclose(self.b0[0], torch.zeros_like(self.b0[0])):
                self._apply_noise(self.b0[0])
            return

        # Keep original FBconv noised as well so the existing named-parameter checks still pass.
        self._apply_noise(self.FFconv.weight)
        if not self.tie_weights:
            self._apply_noise(self.FBconv.weight)

            self._sync_fb_copies_from_base()
            for _fb_conv in self.FBconv_copies:
                self._apply_noise(_fb_conv.weight)

        if not self.tie_bp and self.bypass is not None:
            self._apply_noise(self.bypass.weight)

        if not torch.allclose(self.b0[0], torch.zeros_like(self.b0[0])):
            self._apply_noise(self.b0[0])

    @torch.no_grad()
    def recover_params(self):
        super().recover_params()
        if hasattr(self, "FBconv_copies"):
            self._sync_fb_copies_from_base()


class ODEXInitFFFBPixelSwitchExplicit(ODEXInitFFFBPixelSwitch):
    """
    Explicit mini-solver version of ODEXInitFFFBPixelSwitch.

    Instead of handing one globally time-switched RHS to a single ODE solver,
    this class explicitly scans over pixel intervals and runs aca_ode_solve
    once per mini-interval.
    """

    @staticmethod
    def _pixel_idx_to_active_hw(pix_idx, w):
        active_h = pix_idx // w
        active_w = pix_idx % w
        return active_h, active_w

    def _build_interval_option_aca(self, t0, t1, ref_tensor):
        option_aca = dict(self.option_aca)
        option_aca["t0"] = torch.as_tensor(t0, device=ref_tensor.device, dtype=ref_tensor.dtype)
        option_aca["t1"] = torch.as_tensor(t1, device=ref_tensor.device, dtype=ref_tensor.dtype)
        option_aca["t_eval"] = [option_aca["t0"], option_aca["t1"]]
        if option_aca["h"] is not None:
            option_aca["h"] = min(float(option_aca["h"]), float(t1 - t0))
        return option_aca

    def _run_explicit_pixel_switch(self, x, full_traj=False):
        y = self.init_y(x)
        self.integration_time = self.integration_time.type_as(x)

        t0 = self.option_aca["t0"]
        t1 = self.option_aca["t1"]
        t0 = float(t0.item()) if torch.is_tensor(t0) else float(t0)
        t1 = float(t1.item()) if torch.is_tensor(t1) else float(t1)

        _, _, h, w = y.shape
        n_pix = h * w
        T_s = self._get_switch_period(None)
        delta = T_s / n_pix

        t_cur = t0
        pix_idx = 0
        step_states = [y]
        step_times = [torch.as_tensor(t_cur, device=y.device, dtype=y.dtype)]

        while t_cur < t1 - 1e-15:
            t_next = min(t_cur + delta, t1)

            active_h, active_w = self._pixel_idx_to_active_hw(pix_idx % n_pix, w)
            self._pixel_switch_active_h = active_h
            self._pixel_switch_active_w = active_w

            option_aca = self._build_interval_option_aca(t_cur, t_next, y)
            out = aca_ode_solve(self._make_ode_fn(x), y, option_aca)
            y = out[-1]

            t_cur = t_next
            pix_idx += 1

            if full_traj:
                step_states.append(y)
                step_times.append(torch.as_tensor(t_cur, device=y.device, dtype=y.dtype))

        if full_traj:
            traj = torch.stack(step_states, dim=0)
            steps = torch.stack(step_times, dim=0)
            return traj, steps

        return y

    def forward(self, x, layer_idx=None):
        out = self._run_explicit_pixel_switch(x, full_traj=False)

        if self.bypass is not None:
            out = self.bypass(out) + out
        return out

    def forward_full_steps(self, x, layer_idx=None):
        traj, steps = self._run_explicit_pixel_switch(x, full_traj=True)

        if self.bypass is not None:
            traj = self.bypass(traj) + traj
        return traj, steps


SWITCH_CLASSES = {
    "ODEXInitFFFBPixelSwitch": ODEXInitFFFBPixelSwitch,
    "ODEXInitFFFBPixelSwitchExplicit": ODEXInitFFFBPixelSwitchExplicit,
}
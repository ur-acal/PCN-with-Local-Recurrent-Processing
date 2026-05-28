import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import jvp, vmap
import numpy as np

from copy import deepcopy
from types import MethodType

from TorchDiffEqPack.odesolver import odesolve as aca_ode_solve
from ode_pc import ODEXInitFFFB
from pc_conv import ReLUX


def _clone_conv_like(conv):
    if isinstance(conv, nn.Conv2d):
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
    elif isinstance(conv, nn.ConvTranspose2d):
        new_conv = nn.ConvTranspose2d(
            in_channels=conv.in_channels,
            out_channels=conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            output_padding=conv.output_padding,
            groups=conv.groups,
            bias=(conv.bias is not None),
            dilation=conv.dilation,
            padding_mode=conv.padding_mode,
            device=conv.weight.device,
            dtype=conv.weight.dtype,
        )
    else:
        raise TypeError(f"Unsupported conv type: {type(conv)}")

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
        assert isinstance(self.FBconv, (nn.Conv2d, nn.ConvTranspose2d)), \
            "ODEXInitFFFBPixelSwitch currently expects FBconv to be nn.Conv2d or nn.ConvTranspose2d."

        ff_stride = self.FFconv.stride[0] if isinstance(self.FFconv.stride, tuple) else self.FFconv.stride
        fb_stride = self.FBconv.stride[0] if isinstance(self.FBconv.stride, tuple) else self.FBconv.stride
        assert ff_stride == 1 and fb_stride == 1, \
            "ODEXInitFFFBPixelSwitch currently assumes stride=1."

        if isinstance(self.FBconv, nn.ConvTranspose2d):
            fb_dilation = self.FBconv.dilation[0] if isinstance(self.FBconv.dilation, tuple) else self.FBconv.dilation
            fb_groups = self.FBconv.groups
            fb_outpad = self.FBconv.output_padding[0] if isinstance(self.FBconv.output_padding,
                                                                    tuple) else self.FBconv.output_padding

            assert fb_stride == 1, \
                "ODEXInitFFFBPixelSwitch with ConvTranspose2d currently assumes stride=1."
            assert fb_dilation == 1, \
                "ODEXInitFFFBPixelSwitch with ConvTranspose2d currently assumes dilation=1."
            assert fb_groups == 1, \
                "ODEXInitFFFBPixelSwitch with ConvTranspose2d currently assumes groups=1."
            assert fb_outpad == 0, \
                "ODEXInitFFFBPixelSwitch with ConvTranspose2d currently assumes output_padding=0."

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
        self.FBconv_copies = [_clone_conv_like(self.FBconv) for _ in range(self.ff_kh * self.ff_kw)]
        self._sync_fb_copies_from_base()

    def _fb_weight_to_site_matrix(self, conv):
        """
        Return a flattened weight matrix of shape [Cout, Cin * kh * kw]
        so that one local site can be evaluated as:

            out[b, o] = sum_i patch_flat[b, i] * weight_flat[o, i] + bias[o]

        For Conv2d:
            weight is already [Cout, Cin, kh, kw]

        For ConvTranspose2d with stride=1, dilation=1, groups=1, output_padding=0:
            local site evaluation uses the spatially flipped kernel and
            channel order [Cout, Cin, kh, kw], obtained from the stored
            weight [Cin, Cout, kh, kw].
        """
        if isinstance(conv, nn.Conv2d):
            weight_site = conv.weight
        elif isinstance(conv, nn.ConvTranspose2d):
            weight_site = torch.flip(conv.weight, dims=[2, 3]).permute(1, 0, 2, 3).contiguous()
        else:
            raise TypeError(f"Unsupported FB conv type: {type(conv)}")

        return weight_site.reshape(weight_site.shape[0], -1)

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
        conv: one FB conv copy (Conv2d or ConvTranspose2d)
        patch: [B, Cin, fb_kh, fb_kw]
        return: [B, Cout]
        """
        bsz = patch.shape[0]
        patch_flat = patch.reshape(bsz, -1)
        weight_flat = self._fb_weight_to_site_matrix(conv)  # [Cout, I]
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
            [self._fb_weight_to_site_matrix(conv) for conv in self.FBconv_copies],
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
    def __init__(self, n_iters=1, **kwargs):
        super().__init__(**kwargs)
        assert n_iters is not None and int(n_iters) >= 1, \
            "n_iters must be an integer >= 1."
        self.n_iters = int(n_iters)

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

    def _time_to_active_pixel(self, t, h, w):
        n_pix = h * w
        t0, t1, T_total = self._get_total_horizon()

        if T_total <= 0:
            return 0, 0

        t_val = float(t.item()) if torch.is_tensor(t) else float(t)

        delta = T_total / (self.n_iters * n_pix)
        idx_global = int((t_val - t0) / delta)

        # Clamp for safety at the right endpoint
        idx_global = min(max(idx_global, 0), self.n_iters * n_pix - 1)

        pix_idx = idx_global % n_pix
        active_h = pix_idx // w
        active_w = pix_idx % w
        return active_h, active_w

    def _get_total_horizon(self):
        t0 = self.option_aca["t0"]
        t1 = self.option_aca["t1"]
        t0 = float(t0.item()) if torch.is_tensor(t0) else float(t0)
        t1 = float(t1.item()) if torch.is_tensor(t1) else float(t1)
        return t0, t1, (t1 - t0)

    def _run_explicit_pixel_switch(self, x, full_traj=False):
        y = self.init_y(x)
        self.integration_time = self.integration_time.type_as(x)

        t0, t1, T_total = self._get_total_horizon()

        _, _, h, w = y.shape
        n_pix = h * w
        n_total_intervals = self.n_iters * n_pix
        delta = T_total / n_total_intervals

        t_cur = t0
        step_states = [y]
        step_times = [torch.as_tensor(t_cur, device=y.device, dtype=y.dtype)]

        for interval_idx in range(n_total_intervals):
            t_next = t1 if interval_idx == n_total_intervals - 1 else (t_cur + delta)

            option_aca = self._build_interval_option_aca(t_cur, t_next, y)
            out = aca_ode_solve(self._make_ode_fn(x), y, option_aca)
            y = out[-1]

            t_cur = t_next

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


class ODEXInitFFFBPixelSwitchExplicitStatic(ODEXInitFFFBPixelSwitchExplicit):
    """
    Explicit pixel-switch with Jacobi-style per-iteration update.

    One iteration = visit all pixels once.
    During one iteration, all pixel-local updates are computed from the same
    frozen state y_base. Newly computed pixel values are written into y_next,
    but are NOT visible to later pixel updates within that same iteration.

    After all pixels are visited once, commit:
        y <- y_next
    """

    def _make_ode_fn(self, x, noisy_cu=None, active_h=None, active_w=None):
        assert active_h is not None and active_w is not None, \
            "Jacobi explicit pixel switch requires fixed active_h and active_w."

        def ode_func(t, y):
            bsz, _, h, w = y.shape
            n_pix = h * w

            y_pad = F.pad(y, (self.fb_pad_w, self.fb_pad_w, self.fb_pad_h, self.fb_pad_h))

            patches, valid_mask = self._extract_all_site_patches(
                y_pad, active_h, active_w, h, w
            )
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

    def _run_explicit_pixel_switch(self, x, full_traj=False):
        y = self.init_y(x)
        self.integration_time = self.integration_time.type_as(x)

        t0, t1, T_total = self._get_total_horizon()

        _, _, h, w = y.shape
        n_pix = h * w
        n_total_intervals = self.n_iters * n_pix
        delta = T_total / n_total_intervals

        t_cur = t0
        step_states = [y]
        step_times = [torch.as_tensor(t_cur, device=y.device, dtype=y.dtype)]

        for iter_idx in range(self.n_iters):
            # Freeze the source state for this whole iteration.
            y_base = y
            y_next = y.clone()

            for pix_idx in range(n_pix):
                active_h = pix_idx // w
                active_w = pix_idx % w

                t_next = t1 if (iter_idx == self.n_iters - 1 and pix_idx == n_pix - 1) else (t_cur + delta)

                option_aca = self._build_interval_option_aca(t_cur, t_next, y_base)
                # option_aca = self._build_interval_option_aca(0, delta, y_base)

                out = aca_ode_solve(
                    self._make_ode_fn(x, active_h=active_h, active_w=active_w),
                    y_base,
                    option_aca,
                )
                y_pix = out[-1]

                # Only write the active pixel into the buffer.
                y_next[:, :, active_h, active_w] = y_pix[:, :, active_h, active_w]

                t_cur = t_next

                if full_traj:
                    step_states.append(y_next.clone())
                    step_times.append(torch.as_tensor(t_cur, device=y.device, dtype=y.dtype))

            # Commit once after a full scan over all pixels.
            y = y_next

        if full_traj:
            traj = torch.stack(step_states, dim=0)
            steps = torch.stack(step_times, dim=0)
            return traj, steps

        return y


class ODEXInitFFFBPixelSwitchParallel(ODEXInitFFFBPixelSwitchExplicit):
    """
    Jacobi-style explicit pixel-switch with efficient chunked parallel updates.

    Key behavior:
    - One iteration = visit all pixels once.
    - Within one iteration, every pixel update is computed from the same frozen
      source state y_base.
    - Later pixels/chunks do NOT see earlier updated values from the same iteration.
    - Pixels are processed in chunks of size <= max_pixels.
    - Chunking is only a memory/performance device; it does NOT change the dynamics.

    Hard requirements preserved:
    - still uses _make_ode_fn
    - still uses aca_ode_solve (or whatever solver is in your path)
    """
    def __init__(self, n_iters=1, max_pixels=None, use_cached_patches=True, **kwargs):
        super().__init__(n_iters=n_iters, **kwargs)

        self.use_cached_patches = use_cached_patches
        if max_pixels is not None:
            assert int(max_pixels) >= 1, "max_pixels must be >= 1."
            self.max_pixels = int(max_pixels)
        else:
            self.max_pixels = None

    def _get_valid_indices(self, active_h, active_w, h, w, device):
        # Build the fixed local site offsets for the FF input. [(-1,-1),(-1,0),(-1,1),...]
        dh_list = []
        dw_list = []
        for dh in range(-(self.ff_kh // 2), self.ff_kh // 2 + 1):
            for dw in range(-(self.ff_kw // 2), self.ff_kw // 2 + 1):
                dh_list.append(dh)
                dw_list.append(dw)

        dh = torch.tensor(dh_list, device=device, dtype=torch.long)  # [S], S = 9 for 3x3 kernel
        dw = torch.tensor(dw_list, device=device, dtype=torch.long)  # [S]

        # For every active pixel, compute the FB output positions needed as inputs to FF.
        # They are used to determine which patches are needed in y_ref.
        inter_h = active_h[:, None] + dh[None, :]  # [P, S]
        inter_w = active_w[:, None] + dw[None, :]  # [P, S]

        # On boundaries, some FF input positions needed are in the padding area.
        # For those positions, we don't need to calculate them in the FB conv.
        valid = (inter_h >= 0) & (inter_h < h) & (inter_w >= 0) & (inter_w < w)  # [P, S]

        inter_h_clamped = inter_h.clamp(0, h - 1)
        inter_w_clamped = inter_w.clamp(0, w - 1)
        flat_idx = inter_h_clamped * w + inter_w_clamped  # [P, S]
        return valid, flat_idx, dh, dw

    def _extract_all_patches_for_parallel(self, y_ref, active_h, active_w, flat_idx, valid):
        assert y_ref is not None, "Jacobi chunked efficient mode requires frozen y_ref."
        assert active_h is not None and active_w is not None, \
            "Jacobi chunked efficient mode requires active_h and active_w."

        device = y_ref.device
        dtype = y_ref.dtype

        active_h = torch.as_tensor(active_h, device=device, dtype=torch.long).flatten()
        active_w = torch.as_tensor(active_w, device=device, dtype=torch.long).flatten()
        assert active_h.numel() == active_w.numel(), \
            "active_h and active_w must have the same length."

        bsz, _, h, w = y_ref.shape
        n_active = active_h.numel()
        n_sites = self.ff_kh * self.ff_kw

        # ------------------------------------------------------------
        # Precompute everything that does NOT depend on the solver state y.
        # Since y_ref is frozen for the whole chunk solve, these can be
        # captured in the closure once.
        # ------------------------------------------------------------

        # Extract all patches needed as inputs to FB once.
        y_pad = F.pad(y_ref, (self.fb_pad_w, self.fb_pad_w, self.fb_pad_h, self.fb_pad_h))
        fb_cols = F.unfold(
            y_pad,
            kernel_size=(self.fb_kh, self.fb_kw),
            dilation=1,
            padding=0,
            stride=1,
        )  # [B, I, H*W], where I = Cin * fb_kh * fb_kw
        fb_cols = fb_cols.transpose(1, 2).contiguous()  # [B, H*W, I]

        # Gather all needed FB input patches from the ONE shared y_ref.
        # fb_cols shape: [B, H*W, I]
        # Result: [B, P*S, I] -> [B, P, S, I]
        # P = n_active, S = n_sites
        patches_flat = fb_cols[:, flat_idx, :]
        patches_flat = patches_flat.view(bsz, n_active, n_sites, -1)

        # Zero out invalid intermediate sites corresponding to padding in FB results.
        patches_flat = patches_flat * valid.view(1, n_active, n_sites, 1).to(dtype)

        return patches_flat

    def _make_ode_fn(self, x, noisy_cu=None, y_ref=None, active_h=None, active_w=None):
        """
        Build an RHS that updates a chunk of active pixels in parallel, but
        computes all local quantities from one shared frozen y_ref.

        Args:
            y_ref:    frozen source state for the whole current Jacobi iteration
                      shape [B, C, H, W]
            active_h: 1D tensor/list of active pixel row indices, length P
            active_w: 1D tensor/list of active pixel col indices, length P

        Returns:
            ode_func(t, y): full-state RHS with nonzero entries only at the
                            selected chunk pixels.
        """
        # Get parameter needed
        device = y_ref.device
        dtype = y_ref.dtype

        bsz, _, h, w = y_ref.shape
        n_pix = h * w
        n_active = active_h.numel()
        n_sites = self.ff_kh * self.ff_kw
        cin = y_ref.shape[1]

        valid, flat_idx, dh, dw = self._get_valid_indices(active_h, active_w, h, w, device)

        loc_h = (self.fb_pad_h - dh).to(torch.long)  # [S]
        loc_w = (self.fb_pad_w - dw).to(torch.long)  # [S]

        # Stack copied FB bank weights once.
        weight_flat = torch.stack(
            [self._fb_weight_to_site_matrix(conv) for conv in self.FBconv_copies],
            dim=0
        )  # [S, Cmid, I]

        if self.FBconv_copies[0].bias is not None:
            bias = torch.stack([conv.bias for conv in self.FBconv_copies], dim=0)  # [S, Cmid]
        else:
            bias = None

        patches_flat_cached = None
        def ode_func(t, y):
            # Evaluate all FB copied banks in parallel over both:
            #    - active-pixel dimension P
            #    - local-site dimension S
            #
            # patches_flat: [B, P, S, I]
            # weight_flat:  [S, O, I]
            # z_sites:      [B, P, S, O]
            # O = C_out of W_FB, I = C_in * 9 (C_in and C_out are relative to FB not layer)
            if self.use_cached_patches:
                # Note: If encountered some abnormal acc drop, set do NOT use this branch.
                nonlocal patches_flat_cached
                if patches_flat_cached is None:
                    patches_flat_cached = self._extract_all_patches_for_parallel(
                        y, active_h, active_w, flat_idx.reshape(-1), valid
                    )
                patches_flat = patches_flat_cached
            else:
                patches_flat = self._extract_all_patches_for_parallel(
                    y, active_h, active_w, flat_idx.reshape(-1), valid
                )

            # The clone is removed here. Since we are changing the same place among all calls.
            # Note: If encountered some abnormal acc drop, consider adding clone back here.
            patches_dyn = patches_flat.view(
                bsz, n_active, n_sites, cin, self.fb_kh, self.fb_kw
            )
            # site_idx = 0 -> (-1,-1) relative to active_h/w in FB result
            # -> active_h/w is overlapped with position (2,2) of FB kernel in calculating that site.
            for s in range(n_sites):
                patches_dyn[:, :, s, :, loc_h[s], loc_w[s]] = y[:, :, active_h, active_w].permute(0, 2, 1)
            patches_dyn = patches_dyn * valid.view(1, n_active, n_sites, 1, 1, 1).to(dtype)

            z_sites = torch.einsum(
                "bpsi,soi->bpso",
                patches_dyn.view(bsz, n_active, n_sites, -1),
                weight_flat
            )

            if bias is not None:
                z_sites = z_sites + bias.view(1, 1, n_sites, -1)

            z_sites = z_sites * valid.view(1, n_active, n_sites, 1).to(dtype)

            # Reshape into local FF input tensors,
            # need to permute to place the channel dimension to the second place
            #    [B, P, S, O] -> [B, P, O, S] -> [B*P, (O*ff_kh*ff_kw)]
            z_sites = z_sites.permute(0, 1, 3, 2).contiguous().view(bsz * n_active, -1)

            z_sites = self.act_fn(z_sites)

            # Evaluate FF local center output for all active pixels in batch.
            z_sites = self._eval_ff_center(z_sites)  # [B*P, C_out_FF]
            z_sites = z_sites.view(bsz, n_active, -1).permute(0, 2, 1)  # [B, C_out_FF, P]

            # Scatter chunk outputs back into a full-state RHS.
            out = torch.zeros(
                (bsz, self.FFconv.out_channels, h, w),
                device=device,
                dtype=dtype,
            )

            out[:, :, active_h, active_w] = z_sites

            # Keep the same N scaling as the original pixel-switch math/code.
            return n_pix * out

        return ode_func

    def _post_scan_update(self, y_next, T_iter):
        return y_next

    def _run_explicit_pixel_switch(self, x, full_traj=False):
        y = self.init_y(x)
        self.integration_time = self.integration_time.type_as(x)

        t0, t1, T_total = self._get_total_horizon()

        _, _, h, w = y.shape
        n_pix = h * w

        # One full iteration corresponds to one full scan over all pixels.
        T_iter = T_total / self.n_iters

        # Same mini-interval as the single-pixel explicit version.
        delta = T_iter / n_pix

        max_pixels = n_pix if self.max_pixels is None else min(self.max_pixels, n_pix)

        t_cur = t0
        step_states = [y]
        step_times = [torch.as_tensor(t_cur, device=y.device, dtype=y.dtype)]

        for iter_idx in range(self.n_iters):
            # Freeze the source state for each iteration.
            y_base = y
            y_next = y.clone()

            # Chunking is ONLY an implementation trick.
            # All chunks are still computed from the same y_base and use the same
            # per-pixel mini-interval delta.
            for pix_start in range(0, n_pix, max_pixels):
                pix_end = min(pix_start + max_pixels, n_pix)

                pix_idx = torch.arange(pix_start, pix_end, device=y.device)
                active_h = pix_idx // w
                active_w = pix_idx % w

                # Each chunk solve represents the SAME per-pixel local solve duration delta.
                # Chunk size should NOT stretch time; chunking must not change dynamics.
                if getattr(self, "scale_RHS", True):
                    option_aca = self._build_interval_option_aca(0, delta, y_base)
                else:
                    option_aca = self._build_interval_option_aca(0, delta * n_pix, y_base)

                ode_fn = self._make_ode_fn(
                    x,
                    y_ref=y_base,
                    active_h=active_h,
                    active_w=active_w,
                )

                # solve using the frozen base state, not from y_next
                out = aca_ode_solve(ode_fn, y_base, option_aca)
                y_chunk = out[-1]

                # Commit only the selected chunk pixels into the write buffer.
                y_next[:, :, active_h, active_w] = y_chunk[:, :, active_h, active_w]

            # Optional post-scan correction for possible decay simulation, default does nothing.
            y_next = self._post_scan_update(y_next, T_iter)

            # Commit once after the full iteration.
            y = y_next
            t_cur = t_cur + T_iter

            if full_traj:
                step_states.append(y.clone())
                step_times.append(torch.as_tensor(t_cur, device=y.device, dtype=y.dtype))

        if full_traj:
            traj = torch.stack(step_states, dim=0)
            steps = torch.stack(step_times, dim=0)
            return traj, steps

        return y


class ODEXInitFFFBPixelSwitchEfficient(ODEXInitFFFBPixelSwitchParallel):
    def _make_ode_fn(self, x, noisy_cu=None, y_ref=None, active_h=None, active_w=None):
        """
        Build an RHS that updates a chunk of active pixels in parallel, but
        computes all local quantities from one shared frozen y_ref.

        Args:
            y_ref:    frozen source state for the whole current Jacobi iteration
                      shape [B, C, H, W]
            active_h: 1D tensor/list of active pixel row indices, length P
            active_w: 1D tensor/list of active pixel col indices, length P

        Returns:
            ode_func(t, y): full-state RHS with nonzero entries only at the
                            selected chunk pixels.
        """
        # Get parameter needed
        device = y_ref.device
        dtype = y_ref.dtype

        bsz, _, h, w = y_ref.shape
        n_pix = h * w
        n_active = active_h.numel()
        n_sites = self.ff_kh * self.ff_kw
        cin = y_ref.shape[1]

        valid, flat_idx, dh, dw = self._get_valid_indices(active_h, active_w, h, w, device)

        loc_h = (self.fb_pad_h - dh).to(torch.long)  # [S]
        loc_w = (self.fb_pad_w - dw).to(torch.long)  # [S]

        # Stack copied FB bank weights once.
        # Precompute training-only exact quantities without materializing FB input patches.
        valid_mask = valid.view(1, n_active, n_sites, 1).to(dtype)
        if self.training:
            self.in_training_proc = True
            # Use the exact same local weight layout as _fb_weight_to_site_matrix.
            if isinstance(self.FBconv, nn.Conv2d):
                weight_site = self.FBconv.weight
            elif isinstance(self.FBconv, nn.ConvTranspose2d):
                weight_site = torch.flip(self.FBconv.weight, dims=[2, 3]).permute(1, 0, 2, 3).contiguous()
            else:
                raise TypeError(f"Unsupported FB conv type: {type(self.FBconv)}")
            # Exact local FB correction weights for replacing only the active-pixel entry.
            # weight_site is [O, Cin, fb_kh, fb_kw], so:
            # w_loc[s, o, c] = weight_site[o, c, loc_h[s], loc_w[s]]
            # Result: [S, O, Cin]
            w_loc = weight_site[:, :, loc_h, loc_w].permute(2, 0, 1).contiguous()
        else:
            w_loc = []
            if getattr(self, "in_training_proc", False):
                self._sync_fb_copies_from_base()
            for _s, _conv in enumerate(self.FBconv_copies):
                if isinstance(_conv, nn.Conv2d):
                    weight_site = _conv.weight
                elif isinstance(_conv, nn.ConvTranspose2d):
                    weight_site = torch.flip(_conv.weight, dims=[2, 3]).permute(1, 0, 2, 3).contiguous()
                else:
                    raise TypeError(f"Unsupported FB conv type: {type(_conv)}")
                w_loc.append(weight_site[:, :, loc_h[_s], loc_w[_s]])
            w_loc = torch.stack(w_loc, dim=0).contiguous()  # [S, O, Cin]

        z_sites_ref_cached = None
        y_ref_active_cached = None
        def ode_func(t, y):
            # Evaluate all FB copied banks in parallel over both:
            #    - active-pixel dimension P
            #    - local-site dimension S
            #
            # patches_flat: [B, P, S, I]
            # weight_flat:  [S, O, I]
            # z_sites:      [B, P, S, O]
            # O = C_out of W_FB, I = C_in * 9 (C_in and C_out are relative to FB not layer)
            nonlocal z_sites_ref_cached, y_ref_active_cached
            if z_sites_ref_cached is None:
                if self.training:
                    fb_out_ref = self.FBconv(y)
                    fb_out_ref_flat = fb_out_ref.view(bsz, fb_out_ref.shape[1], -1)
                    z_sites_ref_cached = fb_out_ref_flat[:, :, flat_idx.reshape(-1)]
                    z_sites_ref_cached = z_sites_ref_cached.view(
                        bsz, fb_out_ref.shape[1], n_active, n_sites
                    ).permute(0, 2, 3, 1).contiguous()
                    z_sites_ref_cached = z_sites_ref_cached * valid_mask
                else:
                    z_sites_ref_cached = []
                    for s, conv in enumerate(self.FBconv_copies):
                        fb_out_ref = conv(y)
                        fb_out_ref_flat = fb_out_ref.view(bsz, fb_out_ref.shape[1], -1)
                        z_s_ref = fb_out_ref_flat[:, :, flat_idx[:, s]]
                        z_s_ref = z_s_ref.permute(0, 2, 1).contiguous()
                        z_s_ref = z_s_ref * valid[:, s].view(1, n_active, 1).to(dtype)
                        z_sites_ref_cached.append(z_s_ref)
                    z_sites_ref_cached = torch.stack(z_sites_ref_cached, dim=2)  # [B, P, S, O]

            if y_ref_active_cached is None:
                # Imagine flattening the H and W dimension to be a vector for each b, c_in.
                # Previously it is a 2D matrix of shape (H, W).
                y_ref_active_cached = y[:, :, active_h, active_w].permute(0, 2, 1).contiguous()
            # Current active-pixel values: [B, P, Cin]
            y_active = y[:, :, active_h, active_w].permute(0, 2, 1).contiguous()
            # Exact delta from the frozen source state.
            delta_active = y_active - y_ref_active_cached  # [B, P, Cin]
            # Exact FB correction for replacing only the active-pixel entry:
            # [B, P, S, O]
            corr = torch.einsum("bpc,soc->bpso", delta_active, w_loc)
            z_sites = (z_sites_ref_cached + corr * valid_mask) * valid_mask

            # Reshape into local FF input tensors,
            # need to permute to place the channel dimension to the second place
            #    [B, P, S, O] -> [B, P, O, S] -> [B*P, (O*ff_kh*ff_kw)]
            z_sites = z_sites.permute(0, 1, 3, 2).contiguous().view(bsz * n_active, -1)

            z_sites = self.act_fn(z_sites)

            # Evaluate FF local center output for all active pixels in batch.
            z_sites = self._eval_ff_center(z_sites)  # [B*P, C_out_FF]
            z_sites = z_sites.view(bsz, n_active, -1).permute(0, 2, 1)  # [B, C_out_FF, P]

            # Scatter chunk outputs back into a full-state RHS.
            out = torch.zeros(
                (bsz, self.FFconv.out_channels, h, w),
                device=device,
                dtype=dtype,
            )

            out[:, :, active_h, active_w] = z_sites

            if getattr(self, "scale_RHS", True):
                # Keep the same N scaling as the original pixel-switch math/code.
                return n_pix * out
            else:
                return out

        return ode_func


class ODEXInitFFFBPixelSwitchStretchT(ODEXInitFFFBPixelSwitchEfficient):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.scale_RHS = False


class ODEXInitFFFBPixelSwitchEfficientDecay(ODEXInitFFFBPixelSwitchEfficient):
    def __init__(self, i_leak=None, leak_to=0.0, **kwargs):
        super().__init__(**kwargs)
        self.i_leak = i_leak
        self.leak_to = leak_to
        self.tau_decay = None

    def _init_tau_decay_if_needed(self):
        """
        When using this class, make sure the model is wrapped by the wrapper to provide needed
        parameters including C and v_dd.

        It calculates a fixed R_leak = v * C / i_leak.
        The i_leak is the leak current when v = v_dd.
        If v is a * v_dd, then the i_leak will also be smaller as a * i_leak.
        """
        if self.tau_decay is not None:
            return

        if self.i_leak is None or self.i_leak <= 0:
            self.tau_decay = None
            return

        C_store = getattr(self, "C", 49e-15)
        v_ref = getattr(self, "v_dd", 0.1)

        self.tau_decay = v_ref * C_store / self.i_leak

    def _post_scan_update(self, y_next, T_iter):
        self._init_tau_decay_if_needed()

        if self.tau_decay is None:
            return y_next

        tau_decay = y_next.new_tensor(self.tau_decay)

        _, _, h, w = y_next.shape
        n_pix = h * w

        if getattr(self, "scale_RHS", True):
            dt_slot = T_iter / n_pix
        else:
            dt_slot = T_iter

        pix_idx = torch.arange(n_pix, device=y_next.device, dtype=y_next.dtype)
        wait_steps = (n_pix - 1) - pix_idx

        rho = torch.exp(
            -wait_steps * y_next.new_tensor(dt_slot) / tau_decay
        ).view(1, 1, h, w)

        if self.leak_to == 0.0:
            return y_next * rho

        leak_to = y_next.new_tensor(self.leak_to)
        return leak_to + rho * (y_next - leak_to)


class ODEXInitFFFBPixelSwitchStretchTDecay(ODEXInitFFFBPixelSwitchEfficientDecay):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.scale_RHS = False


class PerturbODEXInitFFFB(ODEXInitFFFBPixelSwitchEfficient):
    def __init__(
        self,
        detach_tangent=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.detach_tangent = bool(detach_tangent)

    def _compute_correction_efficient(self, y):
        """
        Exact and efficient for RHS = FFconv(act_fn(FBconv(y))).

        Uses the closed-form local Jacobian action:
            z = FBconv(y)
            R = FFconv(act_fn(z))
            u_p = F_p(y) or stopgrad(F_p(y))
            delta z_loc,p = M_p u_p
            delta a_loc,p = act'(z_loc,p) ⊙ delta z_loc,p
            C_p = FF_center(delta a_loc,p)

        Returns
        -------
        R : [B, C, H, W]
        C : [B, C, H, W]
        """
        assert isinstance(self.FFconv, nn.Conv2d), \
            "Efficient method currently expects FFconv to be nn.Conv2d."
        assert isinstance(self.FBconv, (nn.Conv2d, nn.ConvTranspose2d)), \
            "Efficient method currently expects FBconv to be nn.Conv2d or nn.ConvTranspose2d."

        z = self.FBconv(y)                  # [B, Cmid, H, W]
        a = self.act_fn(z)
        R = self.FFconv(a)                  # [B, Cy, H, W]

        assert R.shape[1] == y.shape[1], \
            "Need RHS output channels == state channels to use F_p(y) as tangent."

        U = R.detach() if self.detach_tangent else R

        B, Cin, H, W = y.shape
        _, Cmid, _, _ = z.shape
        _, Cy, _, _ = R.shape
        P = H * W

        ff_kh, ff_kw = self.FFconv.kernel_size if isinstance(self.FFconv.kernel_size, tuple) else (
            self.FFconv.kernel_size, self.FFconv.kernel_size
        )
        fb_kh, fb_kw = self.FBconv.kernel_size if isinstance(self.FBconv.kernel_size, tuple) else (
            self.FBconv.kernel_size, self.FBconv.kernel_size
        )

        ff_pad_h = self.FFconv.padding[0] if isinstance(self.FFconv.padding, tuple) else self.FFconv.padding
        ff_pad_w = self.FFconv.padding[1] if isinstance(self.FFconv.padding, tuple) else self.FFconv.padding
        fb_pad_h = self.FBconv.padding[0] if isinstance(self.FBconv.padding, tuple) else self.FBconv.padding
        fb_pad_w = self.FBconv.padding[1] if isinstance(self.FBconv.padding, tuple) else self.FBconv.padding

        S = ff_kh * ff_kw
        device = y.device
        dtype = y.dtype

        # ------------------------------------------------------------
        # 1) Gather all local z-patches used by FF center outputs
        #    z_loc[b, p, s, o] = FB output channel o at local FF site s for pixel p
        # ------------------------------------------------------------
        z_cols = F.unfold(
            z,
            kernel_size=(ff_kh, ff_kw),
            dilation=1,
            padding=(ff_pad_h, ff_pad_w),
            stride=1,
        )   # [B, Cmid*S, P]

        z_loc = z_cols.transpose(1, 2).contiguous().view(B, P, Cmid, S).permute(0, 1, 3, 2).contiguous()
        # z_loc: [B, P, S, Cmid]

        # ------------------------------------------------------------
        # 2) Build local linear map M_p from delta y_p to delta z_loc,p
        # ------------------------------------------------------------
        dh_list = []
        dw_list = []
        for dh in range(-(ff_kh // 2), ff_kh // 2 + 1):
            for dw in range(-(ff_kw // 2), ff_kw // 2 + 1):
                dh_list.append(dh)
                dw_list.append(dw)

        dh = torch.tensor(dh_list, device=device, dtype=torch.long)   # [S]
        dw = torch.tensor(dw_list, device=device, dtype=torch.long)   # [S]

        if isinstance(self.FBconv, nn.Conv2d):
            weight_site = self.FBconv.weight
        else:
            # convert ConvTranspose2d weight to local site matrix layout [Cmid, Cin, kh, kw]
            weight_site = torch.flip(self.FBconv.weight, dims=[2, 3]).permute(1, 0, 2, 3).contiguous()

        # For local FF site offset (dh, dw), y_p contributes to z_{p+(dh,dw)}
        # through FB kernel entry (fb_pad_h - dh, fb_pad_w - dw).
        loc_h = (fb_pad_h - dh).to(torch.long)   # [S]
        loc_w = (fb_pad_w - dw).to(torch.long)   # [S]

        # w_loc[s, o, c] maps delta y_p[c] -> delta z_loc,p[s, o]
        w_loc = weight_site[:, :, loc_h, loc_w].permute(2, 0, 1).contiguous()   # [S, Cmid, Cin]

        # ------------------------------------------------------------
        # 3) Valid mask for boundary pixels
        # ------------------------------------------------------------
        pix = torch.arange(P, device=device)
        ph = pix // W
        pw = pix % W

        site_h = ph[:, None] + dh[None, :]   # [P, S]
        site_w = pw[:, None] + dw[None, :]   # [P, S]
        valid = (site_h >= 0) & (site_h < H) & (site_w >= 0) & (site_w < W)
        valid_mask = valid.view(1, P, S, 1).to(dtype)   # [1, P, S, 1]

        # ------------------------------------------------------------
        # 4) delta z_loc,p = M_p u_p
        # ------------------------------------------------------------
        U_flat = U.permute(0, 2, 3, 1).reshape(B, P, Cin)   # [B, P, Cin]
        delta_z_loc = torch.einsum("bpc,soc->bpso", U_flat, w_loc)   # [B, P, S, Cmid]
        delta_z_loc = delta_z_loc * valid_mask

        # ------------------------------------------------------------
        # 5) delta a_loc,p = act'(z_loc,p) ⊙ delta z_loc,p
        # ------------------------------------------------------------
        if isinstance(self.act_fn, nn.ReLU):
            act_prime = (z_loc > 0).to(dtype)
        elif isinstance(self.act_fn, nn.ReLU6):
            act_prime = ((z_loc > 0) & (z_loc < 6)).to(dtype)
        elif isinstance(self.act_fn, nn.Hardtanh):
            act_prime = ((z_loc > self.act_fn.min_val) & (z_loc < self.act_fn.max_val)).to(dtype)
        elif isinstance(self.act_fn, nn.Identity):
            act_prime = torch.ones_like(z_loc)
        elif isinstance(self.act_fn, ReLUX):
            ub = 6.0 / float(self.act_fn.scale)
            act_prime = ((z_loc > 0) & (z_loc < ub)).to(dtype)
        else:
            raise NotImplementedError(
                f"Need explicit derivative for act_fn type: {type(self.act_fn)}"
            )

        delta_a_loc = act_prime * delta_z_loc   # [B, P, S, Cmid]

        # ------------------------------------------------------------
        # 6) C_p = FF center linear map applied to delta_a_loc,p
        #    No FF bias here, because this is a Jacobian action.
        # ------------------------------------------------------------
        ff_weight_flat = self.FFconv.weight.reshape(self.FFconv.out_channels, -1)   # [Cy, Cmid*S]
        delta_a_flat = delta_a_loc.permute(0, 1, 3, 2).contiguous().view(B * P, Cmid * S)

        C_flat = torch.einsum("bi,oi->bo", delta_a_flat, ff_weight_flat)   # [B*P, Cy]
        C = C_flat.view(B, P, Cy).permute(0, 2, 1).contiguous().view(B, Cy, H, W)

        return R, C

    def _make_ode_fn(self, x, noisy_cu=None):
        def ode_func(t, y):
            R, C = self._compute_correction_efficient(y)
            t0, t1, T_total = self._get_total_horizon()
            t_iter = T_total / self.n_iters
            return R + 0.5 * t_iter * C

        return ode_func

    def forward(self, x, layer_idx=None):
        y0 = self.init_y(x)
        self.integration_time = self.integration_time.type_as(x)

        out = aca_ode_solve(self._make_ode_fn(x), y0, self.option_aca)
        # out = odeint(ode_func, y0, self.integration_time, rtol=self.tol, atol=self.tol, method=self.method)
        out = out[-1]

        if self.bypass is not None:
            out = self.bypass(out) + out
        return out


SWITCH_CLASSES = {
    "ODEXInitFFFBPixelSwitch": ODEXInitFFFBPixelSwitch,
    "ODEXInitFFFBPixelSwitchExplicit": ODEXInitFFFBPixelSwitchExplicit,
    "ODEXInitFFFBPixelSwitchExplicitStatic": ODEXInitFFFBPixelSwitchExplicitStatic,
    "ODEXInitFFFBPixelSwitchParallel": ODEXInitFFFBPixelSwitchParallel,
    "ODEXInitFFFBPixelSwitchEfficient": ODEXInitFFFBPixelSwitchEfficient,
    "ODEXInitFFFBPixelSwitchStretchT": ODEXInitFFFBPixelSwitchStretchT,
    "ODEXInitFFFBPixelSwitchEfficientDecay": ODEXInitFFFBPixelSwitchEfficientDecay,
    "ODEXInitFFFBPixelSwitchStretchTDecay": ODEXInitFFFBPixelSwitchStretchTDecay,
    "PerturbODEXInitFFFB": PerturbODEXInitFFFB,
}
import hashlib
import logging
import pickle
import types
from functools import wraps
from collections import OrderedDict

import numpy as np
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
from copy import deepcopy

from tqdm import tqdm

from pc_model import PCNet
from utils import interpolate_R_eff


def mask_set_val(dst_mat, src_val):
    mask = (dst_mat > 0).flatten()
    # print("src_val shape: {}, dst_mat shape: {}, non-zero vals in dst_mat: {}".format(src_val.shape, dst_mat.shape, torch.sum(mask)))
    dst_shape = dst_mat.shape

    dst_mat = dst_mat.flatten()
    src_val_flat = src_val.flatten()

    dst_mat.masked_scatter_(mask, src_val_flat)
    dst_mat = dst_mat.view(dst_shape)

    return dst_mat


def build_unrolled_csr_padded(C_in, H_pad, W_pad, C_out, k_h, k_w,
                              H_out, W_out, rows, stride, kernel):
    device, dtype = kernel.device, kernel.dtype
    out_dim    = C_out * rows
    in_dim_pad = C_in * H_pad * W_pad
    s = stride

    nnz_per_row = C_in * k_h * k_w
    crow = torch.arange(0, out_dim + 1, device=device, dtype=torch.int64) * nnz_per_row

    # base columns per output-spatial row (within one input channel)
    base_cols_per_row = []
    for i in range(H_out):
        for j in range(W_out):
            cols = []
            for ki in range(k_h):
                for kj in range(k_w):
                    ui = i * s + ki
                    vj = j * s + kj
                    cols.append(ui * W_pad + vj)
            base_cols_per_row.append(cols)
    base_cols = torch.tensor(base_cols_per_row, device=device, dtype=torch.int64)  # [rows, k_h*k_w]

    # expand across input channels for ONE oc-block
    ic_offsets = (torch.arange(C_in, device=device, dtype=torch.int64) * (H_pad * W_pad)).view(1, -1, 1)
    cols_one_block = (base_cols.unsqueeze(1) + ic_offsets).reshape(rows, -1)      # [rows, C_in*k_h*k_w]

    # tile for all output channels
    col_indices = cols_one_block.repeat(C_out, 1).reshape(-1)                      # [out_dim * nnz_per_row]

    # values: per row, concat ker[oc, ic].flatten() for ic over 0..C_in-1
    ker_flat = kernel.reshape(C_out, C_in, -1)                                     # [C_out, C_in, k_h*k_w]
    vals_one_row_order = ker_flat.reshape(-1)                                      # oc-major, then ic, then elems
    values = vals_one_row_order.repeat_interleave(rows).to(dtype)

    return torch.sparse_csr_tensor(crow, col_indices, values,
                                   size=(out_dim, in_dim_pad),
                                   dtype=dtype, device=device)


def build_unrolled_csr_trimmed(C_in, H, W, C_out, k_h, k_w,
                               H_out, W_out, rows, stride, padding, kernel):
    device, dtype = kernel.device, kernel.dtype
    out_dim = C_out * rows
    in_dim  = C_in * H * W
    s, p = stride, padding

    crow = [0]
    cols = []
    vals = []

    for oc in range(C_out):
        for i in range(H_out):
            for j in range(W_out):
                nnz_row = 0
                for ic in range(C_in):
                    for ki in range(k_h):
                        for kj in range(k_w):
                            ui = i * s + ki
                            vj = j * s + kj
                            # keep only if (ui, vj) lies inside unpadded window
                            if (p <= ui < p + H) and (p <= vj < p + W):
                                raw_col = (ui - p) * W + (vj - p)       # within one input channel
                                cols.append(ic * (H * W) + raw_col)     # global column
                                vals.append(kernel[oc, ic, ki, kj].item())
                                nnz_row += 1
                crow.append(crow[-1] + nnz_row)

    crow = torch.tensor(crow, dtype=torch.int64, device=device)
    cols = torch.tensor(cols, dtype=torch.int64, device=device)
    vals = torch.tensor(vals, dtype=dtype, device=device)

    return torch.sparse_csr_tensor(crow, cols, vals,
                                   size=(out_dim, in_dim),
                                   dtype=dtype, device=device)

def conv2d_to_matrix_fixed_padding(input_shape, kernel, stride=1, padding=0, *,
                                   return_padded_input=False):
    C_in, H, W = input_shape
    C_out, _, k_h, k_w = kernel.shape
    p, s = padding, stride

    # Output dims
    H_out = (H + 2 * p - k_h) // s + 1
    W_out = (W + 2 * p - k_w) // s + 1
    rows = H_out * W_out

    # Work on padded grid first
    H_pad, W_pad = H + 2 * p, W + 2 * p
    in_pad_size = H_pad * W_pad

    # Base mask on padded grid
    base_matrix_pad = torch.zeros((rows, in_pad_size), dtype=kernel.dtype, device=kernel.device)
    for i in range(H_out):
        for j in range(W_out):
            r = i * W_out + j
            for ki in range(k_h):
                for kj in range(k_w):
                    ui = i * s + ki
                    vj = j * s + kj
                    c = ui * W_pad + vj
                    base_matrix_pad[r, c] = 1

    # Label on padded grid
    B_labeled_pad = np.full((rows, in_pad_size), "", dtype=object)
    for i in range(H_out):
        for j in range(W_out):
            r = i * W_out + j
            for ki in range(k_h):
                for kj in range(k_w):
                    ui = i * s + ki
                    vj = j * s + kj
                    c = ui * W_pad + vj
                    if base_matrix_pad[r, c] != 0:
                        B_labeled_pad[r, c] = f"{ki + 1}{kj + 1}"

    if not return_padded_input and p > 0:
        # trimmed (unpadded) CSR
        kernel_matrix = build_unrolled_csr_trimmed(
            C_in, H, W, C_out, k_h, k_w, H_out, W_out, rows, s, p, kernel
        )

        # keep mask for labels/mask (unchanged behavior)
        keep = torch.zeros((H_pad, W_pad), dtype=torch.bool, device=kernel.device)
        keep[p:H_pad - p, p:W_pad - p] = True
        keep = keep.flatten()

        base_matrix = base_matrix_pad[:, keep]
        B_labeled = B_labeled_pad[:, keep.cpu().numpy()]
    else:
        # padded CSR
        kernel_matrix = build_unrolled_csr_padded(
            C_in, H_pad, W_pad, C_out, k_h, k_w, H_out, W_out, rows, s, kernel
        )
        base_matrix = base_matrix_pad
        B_labeled = B_labeled_pad

    return kernel_matrix, base_matrix, B_labeled


def build_unrolled_dtc_metadata(mat, input_shape, meta):
    """Map each expanded CSR value back to its original convolution kernel output."""
    inp_chan, inp_h, inp_w = (int(v) for v in input_shape)
    out_chan = int(meta["out_chan"])
    ker_h, ker_w = int(meta["ker_h"]), int(meta["ker_w"])
    stride, padding = int(meta["stride"]), int(meta["padding"])
    out_h = (inp_h + 2 * padding - ker_h) // stride + 1
    out_w = (inp_w + 2 * padding - ker_w) // stride + 1
    spatial_rows = out_h * out_w

    crow = mat.crow_indices()
    row_ids = torch.arange(
        mat.shape[0], device=crow.device, dtype=torch.int64).repeat_interleave(
            crow[1:] - crow[:-1])
    cols = mat.col_indices()

    output_channel = torch.div(row_ids, spatial_rows, rounding_mode="floor")
    output_spatial = row_ids.remainder(spatial_rows)
    output_row = torch.div(output_spatial, out_w, rounding_mode="floor")
    output_col = output_spatial.remainder(out_w)

    input_channel = torch.div(cols, inp_h * inp_w, rounding_mode="floor")
    input_spatial = cols.remainder(inp_h * inp_w)
    input_row = torch.div(input_spatial, inp_w, rounding_mode="floor")
    input_col = input_spatial.remainder(inp_w)

    kernel_row = input_row + padding - output_row * stride
    kernel_col = input_col + padding - output_col * stride
    valid = (
        (output_channel >= 0) & (output_channel < out_chan) &
        (input_channel >= 0) & (input_channel < inp_chan) &
        (kernel_row >= 0) & (kernel_row < ker_h) &
        (kernel_col >= 0) & (kernel_col < ker_w)
    )
    if not bool(valid.all()):
        raise RuntimeError("Could not map every expanded CSR value to a DTC output.")

    dtc_block_ids = output_channel * inp_chan + input_channel
    dtc_output_ids = kernel_row * ker_w + kernel_col
    return dtc_block_ids, dtc_output_ids

@torch.no_grad()
def expanded_weight_cache_fingerprint(model, input_shape):
    """Fingerprint the effective convolutions that Validator will expand."""
    digest = hashlib.sha256()
    digest.update(b"validator-expanded-weights-v1\0")
    digest.update(repr(tuple(int(v) for v in input_shape)).encode("utf-8"))
    digest.update(b"\0")
    num_convs = 0
    for layer_idx, layer in enumerate(model.PcConvs):
        for module_name, module in layer.named_modules():
            if not isinstance(module, nn.Conv2d):
                continue
            num_convs += 1
            geometry = (
                int(layer_idx), module_name, int(module.in_channels),
                int(module.out_channels), tuple(module.kernel_size),
                tuple(module.stride), tuple(module.padding),
                tuple(module.dilation), int(module.groups),
                str(module.padding_mode),
            )
            digest.update(repr(geometry).encode("utf-8"))
            digest.update(b"\0")
            weight = module.weight.detach().cpu().contiguous()
            digest.update(str(weight.dtype).encode("utf-8"))
            digest.update(repr(tuple(weight.shape)).encode("utf-8"))
            digest.update(weight.view(torch.uint8).numpy().tobytes())
            digest.update(b"\0")
    if num_convs == 0:
        raise ValueError("Validator could not find any Conv2d weights to expand.")
    return digest.hexdigest()[:16]


@torch.no_grad()
def cached_unrolled_weight_matches(entry, conv, input_shape):
    """Validate a legacy expanded entry against the current effective weight."""
    if not isinstance(entry, dict) or "weight" not in entry or "meta" not in entry:
        return False
    mat, meta = entry["weight"], entry["meta"]
    if not getattr(mat, "is_sparse_csr", False):
        return False
    if conv.groups != 1 or tuple(conv.dilation) != (1, 1):
        return False

    inp_chan, inp_h, inp_w = (int(v) for v in input_shape)
    stride = conv.stride[0] if isinstance(conv.stride, tuple) else conv.stride
    padding = conv.padding[0] if isinstance(conv.padding, tuple) else conv.padding
    ker_h, ker_w = (
        conv.kernel_size if isinstance(conv.kernel_size, tuple)
        else (conv.kernel_size, conv.kernel_size))
    expected_meta = {
        "padding": int(padding),
        "stride": int(stride),
        "ker_h": int(ker_h),
        "ker_w": int(ker_w),
        "inp_chan": int(conv.in_channels),
        "out_chan": int(conv.out_channels),
    }
    if any(int(meta.get(name, -1)) != value
           for name, value in expected_meta.items()):
        return False
    if inp_chan != conv.in_channels:
        return False

    out_h = (inp_h + 2 * int(padding) - int(ker_h)) // int(stride) + 1
    out_w = (inp_w + 2 * int(padding) - int(ker_w)) // int(stride) + 1
    expected_shape = (
        conv.out_channels * out_h * out_w,
        conv.in_channels * inp_h * inp_w,
    )
    if tuple(mat.shape) != expected_shape:
        return False

    try:
        block_ids, output_ids = build_unrolled_dtc_metadata(
            mat, input_shape, meta)
    except (RuntimeError, ValueError):
        return False
    input_channel = block_ids.remainder(conv.in_channels)
    output_channel = torch.div(
        block_ids, conv.in_channels, rounding_mode="floor")
    kernel_row = torch.div(output_ids, int(ker_w), rounding_mode="floor")
    kernel_col = output_ids.remainder(int(ker_w))
    expected_values = conv.weight[
        output_channel, input_channel, kernel_row, kernel_col]
    return torch.equal(mat.values(), expected_values)


def _save_expanded_weight_cache(stored, path):
    temporary_path = "{}.tmp.{}".format(path, os.getpid())
    torch.save(stored, temporary_path)
    os.replace(temporary_path, path)


def snapshot_clean_mvm_mat_values(net):
    clean_vals = OrderedDict()
    for mod_name, m in net.named_modules():
        if isinstance(m, MVMConv):
            # Cache the clean expanded weights in MVMConv for pulse-based implementation.
            if not hasattr(m, "clean_mat_values"):
                m.clean_mat_values = m.mat.values().detach().clone()
            clean_vals[mod_name] = m.clean_mat_values
    return clean_vals

def assert_mvm_mats_all_values_noised(net, clean_vals, max_print=10, allow_unchanged_frac=1e-4):
    """
    allow_unchanged_frac:
      - 0.0 => fail if ANY unchanged nonzero entry exists
      - e.g. 1e-6 => allow up to 1e-6 fraction unchanged among nonzero entries
    Notice:
     with nnz ~= 0.6M each layer and 20 layers in total, it COULD happen that at least 1 random
     noise is sampled as zero such that the most strict assertion will fail.
    """
    for mod_name, m in net.named_modules():
        if not isinstance(m, MVMConv):
            continue

        cur = m.mat.values()
        ref = clean_vals[mod_name]

        assert cur.numel() == ref.numel(), f"{mod_name}.mat: nnz changed"

        # Multiplicative noise leaves ref==0 unchanged; ignore those.
        nz = (ref != 0)
        total = int(nz.sum().item())
        if total == 0:
            continue

        unchanged_mask = (cur == ref) & nz
        n_unchanged = int(unchanged_mask.sum().item())
        frac_unchanged = n_unchanged / total

        if frac_unchanged > allow_unchanged_frac:
            idx = unchanged_mask.nonzero(as_tuple=False).flatten()
            k = min(max_print, idx.numel())
            idx_s = idx[:k]

            ref_s  = ref[idx_s]
            cur_s  = cur[idx_s]
            diff_s = cur_s - ref_s

            msg = (
                f"{mod_name}.mat: unchanged(nonzero)={n_unchanged}/{total} "
                f"({frac_unchanged:.6g}), allowed_frac={allow_unchanged_frac:.6g}\n"
                f"  dtype={cur.dtype}, device={cur.device}, nnz={cur.numel()}\n"
                f"  Showing {k} unchanged indices (out of {n_unchanged}): {idx_s.tolist()}\n"
                f"  ref:  {ref_s.tolist()}\n"
                f"  cur:  {cur_s.tolist()}\n"
                f"  diff: {diff_s.tolist()}\n"
            )
            raise AssertionError(msg)


class MVMConv(nn.Module):
    def __init__(self, mat, meta):
        super().__init__()
        self.mat = mat
        self.meta = meta
        for _k, _v in meta.items():
            setattr(self, _k, _v)

        self.csv_enabled = False
        self.code_idx_mat = None
        self.v_grid, self.R_codes, self.R_left, self.R_slope = None, None, None, None
        self.proj_fn = None
        self.R = None

        self.mismatch_mat = None
        self.pulse_noisy_values = None
        self.mismatch_type = "mul"
        self.mul_mismatch_mode = None

        self.nonlinear_R_curve_sharing = "shared"
        self.nonlinear_R_curve_seed = None
        self.nonlinear_R_curve_edge_chunk_size = 65536
        self.nonlinear_R_curve_assignment = None
        self.nonlinear_R_curve_group_ids = None
        self.nonlinear_R_curve_row_ids = None
        self.nonlinear_R_curve_group_count = 0
        self.nonlinear_R_curve_bank_v_grid = None
        self.nonlinear_R_curve_bank_R_left = None
        self.nonlinear_R_curve_bank_R_slope = None
        self.nonlinear_R_curve_bank_lengths = None
        self.nonlinear_R_curve_sampling = "empirical_with_replacement"
        self.nonlinear_R_curve_gaussian_v_grid = None
        self.nonlinear_R_curve_gaussian_mean = None
        self.nonlinear_R_curve_gaussian_factor = None
        self.nonlinear_R_curve_gaussian_value_scale = None
        self.nonlinear_R_curve_gaussian_quantity = None
        self.nonlinear_R_curve_gaussian_R_normalized = None

    def set_dtc_metadata(self, dtc_block_ids, dtc_output_ids):
        if dtc_block_ids.numel() != self.mat.values().numel():
            raise ValueError("DTC block metadata must align with every CSR value.")
        if dtc_output_ids.numel() != self.mat.values().numel():
            raise ValueError("DTC output metadata must align with every CSR value.")
        self.register_buffer(
            "dtc_block_ids", dtc_block_ids.to(dtype=torch.int64), persistent=False)
        self.register_buffer(
            "dtc_output_ids", dtc_output_ids.to(dtype=torch.int64), persistent=False)

    def enable_csv(self, v_grid, R_codes, R_left, R_slope, proj_fn, R,
                   mul_mismatch_mode="scale_mismatch", curve_bank=None,
                   nonlinear_R_curve_sharing="shared",
                   nonlinear_R_curve_seed=None,
                   nonlinear_R_curve_edge_chunk_size=65536,
                   nonlinear_R_curve_assignment=None,
                   nonlinear_R_curve_sampling="empirical_with_replacement",
                   curve_gaussian=None):
        self.csv_enabled = True

        self.v_grid = v_grid
        self.R_codes = R_codes
        self.R_left = R_left
        self.R_slope = R_slope
        self.proj_fn = proj_fn
        self.R = R

        self.mul_mismatch_mode = mul_mismatch_mode
        assert self.mul_mismatch_mode in {"scale_mismatch", "static_mismatch"}

        sharing = str(nonlinear_R_curve_sharing).lower()
        aliases = {
            "per_input_element": "per_input",
            "per_input_output_channel": "per_input_output",
        }
        sharing = aliases.get(sharing, sharing)
        supported = {"shared", "per_coupler", "per_input", "per_input_output"}
        if sharing not in supported:
            raise ValueError(
                "Unknown nonlinear-R curve sharing mode: {}.".format(sharing))
        self.nonlinear_R_curve_sharing = sharing
        self.nonlinear_R_curve_seed = nonlinear_R_curve_seed
        self.nonlinear_R_curve_edge_chunk_size = int(
            nonlinear_R_curve_edge_chunk_size)
        if self.nonlinear_R_curve_edge_chunk_size <= 0:
            raise ValueError(
                "nonlinear_R_curve_edge_chunk_size must be positive.")
        sampling = str(nonlinear_R_curve_sampling).lower()
        aliases = {
            "empirical": "empirical_with_replacement",
            "gaussian": "multivariate_gaussian",
        }
        sampling = aliases.get(sampling, sampling)
        supported_sampling = {
            "empirical_with_replacement", "multivariate_gaussian"}
        if sampling not in supported_sampling:
            raise ValueError(
                "Unknown nonlinear-R curve sampling mode: {}.".format(
                    sampling))
        self.nonlinear_R_curve_sampling = sampling
        if sharing != "shared":
            self._build_nonlinear_R_curve_groups()
            if sampling == "multivariate_gaussian":
                if curve_gaussian is None:
                    raise ValueError(
                        "A nonlinear-R Gaussian fit is required for "
                        "multivariate_gaussian sampling.")
                self.set_nonlinear_R_curve_gaussian(curve_gaussian)
                self.sample_nonlinear_R_gaussian_curves()
            else:
                if curve_bank is None:
                    raise ValueError(
                        "A nonlinear-R curve bank is required for {} sharing."
                        .format(sharing))
                self.set_nonlinear_R_curve_bank(curve_bank)
                if nonlinear_R_curve_assignment is None:
                    self.sample_nonlinear_R_curve_assignment()
                else:
                    self.set_nonlinear_R_curve_assignment(
                        nonlinear_R_curve_assignment)

    def set_nonlinear_R_curve_gaussian(self, curve_gaussian):
        """Install a direct full-covariance fit for static curve sampling."""
        required = {"v_grid", "mean", "factor", "value_scale", "quantity"}
        missing = required.difference(curve_gaussian)
        if missing:
            raise ValueError(
                "Curve Gaussian fit is missing {}.".format(sorted(missing)))
        device, dtype = self.mat.device, self.mat.dtype
        self.nonlinear_R_curve_gaussian_v_grid = curve_gaussian["v_grid"].to(
            device=device, dtype=dtype)
        self.nonlinear_R_curve_gaussian_mean = curve_gaussian["mean"].to(
            device=device, dtype=dtype)
        self.nonlinear_R_curve_gaussian_factor = curve_gaussian["factor"].to(
            device=device, dtype=dtype)
        self.nonlinear_R_curve_gaussian_value_scale = float(
            curve_gaussian["value_scale"])
        self.nonlinear_R_curve_gaussian_quantity = str(
            curve_gaussian["quantity"]).lower()
        n_points = self.nonlinear_R_curve_gaussian_v_grid.numel()
        if (self.nonlinear_R_curve_gaussian_v_grid.ndim != 1 or
                self.nonlinear_R_curve_gaussian_mean.shape != (n_points,) or
                self.nonlinear_R_curve_gaussian_factor.shape !=
                (n_points, n_points)):
            raise ValueError("Nonlinear-R Gaussian tensors do not align.")
        if self.nonlinear_R_curve_gaussian_quantity not in {
                "conductance", "resistance"}:
            raise ValueError("Gaussian curve quantity is invalid.")

    def sample_nonlinear_R_gaussian_curves(self):
        """Sample one independent, fixed full curve for every sharing group."""
        n_groups = self.nonlinear_R_curve_group_count
        n_points = self.nonlinear_R_curve_gaussian_v_grid.numel()
        generator = None
        if self.nonlinear_R_curve_seed is not None:
            generator = torch.Generator(device=self.mat.device)
            generator.manual_seed(int(self.nonlinear_R_curve_seed))
        sampled_R_normalized = torch.empty(
            (n_groups, n_points), device=self.mat.device,
            dtype=self.mat.dtype)
        chunk_size = self.nonlinear_R_curve_edge_chunk_size
        mean = self.nonlinear_R_curve_gaussian_mean
        factor_t = self.nonlinear_R_curve_gaussian_factor.t()
        value_scale = self.nonlinear_R_curve_gaussian_value_scale
        quantity = self.nonlinear_R_curve_gaussian_quantity
        positive_floor = torch.finfo(self.mat.dtype).eps * value_scale
        with torch.no_grad():
            for start in range(0, n_groups, chunk_size):
                stop = min(start + chunk_size, n_groups)
                eps = torch.randn(
                    (stop - start, n_points), device=self.mat.device,
                    dtype=self.mat.dtype, generator=generator)
                sampled = (mean + eps @ factor_t) * value_scale
                sampled.clamp_(min=positive_floor)
                if quantity == "conductance":
                    sampled_R_normalized[start:stop] = (
                        sampled.reciprocal() / self.R)
                else:
                    sampled_R_normalized[start:stop] = sampled / self.R
        self.nonlinear_R_curve_gaussian_R_normalized = sampled_R_normalized
        return self.nonlinear_R_curve_gaussian_R_normalized


    def set_nonlinear_R_curve_bank(self, curve_bank):
        """Install evaluated curves without imposing a sampling model."""
        required = {"v_grid", "R_left", "R_slope", "lengths"}
        missing = required.difference(curve_bank)
        if missing:
            raise ValueError(
                "Curve bank is missing {}.".format(sorted(missing)))
        self.nonlinear_R_curve_bank_v_grid = curve_bank["v_grid"].to(
            device=self.mat.device, dtype=self.mat.dtype)
        self.nonlinear_R_curve_bank_R_left = curve_bank["R_left"].to(
            device=self.mat.device, dtype=self.mat.dtype)
        self.nonlinear_R_curve_bank_R_slope = curve_bank["R_slope"].to(
            device=self.mat.device, dtype=self.mat.dtype)
        self.nonlinear_R_curve_bank_lengths = curve_bank["lengths"].to(
            device=self.mat.device, dtype=torch.int64)
        if self.nonlinear_R_curve_bank_v_grid.ndim != 2:
            raise ValueError("Nonlinear-R curve grids must be two-dimensional.")
        n_curves = self.nonlinear_R_curve_bank_v_grid.shape[0]
        if n_curves <= 0:
            raise ValueError("The nonlinear-R curve bank is empty.")
        expected_intervals = self.nonlinear_R_curve_bank_v_grid.shape[1] - 1
        if (self.nonlinear_R_curve_bank_R_left.shape !=
                (n_curves, expected_intervals) or
                self.nonlinear_R_curve_bank_R_slope.shape !=
                (n_curves, expected_intervals) or
                self.nonlinear_R_curve_bank_lengths.numel() != n_curves):
            raise ValueError("Nonlinear-R curve-bank tensors do not align.")
        if bool((self.nonlinear_R_curve_bank_lengths < 2).any()) or bool(
                (self.nonlinear_R_curve_bank_lengths >
                 self.nonlinear_R_curve_bank_v_grid.shape[1]).any()):
            raise ValueError("Nonlinear-R curve lengths are invalid.")

    def _build_nonlinear_R_curve_groups(self):
        nnz = self.mat.values().numel()
        sharing = self.nonlinear_R_curve_sharing
        if sharing == "per_input":
            self.nonlinear_R_curve_group_ids = None
            self.nonlinear_R_curve_row_ids = None
            self.nonlinear_R_curve_group_count = int(self.mat.shape[1])
            return

        crow = self.mat.crow_indices()
        row_ids = torch.arange(
            self.mat.shape[0], device=crow.device,
            dtype=torch.int64).repeat_interleave(crow[1:] - crow[:-1])
        self.nonlinear_R_curve_row_ids = row_ids
        if sharing == "per_coupler":
            group_ids = None
            group_count = nnz
        elif sharing == "per_input_output":
            cols = self.mat.col_indices()
            spatial_rows = self.mat.shape[0] // int(self.meta["out_chan"])
            output_channel = torch.div(
                row_ids, spatial_rows, rounding_mode="floor")
            raw_group_ids = output_channel * self.mat.shape[1] + cols
            _, group_ids = torch.unique(
                raw_group_ids, sorted=True, return_inverse=True)
            group_count = int(group_ids.max().item()) + 1 if nnz else 0
        else:
            raise RuntimeError(
                "Curve groups are only built for non-shared modes.")
        self.nonlinear_R_curve_group_ids = group_ids
        self.nonlinear_R_curve_group_count = int(group_count)

    def set_nonlinear_R_curve_assignment(self, curve_indices):
        """Set externally sampled group-to-curve indices for this hardware."""
        indices = torch.as_tensor(
            curve_indices, device=self.mat.device, dtype=torch.int64).reshape(-1)
        if indices.numel() != self.nonlinear_R_curve_group_count:
            raise ValueError(
                "Expected {} nonlinear-R curve assignments, received {}."
                .format(self.nonlinear_R_curve_group_count, indices.numel()))
        n_curves = self.nonlinear_R_curve_bank_v_grid.shape[0]
        if indices.numel() and (indices.min() < 0 or indices.max() >= n_curves):
            raise ValueError("Nonlinear-R curve assignment is out of range.")
        self.nonlinear_R_curve_assignment = indices

    def sample_nonlinear_R_curve_assignment(self, sampler=None):
        """Sample once; callers may supply any compatible assignment sampler."""
        n_groups = self.nonlinear_R_curve_group_count
        n_curves = self.nonlinear_R_curve_bank_v_grid.shape[0]
        generator = None
        if self.nonlinear_R_curve_seed is not None:
            generator = torch.Generator(device=self.mat.device)
            generator.manual_seed(int(self.nonlinear_R_curve_seed))
        if sampler is None:
            indices = torch.randint(
                n_curves, (n_groups,), device=self.mat.device,
                generator=generator)
        else:
            indices = sampler(
                n_groups=n_groups, n_curves=n_curves,
                device=self.mat.device, generator=generator)
        self.set_nonlinear_R_curve_assignment(indices)
        return self.nonlinear_R_curve_assignment

    def _values_to_code_idx(self, values):
        # The input values is the weight matrix values.
        # This function converts that to the column index in self.R_table.
        w_abs = values.abs()
        zero_mask = w_abs <= 0
        w_abs = w_abs.clamp(min=1e-12)

        # R_ij = R / W_ij
        # W must be quantized before such that R_ij can correspond to one column in the R_table
        R_hat = self.R / w_abs
        code_idx = (R_hat[:, None] - self.R_codes[None, :]).abs().argmin(dim=1)
        code_idx[zero_mask] = -1
        return code_idx

    def _build_code_idx_mat(self, val_sign_and_noise=None):
        # Returns a list of tuples
        # The first element of the tuple is code idx
        # The second element of the tuple is a sparse matrix storing:
        #   no mismatch: sign(weight)
        #   mul mismatch: sign(weight) * (1 + eps)
        # When combining the non-linearity with multiplicative mismatch,
        # if mul_mismatch_mode = "scale_mismatch", the noisy weight is
        # w_noisy = w * (1 + eps). We played a math trick and put the (1 + eps)
        # part along with the sign. So the original weights are left untouched.
        # The mismatch is applied after we get the interpolated weights.
        mat_coo = self.mat.to_sparse_coo().coalesce()
        idx = mat_coo.indices()
        vals = mat_coo.values()

        code_idx_vals = self._values_to_code_idx(vals)  # (nnz,); Get the column index in the R_table.
        sign_vals = vals.sign() if val_sign_and_noise is None else val_sign_and_noise

        mats = []
        uniq = torch.unique(code_idx_vals)
        for j in uniq.tolist():
            if j < 0:
                continue
            sel = (code_idx_vals == j)
            if sel.any():
                idx_j = idx[:, sel]
                val_j = sign_vals[sel].to(vals.dtype)
                mat_j = torch.sparse_coo_tensor(
                    idx_j, val_j,
                    size=self.mat.shape,
                    device=self.mat.device,
                    dtype=self.mat.dtype
                ).coalesce().to_sparse_csr()
                mats.append((int(j), mat_j))
        return mats

    # Three helper functions to support different mismatch levels for
    # different quantization levels
    def _get_quant_magnitude_levels(self, values, q_hi, weight_scale):
        # Maps the weight values to quantization levels.
        k = torch.arange(q_hi + 1, device=values.device, dtype=values.dtype)

        if weight_scale == 1.0:
            levels = k / q_hi
        else:
            first = weight_scale / q_hi
            delta = (1.0 - first) / (q_hi - 1)

            levels = torch.zeros_like(k)
            mask = k > 0
            levels[mask] = first + (k[mask] - 1.0) * delta

        return levels

    def _values_to_level_idx(self, values, q_hi, weight_scale):
        # Get level idx.
        levels = self._get_quant_magnitude_levels(values, q_hi, weight_scale)
        v_abs = values.abs().reshape(-1, 1)
        dist = (v_abs - levels.reshape(1, -1)).abs()
        level_idx = dist.argmin(dim=1).to(torch.long)
        return level_idx.reshape_as(values)

    def _get_sparse_sigma_tensor(self, values, noise_level, q_hi, weight_scale):
        # Get different mismatch levels for each element in the values,
        # depending on the quantization level of each element.
        if not isinstance(noise_level, dict):
            return noise_level

        level_idx = self._values_to_level_idx(values, q_hi, weight_scale)

        # Same as ODEBlockPC: missing keys default to max sigma.
        sigma_lut = torch.full(
            (q_hi + 1,),
            max(noise_level.values()),
            device=values.device,
            dtype=values.dtype
        )
        for _k, _sigma in noise_level.items():
            sigma_lut[_k] = _sigma

        sigma = sigma_lut[level_idx]
        return sigma

    @torch.no_grad()
    def add_noise(self, noise_level, mismatch_type="mul", q_hi=None, weight_scale=1.0):
        if noise_level is None:
            return
        if not isinstance(noise_level, dict) and noise_level <= 0:
            return

        self.mismatch_type = mismatch_type

        if q_hi is None and isinstance(noise_level, dict):
            raise ValueError("q_hi is required when noise_level is a dict.")

        if not self.csv_enabled:
            # Now if not using nonlinear_R, we add mismatch here. Copying the same
            # logic in ODEBlockPC.
            v_ = self.mat.values()
            sigma_ = self._get_sparse_sigma_tensor(v_, noise_level, q_hi, weight_scale)

            if mismatch_type == "mul":
                noise_ = torch.randn_like(v_, device=v_.device, requires_grad=False) * sigma_
                v_.mul_(1 + noise_)
            else:
                max_abs = v_.abs().max()
                noise_ = torch.randn_like(v_, device=v_.device, requires_grad=False) * (sigma_ * max_abs)
                v_.add_(noise_)
            return

        mat_coo = self.mat.to_sparse_coo().coalesce()
        idx = mat_coo.indices()
        vals = mat_coo.values()

        sigma_ = self._get_sparse_sigma_tensor(vals, noise_level, q_hi, weight_scale)

        if mismatch_type == "mul":
            noise_ = torch.randn_like(vals, device=vals.device, requires_grad=False) * sigma_
            self.pulse_noisy_values = vals * (1 + noise_)

            # In this case we sample a fixed val_sign_and_noise, the actual mismatch scales
            # with the input dependent resistance value.
            if self.mul_mismatch_mode == "scale_mismatch":
                val_sign_and_noise = vals.sign() * (1 + noise_)
                self.code_idx_mat = self._build_code_idx_mat(
                    val_sign_and_noise=val_sign_and_noise
                )
                self.mismatch_mat = None

            # In this case, we keep a fixed mismatch.
            elif self.mul_mismatch_mode == "static_mismatch":
                # Static conductance mismatch based on the selected CSV column name.
                # clean self.mat decides the nominal programmed resistance column.
                # mismatch is a fixed extra conductance/weight term and does NOT depend on input voltage.

                self.code_idx_mat = self._build_code_idx_mat()

                clean_code_idx = self._values_to_code_idx(vals)
                valid_mask = clean_code_idx >= 0

                mismatch_vals = torch.zeros_like(vals)

                # calculate R_ij, non-distorted by non-linearity
                clean_R_code = self.R_codes[clean_code_idx[valid_mask]]
                # Compensate the 1/self.R in the wrapper's transform method.
                clean_W_code_abs = self.R / clean_R_code

                # Multiplicative conductance mismatch:
                # real_mismatch = G_ij * sigma * N(0,1) = 1 / R_ij * sigma * N(0,1)
                # The wrapper will divide the result by R, thus we calculate
                # mismatch = R * real_mismatch in this function.
                mismatch_vals[valid_mask] = vals.sign()[valid_mask] * clean_W_code_abs * noise_[valid_mask]

                self.mismatch_mat = torch.sparse_coo_tensor(
                    idx,
                    mismatch_vals,
                    size=self.mat.shape,
                    device=self.mat.device,
                    dtype=self.mat.dtype
                ).coalesce().to_sparse_csr()
            else:
                raise ValueError("Unknown mismatch mode.")

        else:
            max_abs = vals.abs().max()
            noise_ = torch.randn_like(vals, device=vals.device, requires_grad=False) * (sigma_ * max_abs)
            self.pulse_noisy_values = vals + noise_

            # Keep nonlinear-R nominal path unchanged.
            self.code_idx_mat = self._build_code_idx_mat()

            # Additive mismatch path: out += delta_w @ x
            # For additive mismatch, we store it in an extra class attributes.
            self.mismatch_mat = torch.sparse_coo_tensor(
                idx,
                noise_,
                size=self.mat.shape,
                device=self.mat.device,
                dtype=self.mat.dtype
            ).coalesce().to_sparse_csr()

    def _get_R_eff(self, v, code_idx):
        return interpolate_R_eff(
            v, self.v_grid, self.R_codes, self.R_left, self.R_slope,
            code_idx, proj_fn=getattr(self, "proj_fn", None))

    def _get_gaussian_curve_R_eff(self, v, group_index, nominal_R):
        """Evaluate fixed Gaussian-sampled curves on their shared voltage grid."""
        if self.proj_fn is not None:
            v = self.proj_fn(v)
        group_index = torch.as_tensor(
            group_index, device=v.device, dtype=torch.int64).reshape(-1)
        if v.ndim != 2 or group_index.numel() != v.shape[0]:
            raise ValueError(
                "Curve groups must provide one sampled curve per voltage row.")

        grid = self.nonlinear_R_curve_gaussian_v_grid
        query = v.clamp(min=grid[0], max=grid[-1])
        interval = torch.searchsorted(
            grid.contiguous(), query.contiguous(), right=False) - 1
        interval.clamp_(min=0, max=grid.numel() - 2)
        curves = self.nonlinear_R_curve_gaussian_R_normalized[group_index]
        left = curves.gather(1, interval)
        right = curves.gather(1, interval + 1)
        left_v = grid[interval]
        right_v = grid[interval + 1]
        fraction = (query - left_v) / (right_v - left_v)
        return nominal_R * (left + fraction * (right - left))


    def _get_curve_bank_R_eff(self, v, curve_index):
        """Evaluate one selected piecewise-linear R(V) curve per source row."""
        if self.proj_fn is not None:
            v = self.proj_fn(v)
        curve_index = torch.as_tensor(
            curve_index, device=v.device, dtype=torch.int64).reshape(-1)
        if v.ndim != 2 or curve_index.numel() != v.shape[0]:
            raise ValueError(
                "Curve indices must provide one curve for each voltage row.")

        grid = self.nonlinear_R_curve_bank_v_grid[curve_index]
        lengths = self.nonlinear_R_curve_bank_lengths[curve_index]
        last_grid = grid.gather(1, (lengths - 1)[:, None])
        query = torch.maximum(v, grid[:, :1])
        query = torch.minimum(query, last_grid)
        interval = torch.searchsorted(
            grid.contiguous(), query.contiguous(), right=False) - 1
        interval.clamp_(min=0)
        interval = torch.minimum(interval, (lengths - 2)[:, None])
        left_v = grid.gather(1, interval)
        left_R = self.nonlinear_R_curve_bank_R_left[
            curve_index].gather(1, interval)
        slope = self.nonlinear_R_curve_bank_R_slope[
            curve_index].gather(1, interval)
        return left_R + slope * (query - left_v)

    def _forward_pulse_per_input(self, x_flat, pulse_weight, nominal_R):
        if self.nonlinear_R_curve_sampling == "multivariate_gaussian":
            group = torch.arange(
                x_flat.shape[0], device=x_flat.device, dtype=torch.int64)
            R_eff = self._get_gaussian_curve_R_eff(
                x_flat, group, nominal_R)
        else:
            curve_index = self.nonlinear_R_curve_assignment
            R_eff = self._get_curve_bank_R_eff(x_flat, curve_index)
        return torch.sparse.mm(
            pulse_weight, x_flat * nominal_R / R_eff)

    def _forward_pulse_per_edge(self, x_flat, pulse_weight, nominal_R):
        if pulse_weight.layout != torch.sparse_csr:
            raise TypeError(
                "Per-edge nonlinear-R pulse evaluation requires a CSR matrix.")
        pulse_values = pulse_weight.values()
        if pulse_values.numel() != self.mat.values().numel():
            raise ValueError(
                "Pulse values must align with the expanded physical couplers.")

        active_edges = torch.nonzero(
            pulse_values != 0, as_tuple=False).reshape(-1)
        out = x_flat.new_zeros((self.mat.shape[0], x_flat.shape[1]))
        chunk_size = self.nonlinear_R_curve_edge_chunk_size
        for start in range(0, active_edges.numel(), chunk_size):
            edge = active_edges[start:start + chunk_size]
            cols = self.mat.col_indices()[edge]
            rows = self.nonlinear_R_curve_row_ids[edge]
            if self.nonlinear_R_curve_sharing == "per_coupler":
                group = edge
            else:
                group = self.nonlinear_R_curve_group_ids[edge]
            source = x_flat[cols]
            if self.nonlinear_R_curve_sampling == "multivariate_gaussian":
                R_eff = self._get_gaussian_curve_R_eff(
                    source, group, nominal_R)
            else:
                curve = self.nonlinear_R_curve_assignment[group]
                R_eff = self._get_curve_bank_R_eff(source, curve)
            contribution = (
                pulse_values[edge, None] * source * nominal_R / R_eff)
            out.index_add_(0, rows, contribution)
        return out

    def forward_pulse(self, x, pulse_weight, nominal_R=None):
        """Apply an externally generated pulse matrix with configured R(V) sharing."""
        batch_size, _, input_h, input_w = x.shape
        output_h = (input_h + 2 * self.meta["padding"] - self.meta["ker_h"]) // self.meta["stride"] + 1
        output_w = (input_w + 2 * self.meta["padding"] - self.meta["ker_w"]) // self.meta["stride"] + 1
        x_flat = x.reshape(batch_size, -1).t()

        if self.csv_enabled:
            nominal_R = self.R if nominal_R is None else nominal_R
            if self.nonlinear_R_curve_sharing == "shared":
                nominal_idx = (self.R_codes - nominal_R).abs().argmin()
                R_eff = self._get_R_eff(x_flat, nominal_idx)
                x_flat = x_flat * nominal_R / R_eff
                out = torch.sparse.mm(pulse_weight, x_flat)
            elif self.nonlinear_R_curve_sharing == "per_input":
                out = self._forward_pulse_per_input(
                    x_flat, pulse_weight, nominal_R)
            else:
                out = self._forward_pulse_per_edge(
                    x_flat, pulse_weight, nominal_R)
        else:
            out = torch.sparse.mm(pulse_weight, x_flat)
        return out.t().reshape(batch_size, self.meta["out_chan"], output_h, output_w)

    def forward(self, x):
        batch_size, input_channels, input_h, input_w = x.shape
        x = x.view(batch_size, -1).t()
        output_h = (input_h + 2 * self.meta["padding"] - self.meta["ker_h"]) // self.meta["stride"] + 1
        output_w = (input_w + 2 * self.meta["padding"] - self.meta["ker_w"]) // self.meta["stride"] + 1

        if not self.csv_enabled:
            return torch.sparse.mm(self.mat, x).t().view(batch_size, self.meta["out_chan"], output_h, output_w)

        if self.code_idx_mat is None:
            self.code_idx_mat = self._build_code_idx_mat()
        out = None
        # If with scaled multiplicative mismatch, it is already in self.code_idx_mat and mat_j.
        for (j, mat_j) in self.code_idx_mat:
            R_eff_j = self._get_R_eff(x, j)
            x_j = x / R_eff_j
            y_j = torch.sparse.mm(mat_j, x_j)
            out = y_j if out is None else out + y_j

        if out is None:
            out = torch.sparse.mm(self.mat, x)
            return out.t().view(batch_size, self.meta["out_chan"], output_h, output_w)

        ################################################################################
        # This is to compensate the self.R in the transform method of ODEWrapper1State
        # so that we are using R_eff instead of R in the ODE.
        ################################################################################
        out = out * self.R

        # Add additive mismatch or static multiplicative mismatch part.
        if self.mismatch_mat is not None:
            out = out + torch.sparse.mm(self.mismatch_mat, x)

        return out.t().view(batch_size, self.meta["out_chan"], output_h, output_w)

    @property
    def weight(self):
        return self.mat

    def inp_param_sum(self, x):
        _, input_channels, input_h, input_w = x.shape
        output_h = (input_h + 2 * self.meta["padding"] - self.meta["ker_h"]) // self.meta["stride"] + 1
        output_w = (input_w + 2 * self.meta["padding"] - self.meta["ker_w"]) // self.meta["stride"] + 1
        inp_summed = torch.sparse.sum(self.mat.abs().to_sparse_coo(), dim=1).to_dense()
        out_chan = self.meta["out_chan"]
        # inp_summed = inp_summed.view(out_chan, output_h, output_w).sum(dim=1)
        return inp_summed.sqrt().view(1, out_chan, output_h, output_w)

class Validator(nn.Module):
    def __init__(self, model: PCNet, expanded_weight_dir, device, test_dataloader, result_path, wrapper=None,
                 record_full_traj=False, t_end_sf=1.0, **kwargs):
        """
        Do weight unroll_or_load in the init method. The mismatch to weight is added later on
        after unroll_or_load is finished. Make sure the model passed in to init method has
        noise_level=0.0. Otherwise, we will save the unrolled noisy weights.
        """
        super().__init__()
        # The model should contain conv layers only. All transposed conv layers should be converted to conv layers.
        # The model needs to be converted to ode blocks and wrapped with wrapper before.
        self.model = model
        self.model.eval()
        self.device = device
        self.dataloader = test_dataloader
        self._unroll_sample_inputs = next(iter(self.dataloader))[0][:2].to(
            self.device)
        self.expanded_weight_cache_fingerprint = (
            expanded_weight_cache_fingerprint(
                self.model, self._unroll_sample_inputs.shape[1:]))
        self.legacy_exp_w_path = os.path.join(
            expanded_weight_dir, "expanded_weights_{}.pth")
        expanded_weight_dir = os.path.join(
            expanded_weight_dir,
            "cache_{}".format(self.expanded_weight_cache_fingerprint))
        self.exp_w_path = os.path.join(
            expanded_weight_dir, "expanded_weights_{}.pth")
        os.makedirs(expanded_weight_dir, exist_ok=True)
        logging.warning(
            "Expanded-weight cache fingerprint: %s (%s)",
            self.expanded_weight_cache_fingerprint, expanded_weight_dir)
        self.wrappers = wrapper
        self.record_full_traj = record_full_traj
        self.t_end_sf = t_end_sf

        self.unroll_or_load()
        self.result_path = result_path

    @torch.no_grad()
    def _register_hook_for_unroll(self, layer_idx, layer):
        exp_w_path = self.exp_w_path.format(layer_idx)
        legacy_exp_w_path = self.legacy_exp_w_path.format(layer_idx)
        if os.path.exists(exp_w_path):
            stored = torch.load(exp_w_path, map_location=self.device)
            # logging.warning("Found existing expanded weights. Load directly.")
        else:
            stored = {}
            # logging.warning("Needs unrolling...")
        legacy_stored = {}
        if not stored and os.path.exists(legacy_exp_w_path):
            legacy_stored = torch.load(
                legacy_exp_w_path, map_location=self.device)

        def make_hook(parent, mod_name, m):
            hook_handlers = {}
            def pre_hook(mod, inputs):
                # Register hook to unroll the conv weights
                # If the unrolled weights are already stored, load it
                # Otherwise perform unrolling
                if mod_name in stored:
                    return
                legacy_entry = legacy_stored.get(mod_name)
                if legacy_entry is not None and cached_unrolled_weight_matches(
                        legacy_entry, mod, inputs[0].shape[1:]):
                    stored[mod_name] = legacy_entry
                    _save_expanded_weight_cache(stored, exp_w_path)
                    logging.warning(
                        "Migrated matching legacy expanded weights for "
                        "layer %s module %s into cache %s.",
                        layer_idx, mod_name,
                        self.expanded_weight_cache_fingerprint)
                    return
                _, inp_channels, inp_h, inp_w = inputs[0].shape
                stride = mod.stride[0] if isinstance(mod.stride, tuple) else mod.stride
                padding = mod.padding[0] if isinstance(mod.padding, tuple) else mod.padding
                unrolled, _, _ = conv2d_to_matrix_fixed_padding(
                    (inp_channels, inp_h, inp_w), mod.weight.detach(), stride=stride, padding=padding)
                k_h, k_w = mod.kernel_size if isinstance(mod.kernel_size, tuple) else (mod.kernel_size, mod.kernel_size)
                stored[mod_name] = {
                    "weight": unrolled,
                    "meta": {
                        "padding": padding,
                        "stride": stride,
                        "ker_h": int(k_h),
                        "ker_w": int(k_w),
                        "inp_chan": int(mod.in_channels),
                        "out_chan": int(mod.out_channels),
                    }
                }
                _save_expanded_weight_cache(stored, exp_w_path)

            def post_hook(mod, inputs, output):
                # Replace plain conv with unrolled weights
                unrolled, meta = stored[mod_name]["weight"], stored[mod_name]["meta"]
                mvm_conv = MVMConv(unrolled, meta)
                dtc_block_ids, dtc_output_ids = build_unrolled_dtc_metadata(
                    unrolled, inputs[0].shape[1:], meta)
                mvm_conv.set_dtc_metadata(dtc_block_ids, dtc_output_ids)
                mvm_conv.clean_mat_values = mvm_conv.mat.values().detach().clone()
                if hasattr(parent, "_nonlinear_R_pkg"):
                    nonlinear_R_pkg = dict(parent._nonlinear_R_pkg)
                    curve_seed = nonlinear_R_pkg.get(
                        "nonlinear_R_curve_seed")
                    if curve_seed is not None:
                        module_key = "{}:{}".format(
                            layer_idx, mod_name).encode("utf-8")
                        seed_offset = int.from_bytes(
                            hashlib.sha256(module_key).digest()[:8],
                            byteorder="little", signed=False)
                        nonlinear_R_pkg["nonlinear_R_curve_seed"] = (
                            int(curve_seed) + seed_offset) % (2 ** 63 - 1)
                    mvm_conv.enable_csv(**nonlinear_R_pkg)
                setattr(parent, mod_name, mvm_conv)
                hook_handlers["pre_hook"].remove()
                hook_handlers["post_hook"].remove()

            hook_handlers["pre_hook"] = m.register_forward_pre_hook(pre_hook)
            hook_handlers["post_hook"] = m.register_forward_hook(post_hook)

        for _name, _mod in layer.named_modules():
            if isinstance(_mod, nn.Conv2d):
                make_hook(parent=layer, mod_name=_name, m=_mod)

    def _find_unscaled_point(self, steps):
        if np.allclose(self.t_end_sf, 1.0):
            return -1
        t_end_scaled = steps[-1]
        t_end_gt = t_end_scaled / self.t_end_sf
        gt_pos = torch.argmin((steps - t_end_gt).abs()).item()
        return gt_pos

    @torch.no_grad()
    def unroll_or_load(self):
        for _idx, _layer in enumerate(self.model.PcConvs):
            self._register_hook_for_unroll(_idx, _layer)

        # One forward pass to trigger the hooks and unroll the weights
        _ = self.model(self._unroll_sample_inputs)
        self._unroll_sample_inputs = None

        # If record full trajectory, use forward_full_steps of odeblocks.
        if self.record_full_traj:
            assert self.wrappers is not None
            for _idx, _layer in enumerate(self.model.PcConvs):
                _w = self.wrappers[_idx]
                orig_forward = _layer.forward

                def forward_use_full(x, *args, _layer=_layer, w=_w, **kwargs):
                    x_traj = w.wrap_input(x)
                    traj, steps = _layer.forward_full_steps(x_traj)

                    # stash for hooks
                    # should record the exact ode traj; scale output is for accuracy purpose.
                    if isinstance(traj, (tuple, list)):
                        _layer._last_full_traj = tuple(
                            _t.contiguous().reshape(_t.shape[0], _t.shape[1], -1).detach().cpu().numpy()
                            for _t in traj
                        )
                    else:
                        _layer._last_full_traj = traj.contiguous().reshape(traj.shape[0], traj.shape[1],
                                                                     -1).detach().cpu().numpy()
                    _layer._last_full_steps = steps.detach().cpu().numpy() if torch.is_tensor(steps) else steps

                    traj = w.unwrap_output(traj)
                    traj_main = traj[0] if isinstance(traj, (tuple, list)) else traj
                    gt_pos = self._find_unscaled_point(steps)
                    y_last = traj_main[gt_pos]
                    return y_last

                _layer.forward = forward_use_full
                _layer._orig_forward = orig_forward

    @torch.no_grad()
    def test_unroll(self):
        _total, _correct = 0, 0
        for batch_idx, (inputs, targets) in tqdm(enumerate(self.dataloader), total=len(self.dataloader), disable=False):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            with torch.no_grad():
                output_tensor = self.model(inputs)
            _, predicted = torch.max(output_tensor, 1)
            _total += targets.size(0)
            _correct += (predicted == targets).sum().item()
        # Calculate the accuracy
        _acc = 100 * _correct / _total
        logging.warning("Accuracy using unrolled weights: {}".format(_acc))

    @staticmethod
    def patch_init_y_for_capture(mod, update_init_y):
        if getattr(mod, "_init_y_wrapped_for_capture", False):
            return
        # Todo: Do we need to bind all patched functions in the wrapper
        orig_init_y = mod.init_y

        @wraps(orig_init_y)
        def patched_init_y(x, *args, **kwargs):
            # if getattr(orig_init_y, "__self__", None) is not None:
            #     bound_orig = orig_init_y
            # else:
            #     bound_orig = types.MethodType(orig_init_y, mod)
            # return update_init_y(bound_orig, x, *args, **kwargs)
            return update_init_y(orig_init_y, x, *args, **kwargs)

        mod.init_y = patched_init_y
        setattr(mod, "_init_y_wrapped_for_capture", True)

    @staticmethod
    def _get_csr_data(mat):
        mat = mat.detach().to_sparse_csr().cpu()
        return {
            "shape": tuple(mat.size()),
            "row_ptr": mat.crow_indices().numpy(),
            "col_idx": mat.col_indices().numpy(),
            "val": mat.values().numpy(),
        }

    @torch.no_grad()
    def _register_hook_for_record(self, wrappers):
        res = {}
        handlers = []
        for _idx, _layer in enumerate(self.model.PcConvs):
            # dict_keys(['R', 'q', 'kernel_0', 'kernel_1', 'kernel_2', 'kernel_3'])
            # dict_keys(['inp', 'init_res', 'out', 'init_time', 'compute_time', 'FF_mat', 'FB_mat', 'C_ff', 'C_fb'])
            cur_name = "layer_{}".format(_idx)
            res[cur_name] = {}
            res[cur_name]["init_time"] = getattr(_layer, "option_init", {}).get("t1", torch.tensor(0.0)).cpu().item()
            res[cur_name]["compute_time"] = _layer.option_aca["t1"].cpu().item()
            res[cur_name]["FF_mat"] = self._get_csr_data(_layer.FFconv.mat)
            res[cur_name]["FB_mat"] = self._get_csr_data(_layer.FBconv.mat)
            res[cur_name]["s_R"] = getattr(wrappers[_idx], "s_R", None)
            res[cur_name]["R_max"] = getattr(wrappers[_idx], "R_max", None)
            res[cur_name]["R"] = wrappers[_idx].R if res[cur_name]["s_R"] is None else None
            res[cur_name]["k"] = getattr(wrappers[_idx], "k", None)
            res[cur_name]["beta_c"] = wrappers[_idx].beta_c.cpu().item() if hasattr(wrappers[_idx], "beta_c") else None
            res[cur_name]["C"] = wrappers[_idx].C
            res[cur_name]["C_ff"] = wrappers[_idx].C_ff.cpu().item() if hasattr(wrappers[_idx], "C_ff") else None
            res[cur_name]["C_fb"] = wrappers[_idx].C_fb if hasattr(wrappers[_idx], "C_fb") else None
            res[cur_name]["q"] = wrappers[_idx].q
            if hasattr(wrappers[_idx], "beta"):
                res[cur_name]["beta"] = wrappers[_idx].beta.cpu().item() if isinstance(wrappers[_idx].beta, torch.Tensor) else wrappers[_idx].beta
            else:
                res[cur_name]["beta"] = None

            _inp_scale = getattr(wrappers[_idx], "inp_scale", wrappers[_idx].q)
            _out_scale = getattr(wrappers[_idx], "out_scale", wrappers[_idx].q)

            def make_capture_init_res(key=cur_name):
                def capture_init_res(orig_init_y, x, *args, **kwargs):
                    yz = orig_init_y(x, *args, **kwargs)
                    if isinstance(yz, tuple):
                        res[key]["init_res"] = yz[-1].view(yz[-1].shape[0], -1).contiguous().detach().cpu().numpy()
                    else:
                        res[key]["init_res"] = yz.view(yz.shape[0], -1).contiguous().detach().cpu().numpy()
                    return yz
                return capture_init_res

            def pre_hook(mod, inputs, key=cur_name, inp_scale=_inp_scale):
                # Here the input captured hasn't been scaled by inp_scale, thus to
                # match the input to the ode solver, we need to multiple inp_scale
                res[key]["inp"] = inputs[0].view(inputs[0].shape[0], -1).contiguous().detach().cpu().numpy() * inp_scale
                update_init_y = make_capture_init_res(key)
                self.patch_init_y_for_capture(mod, update_init_y)

            def post_hook(mod, inputs, output, key=cur_name, out_scale=_out_scale):
                # res[key]["init_res"] = mod.init_y(inputs[0])[-1]
                # Here the output captured has already been scaled by 1 / out_scale, thus to
                # match the output of the ode solver, we need to multiple out_scale
                res[key]["out"] = output.view(output.shape[0], -1).contiguous().detach().cpu().numpy() * out_scale

                if self.record_full_traj:
                    traj = getattr(mod, "_last_full_traj", None)
                    steps = getattr(mod, "_last_full_steps", None)
                    if traj is not None and steps is not None:
                        res[key]["traj"] = traj
                        res[key]["steps"] = steps

            handlers.append(_layer.register_forward_pre_hook(pre_hook))
            handlers.append(_layer.register_forward_hook(post_hook))
        return res, handlers

    @torch.no_grad()
    def gen_validate_data(self, wrappers, solver, n_samples=20, sample_inp=None, select_layer=6):
        res, handlers = self._register_hook_for_record(wrappers)

        if sample_inp is None:
            _inp, _targets = next(iter(self.dataloader))
            _inp = _inp.to(self.device)
            _targets = _targets.to(self.device)
            _ = self.model(_inp)
            select_key = "layer_{}".format(select_layer)
            if select_key not in res:
                raise ValueError("{} not found in validation results".format(select_key))
            if "traj" not in res[select_key]:
                raise ValueError("{} does not have full trajectory data".format(select_key))
            selected_idx = self._select_top_changed_samples(res[select_key]["traj"], n_samples)
            selected_idx = torch.as_tensor(selected_idx, device=self.device)
            _inp = _inp[selected_idx].contiguous()
            _targets = _targets[selected_idx].contiguous()
            output_tensor = self.model(_inp)
            _, predicted = torch.max(output_tensor, 1)
            _acc = 100 * (predicted == _targets).sum().item() / _targets.size(0)
            logging.warning("Accuracy using selected top changed validation samples: {}".format(_acc))
            save_name = "{}samples_{}.pkl"
        else:
            sample_bs = sample_inp.shape[0]
            shape_target = next(iter(self.dataloader))[0][:sample_bs].to(self.device)
            sample_inp = sample_inp.to(self.device).reshape_as(shape_target) / wrappers[0].inp_scale
            _ = self.model(sample_inp)
            save_name = "{}samples_spec_inp_{}.pkl"

        # Save res and remove hooks via handlers
        os.makedirs(self.result_path, exist_ok=True)
        sample_path = os.path.join(self.result_path, save_name.format(n_samples, solver))
        with open(sample_path, "wb") as fp:
            pickle.dump(res, fp)
        for _h in handlers:
            _h.remove()

        logging.warning("Validation samples dumped to: {}".format(sample_path))


    @staticmethod
    def _select_top_changed_samples(traj, n_samples):
        traj_main = traj[0] if isinstance(traj, (tuple, list)) else traj
        start = traj_main[0]
        end = traj_main[-1]
        denom = np.maximum(np.abs(start), np.finfo(start.dtype).eps)
        change_ratio = np.mean(np.abs(start - end) / denom, axis=1)
        return np.argsort(-change_ratio)[:n_samples]

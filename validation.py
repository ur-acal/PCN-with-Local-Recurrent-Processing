import logging
import pickle
import types
from functools import wraps

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

    def enable_csv(self, v_grid, R_codes, R_left, R_slope, proj_fn, R):
        self.csv_enabled = True

        self.v_grid = v_grid
        self.R_codes = R_codes
        self.R_left = R_left
        self.R_slope = R_slope
        self.proj_fn = proj_fn
        self.R = R

        self.code_idx_mat = self._build_code_idx_mat()

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

    def _build_code_idx_mat(self):
        # Returns a list of tuples
        # The first element of the tuple is code idx
        # The second element of the tuple is a sparse matrix storing
        # the signs of values that corresponds to that code idx in the original matrix.
        mat_coo = self.mat.to_sparse_coo().coalesce()
        idx = mat_coo.indices()
        vals = mat_coo.values()

        code_idx_vals = self._values_to_code_idx(vals)  # (nnz,); Get the column index in the R_table.
        sign_vals = vals.sign()

        # Loop over all possible column indices.
        # Todo: This only works for R_table with limited number of columns (e.g. 15 for 5-bit quantization).
        mats = []
        uniq = torch.unique(code_idx_vals)
        for j in uniq.tolist():
            if j < 0:
                continue
            sel = (code_idx_vals == j)
            if sel.any():
                idx_j = idx[:, sel]
                val_j = sign_vals[sel].to(vals.dtype)
                # Sign only
                mat_j = torch.sparse_coo_tensor(
                    idx_j, val_j,
                    size=self.mat.shape,
                    device=self.mat.device,
                    dtype=self.mat.dtype
                ).coalesce().to_sparse_csr()
                mats.append((int(j), mat_j))
        return mats

    def _get_R_eff(self, v, code_idx):
        # Performs interpolation based on current input value and weight value.
        # v is the current spin state.
        # i,j is to pick a correct interpolant.
        if getattr(self, "proj_fn", None) is not None:
            v = self.proj_fn(v)

        _i = torch.bucketize(v, self.v_grid) - 1
        _i = _i.clamp(min=0, max=self.v_grid.numel() - 2)

        M = self.R_codes.numel()
        v_flat, i_flat = v.reshape(-1), _i.reshape(-1)

        if torch.is_tensor(code_idx):
            if code_idx.numel() == 1:
                j = int(code_idx.item())
                j = min(max(j, 0), M - 1)
                pos = i_flat * M + j
            else:
                j_flat = code_idx.to(torch.long).reshape(-1).clamp(0, M - 1)
                pos = i_flat * M + j_flat
        else:
            j = int(code_idx)
            j = min(max(j, 0), M - 1)
            pos = i_flat * M + j

        R_left_sel = self.R_left.reshape(-1)[pos]
        R_slope_sel = self.R_slope.reshape(-1)[pos]
        v_sel = self.v_grid[i_flat]

        R_eff_flat = R_left_sel + R_slope_sel * (v_flat - v_sel)
        return R_eff_flat.reshape_as(v)

    def forward(self, x):
        batch_size, input_channels, input_h, input_w = x.shape
        x = x.view(batch_size, -1).t()
        output_h = (input_h + 2 * self.meta["padding"] - self.meta["ker_h"]) // self.meta["stride"] + 1
        output_w = (input_w + 2 * self.meta["padding"] - self.meta["ker_w"]) // self.meta["stride"] + 1

        if not self.csv_enabled:
            return torch.sparse.mm(self.mat, x).t().view(batch_size, self.meta["out_chan"], output_h, output_w)

        out = None
        for (j, mat_j) in self.code_idx_mat:
            R_eff_j = self._get_R_eff(x, j)
            x_j = x / R_eff_j
            y_j = torch.sparse.mm(mat_j, x_j)
            out = y_j if out is None else out + y_j

        if out is None:
            out = torch.sparse.mm(self.mat, x)
        out = out * self.R
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
        super().__init__()
        # The model should contain conv layers only. All transposed conv layers should be converted to conv layers.
        # The model needs to be converted to ode blocks and wrapped with wrapper before.
        self.model = model
        self.model.eval()
        self.device = device
        self.exp_w_path = os.path.join(expanded_weight_dir, "expanded_weights_{}.pth")
        os.makedirs(expanded_weight_dir, exist_ok=True)
        self.dataloader = test_dataloader
        self.wrappers = wrapper
        self.record_full_traj = record_full_traj
        self.t_end_sf = t_end_sf

        self.unroll_or_load()
        self.result_path = result_path

    @torch.no_grad()
    def _register_hook_for_unroll(self, layer_idx, layer):
        exp_w_path = self.exp_w_path.format(layer_idx)
        if os.path.exists(exp_w_path):
            stored = torch.load(exp_w_path, map_location=self.device)
        else:
            stored = {}

        def make_hook(parent, mod_name, m):
            hook_handlers = {}
            def pre_hook(mod, inputs):
                # Register hook to unroll the conv weights
                # If the unrolled weights are already stored, load it
                # Otherwise perform unrolling
                if mod_name in stored:
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
                torch.save(stored, exp_w_path)

            def post_hook(mod, inputs, output):
                # Replace plain conv with unrolled weights
                unrolled, meta = stored[mod_name]["weight"], stored[mod_name]["meta"]
                mvm_conv = MVMConv(unrolled, meta)
                if hasattr(parent, "_nonlinear_R_pkg"):
                    mvm_conv.enable_csv(**parent._nonlinear_R_pkg)
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
        _ = self.model(next(iter(self.dataloader))[0][:2].to(self.device))

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
            res[cur_name]["q"] = wrappers[_idx].beta.cpu().item() if isinstance(wrappers[_idx].beta, torch.Tensor) else wrappers[_idx].beta
            res[cur_name]["k"] = getattr(wrappers[_idx], "k", None)
            res[cur_name]["beta_c"] = wrappers[_idx].beta_c.cpu().item() if hasattr(wrappers[_idx], "beta_c") else None
            res[cur_name]["C"] = wrappers[_idx].C
            res[cur_name]["C_ff"] = wrappers[_idx].C_ff.cpu().item() if hasattr(wrappers[_idx], "C_ff") else None
            res[cur_name]["C_fb"] = wrappers[_idx].C_fb if hasattr(wrappers[_idx], "C_fb") else None

            _inp_scale = wrappers[_idx].inp_scale
            _out_scale = wrappers[_idx].out_scale

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
    def gen_validate_data(self, wrappers, solver, n_samples=10, sample_inp=None):
        res, handlers = self._register_hook_for_record(wrappers)

        if sample_inp is None:
            _inp = next(iter(self.dataloader))[0][:n_samples].to(self.device)
            _ = self.model(_inp)
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

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

from example import handleError
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

    # Label on padded grid (strings "11","12",...,"k_h k_w")
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

    # Expand across channels & inject weights
    kernel_matrix = torch.zeros((C_out * rows, C_in * in_pad_size), dtype=kernel.dtype, device=kernel.device)
    base_mat_rows = rows
    for oc in range(C_out):
        for ic in range(C_in):
            start_out = oc * rows
            start_in = ic * in_pad_size
            ker_row = kernel[oc, ic].flatten()
            ker_tiled = ker_row.unsqueeze(0).expand(base_mat_rows, -1)
            kernel_matrix[start_out:start_out + rows, start_in:start_in + in_pad_size] = \
                mask_set_val(base_matrix_pad.clone(), ker_tiled)

    # Optionally drop padded columns (trim to raw input size)
    if not return_padded_input and p > 0:
        keep = torch.zeros((H_pad, W_pad), dtype=torch.bool, device=kernel.device)
        keep[p:H_pad - p, p:W_pad - p] = True
        keep = keep.flatten()  # bool mask over padded columns

        # Trim kernel_matrix per input-channel block
        blocks = []
        for ic in range(C_in):
            lo = ic * in_pad_size
            hi = lo + in_pad_size
            blocks.append(kernel_matrix[:, lo:hi][:, keep])
        kernel_matrix = torch.cat(blocks, dim=1)

        base_matrix = base_matrix_pad[:, keep]
        B_labeled = B_labeled_pad[:, keep.cpu().numpy()]
    else:
        base_matrix = base_matrix_pad
        B_labeled = B_labeled_pad

    return kernel_matrix, base_matrix, B_labeled


class MVMConv(nn.Module):
    def __init__(self, mat, meta):
        super().__init__()
        self.mat = mat
        self.meta = meta

    def forward(self, x):
        batch_size, input_channels, input_h, input_w = x.shape
        x = x.flatten()
        output_h = (input_h + 2 * self.meta["padding"] - self.meta["ker_h"]) // self.meta["stride"] + 1
        output_w = (input_w + 2 * self.meta["padding"] - self.meta["ker_w"]) // self.meta["stride"] + 1
        return self.mat.bmm(x).view(batch_size, self.meta["out_chan"], output_h, output_w)


class Validator(nn.Module):
    def __init__(self, model: PCNet, expanded_weight_dir, device, test_dataloader, result_path, **kwargs):
        super().__init__()
        # The model should contain conv layers only. All transposed conv layers should be converted to conv layers.
        # The model needs to be converted to ode blocks and wrapped with wrapper before.
        self.model = model
        self.model.eval()
        self.device = device
        self.exp_w_path = os.path.join(expanded_weight_dir, "expanded_weights_{}.pth")
        os.makedirs(expanded_weight_dir, exist_ok=True)
        self.dataloader = test_dataloader

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
                unrolled, meta = stored[mod_name]["weight"], stored[mod_name]["meta"]
                setattr(parent, mod_name, MVMConv(unrolled, meta))
                hook_handlers["pre_hook"].remove()
                hook_handlers["post_hook"].remove()

            hook_handlers["pre_hook"] = m.register_forward_pre_hook(pre_hook)
            hook_handlers["post_hook"] = m.register_forward_hook(post_hook)

        for _name, _mod in layer.named_modules():
            if isinstance(_mod, nn.Conv2d):
                make_hook(parent=layer, mod_name=_name, m=_mod)

    @torch.no_grad()
    def unroll_or_load(self):
        for _idx, _layer in enumerate(self.model.PcConvs):
            self._register_hook_for_unroll(_idx, _layer)

        # One forward pass to trigger the hooks and unroll the weights
        _ = self.model(next(iter(self.dataloader))[0][:2].to(self.device))

    @torch.no_grad()
    def test_unroll(self):
        _total, _correct = 0, 0
        for batch_idx, (inputs, targets) in tqdm(enumerate(self.dataloader), total=len(self.dataloader), disable=True):
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
            if getattr(orig_init_y, "__self__", None) is not None:
                bound_orig = orig_init_y
            else:
                bound_orig = types.MethodType(orig_init_y, mod)
            return update_init_y(bound_orig, x, *args, **kwargs)

        mod.init_y = patched_init_y
        setattr(mod, "_init_y_wrapped_for_capture", True)

    @torch.no_grad()
    def _register_hook_for_record(self, wrappers):
        res = {}
        handlers = []
        for _idx, _layer in enumerate(self.model.PcConvs):
            # dict_keys(['R', 'q', 'kernel_0', 'kernel_1', 'kernel_2', 'kernel_3'])
            # dict_keys(['inp', 'init_res', 'out', 'init_time', 'compute_time', 'FF_mat', 'FB_mat', 'C_ff', 'C_fb'])
            cur_name = "layer_{}".format(_idx)
            res[cur_name] = {}
            res[cur_name]["init_time"] = _layer.option_init["t1"].cpu().item()
            res[cur_name]["compute_time"] = _layer.option_aca["t1"].cpu().item()
            res[cur_name]["FF_mat"] = _layer.FFConv.mat.detach().cpu().numpy()
            res[cur_name]["FB_mat"] = _layer.FBConv.mat.detach().cpu().numpy()
            res[cur_name]["R"] = wrappers[_idx].R
            res[cur_name]["q"] = wrappers[_idx].beta.cpu().item() if isinstance(wrappers[_idx].beta, torch.Tensor) else wrappers[_idx].beta
            res[cur_name]["C_ff"] = wrappers[_idx].C_ff.cpu().item()
            res[cur_name]["C_fb"] = wrappers[_idx].C_fb

            def make_capture_init_res(key=cur_name):
                def capture_init_res(orig_init_y, x, *args, **kwargs):
                    yz = orig_init_y(x, *args, **kwargs)
                    res[key]["init_res"] = yz[-1].detach().cpu().numpy()
                    return yz
                return capture_init_res

            def pre_hook(mod, inputs, key=cur_name):
                res[key]["inp"] = inputs[0].detach().cpu().numpy()
                update_init_y = make_capture_init_res(key)
                self.patch_init_y_for_capture(mod, update_init_y)

            def post_hook(mod, inputs, output, key=cur_name):
                # res[key]["init_res"] = mod.init_y(inputs[0])[-1]
                res[key]["out"] = output.detach().cpu().numpy()

            handlers.append(_layer.register_forward_pre_hook(pre_hook))
            handlers.append(_layer.register_forward_hook(post_hook))
        return res, handlers

    @torch.no_grad()
    def gen_validate_data(self, wrappers, n_samples=10):
        res, handlers = self._register_hook_for_record(wrappers)

        _inp = next(iter(self.dataloader))[0][:n_samples].to(self.device)
        _ = self.model(_inp)

        # Save res and remove hooks via handlers
        with open(self.result_path, "wb") as fp:
            pickle.dump(res, fp)
        for _h in handlers:
            _h.remove()

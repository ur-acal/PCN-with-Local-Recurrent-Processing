"""Validator support for feedforward physical CNN blocks."""

from __future__ import annotations

import hashlib
import logging
import os

import torch
import torch.nn as nn

from physical_feedforward import (
    iter_physical_blocks,
    iter_physical_wrappers,
)
from validation import Validator


@torch.no_grad()
def feedforward_expanded_weight_cache_fingerprint(model, input_shape):
    """Fingerprint the exact effective convolutions expanded for this model."""
    digest = hashlib.sha256()
    digest.update(b"feedforward-validator-expanded-weights-v1\0")
    digest.update(repr(tuple(int(v) for v in input_shape)).encode("utf-8"))
    digest.update(b"\0")
    num_convs = 0
    for layer_idx, block in enumerate(iter_physical_blocks(model)):
        for module_name, module in block.named_modules():
            if not isinstance(module, nn.Conv2d):
                continue
            num_convs += 1
            geometry = (
                layer_idx, module_name, module.in_channels,
                module.out_channels, tuple(module.kernel_size),
                tuple(module.stride), tuple(module.padding),
                tuple(module.dilation), module.groups,
                module.padding_mode,
            )
            digest.update(repr(geometry).encode("utf-8"))
            digest.update(b"\0")
            weight = module.weight.detach().cpu().contiguous()
            digest.update(str(weight.dtype).encode("utf-8"))
            digest.update(repr(tuple(weight.shape)).encode("utf-8"))
            digest.update(weight.view(torch.uint8).numpy().tobytes())
            digest.update(b"\0")
    if num_convs == 0:
        raise ValueError("No physical feedforward Conv2d weights were found.")
    return digest.hexdigest()[:16]


class FeedForwardCNNValidator(Validator):
    """PCN Validator specialization for physical feedforward block models.

    The CSR construction, nonlinear-resistance assignment, DTC metadata, and
    cache format remain the existing Validator implementations.  Only model
    traversal and the dry-run hook trigger differ.
    """

    def __init__(self, model, expanded_weight_dir, device, test_dataloader,
                 result_path, wrapper=None, record_full_traj=False,
                 t_end_sf=1.0, **kwargs):
        nn.Module.__init__(self)
        if record_full_traj:
            raise NotImplementedError(
                "Feedforward full-trajectory recording is separate from "
                "PCN recurrent trajectory recording.")
        self.model = model
        self.model.eval()
        self.device = device
        self.dataloader = test_dataloader
        self._physical_blocks = list(iter_physical_blocks(model))
        if not self._physical_blocks:
            raise ValueError("Model has no physical feedforward blocks.")

        self._unroll_sample_inputs = next(iter(self.dataloader))[0][:2].to(
            self.device)
        self.expanded_weight_cache_fingerprint = (
            feedforward_expanded_weight_cache_fingerprint(
                self.model, self._unroll_sample_inputs.shape[1:]))
        self.legacy_exp_w_path = os.path.join(
            expanded_weight_dir, "feedforward_expanded_weights_{}.pth")
        expanded_weight_dir = os.path.join(
            expanded_weight_dir,
            "feedforward_cache_{}".format(
                self.expanded_weight_cache_fingerprint))
        self.exp_w_path = os.path.join(
            expanded_weight_dir, "expanded_weights_{}.pth")
        os.makedirs(expanded_weight_dir, exist_ok=True)
        logging.warning(
            "Feedforward expanded-weight cache fingerprint: %s (%s)",
            self.expanded_weight_cache_fingerprint, expanded_weight_dir)
        self.wrappers = wrapper
        self.record_full_traj = False
        self.t_end_sf = t_end_sf
        self.unroll_or_load()
        self.result_path = result_path

    @torch.no_grad()
    def unroll_or_load(self):
        for layer_idx, block in enumerate(self._physical_blocks):
            block._capture_dense_modules = True
            self._register_hook_for_unroll(layer_idx, block)

        # A physical feedforward block's capture mode calls each Conv2d exactly
        # once.  This triggers the existing Validator hooks without executing
        # a second physical inference or changing the ordinary runtime path.
        try:
            _ = self.model(self._unroll_sample_inputs)
        finally:
            for block in self._physical_blocks:
                block._capture_dense_modules = False
            self._unroll_sample_inputs = None
        # Apply persistent post-quantization mismatch to the expanded physical
        # couplers, matching the PCN validator/wrapper ordering.
        for wrapper in iter_physical_wrappers(self.model):
            wrapper.add_noise()

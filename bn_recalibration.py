"""Reusable BatchNorm recalibration for physical feedforward evaluation."""

from contextlib import contextmanager

import torch
import torch.nn as nn


_DYNAMIC_ENABLE_FLAGS = (
    "enable_summing_current_noise",
    "enable_coupler_noise",
    "enable_slow_summing_current",
    "enable_slow_coupler_noise",
)
_DYNAMIC_DTC_JITTERS = (
    "dtc_leading_edge_jitter_std",
    "dtc_falling_edge_jitter_std",
)
_DYNAMIC_GENERATOR_CACHES = (
    "_summing_noise_generators",
    "_coupler_noise_generators",
    "_slow_summing_current_generators",
    "_slow_coupler_noise_generators",
    "_dtc_timing_generators",
)


@contextmanager
def static_nonideality_calibration_mode(model):
    """Optional context that disables dynamic noise while retaining fixed effects.

    ``recalibrate_batchnorm`` intentionally does not use this context: its
    default hardware-calibration protocol keeps dynamic noise enabled.
    """
    saved = []
    for module in model.modules():
        for name in _DYNAMIC_ENABLE_FLAGS:
            if hasattr(module, name):
                saved.append((module, name, getattr(module, name)))
                setattr(module, name, False)
        for name in _DYNAMIC_DTC_JITTERS:
            if hasattr(module, name):
                saved.append((module, name, getattr(module, name)))
                setattr(module, name, 0.0)
    try:
        yield
    finally:
        for module, name, value in saved:
            setattr(module, name, value)
        # When this optional mode is used, start the subsequent accuracy pass
        # from fresh dynamic-noise generator streams.
        for module in model.modules():
            for name in _DYNAMIC_GENERATOR_CACHES:
                cache = getattr(module, name, None)
                if isinstance(cache, dict):
                    cache.clear()


@torch.no_grad()
def recalibrate_batchnorm(model, calibration_loader, device):
    """Recompute unfused BN running statistics for one hardware trial.

    The caller supplies augmentation-free training data. Quantized weights,
    trial-fixed hardware nonidealities, dynamic current noise, and DTC jitter
    all remain active. Calibration therefore advances the dynamic-noise streams;
    evaluation continues with newly sampled noise from those same streams.
    """
    batchnorms = [
        module for module in model.modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
        and module.track_running_stats
    ]
    if not batchnorms:
        return {"num_batchnorms": 0, "num_batches": 0, "num_samples": 0}

    model.eval()
    original_momenta = {module: module.momentum for module in batchnorms}
    for module in batchnorms:
        module.reset_running_stats()
        module.momentum = None
        module.train()

    num_batches = 0
    num_samples = 0
    try:
        for batch in calibration_loader:
            inputs = batch[0] if isinstance(batch, (tuple, list)) else batch
            inputs = inputs.to(device)
            model(inputs)
            num_batches += 1
            num_samples += int(inputs.shape[0])
    finally:
        for module, momentum in original_momenta.items():
            module.momentum = momentum
        model.eval()

    return {
        "num_batchnorms": len(batchnorms),
        "num_batches": num_batches,
        "num_samples": num_samples,
    }

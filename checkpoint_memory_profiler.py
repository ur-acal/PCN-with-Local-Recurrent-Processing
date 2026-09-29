"""Layer-selective ODE RHS checkpointing and automatic memory profiling."""

from __future__ import annotations

import copy
import logging
import math
import random
import time
from dataclasses import dataclass

import numpy as np
import torch


AUTO_CHECKPOINT_MEMORY_THRESHOLD = 0.8
AUTO_CHECKPOINT_PROFILE_BATCHES = 3
AUTO_CHECKPOINT_PORTION = "auto"
_OPTION_NAMES = ("option_init", "option_patch", "option_aca")
_RUNTIME_TENSOR_NAMES = {
    "_gaussian_curve_samples",
    "_tc_curve_samples",
    "_tc_samples",
    "_spin_factor_y",
    "_spin_factor_z",
    "_nonlinear_R_training_curve_indices",
}


def parse_checkpoint_ode_rhs_portion(value):
    """Parse a checkpoint portion in [0, 1], or the literal ``auto``."""
    if isinstance(value, str) and value.strip().lower() == AUTO_CHECKPOINT_PORTION:
        return AUTO_CHECKPOINT_PORTION
    try:
        portion = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "checkpoint_ode_rhs_portion must be a number in [0, 1] or 'auto'") from exc
    if not math.isfinite(portion) or not 0.0 <= portion <= 1.0:
        raise ValueError(
            "checkpoint_ode_rhs_portion must be a finite number in [0, 1]")
    return portion


def checkpoint_layer_count(portion, num_layers):
    """Round a numeric portion to the nearest layer using round-half-up."""
    if num_layers < 0:
        raise ValueError("num_layers must be non-negative")
    parsed = parse_checkpoint_ode_rhs_portion(portion)
    if parsed == AUTO_CHECKPOINT_PORTION:
        return num_layers
    return min(num_layers, max(0, int(math.floor(parsed * num_layers + 0.5))))


def _unwrap_model(model):
    while hasattr(model, "module"):
        model = model.module
    return model


def checkpoint_ode_rhs_layers(model):
    """Return checkpoint-selectable ODE layers in forward order."""
    model = _unwrap_model(model)
    pc_layers = getattr(model, "PcConvs", None)
    if pc_layers is not None:
        return [layer for layer in pc_layers
                if hasattr(layer, "checkpoint_ode_rhs_training")]

    layers = []
    seen = set()
    for module in model.modules():
        if id(module) in seen:
            continue
        if not hasattr(module, "checkpoint_ode_rhs_training"):
            continue
        if not (hasattr(module, "layer_idx") or hasattr(module, "option_aca")):
            continue
        seen.add(id(module))
        layers.append(module)
    layers.sort(key=lambda layer: int(getattr(layer, "layer_idx", len(layers))))
    return layers


def _hook_owner(hook):
    owner = getattr(hook, "__self__", None)
    if owner is not None:
        return owner
    wrapped = getattr(hook, "hook", None)
    return getattr(wrapped, "__self__", None)


def _set_option(options, enabled):
    if isinstance(options, dict):
        options["checkpoint_ode_rhs_training"] = bool(enabled)


def set_checkpoint_ode_rhs_block(block, enabled):
    """Update one live block and any wrapper snapshots that can replace it."""
    enabled = bool(enabled)
    block.checkpoint_ode_rhs_training = enabled
    for name in _OPTION_NAMES:
        _set_option(getattr(block, name, None), enabled)

    for hook in getattr(block, "_forward_pre_hooks", {}).values():
        owner = _hook_owner(hook)
        if owner is None or getattr(owner, "ode_block", block) is not block:
            continue
        for name in _OPTION_NAMES:
            _set_option(getattr(owner, "orig_" + name, None), enabled)


def set_checkpoint_ode_rhs_layer_count(model, count):
    """Checkpoint the first ``count`` ODE layers and return selection metadata."""
    layers = checkpoint_ode_rhs_layers(model)
    count = int(count)
    if not 0 <= count <= len(layers):
        raise ValueError(
            f"checkpoint layer count {count} is outside [0, {len(layers)}]")
    for index, block in enumerate(layers):
        set_checkpoint_ode_rhs_block(block, index < count)

    model = _unwrap_model(model)
    model.checkpoint_ode_rhs_selected_layers = count
    model.checkpoint_ode_rhs_total_layers = len(layers)
    model.checkpoint_ode_rhs_effective_portion = (
        float(count) / len(layers) if layers else 0.0)
    return {
        "selected_layers": count,
        "total_layers": len(layers),
        "effective_portion": model.checkpoint_ode_rhs_effective_portion,
        "selected_indices": list(range(count)),
    }


def apply_checkpoint_ode_rhs_portion(model, enabled, portion):
    """Apply a manual portion, or the all-on starting point for ``auto``."""
    parsed = parse_checkpoint_ode_rhs_portion(portion)
    layers = checkpoint_ode_rhs_layers(model)
    count = checkpoint_layer_count(parsed, len(layers)) if enabled else 0
    metadata = set_checkpoint_ode_rhs_layer_count(model, count)
    metadata.update(requested=parsed, enabled=bool(enabled))
    return metadata


def configure_trainer_checkpointing(trainer, *, enabled, portion,
                                    memory_fraction=1.0):
    """Install the requested policy on a constructed trainer and its model."""
    parsed = parse_checkpoint_ode_rhs_portion(portion)
    if parsed == AUTO_CHECKPOINT_PORTION and not enabled:
        raise ValueError(
            "checkpoint_ode_rhs_portion=auto requires "
            "checkpoint_ode_rhs_training=true")
    if not 0.0 < float(memory_fraction) <= 1.0:
        raise ValueError("memory_fraction must be in (0, 1]")
    trainer.checkpoint_ode_rhs_training = bool(enabled)
    trainer.checkpoint_ode_rhs_portion = parsed
    trainer.checkpoint_memory_fraction = float(memory_fraction)
    metadata = apply_checkpoint_ode_rhs_portion(
        trainer.model, bool(enabled), parsed)
    logging.warning(
        "ODE RHS checkpoint selection: requested=%s, selected=%d/%d, indices=%s",
        parsed, metadata["selected_layers"], metadata["total_layers"],
        metadata["selected_indices"])
    return metadata


def find_minimum_safe_checkpoint_layers(num_layers, is_safe):
    """Return the smallest safe prefix using a monotone integer search."""
    if num_layers < 1:
        raise ValueError("num_layers must be positive")
    if not is_safe(num_layers):
        raise RuntimeError(
            "Full ODE RHS checkpointing cannot satisfy the memory target.")
    low, high = 0, num_layers
    while low < high:
        middle = (low + high) // 2
        if is_safe(middle):
            high = middle
        else:
            low = middle + 1
    return low


def _clone_to_cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return type(value)((key, _clone_to_cpu(item))
                           for key, item in value.items())
    if isinstance(value, list):
        return [_clone_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_to_cpu(item) for item in value)
    return copy.deepcopy(value)


def _component_state(component):
    return None if component is None else _clone_to_cpu(component.state_dict())


def _capture_runtime(model):
    runtime = []
    for module in model.modules():
        attributes = {}
        for name, value in vars(module).items():
            if isinstance(value, torch.Generator):
                attributes[name] = (
                    "generator", (value, value.get_state().cpu()))
            elif isinstance(value, dict) and (
                    name.endswith("_generators") or
                    any(isinstance(item, torch.Generator)
                        for item in value.values())):
                attributes[name] = (
                    "generator_dict",
                    {key: (item, item.get_state().cpu())
                     for key, item in value.items()
                     if isinstance(item, torch.Generator)})
            elif (name in _RUNTIME_TENSOR_NAMES or
                  name.endswith("_samples")):
                attributes[name] = ("value", _clone_to_cpu(value))
        runtime.append((module, attributes))
    return runtime


def _restore_runtime(runtime, device):
    for module, attributes in runtime:
        for name, (kind, value) in attributes.items():
            if kind == "generator":
                generator, state = value
                generator.set_state(state)
                setattr(module, name, generator)
            elif kind == "generator_dict":
                generators = getattr(module, name, None)
                if isinstance(generators, dict):
                    generators.clear()
                    for key, (generator, state) in value.items():
                        generator.set_state(state)
                        generators[key] = generator
            else:
                def to_device(item):
                    if torch.is_tensor(item):
                        return item.to(device)
                    if isinstance(item, dict):
                        return {key: to_device(entry)
                                for key, entry in item.items()}
                    if isinstance(item, list):
                        return [to_device(entry) for entry in item]
                    if isinstance(item, tuple):
                        return tuple(to_device(entry) for entry in item)
                    return item
                setattr(module, name, to_device(value))


def _clear_checkpoint_caches(model):
    for module in model.modules():
        clear = getattr(module, "_clear_pending_qat_weight_cache", None)
        if clear is not None:
            clear()
        for quantizer in getattr(module, "parametrizations", {}).values() \
                if hasattr(getattr(module, "parametrizations", None), "values") else ():
            for parametrization in quantizer:
                if hasattr(parametrization, "_solve_cache_enabled"):
                    parametrization._solve_cache_enabled = False
                    parametrization._solve_cached_weight = None


class _TrainerSnapshot:
    def __init__(self, trainer):
        self.model = _clone_to_cpu(trainer.model.state_dict())
        self.optimizer = _component_state(trainer.optimizer)
        self.components = {
            name: _component_state(getattr(trainer, name, None))
            for name in ("scheduler", "warmup_scheduler", "grad_scaler", "scaler")
        }
        self.aux = {
            name: _component_state(getattr(trainer, name, None))
            for name in ("_feature_kd_loss", "_crd_loss")
            if getattr(trainer, name, None) is not None
        }
        self.aux_presence = {
            name: getattr(trainer, name, None) is not None
            for name in ("_feature_kd_loss", "_crd_loss")
        }
        self.crd_initialized = getattr(trainer, "_crd_initialized", None)
        self.optimizer_group_count = len(trainer.optimizer.param_groups)
        self.gradients = [
            (parameter, None if parameter.grad is None
             else parameter.grad.detach().cpu().clone())
            for parameter in trainer.model.parameters()
        ]
        self.training_modes = [(module, module.training)
                               for module in trainer.model.modules()]
        self.runtime = _capture_runtime(trainer.model)
        self.python_rng = random.getstate()
        self.numpy_rng = np.random.get_state()
        self.torch_rng = torch.get_rng_state()
        self.cuda_rng = (torch.cuda.get_rng_state_all()
                         if torch.cuda.is_initialized() else None)
        self.loader = trainer.train_dataloader
        generator = getattr(self.loader, "generator", None)
        self.loader_rng = None if generator is None else generator.get_state().cpu()

    def restore(self, trainer):
        _clear_checkpoint_caches(trainer.model)
        if len(trainer.optimizer.param_groups) < self.optimizer_group_count:
            raise RuntimeError(
                "Automatic checkpoint profiling lost an optimizer parameter group.")
        if len(trainer.optimizer.param_groups) > self.optimizer_group_count:
            del trainer.optimizer.param_groups[self.optimizer_group_count:]
            retained = {
                parameter
                for group in trainer.optimizer.param_groups
                for parameter in group["params"]
            }
            for parameter in list(trainer.optimizer.state):
                if parameter not in retained:
                    del trainer.optimizer.state[parameter]
        for name, was_present in self.aux_presence.items():
            if not was_present:
                setattr(trainer, name, None)
        if self.crd_initialized is not None:
            trainer._crd_initialized = self.crd_initialized
        trainer.model.load_state_dict(self.model, strict=True)
        for name, state in self.aux.items():
            module = getattr(trainer, name, None)
            if module is None:
                raise RuntimeError(
                    f"Automatic checkpoint profiling lost auxiliary module {name}.")
            module.load_state_dict(state)
        trainer.optimizer.load_state_dict(self.optimizer)
        for name, state in self.components.items():
            component = getattr(trainer, name, None)
            if state is not None and component is not None:
                component.load_state_dict(state)
        for parameter, gradient in self.gradients:
            parameter.grad = (None if gradient is None else
                              gradient.to(device=parameter.device,
                                          dtype=parameter.dtype))
        for module, training in self.training_modes:
            module.training = training
        device = next(trainer.model.parameters()).device
        _restore_runtime(self.runtime, device)
        random.setstate(self.python_rng)
        np.random.set_state(self.numpy_rng)
        torch.set_rng_state(self.torch_rng)
        if self.cuda_rng is not None:
            torch.cuda.set_rng_state_all(self.cuda_rng)
        generator = getattr(self.loader, "generator", None)
        if self.loader_rng is not None and generator is not None:
            generator.set_state(self.loader_rng)


@dataclass
class CheckpointProfileTrial:
    layers: int
    safe: bool
    oom: bool
    peak_reserved: int | None
    peak_allocated: int | None
    elapsed_seconds: float


class _ProfiledFirstEpochLoader:
    """Replay profiled batches, then continue their live worker iterator."""

    def __init__(self, loader, batches, iterator):
        self.loader = loader
        self.batches = batches
        self.iterator = iterator
        self.used = False

    def __len__(self):
        return len(self.loader)

    def __getattr__(self, name):
        return getattr(self.loader, name)

    def __iter__(self):
        if not self.used:
            self.used = True
            yield from self.batches
            yield from self.iterator
        else:
            yield from self.loader


def _is_cuda_oom(error):
    return isinstance(error, torch.cuda.OutOfMemoryError) or (
        isinstance(error, RuntimeError) and
        "out of memory" in str(error).lower())


def _usable_cuda_capacity(memory_fraction):
    free, total = torch.cuda.mem_get_info()
    reserved = torch.cuda.memory_reserved()
    fraction_capacity = int(total * float(memory_fraction))
    contention_capacity = int(free + reserved)
    return min(fraction_capacity, contention_capacity), int(total), int(free)


def _profile_candidate(trainer, snapshot, batches, layer_count, limit_bytes):
    snapshot.restore(trainer)
    trainer.train_dataloader = _clone_to_cpu(batches)
    set_checkpoint_ode_rhs_layer_count(trainer.model, layer_count)
    trainer.optimizer.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    oom = False
    peak_reserved = None
    peak_allocated = None
    try:
        trainer.train_one_epoch(0)
        torch.cuda.synchronize()
        peak_reserved = int(torch.cuda.max_memory_reserved())
        peak_allocated = int(torch.cuda.max_memory_allocated())
    except BaseException as error:
        if not _is_cuda_oom(error):
            raise
        oom = True
    elapsed = time.perf_counter() - started
    safe = (not oom and peak_reserved is not None and
            peak_reserved <= limit_bytes)
    trainer.optimizer.zero_grad(set_to_none=True)
    _clear_checkpoint_caches(trainer.model)
    torch.cuda.empty_cache()
    snapshot.restore(trainer)
    trainer.optimizer.zero_grad(set_to_none=True)
    _clear_checkpoint_caches(trainer.model)
    torch.cuda.empty_cache()
    return CheckpointProfileTrial(
        layers=layer_count, safe=safe, oom=oom,
        peak_reserved=peak_reserved, peak_allocated=peak_allocated,
        elapsed_seconds=elapsed)


def maybe_profile_checkpoint_ode_rhs(trainer):
    """Choose the smallest safe checkpointed prefix for ``portion=auto``."""
    portion = getattr(trainer, "checkpoint_ode_rhs_portion", 1.0)
    enabled = bool(getattr(trainer, "checkpoint_ode_rhs_training", False))
    if portion != AUTO_CHECKPOINT_PORTION:
        return None
    if not enabled:
        raise ValueError(
            "checkpoint_ode_rhs_portion=auto requires "
            "checkpoint_ode_rhs_training=true")
    if not torch.cuda.is_available():
        raise RuntimeError("Automatic checkpoint memory profiling requires CUDA.")

    layers = checkpoint_ode_rhs_layers(trainer.model)
    if not layers:
        logging.warning(
            "Automatic RHS-checkpoint profiling skipped: the model has no "
            "eligible ODE layers.")
        return {
            "selected_layers": 0,
            "total_layers": 0,
            "effective_portion": 0.0,
            "selected_indices": [],
            "threshold": AUTO_CHECKPOINT_MEMORY_THRESHOLD,
            "profile_batches": 0,
            "trials": [],
        }

    original_loader = trainer.train_dataloader
    worker_count = int(getattr(original_loader, "num_workers", 0))
    # With no workers, restoring the pre-iterator RNG state recreates the same
    # first epoch. Worker RNG state cannot be rewound from the parent process,
    # so multi-worker loaders retain their live iterator and replay the three
    # batches already taken from it.
    snapshot = _TrainerSnapshot(trainer) if worker_count == 0 else None
    iterator = iter(original_loader)
    batches = []
    try:
        for _ in range(AUTO_CHECKPOINT_PROFILE_BATCHES):
            batches.append(next(iterator))
    except StopIteration as exc:
        raise RuntimeError(
            f"Automatic checkpoint profiling requires at least "
            f"{AUTO_CHECKPOINT_PROFILE_BATCHES} training batches.") from exc
    if snapshot is None:
        snapshot = _TrainerSnapshot(trainer)

    trainer.train_dataloader = batches
    torch.cuda.empty_cache()
    capacity, physical_total, initial_free = _usable_cuda_capacity(
        getattr(trainer, "checkpoint_memory_fraction", 1.0))
    limit = int(AUTO_CHECKPOINT_MEMORY_THRESHOLD * capacity)
    trials = []

    def evaluate(count):
        trial = _profile_candidate(
            trainer, snapshot, batches, count, limit)
        trials.append(trial)
        peak = "OOM" if trial.oom else f"{trial.peak_reserved / 2**30:.2f} GiB"
        logging.warning(
            "Automatic RHS-checkpoint trial: layers=%d/%d, peak_reserved=%s, "
            "limit=%.2f GiB, safe=%s",
            count, len(layers), peak, limit / 2**30, trial.safe)
        return trial.safe

    try:
        try:
            selected = find_minimum_safe_checkpoint_layers(
                len(layers), evaluate)
        except RuntimeError as exc:
            raise RuntimeError(
                "Full ODE RHS checkpointing cannot keep the three-batch "
                f"profile below {AUTO_CHECKPOINT_MEMORY_THRESHOLD:.0%} of "
                "usable CUDA memory.") from exc

        # A separate confirmation uses the same three batches and pristine state.
        if not evaluate(selected):
            raise RuntimeError(
                "The automatically selected checkpoint portion failed its "
                "three-batch confirmation.")
    finally:
        trainer.train_dataloader = original_loader
        snapshot.restore(trainer)
        trainer.optimizer.zero_grad(set_to_none=True)
        _clear_checkpoint_caches(trainer.model)
        torch.cuda.empty_cache()

    metadata = set_checkpoint_ode_rhs_layer_count(trainer.model, selected)
    if worker_count > 0:
        trainer.train_dataloader = _ProfiledFirstEpochLoader(
            original_loader, batches, iterator)
    metadata.update(
        threshold=AUTO_CHECKPOINT_MEMORY_THRESHOLD,
        profile_batches=AUTO_CHECKPOINT_PROFILE_BATCHES,
        usable_capacity=capacity,
        physical_total=physical_total,
        initial_free=initial_free,
        trials=[trial.__dict__.copy() for trial in trials],
    )
    trainer.checkpoint_ode_rhs_profile = metadata
    recovery_config = getattr(trainer, "recovery_config", None)
    if isinstance(recovery_config, dict):
        recovery_config["checkpoint_ode_rhs_selected_layers"] = selected
        recovery_config["checkpoint_ode_rhs_effective_portion"] = metadata[
            "effective_portion"]
    logging.warning(
        "Automatic ODE RHS checkpoint selection: %d/%d layers "
        "(effective portion %.6f), threshold %.0f%% of %.2f GiB",
        selected, len(layers), metadata["effective_portion"],
        100 * AUTO_CHECKPOINT_MEMORY_THRESHOLD, capacity / 2**30)
    return metadata

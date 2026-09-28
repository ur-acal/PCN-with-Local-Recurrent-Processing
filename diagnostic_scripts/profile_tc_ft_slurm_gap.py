#!/usr/bin/env python3
"""Bounded, production-path diagnostic for the TC CIFAR fine-tuning gap.

This script receives the exact arguments assembled by
``finetune_one_combo_model``.  It lets ``train_ode_cifar.main`` construct the
real model, teacher, data loaders, wrappers, nonidealities, and optimizer.  The
only behavioral substitution is that ``TrainerCiFarTimmStyle.train`` runs a
bounded prefix of one real training epoch and does not write checkpoints.

Two modes are intentionally separate:

* ``timing``: synchronization only at batch boundaries; suitable for comparing
  end-to-end iteration time with the production log.
* ``profile``: component synchronization, adaptive-solver counters, and a
  PyTorch profiler.  Its timings are diagnostic and must not be compared with
  the uninstrumented production rate.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
MODE = os.environ.get("TC_FT_DIAG_MODE", "timing").strip().lower()
if MODE not in {"timing", "profile"}:
    raise ValueError("TC_FT_DIAG_MODE must be timing or profile")
OUTPUT = Path(os.environ.get("TC_FT_DIAG_OUTPUT", ROOT / "tc_ft_diagnostic"))
MAX_BATCHES = int(os.environ.get(
    "TC_FT_DIAG_BATCHES", "20" if MODE == "timing" else "1"))
WARMUP_BATCHES = int(os.environ.get("TC_FT_DIAG_WARMUP_BATCHES", "2"))
SAVE_TRACE = os.environ.get("TC_FT_DIAG_CHROME_TRACE", "false").lower() == "true"
OUTPUT.mkdir(parents=True, exist_ok=True)


def _jsonable(value: Any, depth: int = 0) -> Any:
    if depth > 7:
        return repr(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if torch.is_tensor(value):
        if value.numel() <= 16:
            return {
                "type": "tensor", "shape": list(value.shape),
                "dtype": str(value.dtype), "device": str(value.device),
                "value": value.detach().cpu().tolist(),
            }
        return {
            "type": "tensor", "shape": list(value.shape),
            "dtype": str(value.dtype), "device": str(value.device),
        }
    if isinstance(value, dict):
        return {str(k): _jsonable(v, depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v, depth + 1) for v in value]
    return repr(value)


RESULT: dict[str, Any] = {
    "status": "starting",
    "mode": MODE,
    "max_batches": MAX_BATCHES,
    "warmup_batches_excluded_from_summary": WARMUP_BATCHES,
    "argv": sys.argv,
    "output": str(OUTPUT),
    "started_unix": time.time(),
}


def _write_result() -> None:
    temporary = OUTPUT / "diagnostic.json.tmp"
    temporary.write_text(json.dumps(_jsonable(RESULT), indent=2, sort_keys=True))
    temporary.replace(OUTPUT / "diagnostic.json")


def _run(command: list[str], timeout: int = 30) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            command, cwd=ROOT, text=True, capture_output=True,
            timeout=timeout, check=False)
        return {
            "command": command, "returncode": proc.returncode,
            "stdout": proc.stdout, "stderr": proc.stderr,
        }
    except Exception as exc:  # diagnostic collection must not abort training
        return {"command": command, "error": repr(exc)}


def _relevant_environment() -> dict[str, str]:
    prefixes = (
        "SLURM_", "CUDA_", "NVIDIA_", "CONDA_", "OMP_", "MKL_",
        "OPENBLAS_", "PYTORCH_", "TORCH_", "TRITON_", "TC_",
        "REUSE_", "CHECKPOINT_",
    )
    exact = {"PATH", "LD_LIBRARY_PATH", "HOSTNAME"}
    return {
        key: value for key, value in sorted(os.environ.items())
        if key in exact or key.startswith(prefixes)
    }


def _collect_environment() -> dict[str, Any]:
    gpu: dict[str, Any] = {"available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        gpu.update({
            "name": props.name,
            "capability": list(torch.cuda.get_device_capability(0)),
            "total_memory_bytes": props.total_memory,
            "multi_processor_count": props.multi_processor_count,
            "cuda_runtime": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
        })
    packages = {}
    for name in ("torchvision", "timm", "triton", "numpy"):
        try:
            module = __import__(name)
            packages[name] = getattr(module, "__version__", "unknown")
        except Exception as exc:
            packages[name] = f"unavailable: {exc!r}"
    try:
        affinity = sorted(os.sched_getaffinity(0))
    except Exception as exc:
        affinity = repr(exc)
    commands = {
        "git_head": _run(["git", "rev-parse", "HEAD"]),
        "git_branch": _run(["git", "branch", "--show-current"]),
        "git_status": _run(["git", "status", "--short"]),
        "git_diff_stat": _run(["git", "diff", "--stat"]),
        "lscpu": _run(["lscpu"]),
        "taskset": _run(["taskset", "-pc", str(os.getpid())]),
        "numactl": _run(["numactl", "--show"]),
        "nvidia_smi_query": _run([
            "nvidia-smi", "--query-gpu=index,name,uuid,pci.bus_id,pstate,"
            "memory.total,memory.used,utilization.gpu,utilization.memory,"
            "temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem,"
            "clocks.max.sm,clocks.max.mem,driver_version",
            "--format=csv,noheader,nounits",
        ]),
        "nvidia_smi_topology": _run(["nvidia-smi", "topo", "-m"]),
        "nvidia_smi_processes": _run([
            "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ]),
    }
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "torch": torch.__version__,
        "packages": packages,
        "gpu": gpu,
        "cpu_count": os.cpu_count(),
        "cpu_affinity": affinity,
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "backend": {
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "allow_tf32_cudnn": torch.backends.cudnn.allow_tf32,
            "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        },
        "environment": _relevant_environment(),
        "torch_config": torch.__config__.show(),
        "commands": commands,
    }


def _hash_batch(batch: Any) -> str:
    """Cheap batch fingerprint; sample large tensors to avoid perturbing timing."""
    digest = hashlib.sha256()

    def update(value: Any) -> None:
        if torch.is_tensor(value):
            tensor = value.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode())
            digest.update(str(tuple(tensor.shape)).encode())
            raw = tensor.numpy().view("uint8").reshape(-1)
            if raw.size <= 8192:
                digest.update(raw.tobytes())
            else:
                digest.update(raw[:4096].tobytes())
                digest.update(raw[-4096:].tobytes())
        elif isinstance(value, (list, tuple)):
            digest.update(type(value).__name__.encode())
            for item in value:
                update(item)
        elif isinstance(value, dict):
            for key in sorted(value):
                digest.update(str(key).encode())
                update(value[key])
        else:
            digest.update(repr(value).encode())

    update(batch)
    return digest.hexdigest()


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class _BoundedTimedLoader:
    """Yield a real loader prefix while measuring complete loop iterations."""

    def __init__(self, loader: Any, limit: int, records: list[dict[str, Any]],
                 on_first_batch=None, on_last_batch=None):
        self.loader = loader
        self.limit = min(limit, len(loader))
        self.records = records
        self.on_first_batch = on_first_batch
        self.on_last_batch = on_last_batch

    def __len__(self) -> int:
        # Preserve the production 390-batch epoch semantics used by progress,
        # warmup calculations, and scheduler setup.
        return len(self.loader)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.loader, name)

    def __iter__(self):
        raw = iter(self.loader)
        active_start = None
        previous = None
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        for index in range(self.limit):
            if active_start is not None:
                _sync()
                previous["compute_seconds"] = time.perf_counter() - active_start
                previous["allocated_bytes_after"] = torch.cuda.memory_allocated()
                previous["reserved_bytes_after"] = torch.cuda.memory_reserved()
                previous["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
                previous["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
                if self.on_last_batch is not None:
                    self.on_last_batch(index - 1)

            _sync()
            wait_start = time.perf_counter()
            batch = next(raw)
            data_wait = time.perf_counter() - wait_start
            hash_start = time.perf_counter()
            batch_hash = _hash_batch(batch)
            hash_seconds = time.perf_counter() - hash_start
            previous = {
                "index": index,
                "data_wait_seconds": data_wait,
                "batch_hash_sha256": batch_hash,
                "batch_hash_seconds_excluded": hash_seconds,
            }
            self.records.append(previous)
            if index == 0 and self.on_first_batch is not None:
                self.on_first_batch()
            active_start = time.perf_counter()
            yield batch

        if active_start is not None:
            _sync()
            previous["compute_seconds"] = time.perf_counter() - active_start
            previous["allocated_bytes_after"] = torch.cuda.memory_allocated()
            previous["reserved_bytes_after"] = torch.cuda.memory_reserved()
            previous["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            previous["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
            if self.on_last_batch is not None:
                self.on_last_batch(self.limit - 1)


class _ComponentTimers:
    def __init__(self):
        self.records: dict[str, list[float]] = defaultdict(list)
        self.restorers: list[Any] = []

    def wrap_method(self, obj: Any, name: str, label: str) -> None:
        original = getattr(obj, name)

        def wrapped(*args, **kwargs):
            _sync()
            start = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                _sync()
                self.records[label].append(time.perf_counter() - start)

        setattr(obj, name, wrapped)
        self.restorers.append(lambda: setattr(obj, name, original))

    def install(self, trainer: Any) -> None:
        for name, label in (
            ("teacher_forward_for_distillation", "teacher_forward"),
            ("_student_forward_feature_kd", "student_forward"),
            ("_compute_feature_kd_loss", "feature_kd_loss"),
        ):
            if hasattr(trainer, name):
                self.wrap_method(trainer, name, label)
        self.wrap_method(trainer.optimizer, "step", "optimizer_step")
        original = torch.autograd.backward

        def timed_backward(*args, **kwargs):
            _sync()
            start = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                _sync()
                self.records["backward"].append(time.perf_counter() - start)

        torch.autograd.backward = timed_backward
        self.restorers.append(lambda: setattr(torch.autograd, "backward", original))

    def restore(self) -> None:
        for restore in reversed(self.restorers):
            restore()


class _SolverCounters:
    """Count production adaptive solves without replacing the solve function."""

    def __init__(self):
        self.rows: list[dict[str, Any]] = []
        self.sequence = 0
        self.restorers: list[Any] = []
        self.rhs_checkpoint_calls = {"grad_enabled": 0, "no_grad": 0}

    def install(self) -> None:
        from TorchDiffEqPack.odesolver.adaptive_grid_solver import (
            AdaptiveGridSolver, Dopri5, ProjDopri5, RK12, RK23)
        from TorchDiffEqPack.odesolver.base import RHSCheckpointWrapper

        original_integrate = AdaptiveGridSolver.integrate_search_grids

        def integrate(solver, *args, **kwargs):
            row = {
                "solve_index": self.sequence,
                "solver_class": type(solver).__name__,
                "method_order": getattr(solver, "order", None),
                "reuse_requested": getattr(solver, "reuse_accepted_step_training", None),
                "reuse_safe": getattr(solver, "accepted_step_reuse_safe", None),
                "checkpoint_requested": getattr(solver, "checkpoint_ode_rhs_training", None),
                "checkpoint_active": getattr(solver, "checkpoint_ode_rhs_active", None),
                "rhs_wrapper_class": type(solver.func).__name__,
                "has_tc_context": getattr(solver, "tc_context", None) is not None,
                "supports_reuse": getattr(solver, "supports_accepted_step_reuse", False),
                "regenerate_graph": getattr(solver, "regenerate_graph", None),
                "has_energy_meter": getattr(solver, "energy_meter", None) is not None,
                "rtol": _jsonable(getattr(solver, "rtol", None)),
                "atol": _jsonable(getattr(solver, "atol", None)),
                "step_calls": 0,
                "step_calls_grad_enabled": 0,
                "step_calls_no_grad": 0,
                "controller_accepts": 0,
                "controller_rejects": 0,
            }
            solver._tc_ft_diag_row = row
            self.rows.append(row)
            self.sequence += 1
            try:
                return original_integrate(solver, *args, **kwargs)
            finally:
                row["neval"] = getattr(solver, "neval", None)
                row["final_h"] = _jsonable(getattr(solver, "h", None))

        AdaptiveGridSolver.integrate_search_grids = integrate
        self.restorers.append(lambda: setattr(
            AdaptiveGridSolver, "integrate_search_grids", original_integrate))

        original_adapt = AdaptiveGridSolver.adapt_stepsize

        def adapt(solver, *args, **kwargs):
            result = original_adapt(solver, *args, **kwargs)
            row = getattr(solver, "_tc_ft_diag_row", None)
            if row is None:
                row = {}
                solver._tc_ft_diag_row = row
                self.rows.append(row)
            if result[1]:
                row["controller_accepts"] = row.get("controller_accepts", 0) + 1
            else:
                row["controller_rejects"] = row.get("controller_rejects", 0) + 1
            return result

        AdaptiveGridSolver.adapt_stepsize = adapt
        self.restorers.append(lambda: setattr(
            AdaptiveGridSolver, "adapt_stepsize", original_adapt))

        for cls in (RK12, RK23, Dopri5, ProjDopri5):
            if "step" not in cls.__dict__:
                continue
            original_step = cls.__dict__["step"]

            def make_step(original):
                def step(solver, *args, **kwargs):
                    row = getattr(solver, "_tc_ft_diag_row", None)
                    if row is None:
                        row = {}
                        solver._tc_ft_diag_row = row
                        self.rows.append(row)
                    row["step_calls"] = row.get("step_calls", 0) + 1
                    grad_key = (
                        "step_calls_grad_enabled" if torch.is_grad_enabled()
                        else "step_calls_no_grad")
                    row[grad_key] = row.get(grad_key, 0) + 1
                    return original(solver, *args, **kwargs)
                return step

            cls.step = make_step(original_step)
            self.restorers.append(lambda cls=cls, original=original_step: setattr(
                cls, "step", original))

        original_rhs = RHSCheckpointWrapper.forward

        def rhs_forward(wrapper, *args, **kwargs):
            key = "grad_enabled" if torch.is_grad_enabled() else "no_grad"
            self.rhs_checkpoint_calls[key] += 1
            return original_rhs(wrapper, *args, **kwargs)

        RHSCheckpointWrapper.forward = rhs_forward
        self.restorers.append(lambda: setattr(
            RHSCheckpointWrapper, "forward", original_rhs))

    def restore(self) -> None:
        for restore in reversed(self.restorers):
            restore()


def _option_audit(model: torch.nn.Module) -> list[dict[str, Any]]:
    rows = []
    option_names = (
        "option_aca", "option_init", "option_patch", "orig_option_aca",
    )
    for name, module in model.named_modules():
        options = {
            key: _jsonable(getattr(module, key))
            for key in option_names if hasattr(module, key)
        }
        if options:
            live = getattr(module, "option_aca", {})
            rows.append({
                "module": name,
                "class": type(module).__name__,
                "layer_idx": getattr(module, "layer_idx", None),
                "live_reuse": live.get("reuse_accepted_step_training"),
                "live_checkpoint": live.get("checkpoint_ode_rhs_training"),
                "options": options,
            })
    return rows


def _summarize_batches(records: list[dict[str, Any]]) -> dict[str, Any]:
    selected = records[min(WARMUP_BATCHES, len(records)):]
    if not selected:
        selected = records
    times = [row["compute_seconds"] for row in selected if "compute_seconds" in row]
    waits = [row["data_wait_seconds"] for row in selected]
    batch_size = int(RESULT.get("parsed_args", {}).get("batch_size", 128))
    return {
        "selected_indices": [row["index"] for row in selected],
        "compute_median_seconds": statistics.median(times) if times else None,
        "compute_mean_seconds": statistics.mean(times) if times else None,
        "compute_min_seconds": min(times) if times else None,
        "compute_max_seconds": max(times) if times else None,
        "data_wait_median_seconds": statistics.median(waits) if waits else None,
        "throughput_images_per_second": batch_size / statistics.median(times) if times else None,
    }


def _profiler_rows(profiler) -> list[dict[str, Any]]:
    rows = []
    for event in profiler.key_averages():
        row = {
            "key": event.key,
            "count": event.count,
            "self_cpu_time_total_us": getattr(event, "self_cpu_time_total", None),
            "cpu_time_total_us": getattr(event, "cpu_time_total", None),
            "self_device_time_total_us": getattr(
                event, "self_device_time_total",
                getattr(event, "self_cuda_time_total", None)),
            "device_time_total_us": getattr(
                event, "device_time_total",
                getattr(event, "cuda_time_total", None)),
            "cpu_memory_usage": getattr(event, "cpu_memory_usage", None),
            "device_memory_usage": getattr(
                event, "device_memory_usage",
                getattr(event, "cuda_memory_usage", None)),
        }
        rows.append(row)
    rows.sort(key=lambda row: row["self_device_time_total_us"] or 0, reverse=True)
    return rows


def _microbenchmarks() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"skipped": "CUDA unavailable"}
    result = {}
    with torch.no_grad():
        x = torch.randn(1024, 1024, device="cuda")
        y = torch.randn(1024, 1024, device="cuda")
        for _ in range(10):
            torch.mm(x, y)
        _sync()
        start = time.perf_counter()
        for _ in range(100):
            torch.mm(x, y)
        _sync()
        result["matmul_1024_100_seconds"] = time.perf_counter() - start

        z = torch.ones(1, device="cuda")
        _sync()
        start = time.perf_counter()
        for _ in range(10000):
            z = z + 1
        _sync()
        result["ten_thousand_small_kernel_chain_seconds"] = time.perf_counter() - start

        start = time.perf_counter()
        for _ in range(1000):
            z.add_(1)
            torch.cuda.synchronize()
        result["one_thousand_launch_and_sync_seconds"] = time.perf_counter() - start
    return result


def main() -> None:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    os.chdir(ROOT)

    RESULT["environment_before_main"] = _collect_environment()
    _write_result()

    import train_ode_cifar as entry
    import trainer_timm
    from training_recovery import restore_latest

    args = entry.get_args()
    RESULT["parsed_args"] = vars(args).copy()
    entry.get_args = lambda: args

    original_evaluate_teacher = entry.evaluate_teacher

    def timed_evaluate_teacher(*call_args, **call_kwargs):
        _sync()
        start = time.perf_counter()
        try:
            return original_evaluate_teacher(*call_args, **call_kwargs)
        finally:
            _sync()
            RESULT["teacher_evaluation_seconds"] = time.perf_counter() - start

    entry.evaluate_teacher = timed_evaluate_teacher

    original_train = trainer_timm.TrainerCiFarTimmStyle.train

    def diagnostic_train(trainer):
        RESULT["trainer_class"] = type(trainer).__name__
        RESULT["loader"] = {
            "class": type(trainer.train_dataloader).__name__,
            "len": len(trainer.train_dataloader),
            "batch_size": getattr(trainer.train_dataloader, "batch_size", None),
            "num_workers": getattr(trainer.train_dataloader, "num_workers", None),
            "pin_memory": getattr(trainer.train_dataloader, "pin_memory", None),
            "persistent_workers": getattr(trainer.train_dataloader, "persistent_workers", None),
            "drop_last": getattr(trainer.train_dataloader, "drop_last", None),
            "sampler": type(getattr(trainer.train_dataloader, "sampler", None)).__name__,
        }
        RESULT["model"] = {
            "class": type(trainer.model).__name__,
            "parameters": sum(p.numel() for p in trainer.model.parameters()),
            "trainable_parameters": sum(
                p.numel() for p in trainer.model.parameters() if p.requires_grad),
        }
        RESULT["block_options_before_train"] = _option_audit(trainer.model)
        try:
            import tc_shared_correction
            RESULT["shared_correction_backend"] = {
                "triton_imported": tc_shared_correction.triton is not None,
                "triton_language_imported": getattr(
                    tc_shared_correction, "tl", None) is not None,
            }
        except Exception as exc:
            RESULT["shared_correction_backend"] = {"error": repr(exc)}
        bad_options = [
            row for row in RESULT["block_options_before_train"]
            if row["live_reuse"] is not True or row["live_checkpoint"] is not True
        ]
        RESULT["optimization_option_mismatches"] = bad_options

        recovered = restore_latest(trainer)
        RESULT["restore_latest_result"] = _jsonable(recovered)
        epoch = recovered[0] if recovered is not None else 0

        batches: list[dict[str, Any]] = []
        component = _ComponentTimers()
        solvers = _SolverCounters()
        profiler = None
        profiler_started = False

        if MODE == "profile":
            component.install(trainer)
            solvers.install()
            activities = [torch.profiler.ProfilerActivity.CPU]
            if torch.cuda.is_available():
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            profiler = torch.profiler.profile(
                activities=activities,
                record_shapes=True,
                profile_memory=True,
                with_stack=False,
            )

        def profile_start():
            nonlocal profiler_started
            if profiler is not None:
                profiler.__enter__()
                profiler_started = True

        def profile_step(_index):
            if profiler is not None:
                profiler.step()

        original_loader = trainer.train_dataloader
        trainer.train_dataloader = _BoundedTimedLoader(
            original_loader, MAX_BATCHES, batches,
            on_first_batch=profile_start, on_last_batch=profile_step)
        _sync()
        train_start = time.perf_counter()
        try:
            loss = trainer.train_one_epoch(epoch)
        finally:
            try:
                _sync()
            except Exception as exc:
                RESULT["final_cuda_sync_error"] = repr(exc)
            RESULT["bounded_epoch_wall_seconds"] = time.perf_counter() - train_start
            trainer.train_dataloader = original_loader
            if profiler is not None and profiler_started:
                try:
                    profiler.__exit__(None, None, None)
                    (OUTPUT / "profiler_cpu.txt").write_text(
                        profiler.key_averages().table(
                            sort_by="self_cpu_time_total", row_limit=200))
                    device_sort = (
                        "self_cuda_time_total" if torch.cuda.is_available()
                        else "self_cpu_time_total")
                    (OUTPUT / "profiler_device.txt").write_text(
                        profiler.key_averages().table(
                            sort_by=device_sort, row_limit=200))
                    RESULT["profiler_events"] = _profiler_rows(profiler)
                    if SAVE_TRACE:
                        profiler.export_chrome_trace(
                            str(OUTPUT / "chrome_trace.json"))
                except Exception as exc:
                    RESULT["profiler_finalize_error"] = repr(exc)
            elif profiler is not None:
                RESULT["profiler_finalize_error"] = "profiler never started"
            component.restore()
            solvers.restore()
            RESULT["batches"] = batches
            RESULT["batch_summary"] = _summarize_batches(batches)
            RESULT["component_seconds"] = dict(component.records)
            RESULT["adaptive_solves"] = sorted(
                solvers.rows, key=lambda row: row.get("solve_index", -1))
            RESULT["rhs_checkpoint_calls"] = solvers.rhs_checkpoint_calls
            RESULT["block_options_after_train"] = _option_audit(trainer.model)
            RESULT["cuda_memory_final"] = {
                "allocated_bytes": torch.cuda.memory_allocated() if torch.cuda.is_available() else None,
                "reserved_bytes": torch.cuda.memory_reserved() if torch.cuda.is_available() else None,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
                "peak_reserved_bytes": torch.cuda.max_memory_reserved() if torch.cuda.is_available() else None,
            }
            _write_result()

        RESULT["train_loss"] = float(loss) if loss is not None else None
        _write_result()

    trainer_timm.TrainerCiFarTimmStyle.train = diagnostic_train
    try:
        entry.main()
        RESULT["microbenchmarks_after_training"] = _microbenchmarks()
        RESULT["status"] = "complete"
    except BaseException as exc:
        RESULT["status"] = "failed"
        RESULT["exception"] = repr(exc)
        RESULT["traceback"] = traceback.format_exc()
        raise
    finally:
        trainer_timm.TrainerCiFarTimmStyle.train = original_train
        RESULT["finished_unix"] = time.time()
        _write_result()


if __name__ == "__main__":
    main()

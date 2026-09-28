#!/usr/bin/env python3
"""Isolate the host-side layers behind TC fine-tuning execution time.

The benchmarks deliberately progress from native single-thread CPU work to
ATen/autograd, direct CUDA-runtime launches, PyTorch CUDA launches, and the two
custom operations that dominate the production TC profile.  CUDA tests report
host enqueue time separately from the device timeline.  No dataset, model, or
checkpoint is loaded.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Callable

import torch


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path(os.environ.get(
    "TC_HOST_DIAG_OUTPUT", ROOT / "tc_host_execution_diagnostic"))
REPEATS = int(os.environ.get("TC_HOST_DIAG_REPEATS", "5"))
SCALE = float(os.environ.get("TC_HOST_DIAG_SCALE", "1.0"))
OUTPUT.mkdir(parents=True, exist_ok=True)

if REPEATS < 1:
    raise ValueError("TC_HOST_DIAG_REPEATS must be positive")
if not (0.0 < SCALE <= 1.0):
    raise ValueError("TC_HOST_DIAG_SCALE must be in (0, 1]")


def _iterations(production_count: int, minimum: int = 1) -> int:
    return max(minimum, round(production_count * SCALE))


COUNTS = {
    "native_cpu_loop": _iterations(50_000_000, 10_000),
    "python_integer_loop": _iterations(5_000_000, 10_000),
    "cpu_aten_add": _iterations(186_992, 100),
    "cpu_autograd": _iterations(10_000, 10),
    "cuda_async_launch": _iterations(186_992, 100),
    "cuda_launch_and_sync": _iterations(10_000, 10),
    "pytorch_cuda_add": _iterations(186_992, 100),
    "pytorch_cuda_add_and_sync": _iterations(10_000, 10),
    "cuda_autograd": _iterations(10_000, 10),
    "quantization_forward": _iterations(4_150, 4),
    "quantization_forward_backward": _iterations(1_608, 2),
    "correction_forward": _iterations(4_062, 4),
    "correction_forward_backward": _iterations(1_607, 2),
}


RESULT: dict[str, Any] = {
    "status": "starting",
    "started_unix": time.time(),
    "output": str(OUTPUT),
    "repeats": REPEATS,
    "iteration_scale": SCALE,
    "counts": COUNTS,
    "results": {},
}


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return repr(value)


def _write_result() -> None:
    temporary = OUTPUT / "host_execution.json.tmp"
    temporary.write_text(json.dumps(_jsonable(RESULT), indent=2, sort_keys=True))
    temporary.replace(OUTPUT / "host_execution.json")


def _run(command: list[str], timeout: int = 60) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command, text=True, capture_output=True, timeout=timeout,
            check=False)
        return {
            "command": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    except Exception as exc:
        return {"command": command, "error": repr(exc)}


def _cpu_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    try:
        fields = Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()
        cpu = int(fields[36])
        snapshot["cpu"] = cpu
        cpu_root = Path(f"/sys/devices/system/cpu/cpu{cpu}")
        nodes = sorted(cpu_root.glob("node[0-9]*"))
        if nodes:
            snapshot["numa_node"] = int(nodes[0].name.removeprefix("node"))
        for name in ("scaling_cur_freq", "cpuinfo_cur_freq", "scaling_max_freq"):
            path = cpu_root / "cpufreq" / name
            if path.is_file():
                snapshot[f"{name}_khz"] = int(path.read_text().strip())
        governor = cpu_root / "cpufreq" / "scaling_governor"
        if governor.is_file():
            snapshot["scaling_governor"] = governor.read_text().strip()
    except Exception as exc:
        snapshot["error"] = repr(exc)
    return snapshot


def _usage() -> dict[str, float | int]:
    value = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "user_seconds": value.ru_utime,
        "system_seconds": value.ru_stime,
        "voluntary_context_switches": value.ru_nvcsw,
        "involuntary_context_switches": value.ru_nivcsw,
        "minor_page_faults": value.ru_minflt,
        "major_page_faults": value.ru_majflt,
    }


def _usage_delta(before: dict[str, float | int],
                 after: dict[str, float | int]) -> dict[str, float | int]:
    return {key: after[key] - before[key] for key in before}


def _summary(samples: list[dict[str, Any]], iterations: int) -> dict[str, Any]:
    numeric_keys = (
        "wall_seconds", "enqueue_seconds", "completion_seconds",
        "device_span_seconds", "user_seconds", "system_seconds",
        "voluntary_context_switches", "involuntary_context_switches",
        "minor_page_faults", "major_page_faults",
    )
    summary: dict[str, Any] = {"iterations": iterations, "samples": samples}
    for key in numeric_keys:
        values = [float(row[key]) for row in samples if key in row]
        if values:
            summary[f"{key}_median"] = statistics.median(values)
            summary[f"{key}_minimum"] = min(values)
            summary[f"{key}_maximum"] = max(values)
            if key.endswith("seconds"):
                summary[f"{key}_median_us_per_iteration"] = (
                    statistics.median(values) * 1e6 / iterations)
    return summary


def _measure_cpu(iterations: int, function: Callable[[int], Any]) -> dict[str, Any]:
    function(min(iterations, max(1, round(iterations * 0.02))))
    samples = []
    for _ in range(REPEATS):
        before_usage = _usage()
        before_cpu = _cpu_snapshot()
        start = time.perf_counter()
        function(iterations)
        elapsed = time.perf_counter() - start
        after_cpu = _cpu_snapshot()
        delta = _usage_delta(before_usage, _usage())
        samples.append({
            "wall_seconds": elapsed,
            "cpu_before": before_cpu,
            "cpu_after": after_cpu,
            **delta,
        })
    return _summary(samples, iterations)


def _sync() -> None:
    torch.cuda.synchronize()


def _measure_cuda(iterations: int,
                  function: Callable[[int], Any]) -> dict[str, Any]:
    function(min(iterations, max(1, round(iterations * 0.01))))
    _sync()
    samples = []
    for _ in range(REPEATS):
        _sync()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        before_usage = _usage()
        before_cpu = _cpu_snapshot()
        start = time.perf_counter()
        function(iterations)
        enqueue_seconds = time.perf_counter() - start
        end_event.record()
        end_event.synchronize()
        completion_seconds = time.perf_counter() - start
        after_cpu = _cpu_snapshot()
        delta = _usage_delta(before_usage, _usage())
        samples.append({
            "enqueue_seconds": enqueue_seconds,
            "completion_seconds": completion_seconds,
            "device_span_seconds": start_event.elapsed_time(end_event) / 1000.0,
            "cpu_before": before_cpu,
            "cpu_after": after_cpu,
            **delta,
        })
    return _summary(samples, iterations)


def _compile_shared_library(name: str, suffix: str, source: str,
                            command: list[str]) -> tuple[Path, dict[str, Any]]:
    digest = hashlib.sha256(source.encode()).hexdigest()[:16]
    build = Path(tempfile.gettempdir()) / f"tc_host_diag_{os.getuid()}"
    build.mkdir(parents=True, exist_ok=True)
    source_path = build / f"{name}_{digest}.{suffix}"
    library_path = build / f"{name}_{digest}.so"
    source_path.write_text(source)
    compile_result: dict[str, Any] = {
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "source_path": str(source_path),
        "library_path": str(library_path),
    }
    if not library_path.is_file():
        expanded = [part.format(source=source_path, output=library_path)
                    for part in command]
        compile_result["compile"] = _run(expanded, timeout=300)
        if compile_result["compile"].get("returncode") != 0:
            raise RuntimeError(
                f"failed to compile {name}: {compile_result['compile']}")
    else:
        compile_result["compile"] = {"cached": True}
    return library_path, compile_result


def _load_native_cpu() -> tuple[Callable[[int], int], dict[str, Any]]:
    source = r"""
#include <stdint.h>
uint64_t tc_host_native_cpu_loop(uint64_t iterations) {
    volatile uint64_t value = 0x12345678ULL;
    for (uint64_t index = 0; index < iterations; ++index) {
        value = value * 1664525ULL + index + 1013904223ULL;
    }
    return value;
}
"""
    compiler = os.environ.get("CC", "cc")
    path, build = _compile_shared_library(
        "native_cpu", "c", source,
        [compiler, "-O3", "-shared", "-fPIC", "{source}", "-o", "{output}"])
    library = ctypes.CDLL(str(path))
    function = library.tc_host_native_cpu_loop
    function.argtypes = [ctypes.c_uint64]
    function.restype = ctypes.c_uint64
    return function, build


def _load_native_cuda() -> tuple[dict[str, Callable[[int], None]], dict[str, Any]]:
    major, minor = torch.cuda.get_device_capability(0)
    architecture = f"{major}{minor}"
    source = r"""
#include <cuda_runtime.h>
#include <stdint.h>

__global__ void tc_host_empty_kernel() {}

extern "C" int tc_host_launch_async(uint64_t iterations) {
    for (uint64_t index = 0; index < iterations; ++index) {
        tc_host_empty_kernel<<<1, 1>>>();
    }
    return static_cast<int>(cudaGetLastError());
}

extern "C" int tc_host_launch_and_sync(uint64_t iterations) {
    for (uint64_t index = 0; index < iterations; ++index) {
        tc_host_empty_kernel<<<1, 1>>>();
        cudaError_t status = cudaDeviceSynchronize();
        if (status != cudaSuccess) {
            return static_cast<int>(status);
        }
    }
    return 0;
}
"""
    nvcc = os.environ.get("NVCC", "nvcc")
    path, build = _compile_shared_library(
        f"native_cuda_sm{architecture}", "cu", source,
        [nvcc, "-O3", "--shared", "-Xcompiler", "-fPIC",
         f"--generate-code=arch=compute_{architecture},code=sm_{architecture}",
         "{source}", "-o", "{output}"])
    library = ctypes.CDLL(str(path))

    def bind(name: str) -> Callable[[int], None]:
        native = getattr(library, name)
        native.argtypes = [ctypes.c_uint64]
        native.restype = ctypes.c_int

        def checked(iterations: int) -> None:
            status = native(iterations)
            if status != 0:
                raise RuntimeError(f"{name} returned CUDA error {status}")

        return checked

    return {
        "async": bind("tc_host_launch_async"),
        "launch_and_sync": bind("tc_host_launch_and_sync"),
    }, build


def _environment() -> dict[str, Any]:
    commands = {
        "lscpu": _run(["lscpu"]),
        "lscpu_extended": _run([
            "lscpu", "-e=CPU,NODE,SOCKET,CORE,ONLINE,MHZ,MAXMHZ,MINMHZ"]),
        "nvidia_smi": _run([
            "nvidia-smi", "--query-gpu=index,name,pci.bus_id,pstate,"
            "clocks.sm,clocks.max.sm,power.draw,power.limit,driver_version",
            "--format=csv,noheader,nounits"]),
        "cc_version": _run([os.environ.get("CC", "cc"), "--version"]),
        "nvcc_version": _run([os.environ.get("NVCC", "nvcc"), "--version"]),
        "perf_version": _run(["perf", "--version"]),
    }
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_capability": (
            list(torch.cuda.get_device_capability(0))
            if torch.cuda.is_available() else None),
        "cpu_count": os.cpu_count(),
        "affinity": sorted(os.sched_getaffinity(0)),
        "cpu_snapshot": _cpu_snapshot(),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "commands": commands,
    }


def _run_suite(native_cpu: Callable[[int], int],
               native_cuda: dict[str, Callable[[int], None]]) -> dict[str, Any]:
    suite: dict[str, Any] = {}

    suite["native_cpu_loop"] = _measure_cpu(
        COUNTS["native_cpu_loop"], native_cpu)

    def python_integer_loop(iterations: int) -> int:
        value = 0
        for index in range(iterations):
            value = (value * 1_664_525 + index + 1_013_904_223) & 0xFFFFFFFF
        return value

    suite["python_integer_loop"] = _measure_cpu(
        COUNTS["python_integer_loop"], python_integer_loop)

    cpu_x = torch.ones(1)
    cpu_y = torch.ones(1)
    cpu_out = torch.empty(1)

    def cpu_aten_add(iterations: int) -> None:
        with torch.no_grad():
            for _ in range(iterations):
                torch.add(cpu_x, cpu_y, out=cpu_out)

    suite["cpu_aten_add_out"] = _measure_cpu(
        COUNTS["cpu_aten_add"], cpu_aten_add)

    cpu_grad_x = torch.ones(1, requires_grad=True)
    cpu_grad = torch.ones(1)

    def cpu_autograd(iterations: int) -> None:
        for _ in range(iterations):
            value = cpu_grad_x * cpu_grad_x
            value.backward(cpu_grad)
            cpu_grad_x.grad = None

    suite["cpu_mul_backward"] = _measure_cpu(
        COUNTS["cpu_autograd"], cpu_autograd)

    suite["native_cuda_empty_async"] = _measure_cuda(
        COUNTS["cuda_async_launch"], native_cuda["async"])
    suite["native_cuda_empty_launch_and_sync"] = _measure_cuda(
        COUNTS["cuda_launch_and_sync"], native_cuda["launch_and_sync"])

    cuda_x = torch.ones(1, device="cuda")
    cuda_y = torch.ones(1, device="cuda")
    cuda_out = torch.empty(1, device="cuda")

    def pytorch_cuda_add(iterations: int) -> None:
        with torch.no_grad():
            for _ in range(iterations):
                torch.add(cuda_x, cuda_y, out=cuda_out)

    suite["pytorch_cuda_add_out"] = _measure_cuda(
        COUNTS["pytorch_cuda_add"], pytorch_cuda_add)

    def pytorch_cuda_add_and_sync(iterations: int) -> None:
        with torch.no_grad():
            for _ in range(iterations):
                torch.add(cuda_x, cuda_y, out=cuda_out)
                torch.cuda.synchronize()

    suite["pytorch_cuda_add_out_and_sync"] = _measure_cuda(
        COUNTS["pytorch_cuda_add_and_sync"], pytorch_cuda_add_and_sync)

    cuda_grad_x = torch.ones(1, device="cuda", requires_grad=True)
    cuda_grad = torch.ones(1, device="cuda")

    def cuda_autograd(iterations: int) -> None:
        for _ in range(iterations):
            value = cuda_grad_x * cuda_grad_x
            value.backward(cuda_grad)
            cuda_grad_x.grad = None

    suite["pytorch_cuda_mul_backward"] = _measure_cuda(
        COUNTS["cuda_autograd"], cuda_autograd)

    from ode_pc import QuantizationImpl

    quant_input = torch.randn(4096, device="cuda", requires_grad=True)
    quant_scale = torch.ones((), device="cuda")
    quant_min = torch.tensor(-15.0, device="cuda")
    quant_max = torch.tensor(15.0, device="cuda")
    quant_grad = torch.ones_like(quant_input)

    def quantization_forward(iterations: int) -> None:
        value = None
        for _ in range(iterations):
            value = QuantizationImpl.apply(
                quant_input, quant_scale, quant_min, quant_max, 1.0)
        del value

    suite["quantization_forward"] = _measure_cuda(
        COUNTS["quantization_forward"], quantization_forward)

    def quantization_forward_backward(iterations: int) -> None:
        for _ in range(iterations):
            value = QuantizationImpl.apply(
                quant_input, quant_scale, quant_min, quant_max, 1.0)
            value.backward(quant_grad)
            quant_input.grad = None

    suite["quantization_forward_backward"] = _measure_cuda(
        COUNTS["quantization_forward_backward"],
        quantization_forward_backward)

    import tc_shared_correction

    suite["shared_correction_backend"] = {
        "triton_imported": tc_shared_correction.triton is not None,
        "fused_correction_available": hasattr(tc_shared_correction, "_correct"),
    }
    if tc_shared_correction.triton is not None:
        correction_input = torch.randn(
            4096, device="cuda", requires_grad=True)
        grid = torch.linspace(-1.0, 1.0, 257, device="cuda")
        table = 10_000.0 * (1.0 + 0.1 * grid.square())
        slope = (table[1:] - table[:-1]) / (grid[1:] - grid[:-1])
        nominal = torch.tensor(10_000.0, device="cuda")
        correction_grad = torch.ones_like(correction_input)

        def correction_forward(iterations: int) -> None:
            value = None
            for _ in range(iterations):
                value = tc_shared_correction._Correction.apply(
                    correction_input, grid, table, slope, nominal,
                    1.0, 1.0)
            del value

        suite["correction_forward"] = _measure_cuda(
            COUNTS["correction_forward"], correction_forward)

        def correction_forward_backward(iterations: int) -> None:
            for _ in range(iterations):
                value = tc_shared_correction._Correction.apply(
                    correction_input, grid, table, slope, nominal,
                    1.0, 1.0)
                value.backward(correction_grad)
                correction_input.grad = None

        suite["correction_forward_backward"] = _measure_cuda(
            COUNTS["correction_forward_backward"],
            correction_forward_backward)
    else:
        suite["correction_skipped"] = "Triton is unavailable"

    return suite


def main() -> None:
    os.chdir(ROOT)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    RESULT["environment"] = _environment()
    _write_result()

    native_cpu, cpu_build = _load_native_cpu()
    RESULT["native_cpu_build"] = cpu_build
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the host execution diagnostic")
    torch.cuda.init()
    native_cuda, cuda_build = _load_native_cuda()
    RESULT["native_cuda_build"] = cuda_build
    _write_result()

    inherited_affinity = set(os.sched_getaffinity(0))
    RESULT["results"]["inherited_affinity"] = _run_suite(
        native_cpu, native_cuda)
    _write_result()

    current_cpu = _cpu_snapshot().get("cpu")
    if current_cpu not in inherited_affinity:
        current_cpu = min(inherited_affinity)
    try:
        os.sched_setaffinity(0, {int(current_cpu)})
        RESULT["main_thread_single_cpu_affinity"] = sorted(
            os.sched_getaffinity(0))
        RESULT["results"]["main_thread_single_cpu_affinity"] = _run_suite(
            native_cpu, native_cuda)
    finally:
        os.sched_setaffinity(0, inherited_affinity)

    RESULT["status"] = "complete"
    RESULT["finished_unix"] = time.time()
    _write_result()


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        RESULT["status"] = "failed"
        RESULT["exception"] = repr(exc)
        RESULT["traceback"] = traceback.format_exc()
        RESULT["finished_unix"] = time.time()
        _write_result()
        raise

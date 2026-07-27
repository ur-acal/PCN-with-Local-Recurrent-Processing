"""Device detection and management for PyTorch operations.

This module provides utilities to detect and manage computing devices (CUDA, MPS, CPU)
for PyTorch operations in a cross-platform way, supporting both NVIDIA GPUs and Apple Silicon.
"""

import logging
from typing import cast

import torch

LOGGER = logging.getLogger("scangen.device")


def get_device(device: str | torch.device | None = None) -> torch.device:
    """Get the best available computing device.

    Automatically detects the best available device in the following priority:
    1. Specified device (if provided and available)
    2. CUDA (NVIDIA GPU)
    3. MPS (Apple Metal Performance Shaders)
    4. CPU (fallback)

    Args:
        device: Optional device string to force a specific device.
                Can be 'cuda', 'mps', 'cpu', or specific device like 'cuda:0'

    Returns:
        torch.device: The selected device

    Examples:
        >>> device = get_device()  # Auto-detect best device
        >>> device = get_device("cuda")  # Force CUDA if available, else fallback
        >>> device = get_device("cuda:0")  # Force CUDA device 0 if available, else fallback
        >>> device = get_device("mps")  # Force MPS if available, else fallback
        >>> device = get_device("cpu")  # Force CPU
    """
    if isinstance(device, torch.device):
        return device
    
    if device not in (None, "auto"):
        requested_device = torch.device(device)
        if is_device_available(requested_device):
            return requested_device
        LOGGER.warning(
            f"Warning: Requested device '{device}' not available, falling back to auto-detection"
        )

    defaults = ["cuda", "mps", "cpu"]
    devices = [d for d in defaults if is_device_available(torch.device(d))]
    if devices:
        return torch.device(devices.pop(0))
    else:
        raise RuntimeError(f"No available Torch devices found in {defaults}")


def is_device_available(device: torch.device) -> bool:
    """Check if a specific device is available.

    Args:
        device: The device to check

    Returns:
        bool: True if the device is available, False otherwise
    """
    try:
        if device.type == "cuda":
            if not torch.cuda.is_available():
                return False
            # If no specific index given, use default device (index is None)
            if device.index is None:
                return True
            # If specific index given, check it's within range
            return device.index < torch.cuda.device_count()
        if device.type == "mps":
            return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        return device.type == "cpu"
    except Exception:
        return False


def get_device_info() -> dict:
    """Get information about available devices.

    Returns:
        dict: Dictionary containing device availability information
    """
    info = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "mps_available": hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
        "selected_device": str(get_device()),
    }

    if info["cuda_available"]:
        info["cuda_devices"] = [
            {
                "index": i,
                "name": torch.cuda.get_device_name(i),
                "memory_total": torch.cuda.get_device_properties(i).total_memory,
            }
            for i in range(cast(int, info["cuda_device_count"]))
        ]

    return info


def to_device(tensor_or_module, device: torch.device | None = None):
    """Move tensor or module to the specified device.

    This is a device-agnostic replacement for .cuda() calls.

    Args:
        tensor_or_module: PyTorch tensor or module to move
        device: Target device (if None, uses get_device())

    Returns:
        The tensor or module moved to the target device
    """
    if device is None:
        device = get_device()

    return tensor_or_module.to(device)


def print_device_info() -> None:
    """Print information about available devices.
    """
    info = get_device_info()
    # ruff: noqa: T201
    print("PyTorch Device Information:")
    print(f"  Selected device: {info['selected_device']}")

    print(f"  CUDA available: {info['cuda_available']}")

    if info["cuda_available"]:
        print(f"  CUDA devices: {info['cuda_device_count']}")
        for device_info in info["cuda_devices"]:
            memory_gb = device_info["memory_total"] / (1024**3)
            print(f"    GPU {device_info['index']}: {device_info['name']} ({memory_gb:.1f} GB)")

    print(f"  MPS available: {info['mps_available']}")

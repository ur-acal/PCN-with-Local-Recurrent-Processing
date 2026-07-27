import pytest

from scangen.device import get_device


def test_device_setup_auto() -> None:
    """Test automatic device selection."""
    device = get_device()  # Use None/auto detection
    assert device.type in ["cpu", "cuda", "mps"]

def test_device_setup_explicit() -> None:
    """Test explicit device selection."""
    device = get_device("cpu")
    assert device.type == "cpu"

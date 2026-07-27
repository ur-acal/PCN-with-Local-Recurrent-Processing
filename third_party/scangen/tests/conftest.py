"""Pytest configuration for scangen tests."""
import os
from pathlib import Path

import pytest


def pytest_addoption(parser):
    """Add custom command-line options."""
    parser.addoption(
        "--all",
        action="store_true",
        default=False,
        help="Run all tests"
    )
    parser.addoption(
        "--slow",
        action="store_true",
        default=False,
        help="Run slow/long-running tests"
    )
    parser.addoption(
        "--data",
        action="store_true",
        default=False,
        help="Run tests that require image data"
    )


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: mark test as slow (requires --slow to run)"
    )
    config.addinivalue_line(
        "markers", "data: mark test as requiring image data (requires --data to run)"
    )


def pytest_collection_modifyitems(config, items):
    """Skip slow tests unless --slow option is provided."""
    run_all = config.getoption("--all")
    # Handle --slow option
    if not config.getoption("--slow") and not run_all:
        skip_slow = pytest.mark.skip(reason="need --slow option to run slow tests")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)

    # Handle --data option
    if not config.getoption("--data") and not run_all:
        skip_data = pytest.mark.skip(reason="need --data option to run data tests")
        for item in items:
            if "data" in item.keywords:
                item.add_marker(skip_data)

@pytest.fixture
def scangen_data():
    environment = os.getenv("SCANGENDATA")
    if environment is not None:
        package_dir = Path(environment)
    else:
        import scangen
        package_dir = Path(scangen.__file__).parent.parent.parent / "data"
    return package_dir

@pytest.fixture
def data_dir(tmp_path):
    """
    Create data subdirectories and change working directory so that
    pydantic will find the files referenced.
    """
    orig_cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        specific_dir = tmp_path / "data"
        specific_dir.mkdir()
        (tmp_path / "output").mkdir()
        yield specific_dir
    finally:
        os.chdir(orig_cwd)


@pytest.fixture
def nn_weights():
    """
    The path to test weights, if they exist.
    """
    environment = os.getenv("SCANGENDATA")
    if environment is not None:
        package_dir = Path(environment)
    else:
        import scangen
        package_dir = Path(scangen.__file__).parent.parent.parent / "data"
    weight_path = package_dir / "weights" / "rgb2raw.pth"
    if not weight_path.exists():
        pytest.skip(f"Test weights not found at {weight_path}")
    yield weight_path

from pathlib import Path
from typing import Tuple

from pydantic import BaseModel, PositiveInt, NonNegativeFloat


class DatasetFormat(BaseModel):
    name: str
    batch_size: PositiveInt
    root: Path  # Relative to SCANGENDATA directory.

class RawModel(BaseModel):
    name: str

class NoiseFormat(BaseModel):
    type: str
    min_shot_noise: NonNegativeFloat
    max_shot_noise: NonNegativeFloat
    read_noise_offset_width: NonNegativeFloat
    read_noise_offset: NonNegativeFloat
    read_noise_slope: NonNegativeFloat

class OutputFormat(BaseModel):
    directory: Path  # Relative to working directory or absolute.
    target_size: Tuple[PositiveInt,PositiveInt]
    save_png: bool
    save_labels: bool
    normalize_png: bool

class DeviceFormat(BaseModel):
    num_workers: int

# This can be hard to test because the Path must exist.
# Look at conftest.py::data_dir to see a test fixture.
class ConfigFormat(BaseModel):
    dataset: DatasetFormat
    rawmodel: RawModel
    noise: NoiseFormat
    output: OutputFormat
    device: DeviceFormat


def create_default_config() -> ConfigFormat:
    """Create default configuration for RAW generation.

    Returns:
        Dictionary with default configuration
    """
    no_directories = ConfigFormat.model_validate({
        "dataset": {
            "name": "cifar10",
            "batch_size": 32,
            "root": "."
        },
        "rawmodel": {
            "name": "CycleISP"
        },
        "noise": {
            "type": "pointwise",
            "min_shot_noise": 0.0001,
            "max_shot_noise": 0.012,
            "read_noise_offset_width": 0.26,
            "read_noise_offset": 1.20,
            "read_noise_slope": 2.18
        },
        "output": {
            "directory": ".",
            "target_size": (16, 16),
            "save_png": True,
            "save_labels": True,
            # False matches CycleISP behavior, True for min-max normalization
            "normalize_png": False,
        },
        "device": {
            "num_workers": 4  # Enable multi-threaded data loading by default
        },
    })
    no_directories.output.directory = "./output"
    return no_directories

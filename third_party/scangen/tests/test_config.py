import os
from pathlib import Path
import tomllib

from pydantic import ValidationError

from scangen.formats.config_format import ConfigFormat, create_default_config

def test_read_config_json(data_dir):
    json_str = """
        {
        "dataset": {
            "name": "cifar10",
            "batch_size": 32,
            "root": "./data"
        },
        "rawmodel": {
            "name": "CycleISP"
        },
        "noise": {
            "type": "cycleisp",
            "min_shot_noise": 0.0001,
            "max_shot_noise": 0.012,
            "read_noise_offset_width": 0.26,
            "read_noise_offset": 1.20,
            "read_noise_slope": 2.18
        },
        "output": {
            "directory": "./output",
            "target_size": [16, 16],
            "save_png": true,
            "save_labels": true,
            "normalize_png": false
        },
        "device": {
            "num_workers": 0
        }
        }"""
    config = ConfigFormat.model_validate_json(json_str)
    assert config.dataset.name == "cifar10"
    assert config.dataset.batch_size == 32


def test_read_config_json_missing(data_dir):
    json_str = """
        {
        "dataset": {
            "name": "cifar10",
            "root": "./data"
        },
        "rawmodel": {
            "name": "CycleISP"
        },
        "noise": {
            "type": "cycleisp",
            "min_shot_noise": 0.0001,
            "max_shot_noise": 0.012,
            "read_noise_offset_width": 0.26,
            "read_noise_offset": 1.20,
            "read_noise_slope": 2.18
        },
        "output": {
            "directory": "./output",
            "target_size": [16, 16],
            "save_png": true,
            "save_labels": true,
            "normalize_png": false
        },
        "device": {
            "num_workers": 0
        }
        }"""
    try:
        config = ConfigFormat.model_validate_json(json_str)
        assert False
    except ValidationError as e:
        assert "missing" in str(e)



def test_read_config_toml(data_dir):
    toml_str = """
        [dataset]
        name = "cifar10"
        batch_size = 32
        root = "./data"
        [rawmodel]
        name = "CycleISP"
        [noise]
        type = "cycleisp"
        min_shot_noise = 0.0001
        max_shot_noise = 0.012
        read_noise_offset_width = 0.26
        read_noise_offset = 1.20
        read_noise_slope = 2.18
        [output]
        directory = "./output"
        target_size = [16, 16]
        save_png = true
        save_labels = true
        normalize_png = false
        [device]
        num_workers = 0
        """
    data = tomllib.loads(toml_str)
    config = ConfigFormat.model_validate(data)
    assert config.dataset.name == "cifar10"

def test_default_config(data_dir):
    config_dict = create_default_config()
    config = ConfigFormat.model_validate(config_dict)
    assert config.dataset.name == "cifar10"

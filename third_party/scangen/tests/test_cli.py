import json
from pathlib import Path

import pytest

from scangen.cli import pregenerate_raw, create_config, print_device_info


def test_cli_create_config(data_dir):
    config_file = data_dir / "config.json"
    output_dir = data_dir / "output"
    create_config(
        output_path=str(config_file),
        dataset="myvacation",
        batch_size=4,
        output_dir=str(output_dir),
        )
    assert config_file.exists()
    with config_file.open("r") as cf:
        cfg = json.load(cf)
        assert str(cfg["dataset"]["name"] == "myvacation")
        assert int(cfg["dataset"]["batch_size"]) == 4
        assert str(cfg["output"]["directory"]) == str(output_dir)


def test_cli_device_info(capsys):
    print_device_info()
    assert "available" in capsys.readouterr().out

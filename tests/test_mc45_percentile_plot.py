"""Plot dataset provenance must not fall back to the other dataset."""
import csv
import json
from pathlib import Path
import subprocess
import sys

import pytest

from diagnostic_scripts.plot_mc45_descending_percentiles import (
    load_corner_means, plot_descending_percentiles,
)


def test_cifar10_relative_plot_records_all_corner_sources(tmp_path):
    source = tmp_path / "cifar10_bank"
    source.mkdir()
    with (source / "corner_trials.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["corner", "trial_index", "accuracy"])
        for process in ("TT", "FF", "SS", "FS", "SF"):
            for voltage in range(3):
                for temperature in range(3):
                    writer.writerow([f"{process}_V{voltage}_T{temperature}", 0,
                                     70 + voltage + temperature])
    series = [dict(label="PCN", model="PCN", color="black", directory=source),
              dict(label="WRN", model="WRN", color="red", directory=source)]
    output = tmp_path / "plots"
    files = plot_descending_percentiles(
        series, output, (10, 15), {"PCN": 89.18, "WRN": 88.83},
        relative_to=88.83, dataset_label="CiFAIR-10")
    assert len(files) == 4 and all(Path(path).is_file() for path in files)
    metadata = json.loads((output / "plot_sources.json").read_text())
    assert metadata["dataset"] == "CiFAIR-10"
    assert metadata["relative_accuracy_reference_percent"] == 88.83
    assert metadata["series"][0]["corners"] == 45
    assert metadata["series"][0]["source"] == str(source)
    assert metadata["series"][0]["mean_accuracy_percent"] == pytest.approx(72)
    for count in (10, 15):
        with (output / f"descending_corner_percentiles_{count}_dots_markers.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 2*(count + 1)
        assert float(rows[0]["relative_accuracy_percent"]) == pytest.approx(89.18/88.83*100)
    with (source / "corner_trials.csv").open("a") as handle:
        handle.write("TT_V0_T0,0,72\n")
    with pytest.raises(ValueError, match="Duplicate"):
        load_corner_means(source)


def test_cifar10_cli_requires_explicit_sources():
    script = Path(__file__).resolve().parents[1] / "diagnostic_scripts/plot_mc45_descending_percentiles.py"
    result = subprocess.run([sys.executable, str(script), "--dataset", "cifar10"],
                            capture_output=True, text=True)
    assert result.returncode == 2
    assert "requires explicit result directories" in result.stderr

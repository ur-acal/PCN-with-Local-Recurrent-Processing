"""Dependency validation and exact configuration replay for queued MC45."""
import argparse
import csv
import json
from pathlib import Path
import sys
from unittest.mock import patch

import pytest
from diagnostic_scripts import queue_feedforward_bn_recal as queue
from scripts.run_feedforward_mc45_ablation import parse_args
from data_utils import MC45CornerData


def fixture(tmp_path):
    with patch.object(sys, "argv", ["mc45", "--model_name", "wrn_flexible_cifar_avgpool_shortcut",
                                    "--checkpoint", str(tmp_path / "source.pth")]):
        config = vars(parse_args())
    config.update(n_trials=1, enob=None, one_over_q=5.0,
                  toggle_timing_mode="fixed", toggle_y_time=15e-9,
                  activation_curve_sharing="per_spin")
    Path(config["checkpoint"]).touch()
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    (baseline / "resolved_config.json").write_text(json.dumps(config))
    corners = MC45CornerData("hardware_data/mc_45_corners").corners
    rows = [{"corner": c["id"], "trial_index": 0, "accuracy": 50.0} for c in corners]
    with (baseline / "corner_trials.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        (baseline / (row["corner"] + ".log")).write_text("trial 0: accuracy=50.0000%\n")
    log = tmp_path / "pipeline.log"
    log.write_text(queue.SUCCESS + "\n")
    return baseline, log, config


def test_replay_changes_only_bn_and_destination(tmp_path):
    baseline, log, config = fixture(tmp_path)
    assert queue.validate_dependency(baseline, log) == config
    command, changes = queue.replay_command(config, tmp_path / "output")
    assert set(changes) == {"output_dir", "bn_recalibrate", "bn_calibration_samples", "bn_calibration_batch_size"}
    with patch.object(sys, "argv", command[2:]):
        restored = vars(parse_args())
    assert restored["enob"] is None  # Must not fall back to the parser's 8 bits.
    assert restored["expanded_weight_dir"] == config["expanded_weight_dir"]
    assert restored["checkpoint"] == config["checkpoint"]
    assert restored["bn_calibration_samples"] == 5120
    assert restored["bn_calibration_batch_size"] == 256


@pytest.mark.parametrize("failure", ["no_success", "partial_csv", "wrong_log"])
def test_incomplete_dependency_is_not_success(tmp_path, failure):
    baseline, log, _ = fixture(tmp_path)
    if failure == "no_success":
        log.write_text("FT completed, eval not finished\n")
    elif failure == "partial_csv":
        path = baseline / "corner_trials.csv"
        path.write_text("\n".join(path.read_text().splitlines()[:-1]) + "\n")
    else:
        next(baseline.glob("TT_*.log")).write_text("trial 0: accuracy=49.0000%\n")
    with pytest.raises(RuntimeError):
        queue.validate_dependency(baseline, log)


def test_waits_then_executes_exact_production_command(tmp_path):
    baseline, log, config = fixture(tmp_path)
    output = tmp_path / "recal"
    args = argparse.Namespace(state_dir=tmp_path / "state", wait_pid=123,
                              wait_start_ticks=456, baseline_dir=baseline, pipeline_log=log,
                              output_dir=output, poll_seconds=1,
                              bn_calibration_samples=5120, bn_calibration_batch_size=256)
    def execute(command, **kwargs):
        assert command == queue.replay_command(config, output)[0]
        output.mkdir()
        (output / "corner_trials.csv").write_bytes((baseline / "corner_trials.csv").read_bytes())
    with patch.object(queue, "process_start_ticks", side_effect=[456, 456, None]), \
         patch.object(queue.time, "sleep") as sleep, \
         patch.object(queue.subprocess, "run", side_effect=execute) as launch, \
         patch.dict("os.environ", {"REHEARSAL_ONLY": "false"}):
        queue.run(args)
    assert sleep.call_count == 2
    assert launch.call_count == 1
    assert json.loads((args.state_dir / "status.json").read_text())["status"] == "COMPLETE"


def test_existing_recal_results_are_not_overwritten(tmp_path):
    baseline, log, _ = fixture(tmp_path)
    output = tmp_path / "recal"
    output.mkdir()
    args = argparse.Namespace(state_dir=tmp_path / "state", wait_pid=123,
                              wait_start_ticks=456, baseline_dir=baseline, pipeline_log=log,
                              output_dir=output, poll_seconds=1,
                              bn_calibration_samples=5120, bn_calibration_batch_size=256)
    with patch.object(queue, "process_start_ticks", return_value=None), \
         patch.object(queue.subprocess, "run") as launch:
        with pytest.raises(FileExistsError):
            queue.run(args)
    launch.assert_not_called()
    assert json.loads((args.state_dir / "status.json").read_text())["status"] == "FAILED"

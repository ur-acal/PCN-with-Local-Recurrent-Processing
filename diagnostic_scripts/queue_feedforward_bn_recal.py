#!/usr/bin/env python3
"""Wait for a complete FT/MC45 pipeline, then replay MC45 with BN recalibration.

The saved baseline configuration is authoritative. No training is launched and
no baseline artifacts are modified. A missing success marker or incomplete
corner set fails visibly rather than treating process disappearance as success.
"""
import argparse
import csv
from datetime import datetime, timezone
import fcntl
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SUCCESS = "Toggle feedforward fine-tuning and MC45 evaluation completed successfully."


def process_start_ticks(pid):
    try:
        # comm may contain spaces or parentheses; fields after its final ')'
        # begin with state (field 3); starttime is field 22.
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return None if fields[0] == "Z" else int(fields[19])
    except FileNotFoundError:
        return None


def validate_dependency(baseline_dir, pipeline_log):
    if SUCCESS not in Path(pipeline_log).read_text():
        raise RuntimeError("Pipeline exited without its final FT+MC45 success marker")
    baseline_dir = Path(baseline_dir)
    config = json.loads((baseline_dir / "resolved_config.json").read_text())
    if config["bn_recalibrate"] or config["n_trials"] != 1 or config["max_eval_batches"] != 0:
        raise ValueError("Expected unrecalibrated, full-dataset, one-trial baseline")
    if config["corner_ids"] is not None:
        raise ValueError("Expected all 45 corners, not a corner subset")
    from data_utils import MC45CornerData
    catalog = MC45CornerData(
        config["mc_45_corner_dir"],
        spin_variation_source=config["mc_spin_variation_source"],
        dtc_pulse_width_variation_source=config["mc_dtc_pulse_width_variation_source"],
        relu_monte_carlo_source=config["mc_relu_monte_carlo_source"],
        coupler_nonlinear_variation_source=config["mc_coupler_nonlinear_variation_source"],
        coupler_nonlinear_variation_quantity=config["mc_coupler_nonlinear_variation_quantity"],
        coupler_nominal_R=config["mc_coupler_nominal_R"])
    expected = {corner["id"] for corner in catalog.corners}
    with (baseline_dir / "corner_trials.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    if len(expected) != 45 or len(rows) != 45 or {row["corner"] for row in rows} != expected:
        raise RuntimeError("Baseline CSV does not contain exactly 45 unique corners")
    for row in rows:
        accuracy = float(row["accuracy"])
        if int(row["trial_index"]) != 0 or not math.isfinite(accuracy) or not 0 <= accuracy <= 100:
            raise RuntimeError("Invalid baseline trial record")
        text = (baseline_dir / (row["corner"] + ".log")).read_text()
        import re
        matches = re.findall(r"^trial\s+0:\s+accuracy=([-+0-9.eE]+)%", text, re.M)
        if len(matches) != 1 or abs(float(matches[0]) - accuracy) > 5e-5:
            raise RuntimeError("Baseline corner log/CSV mismatch: " + row["corner"])
    if not Path(config["checkpoint"]).is_file():
        raise FileNotFoundError(config["checkpoint"])
    return config


def replay_command(baseline, output_dir, samples=5120, batch_size=256):
    from scripts.run_feedforward_mc45_ablation import parse_args
    current = dict(baseline, output_dir=str(output_dir), bn_recalibrate=True,
                   bn_calibration_samples=samples, bn_calibration_batch_size=batch_size)
    with patch.object(sys, "argv", ["mc45", "--model_name", current["model_name"],
                                    "--checkpoint", current["checkpoint"]]):
        defaults = vars(parse_args())
    command = [sys.executable, "-u", str(ROOT / "scripts/run_feedforward_mc45_ablation.py")]
    if current.keys() != defaults.keys():
        raise ValueError("Saved MC45 configuration schema differs from the current parser")
    for name, value in current.items():
        if value is None and defaults[name] is None:
            continue
        command.append("--" + name)
        if isinstance(value, list):
            command.extend(str(item) for item in value)
        else:
            command.append("none" if value is None else
                           str(value).lower() if isinstance(value, bool) else str(value))
    # Parsing for comparison is not an evaluation launch. A neutral argv[0]
    # keeps optional rehearsal hooks from shortening the configuration here;
    # they apply only in the real child process below.
    with patch.object(sys, "argv", ["mc45-config-validation", *command[3:]]):
        resolved = vars(parse_args())
    if resolved != current:
        raise ValueError("Replay arguments do not round-trip exactly")
    allowed = {"output_dir", "bn_recalibrate", "bn_calibration_samples", "bn_calibration_batch_size"}
    differences = {key: [baseline[key], current[key]] for key in current if baseline[key] != current[key]}
    if set(differences) - allowed:
        raise ValueError("Non-BN recipe difference: " + str(differences))
    return command, differences


def run(args):
    state_dir = Path(args.state_dir).resolve()
    state_dir.mkdir(parents=True, exist_ok=True)
    lock = (state_dir / "queue.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    def record(status, **extra):
        entry = dict(status=status, time_utc=datetime.now(timezone.utc).isoformat(),
                     dependency_pid=args.wait_pid, dependency_start_ticks=args.wait_start_ticks,
                     baseline_dir=str(args.baseline_dir), output_dir=str(args.output_dir), **extra)
        temporary = state_dir / "status.json.tmp"
        temporary.write_text(json.dumps(entry, indent=2) + "\n")
        temporary.replace(state_dir / "status.json")
        print(entry["time_utc"], status, json.dumps(extra), flush=True)
    try:
        record("WAITING_FOR_FT_AND_MC45")
        while process_start_ticks(args.wait_pid) == args.wait_start_ticks:
            time.sleep(args.poll_seconds)
        record("VALIDATING_DEPENDENCY")
        baseline = validate_dependency(args.baseline_dir, args.pipeline_log)
        output = Path(args.output_dir).resolve()
        if output.exists():
            raise FileExistsError("Refusing to overwrite BN-recalibration output: " + str(output))
        command, differences = replay_command(baseline, output,
                                             args.bn_calibration_samples, args.bn_calibration_batch_size)
        (state_dir / "baseline_config.json").write_text(json.dumps(baseline, indent=2) + "\n")
        (state_dir / "command.json").write_text(json.dumps(command, indent=2) + "\n")
        record("RUNNING_BN_RECAL_MC45", differences=differences)
        print("COMMAND:", " ".join(command), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
        # Test-only hooks stop after real CSR generation; never label a bounded
        # rehearsal as a completed production accuracy evaluation.
        import os
        if os.environ.get("REHEARSAL_ONLY") == "true":
            record("REHEARSAL_INITIALIZATION_COMPLETE")
            return
        with (output / "corner_trials.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != 45 or len({row["corner"] for row in rows}) != 45:
            raise RuntimeError("Recalibrated run returned without all 45 corner trials")
        record("COMPLETE", corner_trials=len(rows))
    except Exception as error:
        record("FAILED", error=repr(error))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-pid", type=int, required=True)
    parser.add_argument("--wait-start-ticks", type=int, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--pipeline-log", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--bn-calibration-samples", type=int, default=5120)
    parser.add_argument("--bn-calibration-batch-size", type=int, default=256)
    args = parser.parse_args()
    if args.poll_seconds <= 0 or args.bn_calibration_samples <= 0 or args.bn_calibration_batch_size <= 0:
        parser.error("Poll interval and calibration sizes must be positive")
    run(args)


if __name__ == "__main__":
    main()

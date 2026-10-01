#!/usr/bin/env python3
"""Replay the pinned FS_V2_T1 two-trial evaluation across checkpoints.

Only --model_name, --model_dir, and artifact destinations are changed from
results/ff_unroll_fix_FS_V2_T1_2trials/command.json.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results/level3_fs_v2_t1_architecture_comparison_2trials"
BASE_COMMAND = ROOT / "results/ff_unroll_fix_FS_V2_T1_2trials/command.json"


MODELS = {
    "main_96c_4_5_4": {
        "model_dir": "saved_ckpt_runs/coupler_full_range_CiFAIR100_qf1_noENOB_fixedTiming_scaledRecipe1_zOvery1_relu0906_fixed_iq12_toggle_odexinit",
        "model_name": "TIMMQAT5bNoneaNT0p0mulTIMMPCNetNoBatchNorm_PCConvReLU6_0.0eps_ToggleODEXInitFFFB_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_C100_3K1S96C_0.25Dropout_16Layers4l5l4_2Pool_srrlDistill_a0p3_t2p0_CiFAIR_1REP",
    },
    "c36_72_8_10_ideal_pretrain": {
        "model_dir": "saved_ckpt_runs/coupler_full_range_CiFAIR100_twoStage_C36_72_poolPreExp_qf1_noENOB_iq12_toggle_odexinit",
        "model_name": "TIMMQAT5bNoneaNT0p0mulTIMMPCNetNoBatchNorm_PCConvReLU6_0.0eps_ToggleODEXInitFFFB_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_C100_3K1S72C_0.25Dropout_20Layers8l10l0_1Pool9_srrlDistill_a0p3_t2p0_CiFAIR_1REP",
        "expected_checkpoint_acc": 0.621,
    },
    "c36_72_8_10_measured_pretrain": {
        "model_dir": "saved_ckpt_runs/coupler_full_range_CiFAIR100_twoStage_C36_72_poolPreExp_pullbackDirect_qf1_noENOB_iq12_toggle_odexinit",
        "model_name": "TIMMQAT5bNoneaNT0p0mulTIMMPCNetNoBatchNorm_PCConvReLU6_0.0eps_ToggleODEXInitFFFB_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_C100_3K1S72C_0.25Dropout_20Layers8l10l0_1Pool9_srrlDistill_a0p3_t2p0_CiFAIR_1REP",
    },
    "c32_64_11_11_ideal_pretrain": {
        "model_dir": "saved_ckpt_runs/coupler_full_range_CiFAIR100_twoStage_C32_64_poolPreExp_qf1_noENOB_iq12_toggle_odexinit",
        "model_name": "TIMMQAT5bNoneaNT0p0mulTIMMPCNetNoBatchNorm_PCConvReLU6_0.0eps_ToggleODEXInitFFFB_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_C100_3K1S64C_0.25Dropout_24Layers11l11l0_1Pool12_srrlDistill_a0p3_t2p0_CiFAIR_1REP",
    },
    "c32_64_12_10_measured_pretrain": {
        "model_dir": "saved_ckpt_runs/coupler_full_range_CiFAIR100_twoStage_C32_64_poolPreExp_pullbackDirect_warmup5_qf1_noENOB_iq12_toggle_odexinit",
        "model_name": "TIMMQAT5bNoneaNT0p0mulTIMMPCNetNoBatchNorm_PCConvReLU6_0.0eps_ToggleODEXInitFFFB_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_C100_3K1S64C_0.25Dropout_24Layers12l10l0_1Pool13_srrlDistill_a0p3_t2p0_CiFAIR_1REP",
    },
}


def replace_arg(command: list[str], name: str, value: str) -> None:
    command[command.index(name) + 1] = value


def checkpoint_path(spec: dict[str, object]) -> Path:
    model_dir = ROOT / str(spec["model_dir"])
    model_name = str(spec["model_name"])
    return model_dir / model_name / f"{model_name}_best_ckpt.pth"


def checkpoint_metadata(path: Path) -> dict[str, object]:
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return {
        key: checkpoint.get(key)
        for key in ("acc", "epoch", "top1", "best_top1")
        if key in checkpoint
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("tags", nargs="*", choices=MODELS, default=list(MODELS))
    parser.add_argument("--allow-checkpoint-acc-mismatch", action="store_true")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    base = json.loads(BASE_COMMAND.read_text())["rerun"]
    overall: dict[str, object] = {}
    final_status = 0

    for tag in args.tags:
        spec = MODELS[tag]
        ckpt = checkpoint_path(spec)
        if not ckpt.is_file():
            overall[tag] = {"status": "missing_checkpoint", "checkpoint": str(ckpt)}
            final_status = 1
            continue

        metadata = checkpoint_metadata(ckpt)
        expected_acc = spec.get("expected_checkpoint_acc")
        actual_acc = metadata.get("acc")
        if (
            expected_acc is not None
            and actual_acc is not None
            and abs(float(actual_acc) - float(expected_acc)) > 5e-5
            and not args.allow_checkpoint_acc_mismatch
        ):
            overall[tag] = {
                "status": "checkpoint_acc_mismatch",
                "checkpoint": str(ckpt),
                "expected_acc": expected_acc,
                "metadata": metadata,
            }
            final_status = 1
            continue

        run_dir = OUT / tag
        run_dir.mkdir(parents=True, exist_ok=True)
        command = list(base)
        replace_arg(command, "--model_name", str(spec["model_name"]))
        replace_arg(command, "--model_dir", str(spec["model_dir"]))
        replace_arg(command, "--expanded_w_dir", str(run_dir / "expanded_weights"))
        if "--hw_val_path" in command:
            replace_arg(command, "--hw_val_path", str(run_dir / "hw_validation_data"))
        else:
            command += ["--hw_val_path", str(run_dir / "hw_validation_data")]

        (run_dir / "command.json").write_text(
            json.dumps(
                {
                    "base_command": str(BASE_COMMAND.relative_to(ROOT)),
                    "only_numerical_changes": ["--model_name", "--model_dir"],
                    "artifact_destination_changes": ["--expanded_w_dir", "--hw_val_path"],
                    "checkpoint": str(ckpt.relative_to(ROOT)),
                    "checkpoint_metadata": metadata,
                    "command": command,
                },
                indent=2,
            )
            + "\n"
        )

        log_path = run_dir / "FS_V2_T1.log"
        env = os.environ.copy()
        scanbase_bin = str(Path(command[0]).parent)
        env["PATH"] = scanbase_bin + os.pathsep + env.get("PATH", "")
        with log_path.open("w", buffering=1) as log:
            log.write("COMMAND: " + shlex.join(command) + "\n")
            result = subprocess.run(
                command,
                cwd=ROOT,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
            )

        text = log_path.read_text(errors="replace")
        trials = [
            {"trial": int(index), "accuracy_percent": float(accuracy)}
            for index, accuracy in re.findall(
                r"ABLATION_RESULT case=FS_V2_T1 trial_index=(\d+) accuracy=([0-9.]+)",
                text,
            )
        ]
        record: dict[str, object] = {
            "status": "complete" if result.returncode == 0 and len(trials) == 2 else "failed",
            "exitcode": result.returncode,
            "checkpoint": str(ckpt.relative_to(ROOT)),
            "checkpoint_metadata": metadata,
            "trials": trials,
        }
        if trials:
            values = [float(item["accuracy_percent"]) for item in trials]
            record["mean_percent"] = sum(values) / len(values)
        (run_dir / "results.json").write_text(json.dumps(record, indent=2) + "\n")
        overall[tag] = record
        if record["status"] != "complete":
            final_status = 1
        (OUT / "results.json").write_text(json.dumps(overall, indent=2) + "\n")

    (OUT / "results.json").write_text(json.dumps(overall, indent=2) + "\n")
    return final_status


if __name__ == "__main__":
    raise SystemExit(main())

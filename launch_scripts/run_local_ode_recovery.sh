#!/usr/bin/env bash
# Activate scanbase first. Optional first argument: checkpoint or run directory.
# Resumes pretraining only; does not automatically start finetuning.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
checkpoint="${1:-saved_ckpt_runs/cifar100_CiFAIR_C7_14_28_4l5l4_20260908_124210_iq12}"
if (( $# )); then shift; fi
export NUM_WORKERS=0 IS_SLURM=0 PYTHONUNBUFFERED=1
exec python -u scripts/resume_local_ode_training.py "$checkpoint" "$@"

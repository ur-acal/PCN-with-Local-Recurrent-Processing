#!/usr/bin/env bash
#set -euo pipefail

REPO_ROOT="/home/rzeng7/Desktop/research/repos/PCN-with-Local-Recurrent-Processing"
SLURM_SCRIPT="${REPO_ROOT}/launch_scripts/slurm_schedule_ode_train.sh"
LOG_DIR="${REPO_ROOT}/logs/scheduler_slurm/"
LOG_FILE="${LOG_DIR}/slurm_scheduler.log"

mkdir -p "${LOG_DIR}"

module swap slurm slurm/24.05.0.b1

nohup bash "${SLURM_SCRIPT}" >> "${LOG_FILE}" 2>&1 &

tail -n +1 -f "${LOG_FILE}"

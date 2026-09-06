#!/bin/bash

# Optionally wait for another local process, reparameterize a feedforward
# checkpoint for a longer fixed stage time, then launch physical fine-tuning.
set -e

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO_ROOT}"

# Delayed/background launchers do not reliably inherit the interactive conda
# environment or the user-local PATH.  Establish both here before invoking
# Python or the scangen CLI used while constructing the CiFAIR datasets.
CONDA_ENV_NAME="${CONDA_ENV_NAME:-scanbase}"
CONDA_EXE_PATH="${CONDA_EXE:-${HOME}/anaconda3/bin/conda}"
if [[ "${CONDA_DEFAULT_ENV:-}" != "${CONDA_ENV_NAME}" ]]; then
  if [[ ! -x "${CONDA_EXE_PATH}" ]]; then
    echo "Conda executable not found: ${CONDA_EXE_PATH}" >&2
    exit 1
  fi
  source "$("${CONDA_EXE_PATH}" info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV_NAME}"
fi
export PATH="${HOME}/.local/bin:${PATH}"

SOURCE_MODEL_CKPT="${MODEL_CKPT:?MODEL_CKPT is required}"
RESCALED_MODEL_CKPT="${RESCALED_MODEL_CKPT:?RESCALED_MODEL_CKPT is required}"
FIXED_TIMING_SCALE_FACTOR="${FIXED_TIMING_SCALE_FACTOR:-3}"
SOURCE_TOGGLE_Y_TIME="${SOURCE_TOGGLE_Y_TIME:-5e-9}"
TARGET_TOGGLE_Y_TIME="${TARGET_TOGGLE_Y_TIME:-15e-9}"
QUEUE_STATUS_FILE="${QUEUE_STATUS_FILE:-${RESCALED_MODEL_CKPT}.queue_status}"
QUEUE_MARKER_FILE="${QUEUE_STATUS_FILE}.queued_at"
COMPLETION_CHECKPOINT="${COMPLETION_CHECKPOINT:-}"
if [[ -z "${COMPLETION_CHECKPOINT}" &&
      "${SOURCE_MODEL_CKPT}" == *_best_ckpt.pth ]]; then
  COMPLETION_CHECKPOINT="${SOURCE_MODEL_CKPT%_best_ckpt.pth}_last_ckpt.pth"
fi

mkdir -p "$(dirname "${QUEUE_STATUS_FILE}")"
touch "${QUEUE_MARKER_FILE}"
printf 'QUEUED pid=%s source=%s\n' "${WAIT_FOR_PID:-none}" \
  "${SOURCE_MODEL_CKPT}" > "${QUEUE_STATUS_FILE}"

queue_exit_status() {
  status=$?
  if [[ ${status} -ne 0 ]]; then
    message="FAILED exit_status=${status}"
    echo "${message}" >&2
    printf '%s\n' "${message}" > "${QUEUE_STATUS_FILE}"
  fi
}
trap queue_exit_status EXIT

if [[ -n "${WAIT_FOR_PID:-}" ]]; then
  echo "Waiting for PID ${WAIT_FOR_PID} before rescaling and fine-tuning."
  printf 'WAITING pid=%s\n' "${WAIT_FOR_PID}" > "${QUEUE_STATUS_FILE}"
  while kill -0 "${WAIT_FOR_PID}" 2>/dev/null; do
    sleep 30
  done
  if [[ -z "${COMPLETION_CHECKPOINT}" ||
        ! -f "${COMPLETION_CHECKPOINT}" ||
        ! "${COMPLETION_CHECKPOINT}" -nt "${QUEUE_MARKER_FILE}" ]]; then
    echo "Pretraining ended without a fresh completion checkpoint: ${COMPLETION_CHECKPOINT:-unset}" >&2
    exit 1
  fi
fi

printf 'RESCALING source=%s output=%s\n' "${SOURCE_MODEL_CKPT}" \
  "${RESCALED_MODEL_CKPT}" > "${QUEUE_STATUS_FILE}"
rescale_cmd=(
  python scripts/rescale_feedforward_fixed_timing_checkpoint.py
  --input "${SOURCE_MODEL_CKPT}"
  --output "${RESCALED_MODEL_CKPT}"
  --factor "${FIXED_TIMING_SCALE_FACTOR}"
  --source-time "${SOURCE_TOGGLE_Y_TIME}"
  --target-time "${TARGET_TOGGLE_Y_TIME}"
)
if [[ "${RESCALE_OVERWRITE:-false}" =~ ^([Tt][Rr][Uu][Ee]|1|[Yy][Ee][Ss])$ ]]; then
  rescale_cmd+=(--overwrite)
fi
"${rescale_cmd[@]}"

export MODEL_CKPT="${RESCALED_MODEL_CKPT}"
export TOGGLE_TIMING_MODE=fixed
export TOGGLE_Y_TIME="${TARGET_TOGGLE_Y_TIME}"
printf 'FINE_TUNING checkpoint=%s time=%s\n' "${MODEL_CKPT}" \
  "${TOGGLE_Y_TIME}" > "${QUEUE_STATUS_FILE}"
./launch_scripts/run_feedforward_physical_ft.sh
printf 'COMPLETED checkpoint=%s\n' "${MODEL_CKPT}" > "${QUEUE_STATUS_FILE}"
echo "Rescaled feedforward fine-tuning completed successfully."
trap - EXIT

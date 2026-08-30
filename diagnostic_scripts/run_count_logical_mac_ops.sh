#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DEFAULT_MODEL_NAME="TIMMQAT5bNoneaNT0p0mulTIMMPCNetNoBatchNorm_PCConvReLU6_0.0eps_ToggleODEXInitFFFB_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_C100_3K1S96C_0.25Dropout_16Layers4l5l4_2Pool_srrlDistill_a0p3_t2p0_CiFAIR_1REP"
DEFAULT_MODEL_DIR="${REPO_ROOT}/saved_ckpt_runs/coupler_v2_CiFAIR100_qf1_noENOB_matchdistill_srrl_noRE_toggle_odexinit"

MODEL_NAME="${MODEL_NAME:-${DEFAULT_MODEL_NAME}}"
MODEL_DIR="${MODEL_DIR:-${DEFAULT_MODEL_DIR}}"
CHECKPOINT="${CHECKPOINT:-${MODEL_DIR}/${MODEL_NAME}/${MODEL_NAME}_best_ckpt.pth}"
TOGGLE_CYCLES="${TOGGLE_CYCLES:-5}"
INPUT_HEIGHT="${INPUT_HEIGHT:-16}"
INPUT_WIDTH="${INPUT_WIDTH:-16}"
PYTHON="${PYTHON:-python}"

args=(
  --checkpoint "${CHECKPOINT}"
  --toggle-cycles "${TOGGLE_CYCLES}"
  --input-height "${INPUT_HEIGHT}"
  --input-width "${INPUT_WIDTH}"
)
if [[ -n "${OUTPUT_FILE:-}" ]]; then
  args+=(--output "${OUTPUT_FILE}")
fi
if [[ -n "${WEIGHT_BITS:-}" ]]; then
  args+=(--weight-bits "${WEIGHT_BITS}")
fi

cd "${REPO_ROOT}"
exec "${PYTHON}" diagnostic_scripts/count_logical_mac_ops.py "${args[@]}" "$@"

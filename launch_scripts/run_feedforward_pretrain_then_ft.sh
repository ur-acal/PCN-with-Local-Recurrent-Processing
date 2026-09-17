#!/bin/bash

# Sequential pretraining, physical QAT, and (TC only) unrolled evaluation.
set -eo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO_ROOT}"

MODEL_NAME="${MODEL_NAME:?MODEL_NAME is required}"
TASK="${TASK:-cifar100}"
EXP_PREFIX="${EXP_PREFIX:-feedforward_physical}"
PRETRAIN_OUTPUT_DIR="${PRETRAIN_OUTPUT_DIR:-./saved_ckpt_runs/${EXP_PREFIX}_pretrain}"
FT_OUTPUT_DIR="${FT_OUTPUT_DIR:-./saved_ckpt_runs/${EXP_PREFIX}_ft}"
export MODEL_NAME TASK

if [[ "${INPUT_QUANT_BITS:-none}" != "none" &&
      "${PRETRAIN_OUTPUT_DIR,,}" != *"_iq${INPUT_QUANT_BITS,,}"* ]]; then
  PRETRAIN_OUTPUT_DIR+="_iq${INPUT_QUANT_BITS}"
fi
if [[ "${CENTER_STUDENT_INPUT:-false}" =~ ^([Tt][Rr][Uu][Ee]|1|[Yy][Ee][Ss])$ &&
      "${PRETRAIN_OUTPUT_DIR,,}" != *"_ctr"* ]]; then
  PRETRAIN_OUTPUT_DIR+="_ctr"
fi

# Source the stage builders in the worker shell; they invoke Python directly.
# Keep stage-specific optimization/augmentation defaults local so pretraining's
# resolved values cannot become fine-tuning overrides.
feedforward_run_stage() {
  local OUTPUT_DIR="$2"
  local EXTRA_OVERRIDE="${EXTRA_OVERRIDE:-}"
  local TIMM_AUG_LEVEL="${TIMM_AUG_LEVEL:-}"
  local _DEFAULT_OVERRIDE cmd pair env_name arg_name INPUT_PREPROCESS_ARGS
  source "$1"
}

feedforward_run_stage ./launch_scripts/run_feedforward_cifar_pretrain.sh "${PRETRAIN_OUTPUT_DIR}"

checkpoint_name="custom_noresize_${TASK}_${MODEL_NAME}"
checkpoint_selection=best
if [[ "${FINAL_EVAL_ONLY:-false}" == true ]]; then
  checkpoint_selection=last
fi
MODEL_CKPT="${PRETRAIN_OUTPUT_DIR}/${TASK}/custom_noresize/${MODEL_NAME}/${checkpoint_name}/${checkpoint_name}_${checkpoint_selection}_ckpt.pth"
if [[ ! -f "${MODEL_CKPT}" ]]; then
  echo "Pretraining checkpoint not found: ${MODEL_CKPT}" >&2
  exit 1
fi

export MODEL_CKPT
feedforward_run_stage ./launch_scripts/run_feedforward_physical_ft.sh "${FT_OUTPUT_DIR}"

if [[ "${TC_FEEDFORWARD:-false}" == true ]]; then
  # Use the same preprocessing inference/suffix logic as training. FT can infer
  # these values from the pretraining checkpoint even when no env args are set.
  FT_RUN_DIR="$(python - "${FT_OUTPUT_DIR}" "${MODEL_CKPT}" <<'PY'
import os, sys
from input_preprocessing import append_preprocessing_suffix, resolve_preprocessing
bits = os.environ.get('INPUT_QUANT_BITS', 'none').lower()
bits = None if bits in ('', 'none') else int(bits)
center = os.environ.get('CENTER_STUDENT_INPUT', 'auto').lower()
center = None if center in ('', 'none', 'auto') else center in ('true', '1', 'yes')
bits, center = resolve_preprocessing(sys.argv[2], bits, center)
print(append_preprocessing_suffix(sys.argv[1], bits, center))
PY
)"
  MODEL_CKPT="${FT_RUN_DIR}/${TASK}/custom_noresize/${MODEL_NAME}/${checkpoint_name}/${checkpoint_name}_full_param_${checkpoint_selection}_ckpt.pth"
  if [[ ! -f "${MODEL_CKPT}" ]]; then
    echo "Fine-tuning checkpoint not found; refusing evaluation: ${MODEL_CKPT}" >&2
    exit 1
  fi
  export MODEL_CKPT
  export RESULT_PATH="${RESULT_PATH:-./results/${EXP_PREFIX}_tc_eval}"
  EVAL_LOG="${EVAL_LOG:-${RESULT_PATH}/evaluation.log}"
  mkdir -p "${RESULT_PATH}" "$(dirname "${EVAL_LOG}")"
  echo "[TC EVAL] checkpoint=${MODEL_CKPT}; results=${RESULT_PATH}"
  # The stage script resets FT curve sharing/ReLU-bank settings and uses the
  # same common TC hardware defaults as PCN (10 trials unless overridden).
  feedforward_run_stage ./launch_scripts/run_feedforward_physical_eval.sh "${FT_RUN_DIR}" 2>&1 | tee "${EVAL_LOG}"
  echo "TC pretraining, fine-tuning and evaluation completed successfully."
fi

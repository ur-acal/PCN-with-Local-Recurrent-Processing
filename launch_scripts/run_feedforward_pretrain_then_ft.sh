#!/bin/bash

# Sequential pretraining/physical QAT, or explicit FT-only/FT-and-eval modes.
set -eo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO_ROOT}"
source ./launch_scripts/normalize_validation_mode.sh || exit 2
mode="${mode:-default}"
case "$mode" in default|pretrain_only|ft_only|ft_and_eval|pretrain_ft_and_eval) ;; *) echo 'Invalid pipeline mode' >&2; exit 2 ;; esac
if [[ "$mode" == pretrain_ft_and_eval && "${TC_FEEDFORWARD:-false}" == true ]]; then
  echo 'pretrain_ft_and_eval is a toggle pipeline; TC retains mode=default' >&2; exit 2
fi
if [[ "$mode" == pretrain_ft_and_eval ]]; then
  export FINAL_EVAL_ONLY="${FINAL_EVAL_ONLY:-true}"
  export PRETRAIN_ACTIVATION_CORNER_MODE="${PRETRAIN_ACTIVATION_CORNER_MODE:-fixed}"
fi
SELECT_ONE_OVER_Q="${SELECT_ONE_OVER_Q:-$([[ "$mode" == pretrain_ft_and_eval ]] && echo true || echo false)}"
case "$SELECT_ONE_OVER_Q" in true|false) ;; *) echo 'SELECT_ONE_OVER_Q must be true or false' >&2; exit 2;; esac
if [[ "$SELECT_ONE_OVER_Q" == true && "${TC_FEEDFORWARD:-false}" == true ]]; then
  echo 'Automatic q selection currently supports toggle CNNs only' >&2; exit 2
fi
if [[ "$SELECT_ONE_OVER_Q" == true && "${VALIDATION_MODE}" == true ]]; then
  echo 'Automatic q selection does not yet support the held-out validation split' >&2; exit 2
fi
if [[ "$mode" == pretrain_ft_and_eval || "$SELECT_ONE_OVER_Q" == true ]]; then
  _ft_bank="${NONLINEAR_R_TABLE:-coupler_monte}"
  _eval_bank="${MC_COUPLER_NONLINEAR_VARIATION_SOURCE:-${_ft_bank}}"
  _ft_bank="${_ft_bank%/}"; _eval_bank="${_eval_bank%/}"
  [[ "${_ft_bank}" == */* ]] || _ft_bank="hardware_data/mc_45_corners/${_ft_bank}"
  [[ "${_eval_bank}" == */* ]] || _eval_bank="hardware_data/mc_45_corners/${_eval_bank}"
  [[ "$(realpath -m "${_ft_bank}")" == "$(realpath -m "${_eval_bank}")" ]] || {
    echo 'Conflicting FT/evaluation coupler families' >&2; exit 2;
  }
fi

MODEL_NAME="${MODEL_NAME:?MODEL_NAME is required}"
TASK="${TASK:-cifar100}"
EXP_PREFIX="${EXP_PREFIX:-feedforward_physical}"
if [[ "${VALIDATION_MODE}" == true && "${EXP_PREFIX}" != *"_val5k"* ]]; then
  EXP_PREFIX+="_val5k"
fi
PRETRAIN_OUTPUT_DIR="${PRETRAIN_OUTPUT_DIR:-./saved_ckpt_runs/${EXP_PREFIX}_pretrain}"
FT_OUTPUT_DIR="${FT_OUTPUT_DIR:-./saved_ckpt_runs/${EXP_PREFIX}_ft}"
PRETRAIN_OUTPUT_DIR="$(validation_output_path "${PRETRAIN_OUTPUT_DIR}")"
FT_OUTPUT_DIR="$(validation_output_path "${FT_OUTPUT_DIR}")"
export MODEL_NAME TASK
if [[ "${TC_FEEDFORWARD:-false}" != true ]]; then
  # The FT builder and MC45 builder must interpret the CSV identically.
  if [[ -n "${NONLINEAR_R_MC_QUANTITY:-}" && -n "${MC_COUPLER_NONLINEAR_VARIATION_QUANTITY:-}" &&
        "${NONLINEAR_R_MC_QUANTITY}" != "${MC_COUPLER_NONLINEAR_VARIATION_QUANTITY}" ]]; then
    echo "Conflicting nonlinear-R quantity variables" >&2; exit 2
  fi
  export NONLINEAR_R_MC_QUANTITY="${NONLINEAR_R_MC_QUANTITY:-${MC_COUPLER_NONLINEAR_VARIATION_QUANTITY:-conductance}}"
  case "${NONLINEAR_R_MC_QUANTITY}" in conductance|resistance) ;; *) echo "Invalid nonlinear-R quantity" >&2; exit 2;; esac
  export MC_COUPLER_NONLINEAR_VARIATION_QUANTITY="${NONLINEAR_R_MC_QUANTITY}"
fi

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
  local ACTIVATION_CORNER_MODE="${ACTIVATION_CORNER_MODE:-fixed}"
  if [[ "$1" == *run_feedforward_cifar_pretrain.sh ]]; then
    ACTIVATION_CORNER_MODE="${PRETRAIN_ACTIVATION_CORNER_MODE:-${ACTIVATION_CORNER_MODE}}"
  fi
  source "$1"
}

if [[ "$mode" == default || "$mode" == pretrain_only || "$mode" == pretrain_ft_and_eval ]]; then
  feedforward_run_stage ./launch_scripts/run_feedforward_cifar_pretrain.sh "${PRETRAIN_OUTPUT_DIR}"
else
  # Same stage-default initialization as pretraining, without launching Python.
  source ./launch_scripts/tc_feedforward_args.sh pretrain
fi
if [[ "$mode" == pretrain_only ]]; then exit 0; fi

checkpoint_name="custom_noresize_${TASK}_${MODEL_NAME}"
checkpoint_selection=best
if [[ "${FINAL_EVAL_ONLY:-false}" == true && "${VALIDATION_MODE}" != true ]]; then
  checkpoint_selection=last
fi
# Preserve TC's existing checkpoint discovery and default-mode pretraining.
# Only explicit toggle FT modes accept a directly supplied source checkpoint.
if [[ "$mode" == default || "$mode" == pretrain_ft_and_eval || "${TC_FEEDFORWARD:-false}" == true || -z "${MODEL_CKPT:-}" ]]; then
  MODEL_CKPT="${PRETRAIN_OUTPUT_DIR}/${TASK}/custom_noresize/${MODEL_NAME}/${checkpoint_name}/${checkpoint_name}_${checkpoint_selection}_ckpt.pth"
fi
if [[ ! -f "${MODEL_CKPT}" ]]; then
  echo "Pretraining checkpoint not found: ${MODEL_CKPT}" >&2
  exit 1
fi

export MODEL_CKPT
if [[ "$SELECT_ONE_OVER_Q" == true ]]; then
  export SELECTION_OUTPUT_DIR="${SELECTION_OUTPUT_DIR:-./results/${EXP_PREFIX}_selection}"
  # The FT builder supplies the real, resolved hardware/head/architecture args.
  ( export FEEDFORWARD_SELECT_STAGE=true
    feedforward_run_stage ./launch_scripts/run_feedforward_physical_ft.sh "${FT_OUTPUT_DIR}"
  )
  selected_file="${SELECTION_OUTPUT_DIR}/selected_one_over_q.txt"
  [[ -s "$selected_file" ]] || { echo "Missing q selection: $selected_file" >&2; exit 1; }
  export TOGGLE_ONE_OVER_Q="$(<"$selected_file")"
  echo "[SELECTION] one_over_q=${TOGGLE_ONE_OVER_Q}; source=${MODEL_CKPT}"
fi
feedforward_run_stage ./launch_scripts/run_feedforward_physical_ft.sh "${FT_OUTPUT_DIR}"
if [[ "$mode" == ft_only ]]; then exit 0; fi

if [[ "${TC_FEEDFORWARD:-false}" == true || "$mode" == ft_and_eval || "$mode" == pretrain_ft_and_eval ]]; then
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
  if [[ "${TC_FEEDFORWARD:-false}" == true ]]; then
    DEFAULT_RESULT_PATH="./results/${EXP_PREFIX}_tc_eval"
  else
    DEFAULT_RESULT_PATH="./results/${EXP_PREFIX}_mc45"
  fi
  export RESULT_PATH="${RESULT_PATH:-${DEFAULT_RESULT_PATH}}"
  EVAL_LOG="${EVAL_LOG:-${RESULT_PATH}/evaluation.log}"
  mkdir -p "${RESULT_PATH}" "$(dirname "${EVAL_LOG}")"
  echo "[EVAL] checkpoint=${MODEL_CKPT}; results=${RESULT_PATH}"
  if [[ "${TC_FEEDFORWARD:-false}" == true ]]; then
    # Preserve the existing TC evaluation defaults and stage arguments.
    feedforward_run_stage ./launch_scripts/run_feedforward_physical_eval.sh "${FT_RUN_DIR}" 2>&1 | tee "${EVAL_LOG}"
    echo "TC pretraining, fine-tuning and evaluation completed successfully."
  else
    feedforward_run_corner_eval() (
      # FT uses shared curves and averaged operations; MC45 uses full corner
      # banks, per-coupler/per-spin samples, pulse Level 3 and corner DTC.
      table="${NONLINEAR_R_TABLE:-coupler_monte}"
      table="${table%/}"
      if [[ "$mode" == pretrain_ft_and_eval || "$SELECT_ONE_OVER_Q" == true ]]; then
        # MC45 resolves relative sources beneath its bank root, whereas FT
        # resolves them beneath the checkout. Use the validated absolute path.
        export MC_COUPLER_NONLINEAR_VARIATION_SOURCE="$(realpath "${_eval_bank}")"
      fi
      export OUTPUT_DIR="${RESULT_PATH}" PHYSICAL_LEVEL=3 \
        EXPANDED_WEIGHT_DIR="${EXPANDED_WEIGHT_DIR:-./expanded_weights/${EXP_PREFIX}}" \
        ONE_OVER_Q="${TOGGLE_ONE_OVER_Q:-${ONE_OVER_Q:-1}}" \
        MC_COUPLER_NONLINEAR_VARIATION_SOURCE="${MC_COUPLER_NONLINEAR_VARIATION_SOURCE:-${table##*/}}" \
        MC_COUPLER_NONLINEAR_VARIATION_QUANTITY="${NONLINEAR_R_MC_QUANTITY:-conductance}" \
        MC_COUPLER_NOMINAL_R="${MC_COUPLER_NOMINAL_R:-${R_VAL:-50e3}}" \
        FULL_45_CORNER_C="${FULL_45_CORNER_C:-${C_VAL:-500e-15}}" \
        MC_RELU_MONTE_CARLO_SOURCE="${MC_RELU_MONTE_CARLO_SOURCE:-relu_monteCarlo}" \
        NONLINEAR_R_CURVE_SHARING=per_coupler ACTIVATION_CURVE_SHARING=per_spin \
        ENABLE_DTC_NONIDEALITY="${EVAL_ENABLE_DTC_NONIDEALITY:-true}" \
        ENABLE_BN_RECALIBRATION="${ENABLE_BN_RECALIBRATION:-false}" \
        N_TRIALS="${N_TRIALS:-1}" JOBS="${JOBS:-1}"
      bash ./launch_scripts/run_feedforward_mc45_ablation.sh
    )
    feedforward_run_corner_eval 2>&1 | tee "${EVAL_LOG}"
    if [[ "${RUN_BN_RECAL_EVAL:-false}" == true && "${ENABLE_BN_RECALIBRATION:-false}" != true ]]; then
      (
        export RESULT_PATH="${BN_RECAL_RESULT_PATH:-${RESULT_PATH}_bnrecal}"
        export ENABLE_BN_RECALIBRATION=true \
          BN_CALIBRATION_SAMPLES="${BN_CALIBRATION_SAMPLES:-5120}" \
          BN_CALIBRATION_BATCH_SIZE="${BN_CALIBRATION_BATCH_SIZE:-256}"
        # Same checkpoint, seeds, corners and trial count; only calibration
        # and its separate result/log destinations differ.
        mkdir -p "${RESULT_PATH}"
        feedforward_run_corner_eval 2>&1 | tee "${RESULT_PATH}/evaluation.log"
      )
    fi
    echo "Toggle feedforward fine-tuning and MC45 evaluation completed successfully."
  fi
fi

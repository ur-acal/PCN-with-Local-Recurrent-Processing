#!/bin/bash

MODEL_NAME="${MODEL_NAME:-}"
MODEL_DIR="${MODEL_DIR:-}"
MODEL_ARGS=()
if [[ -n "${MODEL_NAME}" ]]; then
  MODEL_ARGS+=(--model_name "${MODEL_NAME}")
fi
if [[ -n "${MODEL_DIR}" ]]; then
  MODEL_ARGS+=(--model_dir "${MODEL_DIR}")
fi

python diagnostic_scripts/plot_toggle_current_distributions.py \
  "${MODEL_ARGS[@]}" \
  --corners "${CORNERS:-FS_V2_T1}" \
  --n_trials "${N_TRIALS:-10}" \
  --batch_size "${BATCH_SIZE:-128}" \
  "$@"

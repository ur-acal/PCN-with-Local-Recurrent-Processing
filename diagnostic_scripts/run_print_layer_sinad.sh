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
CORNER="${CORNER:-TT_V1_T1}"
TRIAL="${TRIAL:-0}"
BATCH_SIZE="${BATCH_SIZE:-128}"

python diagnostic_scripts/print_layer_sinad.py \
  "${MODEL_ARGS[@]}" \
  --corner "${CORNER}" \
  --trial "${TRIAL}" \
  --batch_size "${BATCH_SIZE}" \
  "$@"

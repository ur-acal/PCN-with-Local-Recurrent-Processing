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

python diagnostic_scripts/print_weight_distribution.py "${MODEL_ARGS[@]}" "$@"


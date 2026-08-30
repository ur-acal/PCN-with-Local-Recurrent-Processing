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
TOGGLE_STEP="${TOGGLE_STEP:-1}"
N_SPINS="${N_SPINS:-50}"

LAYER_ARGS=()
if [[ -n "${LAYER:-}" ]]; then
  LAYER_ARGS+=(--layer "${LAYER}")
fi

python diagnostic_scripts/plot_spin_trajectories.py \
  "${MODEL_ARGS[@]}" \
  --corner "${CORNER}" \
  --trial "${TRIAL}" \
  --batch_size "${BATCH_SIZE}" \
  --toggle_step "${TOGGLE_STEP}" \
  --n_spins "${N_SPINS}" \
  "${LAYER_ARGS[@]}" \
  "$@"

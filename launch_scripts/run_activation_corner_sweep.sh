#!/bin/bash
set -euo pipefail

N_TRIALS="${N_TRIALS:-10}"
BASE_SEED="${BASE_SEED:-20260721}"
OUTPUT_DIR="${OUTPUT_DIR:-results/activation_corner_reproduction}"
MODEL_DIR="${MODEL_DIR:-saved_ckpt}"
EXPANDED_W_DIR="${EXPANDED_W_DIR:-expanded_weights}"
TEST_BS="${TEST_BS:-128}"

CORNER_ARGS=()
if [[ -n "${CORNERS:-}" ]]; then
  read -r -a SELECTED_CORNERS <<< "${CORNERS//,/ }"
  CORNER_ARGS=(--corners "${SELECTED_CORNERS[@]}")
fi

python -u scripts/run_activation_corner_sweep.py \
  --n_trials "${N_TRIALS}" \
  --base_seed "${BASE_SEED}" \
  --model_dir "${MODEL_DIR}" \
  --expanded_w_dir "${EXPANDED_W_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --test_bs "${TEST_BS}" \
  "${CORNER_ARGS[@]}"

# Explicit final aggregation step: create and print the Markdown table.
python -u scripts/summarize_activation_corner_sweep.py \
  --input_dir "${OUTPUT_DIR}" \
  --summary_path "${OUTPUT_DIR}/summary.md"

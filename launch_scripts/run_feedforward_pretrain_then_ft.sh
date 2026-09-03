#!/bin/bash

# Sequential unitless physical pretraining and level-2 physical QAT.
set -e

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

OUTPUT_DIR="${PRETRAIN_OUTPUT_DIR}" \
  ./launch_scripts/run_feedforward_cifar_pretrain.sh

checkpoint_name="custom_noresize_${TASK}_${MODEL_NAME}"
MODEL_CKPT="${PRETRAIN_OUTPUT_DIR}/${TASK}/custom_noresize/${MODEL_NAME}/${checkpoint_name}/${checkpoint_name}_best_ckpt.pth"
if [[ ! -f "${MODEL_CKPT}" ]]; then
  echo "Pretraining checkpoint not found: ${MODEL_CKPT}" >&2
  exit 1
fi

export MODEL_CKPT
OUTPUT_DIR="${FT_OUTPUT_DIR}" \
  ./launch_scripts/run_feedforward_physical_ft.sh

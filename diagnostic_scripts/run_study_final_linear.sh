#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export DATALOADER_NUM_WORKERS=0
export PYTHONUNBUFFERED=1
REFERENCE_LOG="${REFERENCE_LOG:-results/coupler_full_range_CiFAIR100_qf1_noENOB_zOvery1_relu0906_fixed_FS_V2_T1_2trials/FS_V2_T1.log}"
OUTPUT_DIR="${OUTPUT_DIR:-results/final_linear_study/pinned_FS_V2_T1}"
PLOT_ARGS=()
if [[ "${PLOT_ONLY:-false}" == true ]]; then
  PLOT_ARGS+=(--plot_only)
fi
python diagnostic_scripts/study_final_linear.py \
  --reference_log "$REFERENCE_LOG" --output_dir "$OUTPUT_DIR" \
  --batch_index "${BATCH_INDEX:-0}" --drop_pp "${DROP_PP:-1}" "${PLOT_ARGS[@]}"

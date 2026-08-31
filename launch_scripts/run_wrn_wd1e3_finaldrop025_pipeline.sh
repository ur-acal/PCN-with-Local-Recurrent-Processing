#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/home/rongzeng/anaconda3/envs/scanbase/bin/python}"
PIPELINE_ROOT="${PIPELINE_ROOT:-$ROOT/logs/wrn_wd1e3_finaldrop025_pipeline}"
TRAIN_OUTPUT_ROOT="${TRAIN_OUTPUT_ROOT:-$ROOT/logs/wrn_wd1e3_finaldrop025}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$ROOT/checkpoint/baselines_wd1e3_finaldrop025}"
MISMATCH_OUTPUT_ROOT="${MISMATCH_OUTPUT_ROOT:-$ROOT/logs/wrn_wd1e3_finaldrop025_mismatch_full}"
REPORT_OUTPUT_ROOT="${REPORT_OUTPUT_ROOT:-$ROOT/logs/combined_pcn_wrn_mismatch_finaldrop025}"
CURRENT_STAGE="initializing"

mkdir -p "$PIPELINE_ROOT"

write_state() {
  local status="$1" stage="$2" detail="${3:-}"
  printf '{\n  "status": "%s",\n  "stage": "%s",\n  "detail": "%s",\n  "controller_pid": %d,\n  "updated_epoch_seconds": %d\n}\n' \
    "$status" "$stage" "$detail" "$$" "$(date +%s)" >"$PIPELINE_ROOT/state.json.tmp"
  mv "$PIPELINE_ROOT/state.json.tmp" "$PIPELINE_ROOT/state.json"
}

on_error() {
  local code=$?
  write_state failed "$CURRENT_STAGE" "exit_code_$code"
  exit "$code"
}
trap on_error ERR

cat >"$PIPELINE_ROOT/paths.json" <<EOF
{
  "training_output": "$TRAIN_OUTPUT_ROOT",
  "checkpoint_root": "$CHECKPOINT_ROOT",
  "mismatch_output": "$MISMATCH_OUTPUT_ROOT",
  "report_output": "$REPORT_OUTPUT_ROOT"
}
EOF

CURRENT_STAGE="training"
write_state running "$CURRENT_STAGE" "reuse_two_pilots_and_train_six_with_gate_bypass"
OUTPUT_ROOT="$TRAIN_OUTPUT_ROOT" CHECKPOINT_ROOT="$CHECKPOINT_ROOT" SKIP_ACCURACY_GATE=true \
  launch_scripts/run_wrn_wd1e3_finaldrop025_train.sh

CURRENT_STAGE="mismatch"
write_state running "$CURRENT_STAGE" "unfolded_recal_bn_max_rms_and_multiplicative"
OUTPUT_ROOT="$MISMATCH_OUTPUT_ROOT" CHECKPOINT_ROOT="$CHECKPOINT_ROOT" \
  launch_scripts/run_wrn_wd1e3_finaldrop025_mismatch_queue.sh

CURRENT_STAGE="completed"
write_state completed "$CURRENT_STAGE" "training_and_mismatch_results_ready"


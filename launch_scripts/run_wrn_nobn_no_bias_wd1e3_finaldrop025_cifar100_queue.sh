#!/bin/bash

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CURRENT_SERVICE="${CURRENT_SERVICE:-scan-wrn-nobn-no-bias-cifar100.service}"
CURRENT_STATE="${CURRENT_STATE:-$ROOT/logs/wrn_nobn_no_bias_cifar100/state.json}"

while systemctl --user is-active --quiet "$CURRENT_SERVICE"; do
  sleep 30
done

if ! grep -q '"status": "complete"' "$CURRENT_STATE"; then
  echo "Prerequisite training did not complete successfully: $CURRENT_STATE" >&2
  exit 1
fi

export OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/logs/wrn_nobn_no_bias_wd1e3_finaldrop025_cifar100}"
export CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$OUTPUT_ROOT/checkpoints}"
export PARALLELISM="${PARALLELISM:-4}"
export OVERRIDE="weight_decay=0.001,final_dropout_rate=0.25,collapse_monitor_enabled=true,collapse_not_learned_deadline_epoch=30,collapse_not_learned_loss_ratio=0.9"
export RECIPE_CHANGES="weight_decay=0.001; final_dropout_rate=0.25"

exec /usr/bin/bash "$ROOT/launch_scripts/run_wrn_nobn_no_bias_cifar100.sh"

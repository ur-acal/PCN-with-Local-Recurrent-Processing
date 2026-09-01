#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/home/rongzeng/anaconda3/envs/scanbase/bin/python}"
DATA_DIR="${DATA_DIR:-$ROOT/../data}"
CHECKPOINT="${CHECKPOINT:-$ROOT/../ScAN-PCN/checkpoint/baselines/cifar100/adapt_noresize_scratch/resnet18/adapt_noresize_scratch_cifar100_resnet18/adapt_noresize_scratch_cifar100_resnet18_best_ckpt.pth}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/logs/resnet18_max_sqrt_bn_study}"
LEVELS="${LEVELS:-0,0.02,0.03,0.05,0.07}"
NOISY_TRIALS="${NOISY_TRIALS:-5}"

[[ -x "$PYTHON_BIN" ]] || { echo "Missing Python: $PYTHON_BIN" >&2; exit 1; }
[[ -s "$CHECKPOINT" ]] || { echo "Missing checkpoint: $CHECKPOINT" >&2; exit 1; }
mkdir -p "$OUTPUT_ROOT/logs"

run_condition() {
  local condition="$1" output="$OUTPUT_ROOT/$1" log="$OUTPUT_ROOT/logs/$1.log"
  shift
  if find "$output" -maxdepth 1 -name 'summary_*.csv' -type f -size +0c 2>/dev/null | grep -q .; then
    echo "Skipping completed condition: $condition"
    return 0
  fi
  mkdir -p "$output"
  "$PYTHON_BIN" -u baseline/run_baseline.py \
    --model_list resnet18 \
    --dataset cifar100 \
    --data_dir "$DATA_DIR" \
    --checkpoint_map "resnet18=$CHECKPOINT" \
    --case adapt_noresize_scratch \
    --pretrained false \
    --batch_size 128 \
    --num_workers 2 \
    --device cuda \
    --seed 123 \
    --noise_level_list "$LEVELS" \
    --noisy_trials "$NOISY_TRIALS" \
    --noise_type additive \
    --additive_scale_mode max_sqrt \
    --noise_to_norm false \
    --exclude_conv_bias_from_mismatch true \
    --results_dir "$output" \
    "$@" >"$log" 2>&1
}

run_condition folded_frozen_bn \
  --fold_norm true \
  --fold_norm_mode resnet_postact_no_mismatch_bias \
  --bn_recalibration_enabled false &
folded_pid=$!

run_condition unfused_recal_bn \
  --fold_norm false \
  --bn_recalibration_enabled true \
  --bn_recalibration_num_samples 5120 \
  --bn_recalibration_batch_size 128 \
  --bn_recalibration_subset_seed 20240618 \
  --bn_recalibration_num_workers 2 \
  --bn_recalibration_diagnostics_dir "$OUTPUT_ROOT/diagnostics/unfused_recal_bn" &
recal_pid=$!

status=0
wait "$folded_pid" || status=$?
wait "$recal_pid" || status=$?
((status == 0)) || exit "$status"

"$PYTHON_BIN" baseline/generate_resnet18_max_sqrt_bn_summary.py --output_root "$OUTPUT_ROOT"
echo "Completed ResNet-18 max_sqrt BN study: $OUTPUT_ROOT"

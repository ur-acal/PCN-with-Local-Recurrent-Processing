#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/home/rongzeng/anaconda3/envs/scanbase/bin/python}"
DATA_DIR="${DATA_DIR:-$ROOT/../data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/logs/resnet18_ours_sigma0p07_selection}"
SELECTION_TRIALS="${SELECTION_TRIALS:-10}"

[[ -x "$PYTHON_BIN" ]] || { echo "Missing scanbase Python: $PYTHON_BIN" >&2; exit 1; }
[[ ! -e "$OUTPUT_ROOT/checkpoints/cifar100/adapt_noresize_scratch/resnet18/adapt_noresize_scratch_cifar100_resnet18/adapt_noresize_scratch_cifar100_resnet18_best_sigma0p07_ckpt.pth" ]] || {
  echo "Completed selected checkpoint already exists under $OUTPUT_ROOT" >&2
  exit 2
}

mkdir -p "$OUTPUT_ROOT"
cat >"$OUTPUT_ROOT/run_manifest.json" <<EOF
{
  "model": "CIFAR-adapted timm ResNet-18",
  "dataset": "cifar100",
  "training_recipe": "existing adapt_noresize_scratch defaults",
  "training_noise": "none",
  "checkpoint_selection": "highest mean max_sqrt additive sigma=0.07 test accuracy",
  "selection_trials": $SELECTION_TRIALS,
  "selection_seed_start": 123,
  "bn_condition": "unfused frozen-BN",
  "fixed_noise_realizations_across_epochs": true,
  "clean_evaluation_during_training": false,
  "train_accuracy_evaluation_during_training": false
}
EOF

exec "$PYTHON_BIN" -u baseline/train_resnet18_sigma_selection.py \
  --data_dir "$DATA_DIR" \
  --output_dir "$OUTPUT_ROOT/checkpoints" \
  --seed 4096 \
  --selection_sigma 0.07 \
  --selection_trials "$SELECTION_TRIALS" \
  --selection_seed 123 \
  --eval_every 1 \
  --skip_eval_epochs 0

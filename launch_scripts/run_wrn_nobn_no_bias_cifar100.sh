#!/bin/bash

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_DIR="${DATA_DIR:-$ROOT/../data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/logs/wrn_nobn_no_bias_cifar100}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$OUTPUT_ROOT/checkpoints}"
PARALLELISM="${PARALLELISM:-4}"
SEED="${SEED:-4096}"

MODELS=(
  wrn_16_2_cifar_nobn_no_bias
  wrn_16_4_cifar_nobn_no_bias
  wrn_28_2_cifar_nobn_no_bias
  wrn_28_4_cifar_nobn_no_bias
)

# The optimization and augmentation recipe comes unchanged from custom_noresize.
# These opt-in fields only terminate a run that is nonfinite, catastrophically
# collapses after learning, or never leaves near-random loss by epoch 30.
OVERRIDE="${OVERRIDE:-collapse_monitor_enabled=true,collapse_not_learned_deadline_epoch=30,collapse_not_learned_loss_ratio=0.9}"
RECIPE_CHANGES="${RECIPE_CHANGES:-none}"

mkdir -p "$OUTPUT_ROOT/train_logs" "$CHECKPOINT_ROOT"

write_state() {
  local status="$1" detail="$2"
  printf '{\n  "status": "%s",\n  "detail": "%s",\n  "parallelism": %d,\n  "controller_pid": %d,\n  "updated_epoch_seconds": %d\n}\n' \
    "$status" "$detail" "$PARALLELISM" "$$" "$(date +%s)" > "$OUTPUT_ROOT/state.json"
}

train_one() {
  local model="$1"
  local log="$OUTPUT_ROOT/train_logs/${model}.log"
  local model_root="$CHECKPOINT_ROOT/cifar100/custom_noresize/$model"

  if [[ -s "$model_root/training_collapse.json" ]]; then
    echo "Skipping previously stopped model: $model"
    return 0
  fi
  if find "$model_root" -name '*_best_ckpt.pth' -type f -print -quit 2>/dev/null | grep -q .; then
    echo "Skipping completed model: $model"
    return 0
  fi
  if [[ -e "$log" || -d "$model_root" ]]; then
    echo "Refusing to overwrite partial run: $model" >&2
    return 4
  fi

  "$PYTHON_BIN" -u baseline/train_baseline_cifar.py \
    --model_name "$model" \
    --dataset cifar100 \
    --data_dir "$DATA_DIR" \
    --output_dir "$CHECKPOINT_ROOT" \
    --case custom_noresize \
    --pretrained false \
    --seed "$SEED" \
    --override "$OVERRIDE" > "$log" 2>&1

  if [[ -s "$model_root/training_collapse.json" ]]; then
    echo "Stopped collapsed/non-learning model: $model"
    return 0
  fi
  find "$model_root" -name '*_best_ckpt.pth' -type f -print -quit | grep -q . || {
    echo "Training produced neither a checkpoint nor a collapse record: $model" >&2
    return 5
  }
}

export ROOT PYTHON_BIN DATA_DIR OUTPUT_ROOT CHECKPOINT_ROOT PARALLELISM SEED OVERRIDE
export -f train_one

cat > "$OUTPUT_ROOT/manifest.json" <<EOF
{
  "dataset": "cifar100",
  "models": [
    "wrn_16_2_cifar_nobn_no_bias",
    "wrn_16_4_cifar_nobn_no_bias",
    "wrn_28_2_cifar_nobn_no_bias",
    "wrn_28_4_cifar_nobn_no_bias"
  ],
  "parallelism": $PARALLELISM,
  "seed": $SEED,
  "base_recipe": "custom_noresize",
  "recipe_changes": "$RECIPE_CHANGES",
  "collapse_monitor": {
    "not_learned_deadline_epoch": 30,
    "not_learned_loss_ratio": 0.9
  }
}
EOF

write_state running "0_of_4_complete"
"$PYTHON_BIN" -m unittest tests.test_wrn_nobn tests.test_training_collapse_monitor || {
  write_state failed "structural_validation_failed"
  exit 1
}
parallel --jobs "$PARALLELISM" --halt never --joblog "$OUTPUT_ROOT/training.joblog" \
  train_one ::: "${MODELS[@]}"
status=$?

completed=0
stopped=0
for model in "${MODELS[@]}"; do
  model_root="$CHECKPOINT_ROOT/cifar100/custom_noresize/$model"
  if [[ -s "$model_root/training_collapse.json" ]]; then
    stopped=$((stopped + 1))
  elif find "$model_root" -name '*_best_ckpt.pth' -type f -print -quit 2>/dev/null | grep -q .; then
    completed=$((completed + 1))
  fi
done

if [[ "$completed" -eq "${#MODELS[@]}" && "$stopped" -eq 0 ]]; then
  write_state complete "${completed}_trained_${stopped}_stopped"
  status=0
else
  write_state failed "${completed}_trained_${stopped}_stopped"
  [[ "$status" -ne 0 ]] || status=1
fi
exit "$status"

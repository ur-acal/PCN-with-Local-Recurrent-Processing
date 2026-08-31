#!/bin/bash -l
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DATA_DIR="${DATA_DIR:-$ROOT/../data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/logs/wrn16_2_wd1e3_pilot}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$ROOT/checkpoint/baselines_wd1e3_pilot}"
SEED="${SEED:-4096}"
PYTHON_BIN="${PYTHON_BIN:-/home/rongzeng/anaconda3/envs/scanbase/bin/python}"
[[ -x "$PYTHON_BIN" ]] || { echo "Missing scanbase Python: $PYTHON_BIN" >&2; exit 1; }

STANDARD_OVERRIDE='num_epochs=300,eval_every=5,weight_decay=0.001'
NOBN_CIFAR10_OVERRIDE='num_epochs=300,eval_every=5,skip_eval_epochs=70,test_batch_size=512,lr=0.1,weight_decay=0.001,max_norm=2.0,bias_lr_multiplier=0.5,bias_weight_decay=0.0,dropout_rate=0.1,warmup_epoch=5,auto_augment=rand-m9-mstd0.5-inc1,mixup_alpha=0.0,cutmix_alpha=0.0,label_smoothing=0.1,re_prob=0.1,color_jitter=0.1'
NOBN_CIFAR100_OVERRIDE='num_epochs=300,eval_every=5,skip_eval_epochs=70,test_batch_size=512,lr=0.1,weight_decay=0.001,max_norm=none,bias_lr_multiplier=1.0,bias_weight_decay=none,dropout_rate=0.0,warmup_epoch=5,auto_augment=rand-m7-mstd0.5-inc1,mixup_alpha=0.1,cutmix_alpha=0.5,label_smoothing=0.05,re_prob=0.1,color_jitter=0.1'

mkdir -p "$OUTPUT_ROOT/train_logs" "$CHECKPOINT_ROOT"

write_state() {
  local status="$1" detail="${2:-}"
  printf '{\n  "status": "%s",\n  "stage": "parallel_training",\n  "detail": "%s",\n  "controller_pid": %d,\n  "updated_epoch_seconds": %d\n}\n' \
    "$status" "$detail" "$$" "$(date +%s)" >"$OUTPUT_ROOT/state.json.tmp"
  mv "$OUTPUT_ROOT/state.json.tmp" "$OUTPUT_ROOT/state.json"
}

train_one() {
  local label="$1" dataset="$2" model="$3" override="$4"
  local log="$OUTPUT_ROOT/train_logs/${label}.log"
  exec "$PYTHON_BIN" -u baseline/train_baseline_cifar.py \
    --model_name "$model" \
    --dataset "$dataset" \
    --data_dir "$DATA_DIR" \
    --output_dir "$CHECKPOINT_ROOT" \
    --case custom_noresize \
    --pretrained false \
    --seed "$SEED" \
    --override "$override" >"$log" 2>&1
}

printf '%s\n' \
  "standard: $STANDARD_OVERRIDE" \
  "nobn_cifar10: $NOBN_CIFAR10_OVERRIDE" \
  "nobn_cifar100: $NOBN_CIFAR100_OVERRIDE" >"$OUTPUT_ROOT/recipes.txt"

write_state running four_jobs

labels=(standard_cifar10 standard_cifar100 nobn_cifar10 nobn_cifar100)
datasets=(cifar10 cifar100 cifar10 cifar100)
models=(wrn_16_2_cifar wrn_16_2_cifar wrn_16_2_cifar_nobn wrn_16_2_cifar_nobn)
overrides=("$STANDARD_OVERRIDE" "$STANDARD_OVERRIDE" "$NOBN_CIFAR10_OVERRIDE" "$NOBN_CIFAR100_OVERRIDE")
pids=()

for i in "${!labels[@]}"; do
  train_one "${labels[$i]}" "${datasets[$i]}" "${models[$i]}" "${overrides[$i]}" &
  pids+=("$!")
done

failed=()
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    failed+=("${labels[$i]}")
  fi
done

if ((${#failed[@]})); then
  write_state failed "${failed[*]}"
  exit 1
fi

write_state completed four_jobs

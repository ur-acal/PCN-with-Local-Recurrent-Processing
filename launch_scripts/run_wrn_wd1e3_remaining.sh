#!/bin/bash -l
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DATA_DIR="${DATA_DIR:-$ROOT/../data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/logs/wrn_wd1e3_remaining}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$ROOT/checkpoint/baselines_wd1e3_pilot}"
SEED="${SEED:-4096}"
PYTHON_BIN="${PYTHON_BIN:-/home/rongzeng/anaconda3/envs/scanbase/bin/python}"
[[ -x "$PYTHON_BIN" ]] || { echo "Missing scanbase Python: $PYTHON_BIN" >&2; exit 1; }

STANDARD_OVERRIDE='num_epochs=300,eval_every=5,weight_decay=0.001'

mkdir -p "$OUTPUT_ROOT/train_logs" "$CHECKPOINT_ROOT"

write_state() {
  local status="$1" detail="${2:-}"
  printf '{\n  "status": "%s",\n  "stage": "parallel_training",\n  "detail": "%s",\n  "controller_pid": %d,\n  "updated_epoch_seconds": %d\n}\n' \
    "$status" "$detail" "$$" "$(date +%s)" >"$OUTPUT_ROOT/state.json.tmp"
  mv "$OUTPUT_ROOT/state.json.tmp" "$OUTPUT_ROOT/state.json"
}

train_one() {
  local label="$1" dataset="$2" model="$3"
  local log="$OUTPUT_ROOT/train_logs/${label}.log"
  exec "$PYTHON_BIN" -u baseline/train_baseline_cifar.py \
    --model_name "$model" \
    --dataset "$dataset" \
    --data_dir "$DATA_DIR" \
    --output_dir "$CHECKPOINT_ROOT" \
    --case custom_noresize \
    --pretrained false \
    --seed "$SEED" \
    --override "$STANDARD_OVERRIDE" >"$log" 2>&1
}

printf '%s\n' "standard: $STANDARD_OVERRIDE" >"$OUTPUT_ROOT/recipes.txt"
write_state running six_jobs

labels=(cifar10_wrn16_4 cifar100_wrn16_4 cifar10_wrn28_2 cifar100_wrn28_2 cifar10_wrn28_4 cifar100_wrn28_4)
datasets=(cifar10 cifar100 cifar10 cifar100 cifar10 cifar100)
models=(wrn_16_4_cifar wrn_16_4_cifar wrn_28_2_cifar wrn_28_2_cifar wrn_28_4_cifar wrn_28_4_cifar)
pids=()

for i in "${!labels[@]}"; do
  train_one "${labels[$i]}" "${datasets[$i]}" "${models[$i]}" &
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

write_state completed six_jobs

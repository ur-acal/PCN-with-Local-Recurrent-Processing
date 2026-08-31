#!/bin/bash -l
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
DATA_DIR="${DATA_DIR:-$ROOT/../data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/logs/wrn_nobn_best_recipe_full}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$ROOT/checkpoint/baselines_nobn_best_recipe}"
SEED="${SEED:-4096}"
TRAIN_OVERRIDE='num_epochs=300,eval_every=5,skip_eval_epochs=70,test_batch_size=512,lr=0.1,weight_decay=0.0001,max_norm=2.0,bias_lr_multiplier=0.5,bias_weight_decay=0.0,dropout_rate=0.1,warmup_epoch=5,auto_augment=rand-m9-mstd0.5-inc1,mixup_alpha=0.0,cutmix_alpha=0.0,label_smoothing=0.1,re_prob=0.1,color_jitter=0.1'
DATASETS=(cifar10 cifar100)
ARCHITECTURES=(WRN_16_2 WRN_16_4 WRN_28_2 WRN_28_4)

mkdir -p "$OUTPUT_ROOT/train_logs" "$OUTPUT_ROOT/eval_logs" "$OUTPUT_ROOT/mismatch" "$CHECKPOINT_ROOT"

write_state() {
  local status="$1" stage="$2" detail="${3:-}"
  printf '{\n  "status": "%s",\n  "stage": "%s",\n  "detail": "%s",\n  "controller_pid": %d,\n  "updated_epoch_seconds": %d\n}\n' \
    "$status" "$stage" "$detail" "$$" "$(date +%s)" > "$OUTPUT_ROOT/state.json.tmp"
  mv "$OUTPUT_ROOT/state.json.tmp" "$OUTPUT_ROOT/state.json"
}

model_name() {
  printf 'wrn_%s_cifar_nobn' "$(printf '%s' "$1" | sed -E 's/^WRN_//; s/[A-Z]/\L&/g')"
}

checkpoint_path() {
  local dataset="$1" name="$2" run="custom_noresize_${1}_${2}"
  printf '%s/%s/custom_noresize/%s/%s/%s_best_ckpt.pth' "$CHECKPOINT_ROOT" "$dataset" "$name" "$run" "$run"
}

run_training_group() {
  local dataset="$1" failures=0
  local -a pids=() labels=()
  write_state running training "$dataset"
  for architecture in "${ARCHITECTURES[@]}"; do
    local name checkpoint log
    name="$(model_name "$architecture")"
    checkpoint="$(checkpoint_path "$dataset" "$name")"
    log="$OUTPUT_ROOT/train_logs/${dataset}_${name}.log"
    if [[ -f "$checkpoint" ]]; then
      printf 'Skipping completed training: %s %s\n' "$dataset" "$name"
      continue
    fi
    (
      exec python baseline/train_baseline_cifar.py \
        --model_name "$name" \
        --dataset "$dataset" \
        --data_dir "$DATA_DIR" \
        --output_dir "$CHECKPOINT_ROOT" \
        --case custom_noresize \
        --pretrained false \
        --seed "$SEED" \
        --override "$TRAIN_OVERRIDE"
    ) >"$log" 2>&1 &
    pids+=("$!")
    labels+=("$dataset/$name")
  done
  for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
      printf 'Training failed: %s (see log)\n' "${labels[$i]}" >&2
      failures=1
    fi
  done
  if (( failures )); then
    write_state failed training "$dataset"
    return 1
  fi
  for architecture in "${ARCHITECTURES[@]}"; do
    local name checkpoint
    name="$(model_name "$architecture")"
    checkpoint="$(checkpoint_path "$dataset" "$name")"
    [[ -f "$checkpoint" ]] || { write_state failed training "missing:$dataset/$name"; return 1; }
  done
}

run_evaluation_group() {
  local dataset="$1" base_index="$2" failures=0
  local -a pids=() labels=()
  write_state running mismatch_evaluation "$dataset"
  for i in "${!ARCHITECTURES[@]}"; do
    local architecture name result_dir log index
    architecture="${ARCHITECTURES[$i]}"
    name="$(model_name "$architecture")"
    result_dir="$OUTPUT_ROOT/mismatch/$dataset/$name"
    log="$OUTPUT_ROOT/eval_logs/${dataset}_${name}.log"
    index=$((base_index + i))
    if [[ -f "$result_dir/full_per_trial.csv" ]] && [[ "$(wc -l < "$result_dir/full_per_trial.csv")" -eq 201 ]]; then
      printf 'Skipping completed mismatch evaluation: %s %s\n' "$dataset" "$name"
      continue
    fi
    mkdir -p "$result_dir"
    (
      exec python baseline/run_wrn_nobn_mismatch_experiment.py \
        --output_dir "$result_dir" \
        --data_dir "$DATA_DIR" \
        --checkpoint_root "$CHECKPOINT_ROOT" \
        --datasets "$dataset" \
        --architectures "$architecture" \
        --mismatch_types additive,multiplicative \
        --noise_levels default \
        --noisy_trials 10 \
        --seed 123 \
        --model_index_offset "$index" \
        --device cuda \
        --batch_size 128 \
        --num_workers 4
    ) >"$log" 2>&1 &
    pids+=("$!")
    labels+=("$dataset/$name")
  done
  for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
      printf 'Mismatch evaluation failed: %s (see log)\n' "${labels[$i]}" >&2
      failures=1
    fi
  done
  if (( failures )); then
    write_state failed mismatch_evaluation "$dataset"
    return 1
  fi
}

write_state running startup eight_models
run_training_group cifar10 || exit 1
run_training_group cifar100 || exit 1
run_evaluation_group cifar10 0 || exit 1
run_evaluation_group cifar100 4 || exit 1
write_state completed complete eight_models

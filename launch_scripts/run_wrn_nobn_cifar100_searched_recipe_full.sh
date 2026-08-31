#!/bin/bash -l
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DATA_DIR="${DATA_DIR:-$ROOT/../data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/logs/wrn_nobn_cifar100_searched_recipe_full}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$ROOT/checkpoint/baselines_nobn_cifar100_searched_recipe}"
SEED="${SEED:-4096}"
TRAIN_OVERRIDE='num_epochs=300,eval_every=5,skip_eval_epochs=70,test_batch_size=512,lr=0.1,weight_decay=0.0005,max_norm=none,bias_lr_multiplier=1.0,bias_weight_decay=none,dropout_rate=0.0,warmup_epoch=5,auto_augment=rand-m7-mstd0.5-inc1,mixup_alpha=0.1,cutmix_alpha=0.5,label_smoothing=0.05,re_prob=0.1,color_jitter=0.1'
ARCHITECTURES=(WRN_16_2 WRN_16_4 WRN_28_2 WRN_28_4)

mkdir -p "$OUTPUT_ROOT/train_logs" "$OUTPUT_ROOT/eval_logs" \
  "$OUTPUT_ROOT/mismatch/cifar100" "$CHECKPOINT_ROOT"

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
  local name="$1" run="custom_noresize_cifar100_${1}"
  printf '%s/cifar100/custom_noresize/%s/%s/%s_best_ckpt.pth' \
    "$CHECKPOINT_ROOT" "$name" "$run" "$run"
}

train_one() {
  local architecture="$1" name log
  name="$(model_name "$architecture")"
  log="$OUTPUT_ROOT/train_logs/cifar100_${name}.log"
  python baseline/train_baseline_cifar.py \
    --model_name "$name" \
    --dataset cifar100 \
    --data_dir "$DATA_DIR" \
    --output_dir "$CHECKPOINT_ROOT" \
    --case custom_noresize \
    --pretrained false \
    --seed "$SEED" \
    --override "$TRAIN_OVERRIDE" >"$log" 2>&1
}

run_training_group() {
  local failures=0 architecture name checkpoint
  local -a pids=() labels=() failed=()
  write_state running parallel_training cifar100
  for architecture in "${ARCHITECTURES[@]}"; do
    name="$(model_name "$architecture")"
    checkpoint="$(checkpoint_path "$name")"
    if [[ -f "$checkpoint" ]]; then
      printf 'Skipping completed training: cifar100 %s\n' "$name"
      continue
    fi
    train_one "$architecture" &
    pids+=("$!")
    labels+=("$architecture")
  done
  for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
      failed+=("${labels[$i]}")
    fi
  done
  if (( ${#failed[@]} )); then
    write_state running sequential_training_retry "${failed[*]}"
    for architecture in "${failed[@]}"; do
      train_one "$architecture" || failures=1
    done
  fi
  for architecture in "${ARCHITECTURES[@]}"; do
    name="$(model_name "$architecture")"
    checkpoint="$(checkpoint_path "$name")"
    [[ -f "$checkpoint" ]] || failures=1
  done
  if (( failures )); then
    write_state failed training cifar100
    return 1
  fi
}

evaluate_one() {
  local architecture="$1" index="$2" name result_dir log
  name="$(model_name "$architecture")"
  result_dir="$OUTPUT_ROOT/mismatch/cifar100/$name"
  log="$OUTPUT_ROOT/eval_logs/cifar100_${name}.log"
  mkdir -p "$result_dir"
  python baseline/run_wrn_nobn_mismatch_experiment.py \
    --output_dir "$result_dir" \
    --data_dir "$DATA_DIR" \
    --checkpoint_root "$CHECKPOINT_ROOT" \
    --datasets cifar100 \
    --architectures "$architecture" \
    --mismatch_types additive,multiplicative \
    --noise_levels default \
    --noisy_trials 10 \
    --seed 123 \
    --model_index_offset "$index" \
    --device cuda \
    --batch_size 128 \
    --num_workers 4 >"$log" 2>&1
}

result_complete() {
  local architecture="$1" name result
  name="$(model_name "$architecture")"
  result="$OUTPUT_ROOT/mismatch/cifar100/$name/full_per_trial.csv"
  [[ -f "$result" ]] && [[ "$(wc -l < "$result")" -eq 201 ]]
}

run_evaluation_group() {
  local failures=0 architecture index
  local -a pids=() labels=() failed=()
  write_state running parallel_mismatch_evaluation cifar100
  for i in "${!ARCHITECTURES[@]}"; do
    architecture="${ARCHITECTURES[$i]}"
    index=$((4 + i))
    if result_complete "$architecture"; then
      printf 'Skipping completed mismatch evaluation: cifar100 %s\n' "$architecture"
      continue
    fi
    evaluate_one "$architecture" "$index" &
    pids+=("$!")
    labels+=("$architecture")
  done
  for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
      failed+=("${labels[$i]}")
    fi
  done
  if (( ${#failed[@]} )); then
    write_state running sequential_mismatch_retry "${failed[*]}"
    for architecture in "${failed[@]}"; do
      for i in "${!ARCHITECTURES[@]}"; do
        [[ "${ARCHITECTURES[$i]}" == "$architecture" ]] && index=$((4 + i))
      done
      evaluate_one "$architecture" "$index" || failures=1
    done
  fi
  for architecture in "${ARCHITECTURES[@]}"; do
    result_complete "$architecture" || failures=1
  done
  if (( failures )); then
    write_state failed mismatch_evaluation cifar100
    return 1
  fi
}

write_state running startup cifar100_four_models
run_training_group || exit 1
run_evaluation_group || exit 1
write_state completed complete cifar100_four_models

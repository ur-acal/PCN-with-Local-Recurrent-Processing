#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/home/rongzeng/anaconda3/envs/scanbase/bin/python}"
DATA_DIR="${DATA_DIR:-$ROOT/../data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/logs/wrn28_2_avgpool_study}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$ROOT/checkpoint/baselines_wrn28_2_avgpool_study}"
COMPARE_DIR="${COMPARE_DIR:-$ROOT/../ScAN-PCN/logs/wrn_like_compare_target_models_by_arch}"
PARALLELISM="${PARALLELISM:-4}"
SEED="${SEED:-4096}"
CURRENT_STAGE="initializing"

DATASETS=(cifar10 cifar100)
MODELS=(
  wrn_28_2_cifar_avgpool
  wrn_28_2_cifar_nobn_avgpool
  wrn_28_2_cifar_avgpool_shortcut
  wrn_28_2_cifar_nobn_avgpool_shortcut
)
FAMILIES=(max_additive rms_additive max_sqrt_additive multiplicative)

NORMAL_OVERRIDE="num_epochs=300,eval_every=5,weight_decay=0.001,dropout_rate=0.0,final_dropout_rate=0.25"
NOBN_CIFAR10_OVERRIDE="num_epochs=300,eval_every=5,skip_eval_epochs=70,test_batch_size=512,lr=0.1,weight_decay=0.001,max_norm=2.0,bias_lr_multiplier=0.5,bias_weight_decay=0.0,dropout_rate=0.1,final_dropout_rate=0.25,warmup_epoch=5,auto_augment=rand-m9-mstd0.5-inc1,mixup_alpha=0.0,cutmix_alpha=0.0,label_smoothing=0.1,re_prob=0.1,color_jitter=0.1"
NOBN_CIFAR100_OVERRIDE="num_epochs=300,eval_every=5,skip_eval_epochs=70,test_batch_size=512,lr=0.1,weight_decay=0.001,max_norm=none,bias_lr_multiplier=1.0,bias_weight_decay=none,dropout_rate=0.0,final_dropout_rate=0.25,warmup_epoch=5,auto_augment=rand-m7-mstd0.5-inc1,mixup_alpha=0.1,cutmix_alpha=0.5,label_smoothing=0.05,re_prob=0.1,color_jitter=0.1"

[[ -x "$PYTHON_BIN" ]] || { echo "Missing scanbase Python: $PYTHON_BIN" >&2; exit 1; }
[[ -d "$COMPARE_DIR" ]] || { echo "Missing WRN comparison inputs: $COMPARE_DIR" >&2; exit 1; }
command -v parallel >/dev/null || { echo "GNU parallel is required" >&2; exit 1; }
mkdir -p "$OUTPUT_ROOT/train_logs" "$OUTPUT_ROOT/eval_logs" "$OUTPUT_ROOT/evaluation" "$CHECKPOINT_ROOT"

write_state() {
  local status="$1" stage="$2" detail="${3:-}"
  printf '{\n  "status": "%s",\n  "stage": "%s",\n  "detail": "%s",\n  "parallelism": %d,\n  "controller_pid": %d,\n  "updated_epoch_seconds": %d\n}\n' \
    "$status" "$stage" "$detail" "$PARALLELISM" "$$" "$(date +%s)" >"$OUTPUT_ROOT/state.json.tmp"
  mv "$OUTPUT_ROOT/state.json.tmp" "$OUTPUT_ROOT/state.json"
}

on_error() {
  local code=$?
  write_state failed "$CURRENT_STAGE" "exit_code_$code"
  exit "$code"
}
trap on_error ERR

is_nobn() {
  [[ "$1" == *"_nobn_"* ]]
}

training_override() {
  local dataset="$1" model="$2"
  if ! is_nobn "$model"; then
    printf '%s\n' "$NORMAL_OVERRIDE"
  elif [[ "$dataset" == cifar10 ]]; then
    printf '%s\n' "$NOBN_CIFAR10_OVERRIDE"
  else
    printf '%s\n' "$NOBN_CIFAR100_OVERRIDE"
  fi
}

checkpoint_path() {
  local dataset="$1" model="$2" run="custom_noresize_${1}_${2}"
  printf '%s/%s/custom_noresize/%s/%s/%s_best_ckpt.pth\n' \
    "$CHECKPOINT_ROOT" "$dataset" "$model" "$run" "$run"
}

training_complete() {
  local dataset="$1" model="$2" log checkpoint
  log="$OUTPUT_ROOT/train_logs/${dataset}_${model}.log"
  checkpoint="$(checkpoint_path "$dataset" "$model")"
  [[ -s "$checkpoint" && -s "$log" ]] && grep -q -- 'Train finished' "$log"
}

train_one() {
  local dataset="$1" model="$2" log checkpoint override
  log="$OUTPUT_ROOT/train_logs/${dataset}_${model}.log"
  checkpoint="$(checkpoint_path "$dataset" "$model")"
  override="$(training_override "$dataset" "$model")"

  if training_complete "$dataset" "$model"; then
    echo "Skipping completed training: $dataset $model"
    return 0
  fi
  if [[ -e "$log" || -e "$checkpoint" ]]; then
    echo "Refusing to overwrite partial training artifacts: $dataset $model" >&2
    return 2
  fi

  "$PYTHON_BIN" -u baseline/train_baseline_cifar.py \
    --model_name "$model" \
    --dataset "$dataset" \
    --data_dir "$DATA_DIR" \
    --output_dir "$CHECKPOINT_ROOT" \
    --case custom_noresize \
    --pretrained false \
    --seed "$SEED" \
    --override "$override" >"$log" 2>&1

  training_complete "$dataset" "$model" || {
    echo "Training did not produce complete artifacts: $dataset $model" >&2
    return 3
  }
}

family_spec() {
  case "$1" in
    max_additive) printf 'additive|max_abs|0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1\n' ;;
    rms_additive) printf 'additive|rms|0,0.25,0.5,0.75,1.0,1.25\n' ;;
    max_sqrt_additive) printf 'additive|max_sqrt|0,0.02,0.03,0.05,0.07,0.09,0.1\n' ;;
    multiplicative) printf 'multiplicative|max_abs|0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4\n' ;;
    *) echo "Unknown mismatch family: $1" >&2; return 2 ;;
  esac
}

evaluation_complete() {
  [[ -s "$OUTPUT_ROOT/evaluation/$1/$2/$3/full_aggregate.csv" ]]
}

eval_one() {
  local dataset="$1" model="$2" family="$3" output log checkpoint spec mismatch_type scale levels
  output="$OUTPUT_ROOT/evaluation/$dataset/$model/$family"
  log="$OUTPUT_ROOT/eval_logs/${dataset}_${model}_${family}.log"
  checkpoint="$(checkpoint_path "$dataset" "$model")"
  IFS='|' read -r mismatch_type scale levels <<<"$(family_spec "$family")"

  if evaluation_complete "$dataset" "$model" "$family"; then
    echo "Skipping completed evaluation: $dataset $model $family"
    return 0
  fi
  [[ -s "$checkpoint" ]] || { echo "Missing checkpoint: $checkpoint" >&2; return 3; }
  if [[ -d "$output" || -e "$log" ]]; then
    echo "Refusing to overwrite partial evaluation: $dataset $model $family" >&2
    return 4
  fi
  mkdir -p "$output"

  if is_nobn "$model"; then
    "$PYTHON_BIN" -u baseline/run_wrn_nobn_mismatch_experiment.py \
      --output_dir "$output" \
      --data_dir "$DATA_DIR" \
      --checkpoint_override "$checkpoint" \
      --model_name_override "$model" \
      --datasets "$dataset" \
      --architectures WRN_28_2 \
      --mismatch_types "$mismatch_type" \
      --additive_scale_mode "$scale" \
      --noise_levels "$levels" \
      --noisy_trials 10 \
      --seed 123 \
      --batch_size 128 \
      --num_workers 2 \
      --case custom_noresize >"$log" 2>&1
  else
    "$PYTHON_BIN" -u baseline/run_wrn_bn_recalibration_experiment.py \
      --mode full \
      --compare_dir "$COMPARE_DIR" \
      --output_dir "$output" \
      --data_dir "$DATA_DIR" \
      --checkpoint_override "$checkpoint" \
      --model_name_override "$model" \
      --datasets "$dataset" \
      --architectures WRN_28_2 \
      --mismatch_types "$mismatch_type" \
      --additive_scale_mode "$scale" \
      --pcn_reference_policy none \
      --noise_levels "$levels" \
      --noisy_trials 10 \
      --seed 123 \
      --batch_size 128 \
      --num_workers 2 \
      --calibration_num_samples 5120 \
      --calibration_batch_size 128 \
      --calibration_subset_seed 20240618 \
      --calibration_num_workers 2 \
      --case custom_noresize \
      --mismatch_parameter_policy existing >"$log" 2>&1
  fi

  evaluation_complete "$dataset" "$model" "$family" || {
    echo "Evaluation did not produce aggregate output: $dataset $model $family" >&2
    return 5
  }
}

export ROOT PYTHON_BIN DATA_DIR OUTPUT_ROOT CHECKPOINT_ROOT COMPARE_DIR PARALLELISM SEED
export NORMAL_OVERRIDE NOBN_CIFAR10_OVERRIDE NOBN_CIFAR100_OVERRIDE
export -f is_nobn training_override checkpoint_path training_complete train_one
export -f family_spec evaluation_complete eval_one

cat >"$OUTPUT_ROOT/manifest.json" <<EOF
{
  "datasets": ["cifar10", "cifar100"],
  "models": [
    "wrn_28_2_cifar_avgpool",
    "wrn_28_2_cifar_nobn_avgpool",
    "wrn_28_2_cifar_avgpool_shortcut",
    "wrn_28_2_cifar_nobn_avgpool_shortcut"
  ],
  "parallelism": $PARALLELISM,
  "normal_override": "$NORMAL_OVERRIDE",
  "nobn_cifar10_override": "$NOBN_CIFAR10_OVERRIDE",
  "nobn_cifar100_override": "$NOBN_CIFAR100_OVERRIDE",
  "mismatch_trials": 10,
  "bn_calibration_samples": 5120,
  "bn_calibration_subset_seed": 20240618
}
EOF

CURRENT_STAGE="structural_validation"
write_state running "$CURRENT_STAGE"
"$PYTHON_BIN" -m unittest tests.test_wrn_avgpool_variants

CURRENT_STAGE="training"
write_state running "$CURRENT_STAGE" "0_of_8_complete"
parallel --jobs "$PARALLELISM" --halt now,fail=1 --joblog "$OUTPUT_ROOT/training.joblog" \
  train_one {1} {2} ::: "${DATASETS[@]}" ::: "${MODELS[@]}"

CURRENT_STAGE="evaluation"
write_state running "$CURRENT_STAGE" "0_of_32_complete"
parallel --jobs "$PARALLELISM" --halt now,fail=1 --joblog "$OUTPUT_ROOT/evaluation.joblog" \
  eval_one {1} {2} {3} ::: "${DATASETS[@]}" ::: "${MODELS[@]}" ::: "${FAMILIES[@]}"

CURRENT_STAGE="complete"
write_state complete "$CURRENT_STAGE" "8_training_and_32_evaluation_units_complete"
echo "WRN-28-2 average-pooling study complete: $OUTPUT_ROOT"

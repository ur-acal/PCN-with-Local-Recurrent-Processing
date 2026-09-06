#!/bin/bash

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_DIR="${DATA_DIR:-$ROOT/../data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/logs/wrn_nobn_no_bias_mismatch_cifar100}"
PARALLELISM="${PARALLELISM:-8}"
MISMATCH_SEED="${MISMATCH_SEED:-123}"
CURRENT_STAGE="waiting_for_training"

TRAIN_SERVICES=(
  scan-wrn-nobn-no-bias-cifar100.service
  scan-wrn-nobn-no-bias-wd1e3-finaldrop025-cifar100.service
)
TRAIN_STATES=(
  "$ROOT/logs/wrn_nobn_no_bias_cifar100/state.json"
  "$ROOT/logs/wrn_nobn_no_bias_wd1e3_finaldrop025_cifar100/state.json"
)
RECIPES=(original_wd5e4_drop0 wd1e3_finaldrop025)
MODELS=(
  wrn_16_2_cifar_nobn_no_bias
  wrn_16_4_cifar_nobn_no_bias
  wrn_28_2_cifar_nobn_no_bias
  wrn_28_4_cifar_nobn_no_bias
)
FAMILIES=(max_additive rms_additive multiplicative)

mkdir -p "$OUTPUT_ROOT/logs" "$OUTPUT_ROOT/results"

write_state() {
  local status="$1" stage="$2" detail="${3:-}"
  printf '{\n  "status": "%s",\n  "stage": "%s",\n  "detail": "%s",\n  "parallelism": %d,\n  "mismatch_seed": %d,\n  "controller_pid": %d,\n  "updated_epoch_seconds": %d\n}\n' \
    "$status" "$stage" "$detail" "$PARALLELISM" "$MISMATCH_SEED" "$$" "$(date +%s)" \
    > "$OUTPUT_ROOT/state.json.tmp"
  mv "$OUTPUT_ROOT/state.json.tmp" "$OUTPUT_ROOT/state.json"
}

wait_for_training() {
  local service
  while true; do
    local active=0
    for service in "${TRAIN_SERVICES[@]}"; do
      if systemctl --user is-active --quiet "$service"; then
        active=1
      fi
    done
    [[ "$active" -eq 0 ]] && break
    sleep 30
  done

  local state
  for state in "${TRAIN_STATES[@]}"; do
    if ! grep -q '"status": "complete"' "$state"; then
      echo "Training prerequisite did not complete successfully: $state" >&2
      return 1
    fi
  done
}

architecture_for() {
  case "$1" in
    wrn_16_2_*) printf 'WRN_16_2|0\n' ;;
    wrn_16_4_*) printf 'WRN_16_4|1\n' ;;
    wrn_28_2_*) printf 'WRN_28_2|2\n' ;;
    wrn_28_4_*) printf 'WRN_28_4|3\n' ;;
    *) echo "Unknown model architecture: $1" >&2; return 2 ;;
  esac
}

checkpoint_for() {
  local recipe="$1" model="$2" checkpoint_root run
  case "$recipe" in
    original_wd5e4_drop0)
      checkpoint_root="$ROOT/logs/wrn_nobn_no_bias_cifar100/checkpoints"
      ;;
    wd1e3_finaldrop025)
      checkpoint_root="$ROOT/logs/wrn_nobn_no_bias_wd1e3_finaldrop025_cifar100/checkpoints"
      ;;
    *) echo "Unknown recipe: $recipe" >&2; return 2 ;;
  esac
  run="custom_noresize_cifar100_${model}"
  printf '%s\n' "$checkpoint_root/cifar100/custom_noresize/$model/$run/${run}_best_ckpt.pth"
}

family_spec() {
  case "$1" in
    max_additive)
      printf 'additive|max_abs|0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1\n'
      ;;
    rms_additive)
      printf 'additive|rms|0,0.25,0.5,0.75,1.0,1.25\n'
      ;;
    multiplicative)
      printf 'multiplicative|max_abs|0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4\n'
      ;;
    *) echo "Unknown mismatch family: $1" >&2; return 2 ;;
  esac
}

verify_checkpoints() {
  local recipe model checkpoint
  for recipe in "${RECIPES[@]}"; do
    for model in "${MODELS[@]}"; do
      checkpoint="$(checkpoint_for "$recipe" "$model")"
      [[ -s "$checkpoint" ]] || {
        echo "Missing checkpoint: $checkpoint" >&2
        return 3
      }
    done
  done
}

run_one() {
  local recipe="$1" model="$2" family="$3"
  local architecture model_index checkpoint mismatch_type scale levels output log
  IFS='|' read -r architecture model_index <<<"$(architecture_for "$model")"
  checkpoint="$(checkpoint_for "$recipe" "$model")"
  IFS='|' read -r mismatch_type scale levels <<<"$(family_spec "$family")"
  output="$OUTPUT_ROOT/results/$recipe/$model/$family"
  log="$OUTPUT_ROOT/logs/${recipe}_${model}_${family}.log"

  if [[ -s "$output/full_aggregate.csv" ]]; then
    echo "Skipping completed evaluation: $recipe $model $family"
    return 0
  fi
  if [[ -d "$output" || -e "$log" ]]; then
    echo "Refusing to overwrite partial evaluation: $recipe $model $family" >&2
    return 4
  fi
  mkdir -p "$output"

  "$PYTHON_BIN" -u baseline/run_wrn_nobn_mismatch_experiment.py \
    --output_dir "$output" \
    --data_dir "$DATA_DIR" \
    --checkpoint_override "$checkpoint" \
    --model_name_override "$model" \
    --datasets cifar100 \
    --architectures "$architecture" \
    --mismatch_types "$mismatch_type" \
    --additive_scale_mode "$scale" \
    --noise_levels "$levels" \
    --noisy_trials 10 \
    --seed "$MISMATCH_SEED" \
    --model_index_offset "$model_index" \
    --batch_size 128 \
    --num_workers 2 \
    --case custom_noresize > "$log" 2>&1

  [[ -s "$output/full_aggregate.csv" ]] || {
    echo "Evaluation did not produce aggregate output: $recipe $model $family" >&2
    return 5
  }
}

export ROOT PYTHON_BIN DATA_DIR OUTPUT_ROOT PARALLELISM MISMATCH_SEED
export -f architecture_for checkpoint_for family_spec run_one
export SHELL=/bin/bash

cat > "$OUTPUT_ROOT/run_manifest.json" <<EOF
{
  "dataset": "cifar100",
  "recipes": ["original_wd5e4_drop0", "wd1e3_finaldrop025"],
  "models_per_recipe": ["WRN_16_2", "WRN_16_4", "WRN_28_2", "WRN_28_4"],
  "mismatch_families": ["max_additive", "rms_additive", "multiplicative"],
  "max_additive_levels": [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1],
  "rms_additive_levels": [0, 0.25, 0.5, 0.75, 1.0, 1.25],
  "multiplicative_levels": [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4],
  "noisy_trials": 10,
  "mismatch_seed": $MISMATCH_SEED,
  "parallelism": $PARALLELISM,
  "parameter_policy": "all non-normalization parameters; convolution biases excluded when present"
}
EOF

write_state running "$CURRENT_STAGE" "waiting_for_8_checkpoints"
if ! wait_for_training; then
  write_state failed "$CURRENT_STAGE" "training_prerequisite_failed"
  exit 1
fi

CURRENT_STAGE="checkpoint_preflight"
write_state running "$CURRENT_STAGE" "checking_8_checkpoints"
if ! verify_checkpoints; then
  write_state failed "$CURRENT_STAGE" "checkpoint_preflight_failed"
  exit 1
fi

CURRENT_STAGE="evaluation"
write_state running "$CURRENT_STAGE" "0_of_24_complete"
parallel --jobs "$PARALLELISM" --halt never \
  --joblog "$OUTPUT_ROOT/evaluation.joblog" \
  run_one {1} {2} {3} ::: "${RECIPES[@]}" ::: "${MODELS[@]}" ::: "${FAMILIES[@]}"
status=$?

if [[ "$status" -eq 0 ]]; then
  write_state complete "$CURRENT_STAGE" "24_of_24_complete"
else
  write_state failed "$CURRENT_STAGE" "one_or_more_evaluations_failed"
fi
exit "$status"

#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/home/rongzeng/anaconda3/envs/scanbase/bin/python}"
DATA_DIR="${DATA_DIR:-$ROOT/../data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/logs/wrn28_2_cifar100_seed456_replication}"
COMPARE_DIR="${COMPARE_DIR:-$ROOT/../ScAN-PCN/logs/wrn_like_compare_target_models_by_arch}"
PARALLELISM="${PARALLELISM:-4}"
MISMATCH_SEED="${MISMATCH_SEED:-456}"
CURRENT_STAGE="initializing"

CONDITIONS=(
  wrn_unfused_recal_bn
  wrn_wd1e3_unfused_recal_bn
  wrn_wd1e3_finaldrop025_unfused_recal_bn
  avgpool_bn
  stride2_bn
  avgpool_bnfree
  stride2_bnfree
)
FAMILIES=(max_additive rms_additive multiplicative)

[[ -x "$PYTHON_BIN" ]] || { echo "Missing scanbase Python: $PYTHON_BIN" >&2; exit 1; }
[[ -d "$COMPARE_DIR" ]] || { echo "Missing comparison directory: $COMPARE_DIR" >&2; exit 1; }
command -v parallel >/dev/null || { echo "GNU parallel is required" >&2; exit 1; }
mkdir -p "$OUTPUT_ROOT/logs" "$OUTPUT_ROOT/results"

write_state() {
  local status="$1" stage="$2" detail="${3:-}"
  printf '{\n  "status": "%s",\n  "stage": "%s",\n  "detail": "%s",\n  "parallelism": %d,\n  "mismatch_seed": %d,\n  "controller_pid": %d,\n  "updated_epoch_seconds": %d\n}\n' \
    "$status" "$stage" "$detail" "$PARALLELISM" "$MISMATCH_SEED" "$$" "$(date +%s)" \
    >"$OUTPUT_ROOT/state.json.tmp"
  mv "$OUTPUT_ROOT/state.json.tmp" "$OUTPUT_ROOT/state.json"
}

on_error() {
  local code=$?
  write_state failed "$CURRENT_STAGE" "exit_code_$code"
  exit "$code"
}
trap on_error ERR

condition_spec() {
  case "$1" in
    wrn_unfused_recal_bn)
      printf 'bn|wrn_28_2_cifar|%s\n' \
        "$ROOT/checkpoint/baselines/cifar100/custom_noresize/wrn_28_2_cifar/custom_noresize_cifar100_wrn_28_2_cifar/custom_noresize_cifar100_wrn_28_2_cifar_best_ckpt.pth"
      ;;
    wrn_wd1e3_unfused_recal_bn)
      printf 'bn|wrn_28_2_cifar|%s\n' \
        "$ROOT/checkpoint/baselines_wd1e3_pilot/cifar100/custom_noresize/wrn_28_2_cifar/custom_noresize_cifar100_wrn_28_2_cifar/custom_noresize_cifar100_wrn_28_2_cifar_best_ckpt.pth"
      ;;
    wrn_wd1e3_finaldrop025_unfused_recal_bn)
      printf 'bn|wrn_28_2_cifar|%s\n' \
        "$ROOT/checkpoint/baselines_wd1e3_finaldrop025/cifar100/custom_noresize/wrn_28_2_cifar/custom_noresize_cifar100_wrn_28_2_cifar/custom_noresize_cifar100_wrn_28_2_cifar_best_ckpt.pth"
      ;;
    avgpool_bn)
      printf 'bn|wrn_28_2_cifar_avgpool|%s\n' \
        "$ROOT/checkpoint/baselines_wrn28_2_avgpool_study/cifar100/custom_noresize/wrn_28_2_cifar_avgpool/custom_noresize_cifar100_wrn_28_2_cifar_avgpool/custom_noresize_cifar100_wrn_28_2_cifar_avgpool_best_ckpt.pth"
      ;;
    stride2_bn)
      printf 'bn|wrn_28_2_cifar_avgpool_shortcut|%s\n' \
        "$ROOT/checkpoint/baselines_wrn28_2_avgpool_study/cifar100/custom_noresize/wrn_28_2_cifar_avgpool_shortcut/custom_noresize_cifar100_wrn_28_2_cifar_avgpool_shortcut/custom_noresize_cifar100_wrn_28_2_cifar_avgpool_shortcut_best_ckpt.pth"
      ;;
    avgpool_bnfree)
      printf 'bnfree|wrn_28_2_cifar_nobn_avgpool|%s\n' \
        "$ROOT/checkpoint/baselines_wrn28_2_avgpool_study/cifar100/custom_noresize/wrn_28_2_cifar_nobn_avgpool/custom_noresize_cifar100_wrn_28_2_cifar_nobn_avgpool/custom_noresize_cifar100_wrn_28_2_cifar_nobn_avgpool_best_ckpt.pth"
      ;;
    stride2_bnfree)
      printf 'bnfree|wrn_28_2_cifar_nobn_avgpool_shortcut|%s\n' \
        "$ROOT/checkpoint/baselines_wrn28_2_avgpool_study/cifar100/custom_noresize/wrn_28_2_cifar_nobn_avgpool_shortcut/custom_noresize_cifar100_wrn_28_2_cifar_nobn_avgpool_shortcut/custom_noresize_cifar100_wrn_28_2_cifar_nobn_avgpool_shortcut_best_ckpt.pth"
      ;;
    *) echo "Unknown condition: $1" >&2; return 2 ;;
  esac
}

family_spec() {
  case "$1" in
    max_additive) printf 'additive|max_abs|0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1\n' ;;
    rms_additive) printf 'additive|rms|0,0.25,0.5,0.75,1.0,1.25\n' ;;
    multiplicative) printf 'multiplicative|max_abs|0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4\n' ;;
    *) echo "Unknown family: $1" >&2; return 2 ;;
  esac
}

verify_checkpoints() {
  local condition kind model checkpoint
  for condition in "${CONDITIONS[@]}"; do
    IFS='|' read -r kind model checkpoint <<<"$(condition_spec "$condition")"
    [[ -s "$checkpoint" ]] || { echo "Missing checkpoint for $condition: $checkpoint" >&2; return 3; }
  done
}

run_one() {
  local condition="$1" family="$2" kind model checkpoint mismatch_type scale levels output log
  IFS='|' read -r kind model checkpoint <<<"$(condition_spec "$condition")"
  IFS='|' read -r mismatch_type scale levels <<<"$(family_spec "$family")"
  output="$OUTPUT_ROOT/results/$condition/$family"
  log="$OUTPUT_ROOT/logs/${condition}_${family}.log"

  if [[ -s "$output/full_aggregate.csv" ]]; then
    echo "Skipping completed result: $condition $family"
    return 0
  fi
  if [[ -d "$output" || -e "$log" ]]; then
    echo "Refusing to overwrite partial result: $condition $family" >&2
    return 4
  fi
  mkdir -p "$output"

  if [[ "$kind" == bn ]]; then
    "$PYTHON_BIN" -u baseline/run_wrn_bn_recalibration_experiment.py \
      --mode full \
      --compare_dir "$COMPARE_DIR" \
      --output_dir "$output" \
      --data_dir "$DATA_DIR" \
      --checkpoint_override "$checkpoint" \
      --model_name_override "$model" \
      --datasets cifar100 \
      --architectures WRN_28_2 \
      --mismatch_types "$mismatch_type" \
      --additive_scale_mode "$scale" \
      --pcn_reference_policy none \
      --noise_levels "$levels" \
      --noisy_trials 10 \
      --seed "$MISMATCH_SEED" \
      --batch_size 128 \
      --num_workers 2 \
      --calibration_num_samples 5120 \
      --calibration_batch_size 128 \
      --calibration_subset_seed 20240618 \
      --calibration_num_workers 2 \
      --case custom_noresize \
      --mismatch_parameter_policy existing >"$log" 2>&1
  else
    "$PYTHON_BIN" -u baseline/run_wrn_nobn_mismatch_experiment.py \
      --output_dir "$output" \
      --data_dir "$DATA_DIR" \
      --checkpoint_override "$checkpoint" \
      --model_name_override "$model" \
      --datasets cifar100 \
      --architectures WRN_28_2 \
      --mismatch_types "$mismatch_type" \
      --additive_scale_mode "$scale" \
      --noise_levels "$levels" \
      --noisy_trials 10 \
      --seed "$MISMATCH_SEED" \
      --batch_size 128 \
      --num_workers 2 \
      --case custom_noresize >"$log" 2>&1
  fi

  [[ -s "$output/full_aggregate.csv" ]] || {
    echo "Missing aggregate output: $condition $family" >&2
    return 5
  }
}

export ROOT PYTHON_BIN DATA_DIR OUTPUT_ROOT COMPARE_DIR PARALLELISM MISMATCH_SEED
export -f condition_spec family_spec run_one
export SHELL=/bin/bash

cat >"$OUTPUT_ROOT/run_manifest.json" <<EOF
{
  "dataset": "cifar100",
  "architecture": "WRN_28_2",
  "conditions": ["wrn_unfused_recal_bn", "wrn_wd1e3_unfused_recal_bn", "wrn_wd1e3_finaldrop025_unfused_recal_bn", "avgpool_bn", "stride2_bn", "avgpool_bnfree", "stride2_bnfree"],
  "mismatch_families": ["max_additive", "rms_additive", "multiplicative"],
  "mismatch_seed": $MISMATCH_SEED,
  "noisy_trials": 10,
  "parallelism": $PARALLELISM,
  "bn_calibration_num_samples": 5120,
  "bn_calibration_subset_seed": 20240618
}
EOF

CURRENT_STAGE="checkpoint_preflight"
write_state running "$CURRENT_STAGE"
verify_checkpoints

CURRENT_STAGE="evaluation"
write_state running "$CURRENT_STAGE" "0_of_21_complete"
parallel --jobs "$PARALLELISM" --halt now,fail=1 \
  --joblog "$OUTPUT_ROOT/evaluation.joblog" \
  run_one {1} {2} ::: "${CONDITIONS[@]}" ::: "${FAMILIES[@]}"

CURRENT_STAGE="complete"
write_state complete "$CURRENT_STAGE" "21_of_21_complete"
echo "Seed-456 WRN-28-2 replication complete: $OUTPUT_ROOT"

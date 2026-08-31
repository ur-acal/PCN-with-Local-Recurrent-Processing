#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/home/rongzeng/anaconda3/envs/scanbase/bin/python}"
DATA_DIR="${DATA_DIR:-$ROOT/../data}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$ROOT/checkpoint/baselines_wd1e3_finaldrop025}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/logs/wrn_wd1e3_finaldrop025_mismatch_full}"
COMPARE_DIR="${COMPARE_DIR:-$ROOT/../ScAN-PCN/logs/wrn_like_compare_target_models_by_arch}"
MAX_PARALLEL="${MAX_PARALLEL:-16}"
CURRENT_STAGE="initializing"

[[ -x "$PYTHON_BIN" ]] || { echo "Missing scanbase Python: $PYTHON_BIN" >&2; exit 1; }
command -v parallel >/dev/null || { echo "GNU parallel is required" >&2; exit 1; }
mkdir -p "$OUTPUT_ROOT/logs"

write_state() {
  local status="$1" stage="$2" detail="${3:-}"
  printf '{\n  "status": "%s",\n  "stage": "%s",\n  "detail": "%s",\n  "controller_pid": %d,\n  "parallelism": %d,\n  "updated_epoch_seconds": %d\n}\n' \
    "$status" "$stage" "$detail" "$$" "$MAX_PARALLEL" "$(date +%s)" >"$OUTPUT_ROOT/state.json.tmp"
  mv "$OUTPUT_ROOT/state.json.tmp" "$OUTPUT_ROOT/state.json"
}

on_error() {
  local code=$?
  write_state failed "$CURRENT_STAGE" "exit_code_$code"
  exit "$code"
}
trap on_error ERR

model_name() {
  local architecture="$1"
  printf 'wrn_%s_cifar\n' "${architecture#WRN_}" | tr '[:upper:]' '[:lower:]'
}

checkpoint_path() {
  local dataset="$1" architecture="$2" model run
  model="$(model_name "$architecture")"
  run="custom_noresize_${dataset}_${model}"
  printf '%s/%s/custom_noresize/%s/%s/%s_best_ckpt.pth\n' \
    "$CHECKPOINT_ROOT" "$dataset" "$model" "$run" "$run"
}

verify_checkpoints() {
  local dataset architecture checkpoint missing=()
  for dataset in cifar10 cifar100; do
    for architecture in WRN_16_2 WRN_16_4 WRN_28_2 WRN_28_4; do
      checkpoint="$(checkpoint_path "$dataset" "$architecture")"
      [[ -s "$checkpoint" ]] || missing+=("$dataset/$architecture")
    done
  done
  if (("${#missing[@]}" > 0)); then
    echo "Missing final-dropout checkpoints: ${missing[*]}" >&2
    return 1
  fi
}

run_one() {
  local family="$1" dataset="$2" architecture="$3" mode="$4"
  local policy mismatch_types scale noise_levels output log
  case "$mode" in
    folded) policy="bn_fold_no_mismatch_bias" ;;
    unfolded) policy="existing" ;;
    *) echo "Unknown mode: $mode" >&2; return 2 ;;
  esac
  case "$family" in
    max_mul)
      mismatch_types="all"
      scale="max_abs"
      noise_levels="csv"
      ;;
    rms)
      mismatch_types="additive"
      scale="rms"
      noise_levels="0.25,0.5,0.75,1.0,1.25"
      ;;
    *)
      echo "Unknown mismatch family: $family" >&2
      return 2
      ;;
  esac

  output="$OUTPUT_ROOT/$family/$mode/$dataset/$architecture"
  log="$OUTPUT_ROOT/logs/${family}_${mode}_${dataset}_${architecture}.log"
  if [[ -s "$output/full_aggregate.csv" ]]; then
    echo "Skipping completed mismatch result: $family $mode $dataset $architecture"
    return 0
  fi
  if [[ -d "$output" || -e "$log" ]]; then
    echo "Refusing to overwrite partial mismatch artifacts: $family $mode $dataset $architecture" >&2
    return 3
  fi
  mkdir -p "$output"

  "$PYTHON_BIN" -u baseline/run_wrn_bn_recalibration_experiment.py \
    --mode full \
    --compare_dir "$COMPARE_DIR" \
    --output_dir "$output" \
    --data_dir "$DATA_DIR" \
    --checkpoint_root "$CHECKPOINT_ROOT" \
    --datasets "$dataset" \
    --architectures "$architecture" \
    --mismatch_types "$mismatch_types" \
    --additive_scale_mode "$scale" \
    --pcn_reference_policy none \
    --noise_levels "$noise_levels" \
    --noisy_trials 10 \
    --seed 123 \
    --device cuda \
    --batch_size 128 \
    --num_workers 2 \
    --calibration_num_samples 5120 \
    --calibration_batch_size 128 \
    --calibration_subset_seed 20240618 \
    --calibration_num_workers 2 \
    --case custom_noresize \
    --mismatch_parameter_policy "$policy" >"$log" 2>&1

  [[ -s "$output/full_aggregate.csv" ]] || {
    echo "Mismatch evaluator did not produce full_aggregate.csv: $output" >&2
    return 4
  }
}

export ROOT PYTHON_BIN DATA_DIR CHECKPOINT_ROOT OUTPUT_ROOT COMPARE_DIR
export -f run_one
export SHELL=/bin/bash

cat >"$OUTPUT_ROOT/run_manifest.json" <<EOF
{
  "checkpoint_root": "$CHECKPOINT_ROOT",
  "pcn_reference_policy": "none",
  "max_additive_scale_mode": "max_abs",
  "rms_additive_scale_mode": "rms",
  "mismatch_types": ["additive_max", "additive_rms", "multiplicative"],
  "max_additive_and_multiplicative_levels": "csv",
  "rms_additive_levels": [0.25, 0.5, 0.75, 1.0, 1.25],
  "noisy_trials": 10,
  "mismatch_seed": 123,
  "calibration_num_samples": 5120,
  "calibration_batch_size": 128,
  "calibration_subset_seed": 20240618,
  "unfolded_policy": "existing",
  "parallelism": $MAX_PARALLEL
}
EOF

CURRENT_STAGE="checkpoint_preflight"
write_state running "$CURRENT_STAGE" "eight_checkpoints"
verify_checkpoints

CURRENT_STAGE="mismatch_evaluation"
write_state running "$CURRENT_STAGE" "16_unfolded_recal_bn_jobs"
parallel --jobs "$MAX_PARALLEL" --halt now,fail=1 \
  --joblog "$OUTPUT_ROOT/mismatch.joblog" \
  run_one {1} {2} {3} {4} \
  ::: max_mul rms \
  ::: cifar10 cifar100 \
  ::: WRN_16_2 WRN_16_4 WRN_28_2 WRN_28_4 \
  ::: unfolded

CURRENT_STAGE="completed"
write_state completed "$CURRENT_STAGE" "16_unfolded_recal_bn_jobs_complete"


#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/home/rongzeng/anaconda3/envs/scanbase/bin/python}"
DATA_DIR="${DATA_DIR:-$ROOT/../data}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$ROOT/checkpoint/baselines_wd1e3_finaldrop025}"
COMPARE_DIR="${COMPARE_DIR:-$ROOT/../ScAN-PCN/logs/wrn_like_compare_target_models_by_arch}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/logs/wrn_wd1e3_finaldrop025_ff_gain_cifar100}"
MAX_PARALLEL="${MAX_PARALLEL:-8}"

ARCHITECTURES=(WRN_16_2 WRN_16_4 WRN_28_2 WRN_28_4)
GAINS=(0.90 0.91 0.92 0.93 0.94 0.95 0.96 0.97 0.98 0.99 1.00 1.01 1.02 1.03 1.04 1.05 1.06 1.07 1.08 1.09 1.10)

[[ -x "$PYTHON_BIN" ]] || { echo "Missing scanbase Python: $PYTHON_BIN" >&2; exit 1; }
command -v parallel >/dev/null || { echo "GNU parallel is required" >&2; exit 1; }
mkdir -p "$OUTPUT_ROOT/logs"

model_name() {
  local architecture="$1"
  printf 'wrn_%s_cifar\n' "${architecture#WRN_}" | tr '[:upper:]' '[:lower:]'
}

checkpoint_path() {
  local architecture="$1" model run
  model="$(model_name "$architecture")"
  run="custom_noresize_cifar100_${model}"
  printf '%s/cifar100/custom_noresize/%s/%s/%s_best_ckpt.pth\n' \
    "$CHECKPOINT_ROOT" "$model" "$run" "$run"
}

verify_checkpoints() {
  local architecture checkpoint missing=()
  for architecture in "${ARCHITECTURES[@]}"; do
    checkpoint="$(checkpoint_path "$architecture")"
    [[ -s "$checkpoint" ]] || missing+=("$architecture")
  done
  if ((${#missing[@]})); then
    echo "Missing CIFAR-100 final-dropout checkpoints: ${missing[*]}" >&2
    return 1
  fi
}

merge_csvs() {
  local filename="$1" destination="$2" expected=84
  local files=()
  mapfile -t files < <(find "$OUTPUT_ROOT" -path "*/gain_*/$filename" -type f | sort)
  if ((${#files[@]} != expected)); then
    echo "Expected $expected $filename files, found ${#files[@]}" >&2
    return 1
  fi
  awk 'FNR == 1 { if (NR == 1) print; next } { print }' "${files[@]}" >"$destination"
}

gain_tag() {
  printf '%s' "$1" | tr '.' 'p'
}

run_one() {
  local architecture="$1" gain="$2" tag output log
  tag="$(gain_tag "$gain")"
  output="$OUTPUT_ROOT/$architecture/gain_$tag"
  log="$OUTPUT_ROOT/logs/${architecture}_gain_${tag}.log"

  if [[ -s "$output/full_aggregate.csv" ]]; then
    echo "Skipping completed result: $architecture gain=$gain"
    return 0
  fi
  if [[ -d "$output" || -e "$log" ]]; then
    echo "Refusing to overwrite partial artifacts: $architecture gain=$gain" >&2
    return 2
  fi
  mkdir -p "$output"

  "$PYTHON_BIN" -u baseline/run_wrn_bn_recalibration_experiment.py \
    --mode full \
    --compare_dir "$COMPARE_DIR" \
    --output_dir "$output" \
    --data_dir "$DATA_DIR" \
    --checkpoint_root "$CHECKPOINT_ROOT" \
    --datasets cifar100 \
    --architectures "$architecture" \
    --mismatch_types additive \
    --additive_scale_mode max_abs \
    --ff_gain "$gain" \
    --pcn_reference_policy none \
    --noise_levels 0 \
    --noisy_trials 1 \
    --seed 123 \
    --device cuda \
    --batch_size 128 \
    --num_workers 2 \
    --calibration_num_samples 5120 \
    --calibration_batch_size 128 \
    --calibration_subset_seed 20240618 \
    --calibration_num_workers 2 \
    --case custom_noresize \
    --mismatch_parameter_policy existing >"$log" 2>&1

  [[ -s "$output/full_aggregate.csv" ]] || {
    echo "Evaluator did not produce full_aggregate.csv: $output" >&2
    return 3
  }
}

export ROOT PYTHON_BIN DATA_DIR CHECKPOINT_ROOT COMPARE_DIR OUTPUT_ROOT
export -f gain_tag run_one
export SHELL=/bin/bash

cat >"$OUTPUT_ROOT/run_manifest.json" <<EOF
{
  "dataset": "cifar100",
  "architectures": ["WRN_16_2", "WRN_16_4", "WRN_28_2", "WRN_28_4"],
  "gains": [0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.00, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.09, 1.10],
  "checkpoint_root": "$CHECKPOINT_ROOT",
  "mismatch": "none",
  "trials_per_gain": 1,
  "bn_condition": "unfolded, all BN statistics recalibrated after gain",
  "calibration_num_samples": 5120,
  "calibration_subset_seed": 20240618,
  "parallelism": $MAX_PARALLEL
}
EOF

verify_checkpoints

parallel --jobs "$MAX_PARALLEL" --halt now,fail=1 \
  --joblog "$OUTPUT_ROOT/sweep.joblog" \
  run_one {1} {2} \
  ::: "${ARCHITECTURES[@]}" \
  ::: "${GAINS[@]}"

merge_csvs full_aggregate.csv "$OUTPUT_ROOT/full_aggregate.csv"
merge_csvs full_clean_sanity.csv "$OUTPUT_ROOT/full_clean_sanity.csv"

echo "Completed CIFAR-100 WRN FF-gain sweep: $OUTPUT_ROOT"

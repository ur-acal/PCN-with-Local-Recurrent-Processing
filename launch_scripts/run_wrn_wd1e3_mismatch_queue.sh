#!/bin/bash -l
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/home/rongzeng/anaconda3/envs/scanbase/bin/python}"
DATA_DIR="${DATA_DIR:-$ROOT/../data}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$ROOT/checkpoint/baselines_wd1e3_pilot}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/logs/wrn_wd1e3_bn_recalibration_full_5120_bn_fold_no_mismatch_bias}"
PCN_KAPPA_CSV="${PCN_KAPPA_CSV:-$ROOT/../ScAN-PCN/logs/kappa_audit/all_primary.csv}"
MAX_PARALLEL="${MAX_PARALLEL:-4}"

[[ -x "$PYTHON_BIN" ]] || { echo "Missing scanbase Python: $PYTHON_BIN" >&2; exit 1; }
[[ -f "$PCN_KAPPA_CSV" ]] || { echo "Missing PCN kappa CSV: $PCN_KAPPA_CSV" >&2; exit 1; }
mkdir -p "$OUTPUT_ROOT/jobs" "$OUTPUT_ROOT/eval_logs"

write_state() {
  local status="$1" detail="${2:-}"
  printf '{\n  "status": "%s",\n  "stage": "mismatch_evaluation",\n  "detail": "%s",\n  "controller_pid": %d,\n  "updated_epoch_seconds": %d\n}\n' \
    "$status" "$detail" "$$" "$(date +%s)" >"$OUTPUT_ROOT/state.json.tmp"
  mv "$OUTPUT_ROOT/state.json.tmp" "$OUTPUT_ROOT/state.json"
}

training_log() {
  local dataset="$1" arch="$2"
  if [[ "$arch" == "WRN_16_2" ]]; then
    printf '%s\n' "$ROOT/logs/wrn16_2_wd1e3_pilot/train_logs/standard_${dataset}.log"
  else
    local compact="wrn${arch#WRN_}"
    compact="${compact,,}"
    printf '%s\n' "$ROOT/logs/wrn_wd1e3_remaining/train_logs/${dataset}_${compact}.log"
  fi
}

checkpoint_path() {
  local dataset="$1" arch="$2"
  local model="wrn_${arch#WRN_}"
  model="${model,,}_cifar"
  local run="custom_noresize_${dataset}_${model}"
  printf '%s\n' "$CHECKPOINT_ROOT/$dataset/custom_noresize/$model/$run/${run}_best_ckpt.pth"
}

wait_for_training() {
  local dataset="$1" arch="$2" log ckpt
  log="$(training_log "$dataset" "$arch")"
  ckpt="$(checkpoint_path "$dataset" "$arch")"
  until [[ -f "$log" ]] && grep -q -- 'Train finished' "$log" && [[ -f "$ckpt" ]]; do
    sleep 30
  done
}

run_mode() {
  local dataset="$1" arch="$2" mode="$3" policy out log
  out="$OUTPUT_ROOT/jobs/${dataset}_${arch}/${mode}"
  log="$OUTPUT_ROOT/eval_logs/${dataset}_${arch}_${mode}.log"
  if [[ -s "$out/full_aggregate.csv" ]]; then
    return 0
  fi
  mkdir -p "$out"
  if [[ "$mode" == "folded" ]]; then
    policy="bn_fold_no_mismatch_bias"
  else
    policy="existing"
  fi
  "$PYTHON_BIN" -u baseline/run_wrn_bn_recalibration_experiment.py \
    --mode full \
    --compare_dir "$ROOT/../ScAN-PCN/logs/wrn_like_compare_target_models_by_arch" \
    --output_dir "$out" \
    --data_dir "$DATA_DIR" \
    --checkpoint_root "$CHECKPOINT_ROOT" \
    --datasets "$dataset" \
    --architectures "$arch" \
    --mismatch_types all \
    --noise_levels csv \
    --noisy_trials 10 \
    --seed 123 \
    --mismatch_parameter_policy "$policy" \
    --calibration_num_samples 5120 \
    --calibration_batch_size 128 \
    --calibration_subset_seed 20240618 \
    --calibration_num_workers 4 \
    --batch_size 128 \
    --num_workers 4 >"$log" 2>&1
}

run_checkpoint() {
  local dataset="$1" arch="$2"
  wait_for_training "$dataset" "$arch"
  run_mode "$dataset" "$arch" folded || return 1
  run_mode "$dataset" "$arch" unfolded
}

merge_aggregates() {
  local mode="$1" output="$2"
  mapfile -t files < <(find "$OUTPUT_ROOT/jobs" -path "*/${mode}/full_aggregate.csv" -type f | sort)
  [[ ${#files[@]} -eq 8 ]] || {
    echo "Expected 8 ${mode} aggregate files, found ${#files[@]}" >&2
    return 1
  }
  awk 'FNR == 1 { if (NR == 1) print; next } { print }' "${files[@]}" >"$output"
}

write_state running queued_eight_checkpoints

datasets=(cifar10 cifar100 cifar10 cifar100 cifar10 cifar100 cifar10 cifar100)
archs=(WRN_16_2 WRN_16_2 WRN_16_4 WRN_16_4 WRN_28_2 WRN_28_2 WRN_28_4 WRN_28_4)
pids=()
labels=()
failed=()

for i in "${!archs[@]}"; do
  while ((${#pids[@]} >= MAX_PARALLEL)); do
    if ! wait "${pids[0]}"; then
      failed+=("${labels[0]}")
    fi
    pids=("${pids[@]:1}")
    labels=("${labels[@]:1}")
  done
  run_checkpoint "${datasets[$i]}" "${archs[$i]}" &
  pids+=("$!")
  labels+=("${datasets[$i]}_${archs[$i]}")
done

for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    failed+=("${labels[$i]}")
  fi
done

if ((${#failed[@]})); then
  write_state failed "${failed[*]}"
  exit 1
fi

merge_aggregates folded "$OUTPUT_ROOT/full_aggregate.csv"
merge_aggregates unfolded "$OUTPUT_ROOT/full_unfolded_aggregate.csv"

"$PYTHON_BIN" baseline/generate_wrn_bn_recalibration_presentation_table.py \
  --aggregate_csv "$OUTPUT_ROOT/full_aggregate.csv" \
  --unfolded_recal_aggregate_csv "$OUTPUT_ROOT/full_unfolded_aggregate.csv" \
  --output_dir "$OUTPUT_ROOT" \
  --pcn_kappa_csv "$PCN_KAPPA_CSV" \
  --checkpoint_root "$CHECKPOINT_ROOT" \
  --mismatch_parameter_policy bn_fold_no_mismatch_bias \
  --device cpu >"$OUTPUT_ROOT/report_generation.log" 2>&1

write_state completed full_report

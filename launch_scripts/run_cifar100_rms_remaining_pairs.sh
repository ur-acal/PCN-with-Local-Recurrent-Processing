#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/home/rongzeng/anaconda3/envs/scanbase/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/logs/rms_cifar100_remaining_pairs}"
DATA_DIR="${DATA_DIR:-$ROOT/../data}"
PCN_ROOT="${PCN_ROOT:-$ROOT/../ScAN-PCN/saved_ckpt}"
WRN_ROOT="${WRN_ROOT:-$ROOT/checkpoint/baselines_wd1e3_pilot}"
COMPARE_DIR="${COMPARE_DIR:-$ROOT/../ScAN-PCN/logs/wrn_like_compare_target_models_by_arch}"
EXISTING_PILOT_ROOT="${EXISTING_PILOT_ROOT:-$ROOT/logs/rms_level_pilot_cifar100_WRN_28_2}"
LEVELS="0.25,0.5,0.75,1.0"
CURRENT_STAGE="initializing"

mkdir -p "$OUTPUT_ROOT/logs" "$OUTPUT_ROOT/pcn" "$OUTPUT_ROOT/wrn"

write_state() {
  local status="$1" stage="$2" detail="${3:-}"
  printf '{\n  "status": "%s",\n  "stage": "%s",\n  "detail": "%s",\n  "controller_pid": %d,\n  "updated_epoch_seconds": %d\n}\n' \
    "$status" "$stage" "$detail" "$$" "$(date +%s)" >"$OUTPUT_ROOT/state.json.tmp"
  mv "$OUTPUT_ROOT/state.json.tmp" "$OUTPUT_ROOT/state.json"
}

on_error() {
  local code=$?
  write_state failed "$CURRENT_STAGE" "exit_code_$code"
  exit "$code"
}
trap on_error ERR

pcn_model_name() {
  case "$1" in
    WRN_16_2)
      printf '%s\n' 'TIMMPCNetNoBatchNorm_PCConv_0.0eps_ODEXInitFFFB_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_C100_3K1S128C_0.25Dropout_7Layers0l1l2_2Pool_1REP'
      ;;
    WRN_16_4)
      printf '%s\n' 'TIMMPCNetNoBatchNorm_PCConv_0.0eps_ODEXInitFFFB_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_C100_3K1S256C_0.25Dropout_7Layers0l1l2_2Pool_1REP'
      ;;
    WRN_28_4)
      printf '%s\n' 'TIMMPCNetNoBatchNorm_PCConv_0.0eps_ODEXInitFFFB_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_C100_3K1S256C_0.25Dropout_13Layers0l3l6_2Pool_1REP'
      ;;
    *) return 2 ;;
  esac
}

run_pcn() {
  local architecture="$1" name output log
  name="$(pcn_model_name "$architecture")"
  output="$OUTPUT_ROOT/pcn/$architecture"
  log="$OUTPUT_ROOT/logs/pcn_${architecture}.log"
  if [[ -s "$output/result.pkl" ]]; then
    return 0
  fi
  [[ ! -e "$log" && ! -e "$output/result.pkl" ]] || {
    echo "Refusing to overwrite partial PCN output for $architecture" >&2
    return 3
  }
  mkdir -p "$output"
  "$PYTHON_BIN" -u ode_inference.py \
    --model_name "$name" \
    --ckpt best \
    --task cifar100 \
    --img_type rgb \
    --model_dir "$PCN_ROOT" \
    --method dopri5 \
    --tol 0.0001 \
    --n_steps 15 \
    --ts_scale 1 \
    --d_start 0 \
    --d_end 1 \
    --n_sweep_left 0 \
    --n_sweep_right 1 \
    --thermal_noise false \
    --mismatch_type add \
    --additive_scale_mode rms \
    --noise_level_list "$LEVELS" \
    --noisy_trials 3 \
    --test_bs 128 \
    --pc_conv PCConvNoisy \
    --ode_block ODEXInitFFFB \
    --output_pickle "$output/result.pkl" >"$log" 2>&1
}

run_wrn() {
  local architecture="$1" output log
  output="$OUTPUT_ROOT/wrn/$architecture"
  log="$OUTPUT_ROOT/logs/wrn_${architecture}.log"
  if [[ -s "$output/full_aggregate.csv" ]]; then
    return 0
  fi
  [[ ! -e "$log" && ! -e "$output/full_aggregate.csv" ]] || {
    echo "Refusing to overwrite partial WRN output for $architecture" >&2
    return 3
  }
  mkdir -p "$output"
  "$PYTHON_BIN" -u baseline/run_wrn_bn_recalibration_experiment.py \
    --mode full \
    --compare_dir "$COMPARE_DIR" \
    --output_dir "$output" \
    --data_dir "$DATA_DIR" \
    --checkpoint_root "$WRN_ROOT" \
    --datasets cifar100 \
    --architectures "$architecture" \
    --mismatch_types additive \
    --additive_scale_mode rms \
    --pcn_reference_policy none \
    --noise_levels "$LEVELS" \
    --noisy_trials 3 \
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
}

cat >"$OUTPUT_ROOT/run_manifest.json" <<EOF
{
  "dataset": "cifar100",
  "architectures": ["WRN_16_2", "WRN_16_4", "WRN_28_4"],
  "reused_architecture": "WRN_28_2",
  "levels": [0.25, 0.5, 0.75, 1.0],
  "trials": 3,
  "additive_scale_mode": "rms",
  "wrn_condition": "weight_decay_1e-3_unfused_recal_bn",
  "parallel_jobs": 6
}
EOF

CURRENT_STAGE="evaluation"
write_state running "$CURRENT_STAGE" "six_parallel_jobs"
pids=()
labels=()
for architecture in WRN_16_2 WRN_16_4 WRN_28_4; do
  run_pcn "$architecture" &
  pids+=("$!")
  labels+=("pcn_$architecture")
  run_wrn "$architecture" &
  pids+=("$!")
  labels+=("wrn_$architecture")
done

failed=()
for index in "${!pids[@]}"; do
  if ! wait "${pids[$index]}"; then
    failed+=("${labels[$index]}")
  fi
done
if (("${#failed[@]}" > 0)); then
  write_state failed "$CURRENT_STAGE" "${failed[*]}"
  exit 1
fi

CURRENT_STAGE="report"
write_state running "$CURRENT_STAGE" "summary"
"$PYTHON_BIN" baseline/generate_cifar100_rms_pilot_summary.py \
  --root "$OUTPUT_ROOT" \
  --existing_pilot_root "$EXISTING_PILOT_ROOT" \
  >"$OUTPUT_ROOT/report.log" 2>&1

CURRENT_STAGE="completed"
write_state completed "$CURRENT_STAGE" "summary_ready"


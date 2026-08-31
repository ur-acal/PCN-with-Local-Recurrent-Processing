#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/logs/rms_additive_full"
STATE="$OUT/state.json"
COMPARE_DIR="/home/rongzeng/_workspce_old/repos/pcn/collaboration/ScAN-PCN/logs/wrn_like_compare_target_models_by_arch"
ORIGINAL_WRN_ROOT="/home/rongzeng/_workspce_old/repos/pcn/collaboration/ScAN-PCN/checkpoint/baselines"
WD_WRN_ROOT="$ROOT/checkpoint/baselines_wd1e3_pilot"
PCN_ROOT="/home/rongzeng/_workspce_old/repos/pcn/collaboration/ScAN-PCN/saved_ckpt"
DATA_DIR="$ROOT/../data"
WRN_JOBS=16
PCN_JOBS=4
CURRENT_STAGE="initializing"

mkdir -p "$OUT"
cd "$ROOT"

write_state() {
  local status="$1" stage="$2" detail="$3"
  printf '{\n  "status": "%s",\n  "stage": "%s",\n  "detail": "%s",\n  "controller_pid": %s,\n  "wrn_parallelism": %s,\n  "pcn_parallelism": %s,\n  "updated_epoch_seconds": %s\n}\n' \
    "$status" "$stage" "$detail" "$$" "$WRN_JOBS" "$PCN_JOBS" "$(date +%s)" > "$STATE"
}

on_error() {
  local code=$?
  write_state failed "$CURRENT_STAGE" "exit_code_${code}"
  exit "$code"
}
trap on_error ERR

run_wrn_job() {
  local family="$1" checkpoint_root="$2" dataset="$3" architecture="$4" mode="$5"
  local policy output log
  if [[ "$mode" == "folded" ]]; then
    policy="bn_fold_no_mismatch_bias"
  else
    policy="existing"
  fi
  output="$OUT/$family/$mode/$dataset/$architecture"
  log="$OUT/logs/${family}_${mode}_${dataset}_${architecture}.log"
  mkdir -p "$output" "$(dirname "$log")"
  python -u baseline/run_wrn_bn_recalibration_experiment.py \
    --compare_dir "$COMPARE_DIR" \
    --output_dir "$output" \
    --data_dir "$DATA_DIR" \
    --checkpoint_root "$checkpoint_root" \
    --mode full \
    --datasets "$dataset" \
    --architectures "$architecture" \
    --mismatch_types additive \
    --additive_scale_mode rms \
    --noise_levels csv \
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
    --mismatch_parameter_policy "$policy" \
    >"$log" 2>&1
}

run_nobn_job() {
  local dataset="$1" architecture="$2"
  local index name checkpoint_root output log
  case "$architecture" in
    WRN_16_2) index=0 ;;
    WRN_16_4) index=1 ;;
    WRN_28_2) index=2 ;;
    WRN_28_4) index=3 ;;
    *) return 2 ;;
  esac
  name="wrn_${architecture#WRN_}_cifar_nobn"
  name="${name,,}"
  if [[ "$dataset" == "cifar100" ]]; then
    checkpoint_root="$ROOT/checkpoint/baselines_nobn_cifar100_searched_recipe"
    index=$((index + 4))
  else
    checkpoint_root="$ROOT/checkpoint/baselines_nobn_best_recipe"
  fi
  output="$OUT/wrn_nobn/$dataset/$architecture"
  log="$OUT/logs/wrn_nobn_${dataset}_${architecture}.log"
  mkdir -p "$output" "$(dirname "$log")"
  python -u baseline/run_wrn_nobn_mismatch_experiment.py \
    --output_dir "$output" \
    --data_dir "$DATA_DIR" \
    --checkpoint_root "$checkpoint_root" \
    --datasets "$dataset" \
    --architectures "$architecture" \
    --mismatch_types additive \
    --additive_scale_mode rms \
    --noise_levels default \
    --noisy_trials 10 \
    --seed 123 \
    --model_index_offset "$index" \
    --device cuda \
    --batch_size 128 \
    --num_workers 2 \
    >"$log" 2>&1
}

pcn_model_name() {
  local dataset="$1" architecture="$2"
  local channels layers layout prefix
  case "$architecture" in
    WRN_16_2) channels=128; layers=7; layout="0l1l2" ;;
    WRN_16_4) channels=256; layers=7; layout="0l1l2" ;;
    WRN_28_2) channels=128; layers=13; layout="0l3l6" ;;
    WRN_28_4) channels=256; layers=13; layout="0l3l6" ;;
    *) return 2 ;;
  esac
  prefix=""
  if [[ "$dataset" == "cifar100" ]]; then
    prefix="C100_"
  fi
  printf 'TIMMPCNetNoBatchNorm_PCConv_0.0eps_ODEXInitFFFB_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_%s3K1S%sC_0.25Dropout_%sLayers%s_2Pool_1REP' \
    "$prefix" "$channels" "$layers" "$layout"
}

run_pcn_job() {
  local scale_mode="$1" dataset="$2" architecture="$3"
  local name output log
  name="$(pcn_model_name "$dataset" "$architecture")"
  output="$OUT/pcn/$scale_mode/$dataset/$architecture"
  log="$OUT/logs/pcn_${scale_mode}_${dataset}_${architecture}.log"
  mkdir -p "$output" "$(dirname "$log")"
  python -u ode_inference.py \
    --model_name "$name" \
    --ckpt best \
    --task "$dataset" \
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
    --additive_scale_mode "$scale_mode" \
    --noise_level_list 0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1 \
    --noisy_trials 10 \
    --test_bs 128 \
    --pc_conv PCConvNoisy \
    --ode_block ODEXInitFFFB \
    --output_pickle "$output/result.pkl" \
    >"$log" 2>&1
}

export ROOT OUT COMPARE_DIR DATA_DIR PCN_ROOT
export -f run_wrn_job run_nobn_job pcn_model_name run_pcn_job
export SHELL=/bin/bash

printf '{\n  "additive_scale_mode": "rms",\n  "levels": [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1],\n  "noisy_trials": 10,\n  "seed": 123,\n  "calibration_samples": 5120,\n  "calibration_batch_size": 128,\n  "calibration_subset_seed": 20240618,\n  "wrn_parallelism": 16,\n  "pcn_parallelism": 4\n}\n' > "$OUT/run_manifest.json"

CURRENT_STAGE="wrn_original_rms"
write_state running "$CURRENT_STAGE" "16_jobs"
parallel --jobs "$WRN_JOBS" --halt now,fail=1 --joblog "$OUT/wrn_original.joblog" \
  run_wrn_job wrn_original "$ORIGINAL_WRN_ROOT" {1} {2} {3} \
  ::: cifar10 cifar100 \
  ::: WRN_16_2 WRN_16_4 WRN_28_2 WRN_28_4 \
  ::: folded unfolded

CURRENT_STAGE="wrn_wd1e3_rms"
write_state running "$CURRENT_STAGE" "16_jobs"
parallel --jobs "$WRN_JOBS" --halt now,fail=1 --joblog "$OUT/wrn_wd1e3.joblog" \
  run_wrn_job wrn_wd1e3 "$WD_WRN_ROOT" {1} {2} {3} \
  ::: cifar10 cifar100 \
  ::: WRN_16_2 WRN_16_4 WRN_28_2 WRN_28_4 \
  ::: folded unfolded

CURRENT_STAGE="wrn_nobn_rms"
write_state running "$CURRENT_STAGE" "8_jobs"
parallel --jobs 8 --halt now,fail=1 --joblog "$OUT/wrn_nobn.joblog" \
  run_nobn_job {1} {2} \
  ::: cifar10 cifar100 \
  ::: WRN_16_2 WRN_16_4 WRN_28_2 WRN_28_4

CURRENT_STAGE="pcn_corrected_max_additive"
write_state running "$CURRENT_STAGE" "8_jobs_at_4_way"
parallel --jobs "$PCN_JOBS" --halt now,fail=1 --joblog "$OUT/pcn_max_abs.joblog" \
  run_pcn_job max_abs {1} {2} \
  ::: cifar10 cifar100 \
  ::: WRN_16_2 WRN_16_4 WRN_28_2 WRN_28_4

CURRENT_STAGE="pcn_rms_additive"
write_state running "$CURRENT_STAGE" "8_jobs_at_4_way"
parallel --jobs "$PCN_JOBS" --halt now,fail=1 --joblog "$OUT/pcn_rms.joblog" \
  run_pcn_job rms {1} {2} \
  ::: cifar10 cifar100 \
  ::: WRN_16_2 WRN_16_4 WRN_28_2 WRN_28_4

CURRENT_STAGE="report_generation"
write_state running "$CURRENT_STAGE" "merged_accuracy_and_kappa"
python -u baseline/generate_combined_rms_mismatch_report.py > "$OUT/report_generation.log" 2>&1

CURRENT_STAGE="completed"
write_state completed "$CURRENT_STAGE" "reports_ready"

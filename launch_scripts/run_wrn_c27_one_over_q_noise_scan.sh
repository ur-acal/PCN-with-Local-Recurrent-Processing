#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO_ROOT}"

PYTHON="${PYTHON:-/home/rongzeng/anaconda3/envs/scanbase/bin/python}"
export PATH="$(dirname "${PYTHON}"):${PATH}"
MODEL_NAME=wrn_flexible_cifar_avgpool_shortcut
MODEL_CKPT="${MODEL_CKPT:-saved_ckpt_runs/coupler_full_range_CiFAIR100_wrn28_c27_avgpool_shortcut_fixedTiming15ns_rescaled_pretrain_iq12/cifar100/custom_noresize/${MODEL_NAME}/custom_noresize_cifar100_${MODEL_NAME}/custom_noresize_cifar100_${MODEL_NAME}_best_ckpt.pth}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/wrn_c27_one_over_q_noise_only_10b}"
LOG_ROOT="${LOG_ROOT:-logs/local_runs/wrn_c27_one_over_q_noise_only_10b}"
PARALLEL_JOBS="${PARALLEL_JOBS:-3}"
[[ "${PARALLEL_JOBS}" =~ ^[1-9][0-9]*$ ]] || {
  echo "PARALLEL_JOBS must be a positive integer" >&2; exit 2;
}
ONE_OVER_Q_VALUES="${ONE_OVER_Q_VALUES:-1 2 4 8 16 25 32 50 64 128}"
ENABLE_CURRENT_NOISE="${ENABLE_CURRENT_NOISE:-true}"
V_DD="${V_DD:-0.1}"
W_BITS="${W_BITS:-5}"
WEIGHT_QUANT_FACTOR_BITS="${WEIGHT_QUANT_FACTOR_BITS:-1}"
TOGGLE_TIMING_MODE="${TOGGLE_TIMING_MODE:-fixed}"
TOGGLE_Y_TIME="${TOGGLE_Y_TIME:-15e-9}"
Z_OVER_Y_TIME="${Z_OVER_Y_TIME:-1}"
PHYSICAL="${PHYSICAL:-true}"
CHECKPOINT_KIND="${CHECKPOINT_KIND:-unitless_physical}"
MAX_EVAL_BATCHES="${MAX_EVAL_BATCHES:-10}"
ENABLE_MEASURED_ACTIVATION="${ENABLE_MEASURED_ACTIVATION:-false}"
ENABLE_NONLINEAR_R="${ENABLE_NONLINEAR_R:-false}"
ENABLE_MEASURED_POOLING="${ENABLE_MEASURED_POOLING:-false}"
ENABLE_SPIN_VARIATION="${ENABLE_SPIN_VARIATION:-false}"
ENABLE_DTC_NONIDEALITY="${ENABLE_DTC_NONIDEALITY:-false}"
NONLINEAR_R_TABLE="${NONLINEAR_R_TABLE:-./hardware_data/mc_45_corners/coupler_monte_tt_25_1_mc81_singleton.csv}"
NONLINEAR_R_TRAIN_MODE="${NONLINEAR_R_TRAIN_MODE:-exact_curve}"

mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"
[[ -f "${MODEL_CKPT}" ]] || { echo "Missing checkpoint: ${MODEL_CKPT}" >&2; exit 1; }

run_one() {
  local one_over_q="$1"
  local tag="qinv${one_over_q}"
  local log="${LOG_ROOT}/${tag}.log"

  echo "$(date -Is) START ${tag}" | tee -a "${LOG_ROOT}/runner.log"
  "${PYTHON}" -u baseline/evaluate_physical_feedforward_cifar.py \
    --model_name "${MODEL_NAME}" \
    --wrn_depth 28 \
    --wrn_first_stage_channels 27 \
    --checkpoint "${MODEL_CKPT}" \
    --dataset cifar100 \
    --img_type CiFAIR \
    --batch_size 128 \
    --max_eval_batches "${MAX_EVAL_BATCHES}" \
    --result_path "${OUTPUT_ROOT}/${tag}" \
    --n_trials 1 \
    --physical_level 2 \
    --physical "${PHYSICAL}" \
    --checkpoint_kind "${CHECKPOINT_KIND}" \
    --R 50e3 \
    --C 500e-15 \
    --v_dd "${V_DD}" \
    --one_over_q "${one_over_q}" \
    --w_bits "${W_BITS}" \
    --weight_quant_factor_bits "${WEIGHT_QUANT_FACTOR_BITS}" \
    --noise_level 0.0 \
    --diff_mismatch false \
    --toggle_timing_mode "${TOGGLE_TIMING_MODE}" \
    --toggle_y_time "${TOGGLE_Y_TIME}" \
    --z_over_y_time "${Z_OVER_Y_TIME}" \
    --enob none \
    --input_quant_bits 12 \
    --center_student_input false \
    --enable_spin_variation "${ENABLE_SPIN_VARIATION}" \
    --spin_variation_seed 20260723 \
    --enable_summing_current_noise "${ENABLE_CURRENT_NOISE}" \
    --summing_current_p 0.6e-12 \
    --summing_noise_seed 20260723 \
    --enable_coupler_noise "${ENABLE_CURRENT_NOISE}" \
    --coupler_noise_p 0.6e-12 \
    --coupler_noise_seed 20260723 \
    --enable_slow_summing_current false \
    --enable_slow_coupler_noise false \
    --enable_dtc_nonideality "${ENABLE_DTC_NONIDEALITY}" \
    --enable_measured_activation "${ENABLE_MEASURED_ACTIVATION}" \
    --measured_activation_scope pc_only \
    --fuse_measured_activation false \
    --activation_curve_path ./hardware_data/relu_current_0p2uA_finer.csv \
    --activation_corner TT \
    --activation_curve_sharing per_model \
    --enable_nonlinear_R "${ENABLE_NONLINEAR_R}" \
    --nonlinear_R_table "${NONLINEAR_R_TABLE}" \
    --nonlinear_R_train_mode "${NONLINEAR_R_TRAIN_MODE}" \
    --nonlinear_R_mc_quantity conductance \
    --nonlinear_R_curve_sharing shared \
    --enable_measured_pooling "${ENABLE_MEASURED_POOLING}" \
    --bn_recalibrate false \
    --use_expanded_weights false \
    --data_seed 20260723 \
    --final_head_type old_ideal \
    >"${log}" 2>&1
  echo "$(date -Is) COMPLETE ${tag}" | tee -a "${LOG_ROOT}/runner.log"
}

status=0
pids=()
for one_over_q in ${ONE_OVER_Q_VALUES}; do
  run_one "${one_over_q}" &
  pids+=("$!")
  if (( ${#pids[@]} >= PARALLEL_JOBS )); then
    wait "${pids[0]}" || status=1
    pids=("${pids[@]:1}")
  fi
done
for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done
if (( status != 0 )); then
  echo "$(date -Is) SCAN FAILED: one or more evaluations failed; see per-job logs" | tee -a "${LOG_ROOT}/runner.log"
  exit "${status}"
fi

echo "$(date -Is) ALL COMPLETE" | tee -a "${LOG_ROOT}/runner.log"

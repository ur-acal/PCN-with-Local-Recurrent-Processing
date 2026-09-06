#!/bin/bash

set -e
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO_ROOT}"

MODEL_NAME="${MODEL_NAME:?MODEL_NAME is required}"
MODEL_CKPT="${MODEL_CKPT:?MODEL_CKPT is required}"

cmd=(
  python baseline/evaluate_physical_feedforward_cifar.py
  --model_name "${MODEL_NAME}"
  --checkpoint "${MODEL_CKPT}"
  --dataset "${TASK:-cifar100}"
  --img_type "${IMG_TYPE:-CiFAIR}"
  --batch_size "${TEST_BATCH_SIZE:-128}"
  --expanded_weight_dir "${EXPANDED_WEIGHT_DIR:-./expanded_weights/feedforward}"
  --result_path "${RESULT_PATH:-./results/feedforward_physical_eval}"
  --n_trials "${N_TRIALS:-1}"
  --physical_level "${PHYSICAL_LEVEL:-3}"
  --R "${R_VAL:-50e3}"
  --C "${C_VAL:-500e-15}"
  --v_dd "${V_DD:-0.1}"
  --one_over_q "${ONE_OVER_Q:-1}"
  --w_bits "${W_BITS:-5}"
  --weight_quant_factor_bits "${WEIGHT_QUANT_FACTOR_BITS:-1}"
  --diff_mismatch "${DIFF_MISMATCH:-false}"
  --toggle_timing_mode "${TOGGLE_TIMING_MODE:-derived}"
  --toggle_y_time "${TOGGLE_Y_TIME:-5e-9}"
  --z_over_y_time "${Z_OVER_Y_TIME:-1}"
  --enob "${ENOB:-8}"
  --input_quant_bits "${INPUT_QUANT_BITS:-none}"
  --center_student_input "${CENTER_STUDENT_INPUT:-auto}"
  --enable_spin_variation "${ENABLE_SPIN_VARIATION:-true}"
  --sigma_spin "${SIGMA_SPIN:-0.10}"
  --spin_variation_mean "${SPIN_VARIATION_MEAN:-1.0}"
  --enable_summing_current_noise "${ENABLE_SUMMING_CURRENT_NOISE:-true}"
  --summing_current_p "${SUMMING_CURRENT_P:-0.6e-12}"
  --enable_coupler_noise "${ENABLE_COUPLER_NOISE:-true}"
  --coupler_noise_p "${COUPLER_NOISE_P:-0.6e-12}"
  --enable_slow_summing_current "${ENABLE_SLOW_SUMMING_CURRENT:-false}"
  --slow_summing_current "${SLOW_SUMMING_CURRENT:-2.47e-9}"
  --enable_slow_coupler_noise "${ENABLE_SLOW_COUPLER_NOISE:-false}"
  --slow_coupler_noise "${SLOW_COUPLER_NOISE:-2.47e-9}"
  --enable_dtc_nonideality "${ENABLE_DTC_NONIDEALITY:-true}"
  --dtc_leading_edge_variation_std "${DTC_LEADING_EDGE_VARIATION_STD:-0.0}"
  --dtc_width_variation_mean "${DTC_WIDTH_VARIATION_MEAN:-0.0}"
  --dtc_width_variation_std "${DTC_WIDTH_VARIATION_STD:-0.018}"
  --dtc_leading_edge_jitter_std "${DTC_LEADING_EDGE_JITTER_STD:-0.005}"
  --dtc_falling_edge_jitter_std "${DTC_FALLING_EDGE_JITTER_STD:-0.005}"
  --enable_measured_activation "${ENABLE_MEASURED_ACTIVATION:-true}"
  --activation_curve_path "${ACTIVATION_CURVE_PATH:-./hardware_data/relu_current_0p2uA_finer.csv}"
  --activation_corner "${ACTIVATION_CORNER:-TT}"
  --activation_curve_sharing "${ACTIVATION_CURVE_SHARING:-per_model}"
  --activation_interpolation "${ACTIVATION_INTERPOLATION:-piecewise_linear}"
  --activation_fit_constraint "${ACTIVATION_FIT_CONSTRAINT:-auto}"
  --activation_normalize_positive_endpoint "${ACTIVATION_NORMALIZE_POSITIVE_ENDPOINT:-false}"
  --compile_measured_activation "${COMPILE_MEASURED_ACTIVATION:-false}"
  --enable_nonlinear_R "${ENABLE_NONLINEAR_R:-true}"
  --nonlinear_R_table "${NONLINEAR_R_TABLE:-./hardware_data/mc_45_corners/coupler_monte}"
  --nonlinear_R_mc_quantity "${MC_COUPLER_NONLINEAR_VARIATION_QUANTITY:-conductance}"
  --nonlinear_R_curve_sharing "${NONLINEAR_R_CURVE_SHARING:-per_coupler}"
  --nonlinear_R_curve_sampling "${NONLINEAR_R_CURVE_SAMPLING:-empirical_with_replacement}"
  --nonlinear_R_curve_edge_chunk_size "${NONLINEAR_R_CURVE_EDGE_CHUNK_SIZE:-65536}"
  --enable_measured_pooling "${ENABLE_MEASURED_POOLING:-true}"
  --bn_recalibrate "${ENABLE_BN_RECALIBRATION:-false}"
  --bn_calibration_batch_size "${BN_CALIBRATION_BATCH_SIZE:-128}"
  --bn_calibration_samples "${BN_CALIBRATION_SAMPLES:-none}"
  --use_expanded_weights "${USE_EXPANDED_WEIGHTS:-true}"
  --nonlinear_R_train_mode "${NONLINEAR_R_TRAIN_MODE:-none}"
  --nonlinear_R_corner_range "${NONLINEAR_R_CORNER_RANGE:-all}"
)

if [[ -n "${WRN_DEPTH:-}" ]]; then
  cmd+=(--wrn_depth "${WRN_DEPTH}")
fi
if [[ -n "${WRN_FIRST_STAGE_CHANNELS:-}" ]]; then
  cmd+=(--wrn_first_stage_channels "${WRN_FIRST_STAGE_CHANNELS}")
fi

for pair in \
  "SPIN_VARIATION_SEED:spin_variation_seed" \
  "SUMMING_NOISE_SEED:summing_noise_seed" \
  "COUPLER_NOISE_SEED:coupler_noise_seed" \
  "DTC_TIMING_SEED:dtc_timing_seed" \
  "ACTIVATION_CURVE_SEED:activation_curve_seed" \
  "NONLINEAR_R_CURVE_SEED:nonlinear_R_curve_seed" \
  "DATA_SEED:data_seed"; do
  env_name="${pair%%:*}"
  arg_name="${pair##*:}"
  if [[ -n "${!env_name:-}" ]]; then
    cmd+=("--${arg_name}" "${!env_name}")
  fi
done

printf 'Running:'
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"

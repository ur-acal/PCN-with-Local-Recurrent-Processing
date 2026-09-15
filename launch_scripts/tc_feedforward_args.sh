# Opt-in defaults only; existing toggle commands remain unchanged.
TC_FEEDFORWARD_ARGS=()
if [[ "${TC_FEEDFORWARD:-false}" == true ]]; then
  source "$(dirname "${BASH_SOURCE[0]}")/tc_hardware_defaults.sh" "$1"
  export PHYSICAL_LEVEL=2
  export WEIGHT_QUANT_FACTOR_BITS=-1 ENOB=none
  export NONLINEAR_R_TABLE="$TC_MEAN_TABLE"
  if [[ "$1" == eval ]]; then
    export NONLINEAR_R_TRAIN_MODE=none
    export NONLINEAR_R_CURVE_SHARING=per_coupler
    export ACTIVATION_CURVE_PATH="${TC_EVAL_ACTIVATION_CURVE_PATH:-./hardware_data/mc_45_corners/0906_RELU_Voltage}"
    export ACTIVATION_CORNER="${TC_EVAL_ACTIVATION_CORNER:-TT_25_1_MC18}"
    export ACTIVATION_CURVE_SHARING=per_spin
  else
    export NONLINEAR_R_TRAIN_MODE=exact_curve
    if [[ "${ENABLE_NONLINEAR_R:-true}" == false ]]; then
      export NONLINEAR_R_TRAIN_MODE=none
    fi
    export ACTIVATION_CURVE_PATH="${ACTIVATION_CURVE_PATH:-./hardware_data/mc_45_corners/0906_RELU_Voltage/tt_25_1.csv}"
    export ACTIVATION_CORNER="${ACTIVATION_CORNER:-MC18}"
    export ACTIVATION_CURVE_SHARING=per_model
  fi
  TC_FEEDFORWARD_ARGS=(--tc_feedforward true --one_shot_conv "${ONE_SHOT_CONV:-false}"
    --w_bits 5 --enob none --weight_quant_factor_bits -1
    --tc_method "${TC_METHOD:-dopri5}" --tc_tol "${TC_TOL:-1e-6}"
    --tc_curve_sampling "${TC_CURVE_SAMPLING:-histogram}"
    --tc_covariance_table "${TC_COVARIANCE_TABLE:-./hardware_data/mc_45_corners/CU_4500_r_vs_vin.csv}"
    --tc_noise_reference_R "${TC_NOISE_REFERENCE_R:-50e3}"
    --measured_pooling_curve_path "${MEASURED_POOLING_CURVE_PATH:-${NONLINEAR_R_TABLE}}"
    --measured_pooling_nominal_R "${MEASURED_POOLING_NOMINAL_R:-10e3}")
  if [[ -n "${TC_STEP_SIZE:-}" ]]; then
    TC_FEEDFORWARD_ARGS+=(--tc_step_size "$TC_STEP_SIZE")
  fi
  if [[ "$1" != pretrain ]]; then
    TC_FEEDFORWARD_ARGS+=(--enable_spin_variation "${ENABLE_SPIN_VARIATION:-true}"
      --sigma_spin "${SIGMA_SPIN:-0.1}"
      --enable_summing_current_noise "${ENABLE_SUMMING_CURRENT_NOISE:-true}"
      --summing_current_p "${SUMMING_CURRENT_P:-0.6e-12}"
      --enable_coupler_noise "${ENABLE_COUPLER_NOISE:-true}"
      --coupler_noise_p "${COUPLER_NOISE_P:-0.6e-12}"
      --enable_measured_activation "${ENABLE_MEASURED_ACTIVATION:-true}"
      --enable_measured_pooling "${ENABLE_MEASURED_POOLING:-true}")
  fi
fi

# Opt-in defaults only; existing toggle commands remain unchanged.
TC_FEEDFORWARD_ARGS=()
# Toggle bank selection mirrors PCN. Explicit training paths still win.
if [[ "${TC_FEEDFORWARD:-false}" != true && -n "${MC_RELU_MONTE_CARLO_SOURCE:-}" ]]; then
  _ff_relu_bank="${MC_RELU_MONTE_CARLO_SOURCE%/}"
  [[ "${_ff_relu_bank}" == */* ]] || _ff_relu_bank="./hardware_data/mc_45_corners/${_ff_relu_bank}"
  case "${_ff_relu_bank##*/}" in
    0906_RELU_Voltage|1022_RELU_Voltage) _ff_relu_corner=TT_25_1_MC18 ;;
    *) _ff_relu_corner=RELU_TTG4_MC1 ;;
  esac
  export ACTIVATION_CURVE_PATH="${ACTIVATION_CURVE_PATH:-${_ff_relu_bank}}"
  export ACTIVATION_CORNER="${ACTIVATION_CORNER:-${_ff_relu_corner}}"
  unset _ff_relu_bank _ff_relu_corner
fi
if [[ "${TC_FEEDFORWARD:-false}" == true ]]; then
  source "$(dirname "${BASH_SOURCE[0]}")/tc_hardware_defaults.sh" "$1"
  if [[ "$1" == pretrain ]]; then
    export WARMUP_PRETRAIN="${WARMUP_PRETRAIN:-5}"
  fi
  export FINAL_EVAL_ONLY="${FINAL_EVAL_ONLY:-true}"
  export PHYSICAL_LEVEL=2
  export WEIGHT_QUANT_FACTOR_BITS=-1 ENOB=none
  export NONLINEAR_R_TABLE="$TC_MEAN_TABLE"
  tc_activation_bank="${MC_RELU_MONTE_CARLO_SOURCE:-0906_RELU_Voltage}"
  tc_activation_bank="${tc_activation_bank%/}"
  if [[ "${tc_activation_bank}" != */* ]]; then
    tc_activation_bank="./hardware_data/mc_45_corners/${tc_activation_bank}"
  fi
  if [[ "$1" == eval ]]; then
    export NONLINEAR_R_TRAIN_MODE=none
    export NONLINEAR_R_CURVE_SHARING=per_coupler
    export ACTIVATION_CURVE_PATH="${TC_EVAL_ACTIVATION_CURVE_PATH:-${tc_activation_bank}}"
    export ACTIVATION_CORNER="${TC_EVAL_ACTIVATION_CORNER:-TT_25_1_MC18}"
    export ACTIVATION_CURVE_SHARING=per_spin
  else
    export NONLINEAR_R_TRAIN_MODE=exact_curve
    if [[ "${ENABLE_NONLINEAR_R:-true}" == false ]]; then
      export NONLINEAR_R_TRAIN_MODE=none
    fi
    if [[ "${ACTIVATION_CORNER_MODE:-fixed}" == random_per_forward ]]; then
      export ACTIVATION_CURVE_PATH="${ACTIVATION_CURVE_PATH:-${tc_activation_bank}}"
      export ACTIVATION_CORNER="${ACTIVATION_CORNER:-TT_25_1_MC18}"
    else
      export ACTIVATION_CURVE_PATH="${ACTIVATION_CURVE_PATH:-${tc_activation_bank}/tt_25_1.csv}"
      export ACTIVATION_CORNER="${ACTIVATION_CORNER:-MC18}"
    fi
    export ACTIVATION_CURVE_SHARING=per_model
  fi
  TC_FEEDFORWARD_ARGS=(--tc_feedforward true
    --compute_time_fixed_std "${COMPUTE_TIME_FIXED_STD:-0}"
    --compute_time_noise_std "${COMPUTE_TIME_NOISE_STD:-0}"
    --timing_noise_scope "${TIMING_NOISE_SCOPE:-per_batch}"
    --tc_intermediate_activation "${TC_INTERMEDIATE_ACTIVATION:-relu6}"
    --one_shot_conv "${ONE_SHOT_CONV:-false}"
    --reuse_accepted_step_training "${REUSE_ACCEPTED_STEP_TRAINING:-false}"
    --checkpoint_ode_rhs_training "${CHECKPOINT_ODE_RHS_TRAINING:-false}"
    --checkpoint_ode_rhs_portion "${CHECKPOINT_ODE_RHS_PORTION:-1.0}"
    --w_bits 5 --enob none --weight_quant_factor_bits -1
    --tc_method "${TC_METHOD:-dopri5}" --tc_tol "${TC_TOL:-1e-6}"
    --tc_curve_sampling "${TC_CURVE_SAMPLING:-histogram}"
    --tc_covariance_table "${TC_COVARIANCE_TABLE:-./hardware_data/mc_45_corners/CU_4500_r_vs_vin.csv}"
    --resistance_floor_ohms "${RESISTANCE_FLOOR_OHMS:-none}"
    --tc_solver_dense_output "${TC_SOLVER_DENSE_OUTPUT:-true}"
    --tc_noise_replay_cache "${TC_NOISE_REPLAY_CACHE:-true}"
    --tc_noise_reference_R "${TC_NOISE_REFERENCE_R:-50e3}"
    --measured_pooling_curve_path "${MEASURED_POOLING_CURVE_PATH:-${NONLINEAR_R_TABLE}}"
    --measured_pooling_nominal_R "${MEASURED_POOLING_NOMINAL_R:-10e3}")
  if [[ -n "${TC_STEP_SIZE:-}" ]]; then
    TC_FEEDFORWARD_ARGS+=(--tc_step_size "$TC_STEP_SIZE")
  fi
  if [[ -n "${TC_NUM_STEPS:-}" ]]; then
    TC_FEEDFORWARD_ARGS+=(--tc_num_steps "$TC_NUM_STEPS")
  fi
  if [[ "$1" != pretrain ]]; then
    TC_FEEDFORWARD_ARGS+=(--enable_spin_variation "${ENABLE_SPIN_VARIATION:-true}"
      --sigma_spin "${SIGMA_SPIN:-0.1}"
      --enable_summing_current_noise "${ENABLE_SUMMING_CURRENT_NOISE:-true}"
      --summing_current_p "${SUMMING_CURRENT_P:-0.6e-12}"
      --enable_coupler_noise "${ENABLE_COUPLER_NOISE:-true}"
      --coupler_noise_p "${COUPLER_NOISE_P:-0.6e-12}"
      --enable_measured_activation "${ENABLE_MEASURED_ACTIVATION:-true}"
      --fuse_measured_activation "${FUSE_MEASURED_ACTIVATION:-true}"
      --enable_measured_pooling "${ENABLE_MEASURED_POOLING:-true}")
  fi
fi

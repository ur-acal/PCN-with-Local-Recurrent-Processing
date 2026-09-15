# Source with argument ft or eval. Empty arrays preserve non-TC callers.
TC_ARGS=()
if [[ "${TC_NONIDEALITIES:-false}" == true ]]; then
  source "$(dirname "${BASH_SOURCE[0]}")/tc_hardware_defaults.sh" "$1"
  export MC_RELU_MONTE_CARLO_SOURCE="${MC_RELU_MONTE_CARLO_SOURCE:-0906_RELU_Voltage}"
  # Fixed TT MC18 for FT; the full measured bank for per-spin evaluation.
  if [[ "$1" == ft ]]; then
    tc_activation_path=./hardware_data/mc_45_corners/0906_RELU_Voltage/tt_25_1.csv
    tc_activation_corner=MC18
  else
    tc_activation_path=./hardware_data/mc_45_corners/0906_RELU_Voltage
    tc_activation_corner=TT_25_1_MC18
  fi
  if [[ "${TOGGLE_MODE:-none}" != none || "${SWITCH_INF:-false}" == true ]]; then
    echo 'TC_NONIDEALITIES cannot be combined with toggle or switched dynamics.' >&2
    return 2
  fi
  case "${TC_STATE:-1}" in
    1) tc_block=ODEXInitFFFB; tc_wrapper=QATWrapper1State; tc_tester=QATTester1State ;;
    2) tc_block=S2NoisyIYAsXZAs0; tc_wrapper=QATWrapper2State; tc_tester=QATTester2State ;;
    *) echo 'TC_STATE must be 1 or 2' >&2; return 2 ;;
  esac
  if [[ "$1" == eval && "${CKPT:-best}" == *full_param* ]]; then
    tc_tester="ODEWrapper${TC_STATE:-1}State"
  fi
  tc_default_conv_method=loop
  if [[ "$1" == ft ]]; then tc_default_conv_method=shared; fi
  export ODE_BLOCK="$tc_block"
  export NONLINEAR_R_CURVE_SEED="${NONLINEAR_R_CURVE_SEED:-4096}"
  TC_ARGS=(--tc_nonidealities true --ode_block "$tc_block"
    --tc_conv_method "${TC_CONV_METHOD:-${tc_default_conv_method}}"
    --tc_curve_sampling "${TC_CURVE_SAMPLING:-histogram}"
    --w_bits 5 --weight_quant_factor_bits none --enob none
    --R "${R_VAL:-10e3}" --R_max "${R_MAX:-150e3}" --C "${C_VAL:-49e-15}" --v_dd "${V_DD:-0.1}"
    --one_over_q "${TOGGLE_ONE_OVER_Q:-1}" --sde_noise_type add
    --enable_spin_variation "${ENABLE_SPIN_VARIATION}" --sigma_spin "${SIGMA_SPIN:-0.1}"
    --spin_variation_mean "${SPIN_VARIATION_MEAN:-1.0}"
    --spin_variation_seed "${SPIN_VARIATION_SEED:-4096}"
    --enable_summing_current_noise "${ENABLE_SUMMING_CURRENT_NOISE}" --summing_current_p "${SUMMING_CURRENT_P}"
    --summing_noise_seed "${SUMMING_NOISE_SEED:-4096}"
    --enable_coupler_noise "${ENABLE_COUPLER_NOISE}" --coupler_noise_p "${COUPLER_NOISE_P}"
    --coupler_noise_seed "${COUPLER_NOISE_SEED:-4096}"
    --enable_slow_summing_current false --enable_slow_coupler_noise false
    --nonlinear_R "${ENABLE_NONLINEAR_R}"
    --nonlinear_R_table "${TC_MEAN_TABLE:-./hardware_data/res_vs_vin_10k_150k.csv}"
    --tc_covariance_table "${TC_COVARIANCE_TABLE:-./hardware_data/mc_45_corners/CU_4500_r_vs_vin.csv}"
    --nonlinear_R_curve_seed "${NONLINEAR_R_CURVE_SEED:-4096}"
    --tc_fb_asd_path "${TC_FB_ASD_PATH:-./hardware_data/coupler_asd_vs_freq.csv}"
    --tc_noise_reference_R "${TC_NOISE_REFERENCE_R:-50e3}" --tc_asd_reference_p "${TC_ASD_REFERENCE_P:-0.6e-12}"
    --enable_measured_activation "${ENABLE_MEASURED_ACTIVATION:-true}"
    --activation_curve_path "${ACTIVATION_CURVE_PATH:-${tc_activation_path}}"
    --activation_corner "${ACTIVATION_CORNER:-${tc_activation_corner}}"
    --enable_measured_pooling "${ENABLE_MEASURED_POOLING:-true}"
    --measured_pooling_curve_path "${MEASURED_POOLING_CURVE_PATH:-${TC_MEAN_TABLE:-./hardware_data/res_vs_vin_10k_150k.csv}}"
    --measured_pooling_nominal_R "${MEASURED_POOLING_NOMINAL_R:-10e3}")
  if [[ "$1" == ft ]]; then
    TC_ARGS+=(--ode_wrapper "$tc_wrapper" --noise_level 0 --method dopri5
      --nonlinear_R_train_mode none --nonlinear_R_curve_sharing shared)
  elif [[ "$1" == eval ]]; then
    TC_ARGS+=(--tc_max_eval_batches "${TC_MAX_EVAL_BATCHES:-0}")
    TC_ARGS+=(--ode_wrapper "$tc_tester" --method "${TC_EVAL_METHOD:-dopri5}"
      --d_start 0 --d_end 1 --n_sweep_left 0 --n_sweep_right 1 --sweep_eps false
      --thermal_noise false --sde_noise_type add --diff_mismatch false --noise_level_list 0
      --test_only false --test_expanded true --nonlinear_R_train_mode none
      --nonlinear_R_curve_sharing per_coupler --noisy_trials "${N_TRIALS:-10}"
      --activation_curve_sharing per_spin --activation_curve_seed "${ACTIVATION_CURVE_SEED:-4096}"
      --hardware_seed "${HARDWARE_SEED:-4096}" --data_seed "${DATA_SEED:-4096}")
    [[ -z "${TC_METADATA_PATH:-}" ]] || TC_ARGS+=(--tc_metadata_path "$TC_METADATA_PATH")
  else
    echo 'TC stage must be ft or eval' >&2; return 2
  fi
fi

# Source only from the opt-in TC argument builders, before legacy defaults.
# Shared hardware specification for PCN and CNN; explicit ablation overrides
# remain possible. Stage-specific CLI spellings belong to the callers.
export R_VAL="${R_VAL:-10e3}" R_MAX="${R_MAX:-150e3}"
export C_VAL="${C_VAL:-49e-15}" V_DD="${V_DD:-0.1}"
export TOGGLE_ONE_OVER_Q="${TOGGLE_ONE_OVER_Q:-${ONE_OVER_Q:-1}}"
export ONE_OVER_Q="$TOGGLE_ONE_OVER_Q"
export TC_MEAN_TABLE="${TC_MEAN_TABLE:-${NONLINEAR_R_TABLE:-./hardware_data/res_vs_vin_10k_150k.csv}}"
export TC_COVARIANCE_TABLE="${TC_COVARIANCE_TABLE:-./hardware_data/mc_45_corners/CU_4500_r_vs_vin.csv}"
export TC_CURVE_SAMPLING="${TC_CURVE_SAMPLING:-histogram}"
export TC_NOISE_REFERENCE_R="${TC_NOISE_REFERENCE_R:-50e3}"
export TC_ASD_REFERENCE_P="${TC_ASD_REFERENCE_P:-0.6e-12}"
export TC_FB_ASD_PATH="${TC_FB_ASD_PATH:-./hardware_data/coupler_asd_vs_freq.csv}"
# A coupler-energy measurement defaults to the smallest physical model needed
# for that measurement: input-dependent coupler conductance only. Explicit
# environment settings still take precedence, so an all-on energy study remains
# available without changing this shared launcher.
if [[ "${MEASURE_COUPLER_ENERGY:-false}" == true ]]; then
  export ENABLE_NONLINEAR_R="${ENABLE_NONLINEAR_R:-true}"
  export ENABLE_MEASURED_ACTIVATION="${ENABLE_MEASURED_ACTIVATION:-false}"
  export ENABLE_MEASURED_POOLING="${ENABLE_MEASURED_POOLING:-false}"
  export ENABLE_SPIN_VARIATION="${ENABLE_SPIN_VARIATION:-false}"
  export ENABLE_SUMMING_CURRENT_NOISE="${ENABLE_SUMMING_CURRENT_NOISE:-false}"
  export ENABLE_COUPLER_NOISE="${ENABLE_COUPLER_NOISE:-false}"
else
  export ENABLE_NONLINEAR_R="${ENABLE_NONLINEAR_R:-true}"
  export ENABLE_MEASURED_ACTIVATION="${ENABLE_MEASURED_ACTIVATION:-true}"
  export ENABLE_MEASURED_POOLING="${ENABLE_MEASURED_POOLING:-true}"
  export ENABLE_SPIN_VARIATION="${ENABLE_SPIN_VARIATION:-true}"
  export ENABLE_SUMMING_CURRENT_NOISE="${ENABLE_SUMMING_CURRENT_NOISE:-true}"
  export ENABLE_COUPLER_NOISE="${ENABLE_COUPLER_NOISE:-true}"
fi
export MEASURED_POOLING_CURVE_PATH="${MEASURED_POOLING_CURVE_PATH:-$TC_MEAN_TABLE}"
export MEASURED_POOLING_NOMINAL_R="${MEASURED_POOLING_NOMINAL_R:-10e3}"
export SIGMA_SPIN="${SIGMA_SPIN:-0.1}" SPIN_VARIATION_MEAN="${SPIN_VARIATION_MEAN:-1.0}"
export SUMMING_CURRENT_P="${SUMMING_CURRENT_P:-0.6e-12}" COUPLER_NOISE_P="${COUPLER_NOISE_P:-0.6e-12}"
export NONLINEAR_R_CURVE_SEED="${NONLINEAR_R_CURVE_SEED:-4096}"
export SPIN_VARIATION_SEED="${SPIN_VARIATION_SEED:-4096}"
export SUMMING_NOISE_SEED="${SUMMING_NOISE_SEED:-4096}" COUPLER_NOISE_SEED="${COUPLER_NOISE_SEED:-4096}"
export ACTIVATION_CURVE_SEED="${ACTIVATION_CURVE_SEED:-4096}"
export HARDWARE_SEED="${HARDWARE_SEED:-4096}" DATA_SEED="${DATA_SEED:-4096}"
export W_BITS=5 ENOB=none MISMATCH_LEVEL=0 NOISE_LEVEL=0 DIFF_MISMATCH=false
export ENABLE_DTC_NONIDEALITY=false ENABLE_SLOW_SUMMING_CURRENT=false ENABLE_SLOW_COUPLER_NOISE=false
if [[ "$1" == eval ]]; then
  export N_TRIALS="${N_TRIALS:-10}"
fi

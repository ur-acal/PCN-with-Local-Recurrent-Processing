#!/bin/bash -l
#SBATCH -N 1
#SBATCH -p ising
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH -t 90:10:00
#SBATCH -o logs/slurm_jobs/feedforward_mc45_%j.out

# PCN-equivalent 45-corner evaluation for a physical feedforward CNN.
set -e

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO_ROOT}"

MODEL_NAME="${MODEL_NAME:?MODEL_NAME is required}"
MODEL_CKPT="${MODEL_CKPT:?MODEL_CKPT is required}"

cmd=(
  python -u scripts/run_feedforward_mc45_ablation.py
  --model_name "${MODEL_NAME}"
  --checkpoint "${MODEL_CKPT}"
  --dataset "${TASK:-cifar100}"
  --img_type "${IMG_TYPE:-CiFAIR}"
  --output_dir "${OUTPUT_DIR:-results/feedforward_mc45}"
  --expanded_weight_dir "${EXPANDED_WEIGHT_DIR:-expanded_weights/feedforward}"
  --n_trials "${N_TRIALS:-5}"
  --jobs "${JOBS:-1}"
  --test_bs "${TEST_BATCH_SIZE:-128}"
  --base_seed "${BASE_SEED:-20260723}"
  --mc_45_corner_dir "${MC_45_CORNER_DIR:-hardware_data/mc_45_corners}"
  --mc_spin_variation_source "${MC_SPIN_VARIATION_SOURCE:-PVT_Monte_Carlo_Results_SPIN.csv}"
  --mc_dtc_pulse_width_variation_source "${MC_DTC_PULSE_WIDTH_VARIATION_SOURCE:-PVT_45corner_DTC_pulse_width.csv}"
  --mc_relu_monte_carlo_source "${MC_RELU_MONTE_CARLO_SOURCE:-relu_monteCarlo}"
  --mc_coupler_nonlinear_variation_source "${MC_COUPLER_NONLINEAR_VARIATION_SOURCE:-coupler_monte_v2}"
  --mc_coupler_nonlinear_variation_quantity "${MC_COUPLER_NONLINEAR_VARIATION_QUANTITY:-conductance}"
  --mc_coupler_nominal_R "${MC_COUPLER_NOMINAL_R:-50e3}"
  --full_45_corner_C "${FULL_45_CORNER_C:-500e-15}"
  --room_temperature_c "${ROOM_TEMPERATURE_C:-25.0}"
  --physical_level "${PHYSICAL_LEVEL:-3}"
  --v_dd "${V_DD:-0.1}"
  --one_over_q "${ONE_OVER_Q:-1}"
  --weight_quant_factor_bits "${WEIGHT_QUANT_FACTOR_BITS:-1}"
  --enob "${ENOB:-8}"
  --toggle_timing_mode "${TOGGLE_TIMING_MODE:-derived}"
  --toggle_y_time "${TOGGLE_Y_TIME:-5e-9}"
  --z_over_y_time "${Z_OVER_Y_TIME:-1}"
  --input_quant_bits "${INPUT_QUANT_BITS:-none}"
  --center_student_input "${CENTER_STUDENT_INPUT:-auto}"
  --enable_nonlinear_R "${ENABLE_NONLINEAR_R:-true}"
  --enable_diff_mismatch "${ENABLE_DIFF_MISMATCH:-false}"
  --nonlinear_R_curve_sharing "${NONLINEAR_R_CURVE_SHARING:-per_coupler}"
  --nonlinear_R_curve_sampling "${NONLINEAR_R_CURVE_SAMPLING:-empirical_with_replacement}"
  --nonlinear_R_curve_edge_chunk_size "${NONLINEAR_R_CURVE_EDGE_CHUNK_SIZE:-65536}"
  --enable_spin_variation "${ENABLE_SPIN_VARIATION:-true}"
  --enable_measured_activation "${ENABLE_MEASURED_ACTIVATION:-true}"
  --activation_curve_sharing "${ACTIVATION_CURVE_SHARING:-per_model}"
  --activation_interpolation "${ACTIVATION_INTERPOLATION:-piecewise_linear}"
  --activation_fit_constraint "${ACTIVATION_FIT_CONSTRAINT:-auto}"
  --activation_normalize_positive_endpoint "${ACTIVATION_NORMALIZE_POSITIVE_ENDPOINT:-false}"
  --compile_measured_activation "${COMPILE_MEASURED_ACTIVATION:-false}"
  --enable_measured_pooling "${ENABLE_MEASURED_POOLING:-true}"
  --enable_summing_current_noise "${ENABLE_SUMMING_CURRENT_NOISE:-true}"
  --summing_current_p "${SUMMING_CURRENT_P:-0.6e-12}"
  --enable_coupler_noise "${ENABLE_COUPLER_NOISE:-true}"
  --coupler_noise_p "${COUPLER_NOISE_P:-0.6e-12}"
  --enable_slow_summing_current "${ENABLE_SLOW_SUMMING_CURRENT:-false}"
  --slow_summing_current "${SLOW_SUMMING_CURRENT:-2.47e-9}"
  --enable_slow_coupler_noise "${ENABLE_SLOW_COUPLER_NOISE:-false}"
  --slow_coupler_noise "${SLOW_COUPLER_NOISE:-2.47e-9}"
  --enable_dtc_nonideality "${ENABLE_DTC_NONIDEALITY:-true}"
  --dtc_characterized_nominal_fraction "${DTC_CHARACTERIZED_NOMINAL_FRACTION:-0.0625}"
  --dtc_leading_edge_variation_std "${DTC_LEADING_EDGE_VARIATION_STD:-0.0}"
  --dtc_leading_edge_jitter_std "${DTC_LEADING_EDGE_JITTER_STD:-0.005}"
  --dtc_falling_edge_jitter_std "${DTC_FALLING_EDGE_JITTER_STD:-0.005}"
)

if [[ -n "${CORNER_IDS:-}" ]]; then
  read -r -a corner_array <<< "${CORNER_IDS}"
  cmd+=(--corner_ids "${corner_array[@]}")
fi
if [[ -n "${FIXED_RELU_MC_INDEX:-}" ]]; then
  cmd+=(--fixed_relu_mc_index "${FIXED_RELU_MC_INDEX}")
fi
if [[ -n "${FIXED_COUPLER_MC_INDEX:-}" ]]; then
  cmd+=(--fixed_coupler_mc_index "${FIXED_COUPLER_MC_INDEX}")
fi

printf 'Running:'
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"

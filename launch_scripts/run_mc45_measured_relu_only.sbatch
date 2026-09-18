#!/bin/bash -l
#SBATCH -N 1
#SBATCH -p ising
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH -t 90:10:00

# Explicit 45-corner reference run: quantized weights plus the finetuning
# measured-ReLU curve only.  All other hardware nonidealities are disabled.
set -euo pipefail

if [[ "${IS_SLURM:-0}" == 1 ]]; then
  source activate base
  conda activate scanbase
fi

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO_ROOT}"

python scripts/run_toggle_nonideality_ablation.py \
  --full_45_corner_test true \
  --mc_45_corner_dir "${MC_45_CORNER_DIR:-hardware_data/mc_45_corners}" \
  --mc_spin_variation_source "${MC_SPIN_VARIATION_SOURCE:-PVT_Monte_Carlo_Results_SPIN.csv}" \
  --mc_dtc_pulse_width_variation_source "${MC_DTC_PULSE_WIDTH_VARIATION_SOURCE:-PVT_45corner_DTC_pulse_width.csv}" \
  --mc_relu_monte_carlo_source "${MC_RELU_MONTE_CARLO_SOURCE:-relu_monteCarlo}" \
  --mc_coupler_nonlinear_variation_source "${MC_COUPLER_NONLINEAR_VARIATION_SOURCE:-coupler_monte_v2}" \
  --mc_coupler_nonlinear_variation_quantity "${MC_COUPLER_NONLINEAR_VARIATION_QUANTITY:-conductance}" \
  --mc_coupler_nominal_R "${MC_COUPLER_NOMINAL_R:-50e3}" \
  --full_45_corner_enable_spin_variation false \
  --full_45_corner_enable_measured_activation true \
  --full_45_corner_enable_measured_pooling false \
  --full_45_corner_enable_nonlinear_R false \
  --full_45_corner_enable_diff_mismatch false \
  --full_45_corner_enable_summing_current_noise false \
  --full_45_corner_enable_coupler_noise false \
  --full_45_corner_enable_slow_summing_current false \
  --full_45_corner_enable_slow_coupler_noise false \
  --full_45_corner_enable_dtc_nonideality false \
  --n_trials "${N_TRIALS:-10}" \
  --base_seed "${BASE_SEED:-20260723}" \
  --model_name "${MODEL_NAME}" \
  --model_dir "${MODEL_DIR}" \
  --ckpt best \
  --output_dir "${OUTPUT_DIR}" \
  --test_bs 128 \
  --jobs 1 \
  --toggle_level 3 \
  --toggle_n_cycles "${TOGGLE_N_CYCLES:-5}" \
  --toggle_time_split "${TOGGLE_TIME_SPLIT:-0.5}" \
  --toggle_fast_path "${TOGGLE_FAST_PATH:-true}" \
  --odexinit_scaling_mode direct \
  --toggle_timing_mode "${TOGGLE_TIMING_MODE:-derived}" \
  --toggle_y_time "${TOGGLE_Y_TIME:-5e-9}" \
  --z_over_y_time "${Z_OVER_Y_TIME:-3}" \
  --v_dd "${V_DD:-0.1}" \
  --one_over_q "${ONE_OVER_Q:-1}" \
  --weight_quant_factor_bits "${WEIGHT_QUANT_FACTOR_BITS:-1}" \
  --enob "${ENOB:-none}" \
  --R_max none \
  --full_45_corner_C "${FULL_45_CORNER_C:-500e-15}" \
  --room_temperature_c "${ROOM_TEMPERATURE_C:-25.0}" \
  --nonlinear_R_curve_sharing shared \
  --nonlinear_R_curve_sampling empirical_with_replacement \
  --activation_curve_sharing per_layer \
  --activation_interpolation piecewise_linear \
  --activation_fit_constraint auto \
  --activation_normalize_positive_endpoint false \
  --compile_measured_activation false \
  --activation_curve_override "${ACTIVATION_CURVE_PATH:-hardware_data/relu_current_0p2uA_finer.csv}" \
  --expanded_w_dir expanded_weights \
  --cases ideal \
  --measured_activation_baseline

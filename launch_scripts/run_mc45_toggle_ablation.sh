#!/bin/bash -l
#SBATCH -N 1
#SBATCH -p ising
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH -t 90:10:00
#SBATCH -o /scratch/rzeng7/repos/PCN-with-Local-Recurrent-Processing/logs/slurm_jobs/slurm_%j.out

IS_SLURM="${IS_SLURM:-0}"

if [[ "${IS_SLURM}" == 1 ]]; then
  source activate base
  conda activate scanbase
fi

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO_ROOT}"

MODEL_NAME="${MODEL_NAME:-TIMMQAT5b8aNT0p0mulTIMMPCNetNoBatchNorm_PCConvReLU6_0.0eps_ToggleODEXInitFFFB_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_C100_3K1S96C_0.25Dropout_16Layers4l5l4_2Pool_srrlDistill_a0p3_t2p0_scanGFI_1REP}"
MODEL_DIR="${MODEL_DIR:-saved_ckpt_runs/coupler_v2_cifar100_qf1}"
N_TRIALS="${N_TRIALS:-5}"
BASE_SEED="${BASE_SEED:-20260723}"
CORNER_IDS="${CORNER_IDS:-}"
FIXED_RELU_MC_INDEX="${FIXED_RELU_MC_INDEX:-}"
FIXED_COUPLER_MC_INDEX="${FIXED_COUPLER_MC_INDEX:-}"
OUTPUT_DIR="${OUTPUT_DIR:-results/coupler_monte_v2_cifar100_qf1}"
WEIGHT_QUANT_FACTOR_BITS="${WEIGHT_QUANT_FACTOR_BITS:-1}"
ENOB="${ENOB:-8}"
TOGGLE_TIMING_MODE="${TOGGLE_TIMING_MODE:-derived}"
TOGGLE_Y_TIME="${TOGGLE_Y_TIME:-5e-9}"
Z_OVER_Y_TIME="${Z_OVER_Y_TIME:-3}"
V_DD="${V_DD:-0.1}"
ONE_OVER_Q="${ONE_OVER_Q:-1}"

INPUT_PREPROCESS_ARGS=()
if [[ -n "${INPUT_QUANT_BITS+x}" ]]; then
  INPUT_PREPROCESS_ARGS+=(--input_quant_bits "${INPUT_QUANT_BITS}")
fi
if [[ -n "${CENTER_STUDENT_INPUT+x}" ]]; then
  INPUT_PREPROCESS_ARGS+=(--center_student_input "${CENTER_STUDENT_INPUT}")
fi

MC_45_CORNER_DIR="${MC_45_CORNER_DIR:-hardware_data/mc_45_corners}"
ROOM_TEMPERATURE_C="${ROOM_TEMPERATURE_C:-25.0}"
FULL_45_CORNER_C="${FULL_45_CORNER_C:-500e-15}"
# coupler non-linearity and variation
MC_COUPLER_NONLINEAR_VARIATION_SOURCE="${MC_COUPLER_NONLINEAR_VARIATION_SOURCE:-coupler_monte_v2}"
MC_COUPLER_NONLINEAR_VARIATION_QUANTITY="${MC_COUPLER_NONLINEAR_VARIATION_QUANTITY:-conductance}"
MC_COUPLER_NOMINAL_R="${MC_COUPLER_NOMINAL_R:-50e3}"
ENABLE_NONLINEAR_R="${ENABLE_NONLINEAR_R:-true}"
ENABLE_DIFF_MISMATCH="${ENABLE_DIFF_MISMATCH:-false}"
NONLINEAR_R_CURVE_SHARING="${NONLINEAR_R_CURVE_SHARING:-per_coupler}"
NONLINEAR_R_CURVE_SAMPLING="${NONLINEAR_R_CURVE_SAMPLING:-empirical_with_replacement}"
NONLINEAR_R_CURVE_EDGE_CHUNK_SIZE="${NONLINEAR_R_CURVE_EDGE_CHUNK_SIZE:-65536}"
# Spin variation
MC_SPIN_VARIATION_SOURCE="${MC_SPIN_VARIATION_SOURCE:-PVT_Monte_Carlo_Results_SPIN.csv}"
ENABLE_SPIN_VARIATION="${ENABLE_SPIN_VARIATION:-true}"
# measured act function
MC_RELU_MONTE_CARLO_SOURCE="${MC_RELU_MONTE_CARLO_SOURCE:-relu_monteCarlo}"
ENABLE_MEASURED_ACTIVATION="${ENABLE_MEASURED_ACTIVATION:-true}"
ACTIVATION_CURVE_SHARING="${ACTIVATION_CURVE_SHARING:-per_model}"
ACTIVATION_INTERPOLATION="${ACTIVATION_INTERPOLATION:-piecewise_linear}"
ACTIVATION_FIT_CONSTRAINT="${ACTIVATION_FIT_CONSTRAINT:-auto}"
ACTIVATION_NORMALIZE_POSITIVE_ENDPOINT="${ACTIVATION_NORMALIZE_POSITIVE_ENDPOINT:-false}"
COMPILE_MEASURED_ACTIVATION="${COMPILE_MEASURED_ACTIVATION:-false}"
# measured pooling
ENABLE_MEASURED_POOLING="${ENABLE_MEASURED_POOLING:-true}"
# noise
ENABLE_SUMMING_CURRENT_NOISE="${ENABLE_SUMMING_CURRENT_NOISE:-true}"
SUMMING_CURRENT_P="${SUMMING_CURRENT_P:-0.6e-12}"
ENABLE_COUPLER_NOISE="${ENABLE_COUPLER_NOISE:-true}"
COUPLER_NOISE_P="${COUPLER_NOISE_P:-0.6e-12}"
ENABLE_SLOW_SUMMING_CURRENT="${ENABLE_SLOW_SUMMING_CURRENT:-false}"
SLOW_SUMMING_CURRENT="${SLOW_SUMMING_CURRENT:-2.47e-9}"
ENABLE_SLOW_COUPLER_NOISE="${ENABLE_SLOW_COUPLER_NOISE:-false}"
SLOW_COUPLER_NOISE="${SLOW_COUPLER_NOISE:-2.47e-9}"
# DTC pulse
ENABLE_DTC_NONIDEALITY="${ENABLE_DTC_NONIDEALITY:-true}"
MC_DTC_PULSE_WIDTH_VARIATION_SOURCE="${MC_DTC_PULSE_WIDTH_VARIATION_SOURCE:-PVT_45corner_DTC_pulse_width.csv}"
DTC_CHARACTERIZED_NOMINAL_FRACTION="${DTC_CHARACTERIZED_NOMINAL_FRACTION:-0.0625}"
DTC_LEADING_EDGE_VARIATION_STD="${DTC_LEADING_EDGE_VARIATION_STD:-0.0}"
DTC_LEADING_EDGE_JITTER_STD="${DTC_LEADING_EDGE_JITTER_STD:-0.005}"
DTC_FALLING_EDGE_JITTER_STD="${DTC_FALLING_EDGE_JITTER_STD:-0.005}"

CORNER_ARGS=()
FIXED_RELU_ARGS=()
FIXED_COUPLER_ARGS=()
if [[ -n "${CORNER_IDS}" ]]; then
  read -r -a CORNER_ID_ARRAY <<< "${CORNER_IDS}"
  CORNER_ARGS=(--corner_ids "${CORNER_ID_ARRAY[@]}")
fi
if [[ -n "${FIXED_RELU_MC_INDEX}" ]]; then
  FIXED_RELU_ARGS=(--fixed_relu_mc_index "${FIXED_RELU_MC_INDEX}")
fi
if [[ -n "${FIXED_COUPLER_MC_INDEX}" ]]; then
  FIXED_COUPLER_ARGS=(--fixed_coupler_mc_index "${FIXED_COUPLER_MC_INDEX}")
fi

RELU_CURVE_SELECTION="${FIXED_RELU_MC_INDEX:-random_without_replacement}"
if [[ "${ACTIVATION_CURVE_SHARING}" != "per_model" ]]; then
  RELU_CURVE_SELECTION="sample_with_replacement"
fi

mkdir -p "${OUTPUT_DIR}"
{
  echo "MC45 resolved configuration"
  echo "model_name=${MODEL_NAME}"
  echo "model_dir=${MODEL_DIR}"
  echo "weight_quant_factor_bits=${WEIGHT_QUANT_FACTOR_BITS}"
  echo "enob=${ENOB}"
  echo "toggle_timing_mode=${TOGGLE_TIMING_MODE}"
  echo "toggle_y_time=${TOGGLE_Y_TIME}"
  echo "z_over_y_time=${Z_OVER_Y_TIME}"
  echo "n_trials=${N_TRIALS}"
  echo "base_seed=${BASE_SEED}"
  echo "corner_ids=${CORNER_IDS:-all}"
  echo "fixed_relu_mc_index=${FIXED_RELU_MC_INDEX:-none}"
  echo "fixed_coupler_mc_index=${FIXED_COUPLER_MC_INDEX:-none}"
  echo "relu_curve_selection=${RELU_CURVE_SELECTION}"
  echo "output_dir=${OUTPUT_DIR}"
  echo "mc_45_corner_dir=${MC_45_CORNER_DIR}"
  echo "room_temperature_c=${ROOM_TEMPERATURE_C}"
  echo "C=${FULL_45_CORNER_C}"
  echo "coupler_source=${MC_COUPLER_NONLINEAR_VARIATION_SOURCE}"
  echo "coupler_quantity=${MC_COUPLER_NONLINEAR_VARIATION_QUANTITY}"
  echo "coupler_nominal_R=${MC_COUPLER_NOMINAL_R}"
  echo "enable_nonlinear_R=${ENABLE_NONLINEAR_R}"
  echo "enable_diff_mismatch=${ENABLE_DIFF_MISMATCH}"
  echo "nonlinear_R_curve_sharing=${NONLINEAR_R_CURVE_SHARING}"
  echo "nonlinear_R_curve_sampling=${NONLINEAR_R_CURVE_SAMPLING}"
  echo "nonlinear_R_curve_edge_chunk_size=${NONLINEAR_R_CURVE_EDGE_CHUNK_SIZE}"
  echo "spin_source=${MC_SPIN_VARIATION_SOURCE}"
  echo "enable_spin_variation=${ENABLE_SPIN_VARIATION}"
  echo "relu_source=${MC_RELU_MONTE_CARLO_SOURCE}"
  echo "enable_measured_activation=${ENABLE_MEASURED_ACTIVATION}"
  echo "activation_curve_sharing=${ACTIVATION_CURVE_SHARING}"
  echo "activation_interpolation=${ACTIVATION_INTERPOLATION}"
  echo "activation_fit_constraint=${ACTIVATION_FIT_CONSTRAINT}"
  echo "activation_normalize_positive_endpoint=${ACTIVATION_NORMALIZE_POSITIVE_ENDPOINT}"
  echo "compile_measured_activation=${COMPILE_MEASURED_ACTIVATION}"
  echo "enable_measured_pooling=${ENABLE_MEASURED_POOLING}"
  echo "enable_summing_current_noise=${ENABLE_SUMMING_CURRENT_NOISE}"
  echo "summing_current_p=${SUMMING_CURRENT_P}"
  echo "enable_coupler_noise=${ENABLE_COUPLER_NOISE}"
  echo "coupler_noise_p=${COUPLER_NOISE_P}"
  echo "enable_slow_summing_current=${ENABLE_SLOW_SUMMING_CURRENT}"
  echo "slow_summing_current=${SLOW_SUMMING_CURRENT}"
  echo "enable_slow_coupler_noise=${ENABLE_SLOW_COUPLER_NOISE}"
  echo "slow_coupler_noise=${SLOW_COUPLER_NOISE}"
  echo "dtc_source=${MC_DTC_PULSE_WIDTH_VARIATION_SOURCE}"
  echo "enable_dtc_nonideality=${ENABLE_DTC_NONIDEALITY}"
  echo "dtc_characterized_nominal_fraction=${DTC_CHARACTERIZED_NOMINAL_FRACTION}"
  echo "dtc_leading_edge_variation_std=${DTC_LEADING_EDGE_VARIATION_STD}"
  echo "dtc_leading_edge_jitter_std=${DTC_LEADING_EDGE_JITTER_STD}"
  echo "dtc_falling_edge_jitter_std=${DTC_FALLING_EDGE_JITTER_STD}"
} | tee "${OUTPUT_DIR}/run_config.txt"

python scripts/run_toggle_nonideality_ablation.py \
  --full_45_corner_test true \
  --mc_45_corner_dir "${MC_45_CORNER_DIR}" \
  --mc_spin_variation_source "${MC_SPIN_VARIATION_SOURCE}" \
  --mc_dtc_pulse_width_variation_source "${MC_DTC_PULSE_WIDTH_VARIATION_SOURCE}" \
  --mc_relu_monte_carlo_source "${MC_RELU_MONTE_CARLO_SOURCE}" \
  --mc_coupler_nonlinear_variation_source "${MC_COUPLER_NONLINEAR_VARIATION_SOURCE}" \
  --mc_coupler_nonlinear_variation_quantity "${MC_COUPLER_NONLINEAR_VARIATION_QUANTITY}" \
  --mc_coupler_nominal_R "${MC_COUPLER_NOMINAL_R}" \
  --full_45_corner_enable_spin_variation "${ENABLE_SPIN_VARIATION}" \
  --full_45_corner_enable_measured_activation "${ENABLE_MEASURED_ACTIVATION}" \
  --full_45_corner_enable_measured_pooling "${ENABLE_MEASURED_POOLING}" \
  --full_45_corner_enable_nonlinear_R "${ENABLE_NONLINEAR_R}" \
  --full_45_corner_enable_diff_mismatch "${ENABLE_DIFF_MISMATCH}" \
  --full_45_corner_enable_summing_current_noise "${ENABLE_SUMMING_CURRENT_NOISE}" \
  --full_45_corner_enable_coupler_noise "${ENABLE_COUPLER_NOISE}" \
  --full_45_corner_enable_slow_summing_current "${ENABLE_SLOW_SUMMING_CURRENT}" \
  --full_45_corner_slow_summing_current "${SLOW_SUMMING_CURRENT}" \
  --full_45_corner_enable_slow_coupler_noise "${ENABLE_SLOW_COUPLER_NOISE}" \
  --full_45_corner_slow_coupler_noise "${SLOW_COUPLER_NOISE}" \
  --full_45_corner_enable_dtc_nonideality "${ENABLE_DTC_NONIDEALITY}" \
  --n_trials "${N_TRIALS}" \
  --base_seed "${BASE_SEED}" \
  --model_name "${MODEL_NAME}" \
  --model_dir "${MODEL_DIR}" \
  --ckpt best \
  --output_dir "${OUTPUT_DIR}" \
  --test_bs 128 \
  --jobs 1 \
  --toggle_level 3 \
  --odexinit_scaling_mode direct \
  --toggle_timing_mode "${TOGGLE_TIMING_MODE}" \
  --toggle_y_time "${TOGGLE_Y_TIME}" \
  --z_over_y_time "${Z_OVER_Y_TIME}" \
  --v_dd "${V_DD}" \
  --one_over_q "${ONE_OVER_Q}" \
  --weight_quant_factor_bits "${WEIGHT_QUANT_FACTOR_BITS}" \
  --enob "${ENOB}" \
  --R_max none \
  --full_45_corner_C "${FULL_45_CORNER_C}" \
  --room_temperature_c "${ROOM_TEMPERATURE_C}" \
  --full_45_corner_summing_current_p "${SUMMING_CURRENT_P}" \
  --coupler_noise_p "${COUPLER_NOISE_P}" \
  --dtc_characterized_nominal_fraction "${DTC_CHARACTERIZED_NOMINAL_FRACTION}" \
  --dtc_leading_edge_variation_std "${DTC_LEADING_EDGE_VARIATION_STD}" \
  --dtc_leading_edge_jitter_std "${DTC_LEADING_EDGE_JITTER_STD}" \
  --dtc_falling_edge_jitter_std "${DTC_FALLING_EDGE_JITTER_STD}" \
  --nonlinear_R_curve_sharing "${NONLINEAR_R_CURVE_SHARING}" \
  --nonlinear_R_curve_sampling "${NONLINEAR_R_CURVE_SAMPLING}" \
  --nonlinear_R_curve_edge_chunk_size "${NONLINEAR_R_CURVE_EDGE_CHUNK_SIZE}" \
  --activation_interpolation "${ACTIVATION_INTERPOLATION}" \
  --activation_curve_sharing "${ACTIVATION_CURVE_SHARING}" \
  --activation_fit_constraint "${ACTIVATION_FIT_CONSTRAINT}" \
  --activation_normalize_positive_endpoint "${ACTIVATION_NORMALIZE_POSITIVE_ENDPOINT}" \
  --compile_measured_activation "${COMPILE_MEASURED_ACTIVATION}" \
  --expanded_w_dir expanded_weights \
  "${INPUT_PREPROCESS_ARGS[@]}" \
  "${CORNER_ARGS[@]}" \
  "${FIXED_RELU_ARGS[@]}" \
  "${FIXED_COUPLER_ARGS[@]}"

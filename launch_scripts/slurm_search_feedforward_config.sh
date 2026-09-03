#!/bin/bash

# Slurm entry point for the feedforward pretrain -> physical-FT pipeline.
# It mirrors the PCN search launcher's applicable configuration surface; the
# recurrent/ODE-only controls intentionally do not exist on this path.
set -e

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO_ROOT}"
mkdir -p logs/slurm_jobs

export MODEL_NAME="${MODEL_NAME:?MODEL_NAME is required}"
export TASK="${TASK:-cifar100}"
export IMG_TYPE="${IMG_TYPE:-CiFAIR}"
case "${IMG_TYPE,,}" in
  cifair) export IMG_TYPE="CiFAIR" ;;
  scangfi|raw|_raw) export IMG_TYPE="scanGFI" ;;
esac
export EXP_PREFIX="${EXP_PREFIX:-feedforward_physical}"
export DATA_DIR="${DATA_DIR:-../data}"
export TIMM_AUG_LEVEL="${TIMM_AUG_LEVEL:-none}"

# Teacher and distillation.
if [[ "${IMG_TYPE}" == "CiFAIR" ]]; then
  _DEFAULT_TEACHER_CKPT="checkpoint/efficientnet_v2_l_${TASK}_CiFAIR_timm.pth"
  _DEFAULT_TEACHER_ARCH="efficientnet_v2_l"
elif [[ "${TASK}" == "cifar10" ]]; then
  _DEFAULT_TEACHER_CKPT="checkpoint/b4.pth"
  _DEFAULT_TEACHER_ARCH="efficientnet-b4"
else
  _DEFAULT_TEACHER_CKPT="checkpoint/b4_100.pth"
  _DEFAULT_TEACHER_ARCH="efficientnet_v2_l"
fi
export TEACHER_CKPT="${TEACHER_CKPT:-${_DEFAULT_TEACHER_CKPT}}"
export TEACHER_ARCH="${TEACHER_ARCH:-${_DEFAULT_TEACHER_ARCH}}"
export DISTILL_METHOD="${DISTILL_METHOD:-srrl}"
export TEACHER_ARCH_SOURCE="${TEACHER_ARCH_SOURCE:-auto}"
export TEACHER_INPUT_SIZE="${TEACHER_INPUT_SIZE:-224}"
export TEACHER_CENTER_CROP="${TEACHER_CENTER_CROP:-true}"
export ADAPT_PIL_TEACHER="${ADAPT_PIL_TEACHER:-false}"
export ORIG_T_INP="${ORIG_T_INP:-false}"
export DISTILL_ALPHA="${DISTILL_ALPHA:-0.3}"
export DISTILL_TEMPERATURE="${DISTILL_TEMPERATURE:-2.0}"
export SRRL_WEIGHT="${SRRL_WEIGHT:-1.0}"
export MGD_ALPHA="${MGD_ALPHA:-7e-5}"
export MGD_LAMBDA="${MGD_LAMBDA:-0.5}"
export MGD_MASK_MODE="${MGD_MASK_MODE:-channel}"
export REVIEWKD_WEIGHT="${REVIEWKD_WEIGHT:-1.0}"
export REVIEWKD_WARMUP_EPOCHS="${REVIEWKD_WARMUP_EPOCHS:-20}"
export REVIEWKD_NUM_STAGES="${REVIEWKD_NUM_STAGES:-4}"

# Feedforward physical mapping. Both derived stage durations are one; fixed
# timing defaults to equal first/second-convolution durations.
export R_VAL="${R_VAL:-67e3}"
export C_VAL="${C_VAL:-282e-15}"
export V_DD="${V_DD:-0.1}"
export TOGGLE_ONE_OVER_Q="${TOGGLE_ONE_OVER_Q:-1}"
export TOGGLE_TIMING_MODE="${TOGGLE_TIMING_MODE:-derived}"
export TOGGLE_Y_TIME="${TOGGLE_Y_TIME:-5e-9}"
export Z_OVER_Y_TIME="${Z_OVER_Y_TIME:-1}"
export SCALE_TRAIN_RECIPE="${SCALE_TRAIN_RECIPE:-false}"
export INPUT_QUANT_BITS="${INPUT_QUANT_BITS:-none}"
export CENTER_STUDENT_INPUT="${CENTER_STUDENT_INPUT:-false}"

# Measured activation and pooling.
export ACTIVATION_CORNER_MODE="${ACTIVATION_CORNER_MODE:-fixed}"
export ENABLE_PRETRAIN_MEASURED_ACTIVATION="${ENABLE_PRETRAIN_MEASURED_ACTIVATION:-false}"
export ENABLE_MEASURED_ACTIVATION="${ENABLE_MEASURED_ACTIVATION:-true}"
export UNITLESS_MEASURED_PULLBACK_MODE="${UNITLESS_MEASURED_PULLBACK_MODE:-none}"
export ACTIVATION_CORNER="${ACTIVATION_CORNER:-TT}"
if [[ "${ACTIVATION_CORNER_MODE}" == "random_per_forward" ]]; then
  export ACTIVATION_CURVE_PATH="${ACTIVATION_CURVE_PATH:-./hardware_data/relu_current_0p2uA_all.csv}"
else
  export ACTIVATION_CURVE_PATH="${ACTIVATION_CURVE_PATH:-./hardware_data/relu_current_0p2uA_finer.csv}"
fi
export ACTIVATION_CURVE_SHARING="${ACTIVATION_CURVE_SHARING:-per_model}"
export ACTIVATION_INTERPOLATION="${ACTIVATION_INTERPOLATION:-piecewise_linear}"
export ACTIVATION_SPLINE_PARAMETERS="${ACTIVATION_SPLINE_PARAMETERS:-10}"
export ACTIVATION_FIT_CONSTRAINT="${ACTIVATION_FIT_CONSTRAINT:-auto}"
export SCALE_MEASURED_ACTIVATION="${SCALE_MEASURED_ACTIVATION:-false}"
export COMPILE_MEASURED_ACTIVATION="${COMPILE_MEASURED_ACTIVATION:-false}"
export ENABLE_MEASURED_POOLING="${ENABLE_MEASURED_POOLING:-false}"

# Physical FT nonidealities and quantization.
export W_BITS="${W_BITS:-5}"
export WEIGHT_QUANT_FACTOR_BITS="${WEIGHT_QUANT_FACTOR_BITS:-none}"
export ENOB="${ENOB:-8}"
export PULSE_MISMATCH_TRAINING_MODE="${PULSE_MISMATCH_TRAINING_MODE:-post_quant_amplitude}"
export MISMATCH_LEVEL="${MISMATCH_LEVEL:-0.0}"
export MISMATCH_TYPE="${MISMATCH_TYPE:-mul}"
export ENABLE_SPIN_VARIATION="${ENABLE_SPIN_VARIATION:-true}"
export SIGMA_SPIN="${SIGMA_SPIN:-0.10}"
export SPIN_VARIATION_MEAN="${SPIN_VARIATION_MEAN:-1.0}"
export ENABLE_SUMMING_CURRENT_NOISE="${ENABLE_SUMMING_CURRENT_NOISE:-true}"
export SUMMING_CURRENT_P="${SUMMING_CURRENT_P:-0.6e-12}"
export ENABLE_COUPLER_NOISE="${ENABLE_COUPLER_NOISE:-true}"
export COUPLER_NOISE_P="${COUPLER_NOISE_P:-0.6e-12}"
export ENABLE_SLOW_SUMMING_CURRENT="${ENABLE_SLOW_SUMMING_CURRENT:-false}"
export SLOW_SUMMING_CURRENT="${SLOW_SUMMING_CURRENT:-2.47e-9}"
export ENABLE_SLOW_COUPLER_NOISE="${ENABLE_SLOW_COUPLER_NOISE:-false}"
export SLOW_COUPLER_NOISE="${SLOW_COUPLER_NOISE:-2.47e-9}"
export ENABLE_DTC_NONIDEALITY="${ENABLE_DTC_NONIDEALITY:-false}"
export DTC_LEADING_EDGE_VARIATION_STD="${DTC_LEADING_EDGE_VARIATION_STD:-0.0}"
export DTC_WIDTH_VARIATION_MEAN="${DTC_WIDTH_VARIATION_MEAN:-0.0}"
export DTC_WIDTH_VARIATION_STD="${DTC_WIDTH_VARIATION_STD:-0.018}"
export DTC_LEADING_EDGE_JITTER_STD="${DTC_LEADING_EDGE_JITTER_STD:-0.005}"
export DTC_FALLING_EDGE_JITTER_STD="${DTC_FALLING_EDGE_JITTER_STD:-0.005}"

# Nonlinear-R training and optimization.
export NONLINEAR_R_TABLE="${NONLINEAR_R_TABLE:-coupler_monte}"
export NONLINEAR_R_TRAIN_MODE="${NONLINEAR_R_TRAIN_MODE:-exact_curve}"
export NONLINEAR_R_MC_QUANTITY="${NONLINEAR_R_MC_QUANTITY:-conductance}"
export NONLINEAR_R_CURVE_SHARING="${NONLINEAR_R_CURVE_SHARING:-shared}"
export NONLINEAR_R_CORNER_RANGE="${NONLINEAR_R_CORNER_RANGE:-all}"
export NONLINEAR_R_CURVE_SEED="${NONLINEAR_R_CURVE_SEED:-none}"
export WARMUP_PRETRAIN="${WARMUP_PRETRAIN:-0}"
export WARMUP_FT="${WARMUP_FT:-0}"
export PRETRAIN_LEARNING_RATE="${PRETRAIN_LEARNING_RATE:-0.01}"
export PRETRAIN_NUM_EPOCHS="${PRETRAIN_NUM_EPOCHS:-300}"
export FT_LEARNING_RATE="${FT_LEARNING_RATE:-0.005}"
export FT_NUM_EPOCHS="${FT_NUM_EPOCHS:-140}"
export PRETRAIN_EVAL_EVERY="${PRETRAIN_EVAL_EVERY:-5}"
export FT_EVAL_EVERY="${FT_EVAL_EVERY:-2}"
export TIMM_RE_PROB="${TIMM_RE_PROB:-0.0}"

sbatch -N "${N_NODES:-1}" \
  --time="${SBATCH_TIMELIMIT:-48:00:00}" \
  --export=ALL \
  ./launch_scripts/feedforward_pretrain_then_ft.sbatch

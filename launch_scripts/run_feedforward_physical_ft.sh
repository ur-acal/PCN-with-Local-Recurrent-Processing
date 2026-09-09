#!/bin/bash

# Hardware-aware feedforward fine-tuning.  Level 2 uses averaged physical
# convolutions for training.  Level 3 is available for pulse-level diagnostic
# training, but is much slower.
set -e

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO_ROOT}"

MODEL_NAME="${MODEL_NAME:?MODEL_NAME is required}"
MODEL_CKPT="${MODEL_CKPT:?MODEL_CKPT is required}"
TASK="${TASK:-cifar100}"
IMG_TYPE="${IMG_TYPE:-CiFAIR}"
OUTPUT_DIR="${OUTPUT_DIR:-./saved_ckpt_runs/feedforward_physical_ft}"
INPUT_PREPROCESS_ARGS=()
if [[ -n "${INPUT_QUANT_BITS+x}" ]]; then
  INPUT_PREPROCESS_ARGS+=(--input_quant_bits "${INPUT_QUANT_BITS}")
fi
if [[ -n "${CENTER_STUDENT_INPUT+x}" ]]; then
  INPUT_PREPROCESS_ARGS+=(--center_student_input "${CENTER_STUDENT_INPUT}")
fi
DATA_DIR="${DATA_DIR:-../data}"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  NUM_WORKERS="${NUM_WORKERS:-2}"
else
  NUM_WORKERS="${NUM_WORKERS:-0}"
fi
PHYSICAL_LEVEL="${PHYSICAL_LEVEL:-2}"
R_VAL="${R_VAL:-67e3}"
C_VAL="${C_VAL:-282e-15}"
V_DD="${V_DD:-0.1}"
ONE_OVER_Q="${TOGGLE_ONE_OVER_Q:-${ONE_OVER_Q:-1}}"
W_BITS="${W_BITS:-5}"
WEIGHT_QUANT_FACTOR_BITS="${WEIGHT_QUANT_FACTOR_BITS:--1}"
if [[ "${WEIGHT_QUANT_FACTOR_BITS,,}" == "none" ]]; then
  WEIGHT_QUANT_FACTOR_BITS=-1
fi
TOGGLE_TIMING_MODE="${TOGGLE_TIMING_MODE:-derived}"
TOGGLE_Y_TIME="${TOGGLE_Y_TIME:-5e-9}"
_DEFAULT_OVERRIDE="lr=${FT_LEARNING_RATE:-0.005},num_epochs=${FT_NUM_EPOCHS:-140},warmup_epoch=${WARMUP_FT:-0},weight_decay=${WEIGHT_DECAY:-0.001},batch_size=${BATCH_SIZE:-128}"
case "${MODEL_NAME}" in
  wrn_28_2_cifar_nobn_avgpool|wrn_28_2_cifar_nobn_avgpool_shortcut)
    if [[ "${TASK}" == "cifar10" ]]; then
      _DEFAULT_OVERRIDE+=",dropout_rate=${DROPOUT_RATE:-0.1},max_norm=${MAX_NORM:-2.0},bias_lr_multiplier=${BIAS_LR_MULTIPLIER:-0.5},bias_weight_decay=${BIAS_WEIGHT_DECAY:-0.0}"
    fi
    ;;
esac
EXTRA_OVERRIDE="${EXTRA_OVERRIDE:-${_DEFAULT_OVERRIDE}}"
NONLINEAR_R_TABLE="${NONLINEAR_R_TABLE:-coupler_monte}"
if [[ "${NONLINEAR_R_TABLE}" != */* ]]; then
  NONLINEAR_R_TABLE="./hardware_data/mc_45_corners/${NONLINEAR_R_TABLE}"
fi
ACTIVATION_CORNER_MODE="${ACTIVATION_CORNER_MODE:-fixed}"
if [[ "${ACTIVATION_CORNER_MODE}" == "random_per_forward" ]]; then
  ACTIVATION_CURVE_PATH="${ACTIVATION_CURVE_PATH:-./hardware_data/relu_current_0p2uA_all.csv}"
else
  ACTIVATION_CURVE_PATH="${ACTIVATION_CURVE_PATH:-./hardware_data/relu_current_0p2uA_finer.csv}"
fi

cmd=(
  python baseline/train_baseline_cifar.py
  --model_name "${MODEL_NAME}"
  --dataset "${TASK}"
  --data_dir "${DATA_DIR}"
  --output_dir "${OUTPUT_DIR}"
  --case custom_noresize
  --seed "${TRAINING_SEED:-4096}"
  --eval_every "${FT_EVAL_EVERY:-2}"
  --num_workers "${NUM_WORKERS}"
  --img_type "${IMG_TYPE}"
  --resume_checkpoint "${MODEL_CKPT}"
  --physical_feedforward true
  --physical_level "${PHYSICAL_LEVEL}"
  --R "${R_VAL}"
  --C "${C_VAL}"
  --v_dd "${V_DD}"
  --one_over_q "${ONE_OVER_Q}"
  --w_bits "${W_BITS}"
  --weight_quant_factor_bits "${WEIGHT_QUANT_FACTOR_BITS}"
  --noise_level "${MISMATCH_LEVEL:-0.0}"
  --noise_type "${MISMATCH_TYPE:-mul}"
  --pulse_mismatch_training_mode "${PULSE_MISMATCH_TRAINING_MODE:-post_quant_amplitude}"
  --toggle_timing_mode "${TOGGLE_TIMING_MODE}"
  --toggle_y_time "${TOGGLE_Y_TIME}"
  --z_over_y_time "${Z_OVER_Y_TIME:-1}"
  --enob "${ENOB:-8}"
  --timm_aug_level "${TIMM_AUG_LEVEL:-no_aug}"
  --timm_re_prob "${TIMM_RE_PROB:-0.0}"
  --distill_method "${DISTILL_METHOD:-srrl}"
  --distill_alpha "${DISTILL_ALPHA:-0.3}"
  --distill_temperature "${DISTILL_TEMPERATURE:-2.0}"
  --srrl_weight "${SRRL_WEIGHT:-1.0}"
  --mgd_alpha "${MGD_ALPHA:-7e-5}"
  --mgd_lambda "${MGD_LAMBDA:-0.5}"
  --mgd_mask_mode "${MGD_MASK_MODE:-channel}"
  --reviewkd_weight "${REVIEWKD_WEIGHT:-1.0}"
  --reviewkd_warmup_epochs "${REVIEWKD_WARMUP_EPOCHS:-20}"
  --reviewkd_num_stages "${REVIEWKD_NUM_STAGES:-4}"
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
  --enable_dtc_nonideality "${ENABLE_DTC_NONIDEALITY:-false}"
  --dtc_leading_edge_variation_std "${DTC_LEADING_EDGE_VARIATION_STD:-0.0}"
  --dtc_width_variation_mean "${DTC_WIDTH_VARIATION_MEAN:-0.0}"
  --dtc_width_variation_std "${DTC_WIDTH_VARIATION_STD:-0.018}"
  --dtc_leading_edge_jitter_std "${DTC_LEADING_EDGE_JITTER_STD:-0.005}"
  --dtc_falling_edge_jitter_std "${DTC_FALLING_EDGE_JITTER_STD:-0.005}"
  --enable_measured_activation "${ENABLE_MEASURED_ACTIVATION:-true}"
  --activation_curve_path "${ACTIVATION_CURVE_PATH}"
  --activation_corner "${ACTIVATION_CORNER:-TT}"
  --activation_corner_mode "${ACTIVATION_CORNER_MODE}"
  --activation_curve_sharing "${ACTIVATION_CURVE_SHARING:-per_model}"
  --activation_interpolation "${ACTIVATION_INTERPOLATION:-piecewise_linear}"
  --activation_spline_parameters "${ACTIVATION_SPLINE_PARAMETERS:-10}"
  --activation_fit_constraint "${ACTIVATION_FIT_CONSTRAINT:-auto}"
  --activation_normalize_positive_endpoint "${SCALE_MEASURED_ACTIVATION:-false}"
  --compile_measured_activation "${COMPILE_MEASURED_ACTIVATION:-false}"
  --nonlinear_R_train_mode "${NONLINEAR_R_TRAIN_MODE:-exact_curve}"
  --nonlinear_R_table "${NONLINEAR_R_TABLE}"
  --nonlinear_R_mc_quantity "${MC_COUPLER_NONLINEAR_VARIATION_QUANTITY:-conductance}"
  --nonlinear_R_corner_range "${NONLINEAR_R_CORNER_RANGE:-all}"
  --nonlinear_R_curve_sharing "${NONLINEAR_R_CURVE_SHARING:-shared}"
  --enable_measured_pooling "${ENABLE_MEASURED_POOLING:-false}"
  --scale_train_recipe "${SCALE_TRAIN_RECIPE:-false}"
  --ff_train_scale "${FF_TRAIN_SCALE:-1.0}"
  --fb_train_scale "${FB_TRAIN_SCALE:-1.0}"
  --override "${EXTRA_OVERRIDE}"
  --print_only "${PRINT_ONLY:-false}"
)
cmd+=("${INPUT_PREPROCESS_ARGS[@]}")

if [[ -n "${WRN_DEPTH:-}" ]]; then
  cmd+=(--wrn_depth "${WRN_DEPTH}")
fi
if [[ -n "${WRN_FIRST_STAGE_CHANNELS:-}" ]]; then
  cmd+=(--wrn_first_stage_channels "${WRN_FIRST_STAGE_CHANNELS}")
fi

if [[ -n "${ACTIVATION_CURVE_SEED:-}" ]]; then
  cmd+=(--activation_curve_seed "${ACTIVATION_CURVE_SEED}")
fi
if [[ -n "${NONLINEAR_R_CURVE_SEED:-}" ]]; then
  cmd+=(--nonlinear_R_curve_seed "${NONLINEAR_R_CURVE_SEED}")
fi
for pair in \
  "SPIN_VARIATION_SEED:spin_variation_seed" \
  "SUMMING_NOISE_SEED:summing_noise_seed" \
  "COUPLER_NOISE_SEED:coupler_noise_seed" \
  "DTC_TIMING_SEED:dtc_timing_seed"; do
  env_name="${pair%%:*}"
  arg_name="${pair##*:}"
  if [[ -n "${!env_name:-}" ]]; then
    cmd+=("--${arg_name}" "${!env_name}")
  fi
done
if [[ -n "${TEACHER_CKPT:-}" ]]; then
  cmd+=(--teacher_ckpt "${TEACHER_CKPT}")
fi
if [[ -n "${TEACHER_ARCH:-}" ]]; then
  cmd+=(--teacher_arch "${TEACHER_ARCH}")
fi
cmd+=(
  --teacher_arch_source "${TEACHER_ARCH_SOURCE:-auto}"
  --teacher_input_size "${TEACHER_INPUT_SIZE:-224}"
  --teacher_center_crop "${TEACHER_CENTER_CROP:-true}"
  --adapt_PIL_teacher "${ADAPT_PIL_TEACHER:-false}"
  --orig_t_inp "${ORIG_T_INP:-false}"
)

printf 'Running:'
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"

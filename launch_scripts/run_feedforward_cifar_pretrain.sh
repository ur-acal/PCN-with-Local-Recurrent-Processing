#!/bin/bash

# Unitless physical WRN/ResNet-like pretraining. CiFAIR uses the same existing
# TrainerCiFarTimmStyle non-RGB augmentation path as PCN pretraining.
set -e

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO_ROOT}"

MODEL_NAME="${MODEL_NAME:?MODEL_NAME is required}"
TASK="${TASK:-cifar100}"
IMG_TYPE="${IMG_TYPE:-CiFAIR}"
OUTPUT_DIR="${OUTPUT_DIR:-./saved_ckpt_runs/feedforward_pretrain}"
INPUT_QUANT_BITS="${INPUT_QUANT_BITS:-none}"
CENTER_STUDENT_INPUT="${CENTER_STUDENT_INPUT:-false}"
if [[ "${INPUT_QUANT_BITS,,}" != "none" && -n "${INPUT_QUANT_BITS}" &&
      "${OUTPUT_DIR,,}" != *"_iq${INPUT_QUANT_BITS,,}"* ]]; then
  OUTPUT_DIR+="_iq${INPUT_QUANT_BITS}"
fi
if [[ "${CENTER_STUDENT_INPUT,,}" == "true" && "${OUTPUT_DIR,,}" != *"_ctr"* ]]; then
  OUTPUT_DIR+="_ctr"
fi
DATA_DIR="${DATA_DIR:-../data}"
TIMM_AUG_LEVEL="${TIMM_AUG_LEVEL:-none}"
TIMM_RE_PROB="${TIMM_RE_PROB:-0.0}"
DISTILL_METHOD="${DISTILL_METHOD:-srrl}"
_DEFAULT_OVERRIDE="lr=${PRETRAIN_LEARNING_RATE:-0.1},num_epochs=${PRETRAIN_NUM_EPOCHS:-300},warmup_epoch=${WARMUP_PRETRAIN:-0},weight_decay=${WEIGHT_DECAY:-0.001},batch_size=${BATCH_SIZE:-128}"
case "${MODEL_NAME}" in
  wrn_28_2_cifar_nobn_avgpool|wrn_28_2_cifar_nobn_avgpool_shortcut)
    if [[ "${TASK}" == "cifar10" ]]; then
      _DEFAULT_OVERRIDE+=",dropout_rate=${DROPOUT_RATE:-0.1},max_norm=${MAX_NORM:-2.0},bias_lr_multiplier=${BIAS_LR_MULTIPLIER:-0.5},bias_weight_decay=${BIAS_WEIGHT_DECAY:-0.0}"
    fi
    ;;
esac
EXTRA_OVERRIDE="${EXTRA_OVERRIDE:-${_DEFAULT_OVERRIDE}}"
ONE_OVER_Q="${TOGGLE_ONE_OVER_Q:-${ONE_OVER_Q:-1}}"
ACTIVATION_CORNER_MODE="${ACTIVATION_CORNER_MODE:-fixed}"
if [[ "${ACTIVATION_CORNER_MODE}" == "random_per_forward" ]]; then
  ACTIVATION_CURVE_PATH="${ACTIVATION_CURVE_PATH:-./hardware_data/relu_current_0p2uA_all.csv}"
else
  ACTIVATION_CURVE_PATH="${ACTIVATION_CURVE_PATH:-./hardware_data/relu_current_0p2uA_finer.csv}"
fi
UNITLESS_MEASURED_PULLBACK_MODE="${UNITLESS_MEASURED_PULLBACK_MODE:-none}"
if [[ "${UNITLESS_MEASURED_PULLBACK_MODE}" == "none" ]]; then
  ENABLE_PRETRAIN_MEASURED_ACTIVATION=false
else
  ENABLE_PRETRAIN_MEASURED_ACTIVATION=true
fi

cmd=(
  python baseline/train_baseline_cifar.py
  --model_name "${MODEL_NAME}"
  --dataset "${TASK}"
  --data_dir "${DATA_DIR}"
  --output_dir "${OUTPUT_DIR}"
  --case custom_noresize
  --seed "${TRAINING_SEED:-4096}"
  --eval_every "${PRETRAIN_EVAL_EVERY:-5}"
  --img_type "${IMG_TYPE}"
  --timm_aug_level "${TIMM_AUG_LEVEL}"
  --timm_re_prob "${TIMM_RE_PROB}"
  --distill_method "${DISTILL_METHOD}"
  --distill_alpha "${DISTILL_ALPHA:-0.3}"
  --distill_temperature "${DISTILL_TEMPERATURE:-2.0}"
  --srrl_weight "${SRRL_WEIGHT:-1.0}"
  --mgd_alpha "${MGD_ALPHA:-7e-5}"
  --mgd_lambda "${MGD_LAMBDA:-0.5}"
  --mgd_mask_mode "${MGD_MASK_MODE:-channel}"
  --reviewkd_weight "${REVIEWKD_WEIGHT:-1.0}"
  --reviewkd_warmup_epochs "${REVIEWKD_WARMUP_EPOCHS:-20}"
  --reviewkd_num_stages "${REVIEWKD_NUM_STAGES:-4}"
  --physical_feedforward true
  --physical_pretraining true
  --physical_level 2
  --R "${R_VAL:-67e3}"
  --C "${C_VAL:-282e-15}"
  --v_dd "${V_DD:-0.1}"
  --one_over_q "${ONE_OVER_Q}"
  --toggle_timing_mode "${TOGGLE_TIMING_MODE:-derived}"
  --toggle_y_time "${TOGGLE_Y_TIME:-5e-9}"
  --z_over_y_time "${Z_OVER_Y_TIME:-1}"
  --scale_train_recipe "${SCALE_TRAIN_RECIPE:-false}"
  --input_quant_bits "${INPUT_QUANT_BITS}"
  --center_student_input "${CENTER_STUDENT_INPUT}"
  --enable_pretrain_measured_activation "${ENABLE_PRETRAIN_MEASURED_ACTIVATION}"
  --activation_curve_path "${ACTIVATION_CURVE_PATH}"
  --activation_corner "${ACTIVATION_CORNER:-TT}"
  --activation_corner_mode "${ACTIVATION_CORNER_MODE}"
  --activation_curve_sharing "${ACTIVATION_CURVE_SHARING:-per_model}"
  --activation_interpolation "${ACTIVATION_INTERPOLATION:-piecewise_linear}"
  --activation_spline_parameters "${ACTIVATION_SPLINE_PARAMETERS:-10}"
  --activation_fit_constraint "${ACTIVATION_FIT_CONSTRAINT:-auto}"
  --activation_normalize_positive_endpoint "${SCALE_MEASURED_ACTIVATION:-false}"
  --compile_measured_activation "${COMPILE_MEASURED_ACTIVATION:-false}"
  --unitless_measured_pullback_mode "${UNITLESS_MEASURED_PULLBACK_MODE}"
  --print_only "${PRINT_ONLY:-false}"
)

if [[ -n "${WRN_DEPTH:-}" ]]; then
  cmd+=(--wrn_depth "${WRN_DEPTH}")
fi
if [[ -n "${WRN_FIRST_STAGE_CHANNELS:-}" ]]; then
  cmd+=(--wrn_first_stage_channels "${WRN_FIRST_STAGE_CHANNELS}")
fi

if [[ -n "${TEACHER_CKPT:-}" ]]; then
  cmd+=(--teacher_ckpt "${TEACHER_CKPT}")
fi
if [[ -n "${PRETRAIN_RESUME_CKPT:-}" ]]; then
  cmd+=(--resume_checkpoint "${PRETRAIN_RESUME_CKPT}")
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
if [[ -n "${UNITLESS_PULLBACK_Q:-}" && "${UNITLESS_PULLBACK_Q,,}" != "none" ]]; then
  cmd+=(--unitless_pullback_q "${UNITLESS_PULLBACK_Q}")
fi
if [[ -n "${ACTIVATION_CURVE_SEED:-}" ]]; then
  cmd+=(--activation_curve_seed "${ACTIVATION_CURVE_SEED}")
fi
if [[ -n "${EXTRA_OVERRIDE}" ]]; then
  cmd+=(--override "${EXTRA_OVERRIDE}")
fi

printf 'Running:'
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"

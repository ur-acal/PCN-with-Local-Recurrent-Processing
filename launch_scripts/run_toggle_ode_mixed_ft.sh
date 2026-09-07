#!/usr/bin/env bash

#######################################################
# An example launch
# sbatch -N 1 --export=ALL,TIMM_AUG_LEVEL=no_aug,ENOB=6,SWITCH_INF=false ./launch_scripts/run_slurm_ode_mixed_ft.sh
#######################################################

R_VAL="${R_VAL:-67e3}"
R_MAX="${R_MAX:-none}"
C_VAL="${C_VAL:-282e-15}"
V_DD="${V_DD:-0.1}"
MISMATCH_LEVEL="${MISMATCH_LEVEL:-0.0}"

declare -A NOISE_LEVELS=(
  [mul]="${MISMATCH_LEVEL}"
  [add]="0.05 0.08 0.1 0.15 0.2"
)
NOISE_TYPES=(
  "mul"
#  "add"
)
NBITS=(5)
R_MAX_LIST=("${R_MAX}")
TOGGLE_ONE_OVER_Q="${TOGGLE_ONE_OVER_Q:-1}"
ONE_OVER_Q_LIST=("${TOGGLE_ONE_OVER_Q}")
EXP="${EXP_OVERRIDE:-NODE_0602_QAT_with_noise_inject_kd_crd_training_C100}"
LOGDIR="./logs/${EXP}"
mkdir -p "${LOGDIR}"
MODEL_NAME="${MODEL_NAME_OVERRIDE:-TIMMPCNetNoBatchNorm_PCConvReLU6_0.0eps_ODEXInitFFFB_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_C100_3K1S96C_0.25Dropout_16Layers4l5l4_2Pool_srrlDistill_a0p3_t2p0_scanGFI_6REP}"
#MODEL_NAME="${MODEL_NAME_OVERRIDE:-TIMMPCNetNoBatchNorm_PCConvReLU6_0.0eps_ToggleODEXInitFFFB_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_C100_3K1S96C_0.25Dropout_16Layers4l5l4_2Pool_srrlDistill_a0p3_t2p0_scanGFI_1REP}"
#MODEL_NAME="TIMMPCNetNoBatchNorm_PCConvReLU6_0.0eps_ODEXInitFFFB_eulerSolver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_C100_3K1S72C_0.25Dropout_12Layers11l0l0_1Pool5_srrlDistill_a0p3_t2p0_scanGFI_2REP"

if [[ -z "${IMG_TYPE:-}" ]]; then
  if [[ "${MODEL_NAME,,}" == *cifair* ]]; then
    IMG_TYPE="CiFAIR"
  else
    IMG_TYPE="scanGFI"
  fi
fi
case "${IMG_TYPE,,}" in
  cifair) IMG_TYPE="CiFAIR" ;;
  scangfi|raw|_raw) IMG_TYPE="scanGFI" ;;
esac

SWITCH_INF=${SWITCH_INF:-false} # Change depending on model name; true if using Euler solver.
TIMM_AUG_LEVEL=${TIMM_AUG_LEVEL:-no_aug}
ENOB="${ENOB:-8}"
K_VAL="${K_VAL:-1e3}"
MODEL_DIR="${MODEL_DIR:-./saved_ckpt}"
INPUT_PREPROCESS_ARGS=()
if [[ -n "${INPUT_QUANT_BITS+x}" ]]; then
  INPUT_PREPROCESS_ARGS+=(--input_quant_bits "${INPUT_QUANT_BITS}")
fi
if [[ -n "${CENTER_STUDENT_INPUT+x}" ]]; then
  INPUT_PREPROCESS_ARGS+=(--center_student_input "${CENTER_STUDENT_INPUT}")
fi
PCN="${PCN:-PCNetNoBatchNorm}"
WARMUP_FT="${WARMUP_FT:-0}"
FT_LEARNING_RATE="${FT_LEARNING_RATE:-0.005}"
FT_NUM_EPOCHS="${FT_NUM_EPOCHS:-${NUM_EPOCHS:-140}}"
MEM_FRAC="${MEM_FRAC:-0.9}"
DISTILL_METHOD="${DISTILL_METHOD:-srrl}"
SRRL_WEIGHT="${SRRL_WEIGHT:-1.0}"
SRRL_ARGS=(--srrl_weight "${SRRL_WEIGHT}")
TIMM_RE_PROB="${TIMM_RE_PROB:-0.0}"
TIMM_RE_ARGS=(--timm_re_prob "${TIMM_RE_PROB}")
REVIEWKD_WEIGHT="${REVIEWKD_WEIGHT:-1.0}"
REVIEWKD_WARMUP_EPOCHS="${REVIEWKD_WARMUP_EPOCHS:-20}"
REVIEWKD_NUM_STAGES="${REVIEWKD_NUM_STAGES:-4}"
REVIEWKD_ARGS=(
  --reviewkd_weight "${REVIEWKD_WEIGHT}"
  --reviewkd_warmup_epochs "${REVIEWKD_WARMUP_EPOCHS}"
  --reviewkd_num_stages "${REVIEWKD_NUM_STAGES}"
)
PULSE_MISMATCH_TRAINING_MODE="${PULSE_MISMATCH_TRAINING_MODE:-post_quant_amplitude}"
WEIGHT_QUANT_FACTOR_BITS="${WEIGHT_QUANT_FACTOR_BITS:-none}"
# This controls if we are using different measure activation curves per forward pass in training.
ACTIVATION_CORNER_MODE="${ACTIVATION_CORNER_MODE:-fixed}"
ACTIVATION_RANDOM_CURVE_SHARING="${ACTIVATION_RANDOM_CURVE_SHARING:-per_layer}"
MC_RELU_MONTE_CARLO_SOURCE="${MC_RELU_MONTE_CARLO_SOURCE:-relu_monteCarlo}"
ENABLE_MEASURED_POOLING="${ENABLE_MEASURED_POOLING:-false}"
NONLINEAR_R_TABLE="${NONLINEAR_R_TABLE:-coupler_monte}"
NONLINEAR_R_TRAIN_MODE="${NONLINEAR_R_TRAIN_MODE:-exact_curve}"
ENABLE_NONLINEAR_R="${ENABLE_NONLINEAR_R:-true}"
if [[ "${MC_RELU_MONTE_CARLO_SOURCE%/}" == "0906_RELU_Voltage" ]]; then
  if [[ "${ACTIVATION_CORNER_MODE}" == "random_per_forward" ]]; then
    ACTIVATION_CURVE_PATH="${ACTIVATION_CURVE_PATH:-./hardware_data/mc_45_corners/0906_RELU_Voltage}"
    ACTIVATION_CORNER="${ACTIVATION_CORNER:-TT_25_1_MC18}"
  else
    # MC18 is the TT/V1/T1 realization closest to this corner's 100-curve mean.
    ACTIVATION_CURVE_PATH="${ACTIVATION_CURVE_PATH:-./hardware_data/mc_45_corners/0906_RELU_Voltage/tt_25_1.csv}"
    ACTIVATION_CORNER="${ACTIVATION_CORNER:-MC18}"
  fi
elif [[ "${ACTIVATION_CORNER_MODE}" == "random_per_forward" ]]; then
  ACTIVATION_CURVE_PATH="${ACTIVATION_CURVE_PATH:-./hardware_data/relu_current_0p2uA_all.csv}"
else
  ACTIVATION_CURVE_PATH="${ACTIVATION_CURVE_PATH:-./hardware_data/relu_current_0p2uA_finer.csv}"
fi
ACTIVATION_CORNER="${ACTIVATION_CORNER:-TT}"
VARIATION_AWARE_ARGS=(
  --enable_spin_variation "${ENABLE_SPIN_VARIATION:-true}"
  --sigma_spin "${SIGMA_SPIN:-0.10}"
  --enable_summing_current_noise "${ENABLE_SUMMING_CURRENT_NOISE:-true}"
  --summing_current_p "${SUMMING_CURRENT_P:-0.6e-12}"
  --enable_coupler_noise "${ENABLE_COUPLER_NOISE:-true}"
  --coupler_noise_p "${COUPLER_NOISE_P:-0.6e-12}"
  --enable_slow_summing_current "${ENABLE_SLOW_SUMMING_CURRENT:-false}"
  --slow_summing_current "${SLOW_SUMMING_CURRENT:-2.47e-9}"
  --enable_slow_coupler_noise "${ENABLE_SLOW_COUPLER_NOISE:-false}"
  --slow_coupler_noise "${SLOW_COUPLER_NOISE:-2.47e-9}"
)
if [[ "${NONLINEAR_R_TRAIN_MODE:-none}" != "none" ]]; then
  ENABLE_NONLINEAR_R=true
fi
NONLINEAR_R_TRAIN_ARGS=(
  --nonlinear_R "${ENABLE_NONLINEAR_R:-false}"
  --nonlinear_R_table "./hardware_data/mc_45_corners/${NONLINEAR_R_TABLE}"
  --nonlinear_R_mc_quantity "${NONLINEAR_R_MC_QUANTITY:-conductance}"
  --nonlinear_R_curve_sharing "${NONLINEAR_R_CURVE_SHARING:-shared}"
  --nonlinear_R_curve_seed "${NONLINEAR_R_CURVE_SEED:-none}"
  --train_conv_expanded "${TRAIN_CONV_EXPANDED:-false}"
  --nonlinear_R_train_mode "${NONLINEAR_R_TRAIN_MODE:-none}"
  --nonlinear_R_corner_range "${NONLINEAR_R_CORNER_RANGE:-all}"
)
ODE_BLOCK_OVERRIDE="${ODE_BLOCK_OVERRIDE:-ToggleODEXInitFFFB}"
# This controls the scaling for the toggle class. approximating the old 1state or directly scale.
ODEXINIT_SCALING_MODE="${ODEXINIT_SCALING_MODE:-direct}" # "approx", "direct"
TOGGLE_N_CYCLES="${TOGGLE_N_CYCLES:-5}"
TOGGLE_TIME_SPLIT="${TOGGLE_TIME_SPLIT:-0.5}"
TOGGLE_FAST_PATH="${TOGGLE_FAST_PATH:-true}"
TOGGLE_TIMING_MODE="${TOGGLE_TIMING_MODE:-derived}"
TOGGLE_Y_TIME="${TOGGLE_Y_TIME:-5e-9}"
Z_OVER_Y_TIME="${Z_OVER_Y_TIME:-3}"
SCALE_TRAIN_RECIPE="${SCALE_TRAIN_RECIPE:-false}"
TOGGLE_ARGS=(--odexinit_scaling_mode "${ODEXINIT_SCALING_MODE}")
TOGGLE_ARGS+=(--toggle_timing_mode "${TOGGLE_TIMING_MODE}")
TOGGLE_ARGS+=(--toggle_y_time "${TOGGLE_Y_TIME}")
TOGGLE_ARGS+=(--z_over_y_time "${Z_OVER_Y_TIME}")
if [[ -n "${TOGGLE_N_CYCLES:-}" ]]; then TOGGLE_ARGS+=(--toggle_n_cycles "${TOGGLE_N_CYCLES}"); fi
if [[ -n "${TOGGLE_TIME_SPLIT:-}" ]]; then TOGGLE_ARGS+=(--toggle_time_split "${TOGGLE_TIME_SPLIT}"); fi
if [[ -n "${TOGGLE_FAST_PATH:-}" ]]; then TOGGLE_ARGS+=(--toggle_fast_path "${TOGGLE_FAST_PATH}"); fi
echo "=========== R: ${R_VAL}; R_max: ${R_MAX}; C: ${C_VAL}; mismatch level: ${MISMATCH_LEVEL} ==========="
echo "=========== TIMM_AUG_LEVEL: ${TIMM_AUG_LEVEL}, SWITCH_INF: ${SWITCH_INF}, ENOB: ${ENOB}, ODE_BLOCK_OVERRIDE: ${ODE_BLOCK_OVERRIDE} ==========="
echo "=========== FT spin variation: ${ENABLE_SPIN_VARIATION:-true} (sigma=${SIGMA_SPIN:-0.10}); summing-current noise: ${ENABLE_SUMMING_CURRENT_NOISE:-true} (p=${SUMMING_CURRENT_P:-0.6e-12}); coupler noise: ${ENABLE_COUPLER_NOISE:-true} (p=${COUPLER_NOISE_P:-0.6e-12}) ==========="
echo "=========== FT slow summing current: ${ENABLE_SLOW_SUMMING_CURRENT:-false} (std=${SLOW_SUMMING_CURRENT:-2.47e-9} A); slow coupler noise: ${ENABLE_SLOW_COUPLER_NOISE:-false} (per-coupler std=${SLOW_COUPLER_NOISE:-2.47e-9} A) ==========="
echo "=========== nonlinear-R training: mode=${NONLINEAR_R_TRAIN_MODE:-none}, corners=${NONLINEAR_R_CORNER_RANGE:-all}, expanded=${TRAIN_CONV_EXPANDED:-false} ==========="
echo "=========== measured pooling training: ${ENABLE_MEASURED_POOLING} ==========="
echo "=========== weight quant factor bits: ${WEIGHT_QUANT_FACTOR_BITS} ==========="
echo "=========== toggle timing: ${TOGGLE_TIMING_MODE}, T_y=${TOGGLE_Y_TIME}s, T_z/T_y=${Z_OVER_Y_TIME}, scale_train_recipe=${SCALE_TRAIN_RECIPE} ==========="
echo "=========== image data: ${IMG_TYPE} ==========="
echo "=========== FT objective: ${DISTILL_METHOD}, SRRL weight: ${SRRL_WEIGHT}; LR: ${FT_LEARNING_RATE}, epochs: ${FT_NUM_EPOCHS}, warmup: ${WARMUP_FT} ==========="

#######################################################################################################################
# For QAT models, keep finetuning with full_param checkpoint, which keeps the original un-parametrized weights
#MODEL_NAME="QAT5bNT0p1mulPCNetNoBatchNorm_PCConvReLU6_0.0eps_S2NoisyIYAsXZAs0_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_3K1S48C_0.25Dropout_18Layers_2Pool_scanGFI_5REP"
#######################################################################################################################
CKPT="best"
if [[ "$MODEL_NAME" == *QAT* ]]; then
  CKPT="full_param_best"
fi

if [[ "$MODEL_NAME" == *C100* ]]; then
  _task="cifar100"
else
  _task="cifar10"
fi
if [[ "${_task}" == "cifar100" ]]; then
  N_CLASSES=100
else
  N_CLASSES=10
fi
# Set Wrapper
QAT_WRAPPER="QATWrapper1State"
if [[ "$MODEL_NAME" == *S2* || "$MODEL_NAME" == *State2* ]]; then
  QAT_WRAPPER="QATWrapper2State"
fi
#QAT_WRAPPER="QATWrapper1StateWithX"
# Get ODE Block
IFS='_' read -r -a parts <<< "$MODEL_NAME"
ODE_BLK="${parts[3]}"
if [[ -n "${ODE_BLOCK_OVERRIDE}" ]]; then
  ODE_BLK="${ODE_BLOCK_OVERRIDE}"
fi
if [[ "${ODE_BLK}" == TogglePulseBlk* ]]; then
  QAT_WRAPPER="TogglePulseQATWrapper1State"
elif [[ "${ODE_BLK}" == "ToggleResetZ" || "${ODE_BLK}" == "ToggleKeepZ" || "${ODE_BLK}" == ToggleODEXInit* || "${ODE_BLK}" == TogglePulse* ]]; then
  QAT_WRAPPER="ToggleQATWrapper1State"
fi
# Teacher model setting
if [[ "${IMG_TYPE}" == "CiFAIR" ]]; then
  _DEFAULT_TEACHER_CKPT="checkpoint/efficientnet_v2_l_${_task}_CiFAIR_timm.pth"
  _DEFAULT_TEACHER_ARCH="efficientnet_v2_l"
elif [[ "${_task}" == "cifar10" ]]; then
  _DEFAULT_TEACHER_CKPT="checkpoint/b4.pth"
  _DEFAULT_TEACHER_ARCH="efficientnet-b4"
else
  _DEFAULT_TEACHER_CKPT="checkpoint/b4_100.pth"
  _DEFAULT_TEACHER_ARCH="efficientnet_v2_l"
fi
TEACHER_CKPT="${TEACHER_CKPT:-${_DEFAULT_TEACHER_CKPT}}"
TEACHER_ARCH="${TEACHER_ARCH:-${_DEFAULT_TEACHER_ARCH}}"
TEACHER_ARCH_SOURCE="${TEACHER_ARCH_SOURCE:-auto}"
TEACHER_INPUT_SIZE="${TEACHER_INPUT_SIZE:-224}"
TEACHER_CENTER_CROP="${TEACHER_CENTER_CROP:-true}"
# Enable only for legacy run_teacher PIL checkpoints (match_distill_preprocess=false).
ADAPT_PIL_TEACHER="${ADAPT_PIL_TEACHER:-false}"

train_solver="dopri5"
if [[ "${SWITCH_INF}" == "true" ]]; then
  train_solver="euler"
fi

# No 2
for one_over_q in "${ONE_OVER_Q_LIST[@]}"; do
  for nt in "${NOISE_TYPES[@]}"; do
    for nl in ${NOISE_LEVELS[$nt]}; do
      for n_bits in "${NBITS[@]}"; do
        for R_max in "${R_MAX_LIST[@]}"; do
          echo "log dir: ${LOGDIR}/train_${EXP}_No_2_ReLU6_2State_${n_bits}_${nl}_${nt}_${R_max}.log"
          python train_ode_cifar.py \
            --dataset       "${_task}" \
            --num_classes   "${N_CLASSES}" \
            --save_path     "${MODEL_DIR}" \
            --ckpt          "${CKPT}" \
            --timm_trainer  "true" \
            --timm_sched    "cosine" \
            --timm_aug_level "${TIMM_AUG_LEVEL}" \
            "${TIMM_RE_ARGS[@]}" \
            --rggb_to_rgb   "false" \
            --optim         "SGD" \
            --learning_rate "${FT_LEARNING_RATE}" \
            --eval_every    "${EVAL_EVERY:-2}" \
            --num_epochs    "${FT_NUM_EPOCHS}" \
            --warmup_epoch  "${WARMUP_FT}" \
            --img_type      "${IMG_TYPE}" \
            "${INPUT_PREPROCESS_ARGS[@]}" \
            --model_name    "${MODEL_NAME}" \
            --output_save_path "${OUTPUT_SAVE_PATH:-./saved_ckpt}" \
            --offset_eps    0.0 \
            --dropout       0.25 \
            --avg_pooling   "true" \
            --tie_weights   "false" \
            --tie_bp        "false" \
            --bypass        "false" \
            --batch_size    128 \
            --method        "${train_solver}" \
            --n_steps       5 \
            --tol           "1e-6" \
            --t_end         "1.75" \
            --R             "${R_VAL}" \
            --R_max         "${R_max}" \
            --C             "${C_VAL}" \
            --k             "${K_VAL}" \
            --v_dd          "${V_DD}" \
            --enable_measured_activation "${ENABLE_MEASURED_ACTIVATION:-true}" \
            --enable_measured_pooling "${ENABLE_MEASURED_POOLING}" \
            --activation_curve_path "${ACTIVATION_CURVE_PATH}" \
            --activation_corner "${ACTIVATION_CORNER:-TT}" \
            --activation_corner_mode "${ACTIVATION_CORNER_MODE}" \
            --activation_random_curve_sharing "${ACTIVATION_RANDOM_CURVE_SHARING}" \
            --activation_interpolation "${ACTIVATION_INTERPOLATION:-piecewise_linear}" \
            --activation_spline_parameters "${ACTIVATION_SPLINE_PARAMETERS:-10}" \
            --activation_fit_constraint "${ACTIVATION_FIT_CONSTRAINT:-auto}" \
            --activation_normalize_positive_endpoint "${SCALE_MEASURED_ACTIVATION:-false}" \
            --enob          "${ENOB}" \
            --w_bits        "${n_bits}" \
            --weight_quant_factor_bits "${WEIGHT_QUANT_FACTOR_BITS}" \
            --scale_train_recipe "${SCALE_TRAIN_RECIPE}" \
            --patch_node    "${PATCH_NODE:-}" \
            --patch_stride  "${PATCH_STRIDE:-}" \
            --patch_cycle   "${PATCH_CYCLE:-}" \
            --patch_pad     "${PATCH_PAD:-}" \
            --fold_scalar   "${FOLD_SCALAR:-}" \
            --tie_cap       "false" \
            --one_over_q    "${one_over_q}" \
            "${TOGGLE_ARGS[@]}" \
            "${VARIATION_AWARE_ARGS[@]}" \
            "${NONLINEAR_R_TRAIN_ARGS[@]}" \
            --qat_cls       "SymQuantizeWeight" \
            --pcn           "${PCN}" \
            --ode_wrapper   "$QAT_WRAPPER" \
            --pc_conv       "PCConvReLU6" \
            --ode_block     "$ODE_BLK" \
            --noise_level   "${nl}" \
            --noise_type    "${nt}" \
            --pulse_mismatch_training_mode "${PULSE_MISMATCH_TRAINING_MODE}" \
            --teacher_ckpt  "${TEACHER_CKPT}" \
            --teacher_arch  "${TEACHER_ARCH}" \
            --teacher_arch_source "${TEACHER_ARCH_SOURCE}" \
            --teacher_input_size  "${TEACHER_INPUT_SIZE}" \
            --teacher_center_crop "${TEACHER_CENTER_CROP}" \
            --adapt_PIL_teacher "${ADAPT_PIL_TEACHER}" \
            --distill_method "${DISTILL_METHOD}" \
            "${SRRL_ARGS[@]}" \
            "${REVIEWKD_ARGS[@]}" \
            --contrast_method "memory" \
            --distill_alpha  0.3 \
            --distill_temperature 2.0 \
            --test_only     "false" \
            --mem_frac      "${MEM_FRAC}" \
            2>&1 | tee "${LOGDIR}/train_${EXP}_No_2_ReLU6_2State_${n_bits}_${nl}_${nt}_${R_max}_${one_over_q}.log"
        done
      done
    done
  done
done
echo "Completed."

############################################################
# launch in this way:
# nohup bash launch_scripts/run_ode_train.sh > ./logs/master_single.log 2>&1 &
# tail -f ./logs/master_single.log
# after train finished
# cat ./logs/master_single.log | grep "Train finished" -A 3
############################################################

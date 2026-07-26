#!/usr/bin/env bash

#######################################################
# An example launch
# sbatch -N 1 --export=ALL,TIMM_AUG_LEVEL=no_aug,ENOB=6,SWITCH_INF=false ./launch_scripts/run_slurm_ode_mixed_ft.sh
#######################################################

declare -A NOISE_LEVELS=(
  [mul]="0.25"
  [add]="0.05 0.08 0.1 0.15 0.2"
)
NOISE_TYPES=(
  "mul"
#  "add"
)
NBITS=(5)
R_MAX_LIST=("")
ONE_OVER_Q_LIST=("1")
EXP="${EXP_OVERRIDE:-NODE_0602_QAT_with_noise_inject_kd_crd_training_C100}"
LOGDIR="./logs/${EXP}"
mkdir -p "${LOGDIR}"
MODEL_NAME="${MODEL_NAME_OVERRIDE:-TIMMPCNetNoBatchNorm_PCConvReLU6_0.0eps_ODEXInitFFFB_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_C100_3K1S96C_0.25Dropout_16Layers4l5l4_2Pool_srrlDistill_a0p3_t2p0_scanGFI_6REP}"
#MODEL_NAME="${MODEL_NAME_OVERRIDE:-TIMMPCNetNoBatchNorm_PCConvReLU6_0.0eps_ToggleODEXInitFFFB_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_C100_3K1S96C_0.25Dropout_16Layers4l5l4_2Pool_srrlDistill_a0p3_t2p0_scanGFI_1REP}"
#MODEL_NAME="TIMMPCNetNoBatchNorm_PCConvReLU6_0.0eps_ODEXInitFFFB_eulerSolver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_C100_3K1S72C_0.25Dropout_12Layers11l0l0_1Pool5_srrlDistill_a0p3_t2p0_scanGFI_2REP"

SWITCH_INF=${SWITCH_INF:-false} # Change depending on model name; true if using Euler solver.
TIMM_AUG_LEVEL=${TIMM_AUG_LEVEL:-no_aug}
ENOB="${ENOB:-8}"
K_VAL="${K_VAL:-1e3}"
PULSE_MISMATCH_TRAINING_MODE="${PULSE_MISMATCH_TRAINING_MODE:-post_quant_amplitude}"
# This controls if we are using different measure activation curves per forward pass in training.
ACTIVATION_CORNER_MODE="${ACTIVATION_CORNER_MODE:-fixed}"
if [[ "${ACTIVATION_CORNER_MODE}" == "random_per_forward" ]]; then
  ACTIVATION_CURVE_PATH="${ACTIVATION_CURVE_PATH:-./hardware_data/relu_current_0p2uA_all.csv}"
else
  ACTIVATION_CURVE_PATH="${ACTIVATION_CURVE_PATH:-./hardware_data/relu_current_0p2uA_finer.csv}"
fi
VARIATION_AWARE_ARGS=(
  --enable_spin_variation "${ENABLE_SPIN_VARIATION:-true}"
  --sigma_spin "${SIGMA_SPIN:-0.10}"
  --enable_summing_current_noise "${ENABLE_SUMMING_CURRENT_NOISE:-false}"
  --summing_current_p "${SUMMING_CURRENT_P:-18.5e-12}"
  --enable_coupler_noise "${ENABLE_COUPLER_NOISE:-true}"
  --coupler_noise_p "${COUPLER_NOISE_P:-0.6e-12}"
)
ODE_BLOCK_OVERRIDE="${ODE_BLOCK_OVERRIDE:-ToggleODEXInitFFFB}"
# This controls the scaling for the toggle class. approximating the old 1state or directly scale.
ODEXINIT_SCALING_MODE="${ODEXINIT_SCALING_MODE:-direct}" # "approx", "direct"
TOGGLE_ARGS=(--odexinit_scaling_mode "${ODEXINIT_SCALING_MODE}")
TOGGLE_N_CYCLES="${TOGGLE_N_CYCLES:-5}"
if [[ -n "${TOGGLE_N_CYCLES:-}" ]]; then TOGGLE_ARGS+=(--toggle_n_cycles "${TOGGLE_N_CYCLES}"); fi
if [[ -n "${TOGGLE_TIME_SPLIT:-}" ]]; then TOGGLE_ARGS+=(--toggle_time_split "${TOGGLE_TIME_SPLIT}"); fi
if [[ -n "${TOGGLE_FAST_PATH:-}" ]]; then TOGGLE_ARGS+=(--toggle_fast_path "${TOGGLE_FAST_PATH}"); fi
echo "=========== TIMM_AUG_LEVEL: ${TIMM_AUG_LEVEL}, SWITCH_INF: ${SWITCH_INF}, ENOB: ${ENOB}, ODE_BLOCK_OVERRIDE: ${ODE_BLOCK_OVERRIDE} ==========="
echo "=========== FT spin variation: ${ENABLE_SPIN_VARIATION:-true} (sigma=${SIGMA_SPIN:-0.10}); summing-current noise: ${ENABLE_SUMMING_CURRENT_NOISE:-false} (p=${SUMMING_CURRENT_P:-18.5e-12}); coupler noise: ${ENABLE_COUPLER_NOISE:-true} (p=${COUPLER_NOISE_P:-0.6e-12}) ==========="

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
TEACHER_CKPT="${TEACHER_CKPT:-checkpoint/b4_100.pth}"
TEACHER_ARCH="${TEACHER_ARCH:-efficientnet_v2_l}"
TEACHER_ARCH_SOURCE="${TEACHER_ARCH_SOURCE:-auto}"
TEACHER_INPUT_SIZE="${TEACHER_INPUT_SIZE:-224}"
TEACHER_CENTER_CROP="${TEACHER_CENTER_CROP:-true}"
if [[ "${DATASET_NAME}" == "cifar10" ]]; then
  TEACHER_CKPT="checkpoint/b4.pth"
  TEACHER_ARCH="efficientnet-b4"
fi

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
            --ckpt          "${CKPT}" \
            --timm_trainer  "true" \
            --timm_sched    "cosine" \
            --timm_aug_level "${TIMM_AUG_LEVEL}" \
            --rggb_to_rgb   "false" \
            --optim         "SGD" \
            --learning_rate 0.005 \
            --eval_every    "${EVAL_EVERY:-2}" \
            --num_epochs    "${NUM_EPOCHS:-140}" \
            --img_type      "scanGFI" \
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
            --R             "27.8e3" \
            --R_max         "${R_max}" \
            --C             "282e-15" \
            --k             "${K_VAL}" \
            --v_dd          "0.1" \
            --enable_measured_activation "${ENABLE_MEASURED_ACTIVATION:-true}" \
            --activation_curve_path "${ACTIVATION_CURVE_PATH}" \
            --activation_corner "${ACTIVATION_CORNER:-TT}" \
            --activation_corner_mode "${ACTIVATION_CORNER_MODE}" \
            --activation_interpolation "${ACTIVATION_INTERPOLATION:-piecewise_linear}" \
            --activation_spline_parameters "${ACTIVATION_SPLINE_PARAMETERS:-10}" \
            --activation_fit_constraint "${ACTIVATION_FIT_CONSTRAINT:-auto}" \
            --activation_normalize_positive_endpoint "${SCALE_MEASURED_ACTIVATION:-false}" \
            --enob          "${ENOB}" \
            --w_bits        "${n_bits}" \
            --patch_node    "8" \
            --patch_stride  "8" \
            --patch_cycle   "1" \
            --patch_pad     "0" \
            --fold_scalar   "1" \
            --tie_cap       "false" \
            --one_over_q    "${one_over_q}" \
            "${TOGGLE_ARGS[@]}" \
            "${VARIATION_AWARE_ARGS[@]}" \
            --qat_cls       "SymQuantizeWeight" \
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
            --distill_method srrl \
            --contrast_method "memory" \
            --distill_alpha  0.3 \
            --distill_temperature 2.0 \
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

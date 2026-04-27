#!/usr/bin/env bash

declare -A NOISE_LEVELS=(
  [mul]="0.0"
  [add]="0.05 0.08 0.1 0.15 0.2"
)
NOISE_TYPES=(
  "mul"
#  "add"
)
NBITS=(5)
R_MAX_LIST=("300e3")
ONE_OVER_Q_LIST=("1")
EXP="NODE_0324_QAT_with_noise_inject_kd_crd_training_C100"
LOGDIR="./logs/${EXP}"
mkdir -p "${LOGDIR}"
#MODEL_NAME="PCNetNoBatchNorm_PCConvReLU6_0.0eps_ODEXInitFFFB_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_3K1S96C_0.25Dropout_16Layers4l5l4_2Pool_kd_crdDistill_a0p3_t2p0_scanGFI_1REP"

MODEL_NAME="PCNetNoBatchNorm_PCConvReLU6_0.0eps_ODEXInitFFFB_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_C100_3K1S96C_0.25Dropout_16Layers4l5l4_2Pool_kd_crdDistill_a0p3_t2p0_scanGFI_1REP"
#######################################################################################################################
# For QAT models, keep finetuning with full_param checkpoint, which keeps the original un-parametrized weights
#MODEL_NAME="QAT5bNT0p1mulPCNetNoBatchNorm_PCConvReLU6_0.0eps_S2NoisyIYAsXZAs0_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_3K1S48C_0.25Dropout_18Layers_2Pool_scanGFI_5REP"
#MODEL_NAME="QAT5bNT0p1mulPCNetNoBatchNorm_PCConvReLU6_0.0eps_S2NoisyIYAsXZAs0_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_3K1S72C_0.25Dropout_13Layers_2Pool_scanGFI_4REP"
#MODEL_NAME="QAT5bNT0p15mulPCNetNoBatchNorm_PCConvReLU6_0.0eps_ODEXInitFFFB_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_3K1S72C_0.25Dropout_13Layers_2Pool_scanGFI_1REP"
#MODEL_NAME="QAT5bNT0p25mulPCNetNoBatchNorm_PCConvReLU6_0.0eps_ODEXInitFFFB_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_C100_3K1S96C_0.25Dropout_16Layers4l5l4_2Pool_kd_crdDistill_a0p3_t2p0_scanGFI_2REP"
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
            --rggb_to_rgb   "false" \
            --optim         "SGD" \
            --learning_rate 0.005 \
            --cosine_t0     20 \
            --eval_every    2 \
            --num_epochs    140 \
            --img_type      "scanGFI" \
            --model_name    "${MODEL_NAME}" \
            --offset_eps    0.0 \
            --dropout       0.25 \
            --tie_weights   "false" \
            --tie_bp        "false" \
            --bypass        "false" \
            --batch_size    128 \
            --method        "euler" \
            --n_steps       5 \
            --tol           "1e-6" \
            --t_end         "1.75" \
            --R             "20e3" \
            --R_max         "${R_max}" \
            --C             "49e-15" \
            --v_dd          "0.2" \
            --enob          "8" \
            --w_bits        "${n_bits}" \
            --patch_node    "8" \
            --patch_stride  "8" \
            --patch_cycle   "1" \
            --patch_pad     "0" \
            --fold_scalar   "1" \
            --tie_cap       "false" \
            --one_over_q    "${one_over_q}" \
            --qat_cls       "SymQuantizeWeight" \
            --ode_wrapper   "$QAT_WRAPPER" \
            --pc_conv       "PCConvReLU6" \
            --ode_block     "PerturbODEXInitFFFB" \
            --noise_level   "${nl}" \
            --noise_type    "${nt}" \
            --teacher_ckpt  "${TEACHER_CKPT}" \
            --teacher_arch  "${TEACHER_ARCH}" \
            --teacher_arch_source "${TEACHER_ARCH_SOURCE}" \
            --teacher_input_size  "${TEACHER_INPUT_SIZE}" \
            --teacher_center_crop "${TEACHER_CENTER_CROP}" \
            --distill_method kd_crd \
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
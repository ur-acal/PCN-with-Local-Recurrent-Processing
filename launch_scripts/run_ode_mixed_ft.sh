#!/usr/bin/env bash

declare -A NOISE_LEVELS=(
  [mul]="0.15"
  [add]="0.05 0.08 0.1 0.15 0.2"
)
NOISE_TYPES=(
  "mul"
#  "add"
)
NBITS=(5)
R_MAX_LIST=("300e3")
EXP="NODE_0128_QAT_with_noise_inject_training"
LOGDIR="./logs/${EXP}"
mkdir -p "${LOGDIR}"
#MODEL_NAME="PCNetNoBatchNorm_PCConvReLU6_0.002eps_S2NoMinusZChgZNoisyI_dopri5Solver_1.5TEnd_0.0001Tol_0.001WD_128BS_0.01LR_0.25Dropout_7Layers_2Pool_2REP"
#MODEL_NAME="PCNetNoBatchNorm_PCConvReLU6_0.0eps_S2NoisyIYAsXZAs0_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_3K1S48C_0.25Dropout_18Layers_2Pool_scanGFI_2REP"
#MODEL_NAME="PCNetNoBatchNorm_PCConvReLU6_0.0eps_S2NoisyIYAsXZAs0_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_C100_3K1S80C_0.25Dropout_10Layers_2Pool_scanGFI_1REP"
#MODEL_NAME="PCNetNoBatchNorm_PCConvReLU6_0.0eps_S2NoisyIYAsXZAs0_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_3K1S72C_0.25Dropout_13Layers_2Pool_scanGFI_1REP"
#MODEL_NAME="PCNetNoBatchNorm_PCConvReLU6_0.0eps_S2NoisyIYAsXZAs0_dopri5Solver_1.75TEnd_0.0001Tol_0.0001WD_128BS_0.01LR_C100_3K1S128C_0.0Dropout_7Layers_2Pool_kd_crdDistill_a0p3_t2p0_scanGFI_1REP"
#MODEL_NAME="PCNetNoBatchNorm_PCConvReLU6_0.0eps_ODEXInitFFFB_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_3K1S72C_0.25Dropout_13Layers_2Pool_scanGFI_1REP"
#MODEL_NAME="PCNetNoBatchNorm_PCConvReLU6_0.0eps_ODEXInitFFFB_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_C100_3K1S128C_0.25Dropout_10Layers2l2l3_2Pool_scanGFI_2REP"
MODEL_NAME="PCNetNoBatchNorm_PCConvReLU6_0.0eps_ODEXInitFFFB_dopri5Solver_1.75TEnd_0.0001Tol_0.0001WD_128BS_0.01LR_C100_3K1S128C_0.0Dropout_10Layers_2Pool_kd_crdDistill_a0p3_t2p0_scanGFI_2REP"
#######################################################################################################################
# For QAT models, keep finetuning with full_param checkpoint, which keeps the original un-parametrized weights
#MODEL_NAME="QAT5bNT0p1mulPCNetNoBatchNorm_PCConvReLU6_0.0eps_S2NoisyIYAsXZAs0_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_3K1S48C_0.25Dropout_18Layers_2Pool_scanGFI_5REP"
#MODEL_NAME="QAT5bNT0p1mulPCNetNoBatchNorm_PCConvReLU6_0.0eps_S2NoisyIYAsXZAs0_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_3K1S72C_0.25Dropout_13Layers_2Pool_scanGFI_4REP"
#MODEL_NAME="QAT5bNT0p15mulPCNetNoBatchNorm_PCConvReLU6_0.0eps_ODEXInitFFFB_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_3K1S72C_0.25Dropout_13Layers_2Pool_scanGFI_1REP"
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
# Get ODE Block
IFS='_' read -r -a parts <<< "$MODEL_NAME"
ODE_BLK="${parts[3]}"
# No 2
for nt in "${NOISE_TYPES[@]}"; do
  for nl in ${NOISE_LEVELS[$nt]}; do
    for n_bits in "${NBITS[@]}"; do
      for R_max in "${R_MAX_LIST[@]}"; do
        echo "log dir: ${LOGDIR}/train_${EXP}_No_2_ReLU6_2State_${n_bits}_${nl}_${nt}_${R_max}.log"
        python train_ode_cifar.py \
          --task          "${_task}" \
          --ckpt          "${CKPT}" \
          --optim         "SGD" \
          --learning_rate 0.005 \
          --cosine_t0     20 \
          --eval_every    2 \
          --num_epochs    80 \
          --img_type      "scanGFI" \
          --model_name    "${MODEL_NAME}" \
          --offset_eps    0.0 \
          --dropout       0.25 \
          --tie_weights   "false" \
          --tie_bp        "false" \
          --bypass        "false" \
          --batch_size    128 \
          --method        "dopri5" \
          --tol           "1e-6" \
          --t_end         "1.75" \
          --R             "20e3" \
          --R_max         "${R_max}" \
          --C             "49e-15" \
          --v_dd          "0.2" \
          --w_bits        "${n_bits}" \
          --patch_node    "8" \
          --patch_stride  "8" \
          --patch_cycle   "1" \
          --patch_pad     "0" \
          --fold_scalar   "1" \
          --tie_cap       "false" \
          --one_over_q    "6" \
          --qat_cls       "SymQuantizeWeight" \
          --ode_wrapper   "$QAT_WRAPPER" \
          --pc_conv       "PCConvReLU6" \
          --ode_block     "$ODE_BLK" \
          --noise_level   "${nl}" \
          --noise_type    "${nt}" \
          2>&1 | tee "${LOGDIR}/train_${EXP}_No_2_ReLU6_2State_${n_bits}_${nl}_${nt}_${R_max}.log"
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
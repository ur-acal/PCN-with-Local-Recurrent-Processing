#!/usr/bin/env bash

declare -A NOISE_LEVELS=(
  [mul]="0.1"
  [add]="0.05 0.08 0.1 0.15 0.2"
)
NOISE_TYPES=(
  "mul"
#  "add"
)
NBITS=(5)
EXP="NODE_1229_QAT_with_noise_inject_training"
LOGDIR="./logs/${EXP}"
mkdir -p "${LOGDIR}"
#MODEL_NAME="PCNetNoBatchNorm_PCConvReLU6_0.002eps_S2NoMinusZChgZNoisyI_dopri5Solver_1.5TEnd_0.0001Tol_0.001WD_128BS_0.01LR_0.25Dropout_7Layers_2Pool_2REP"
#MODEL_NAME="PCNetNoBatchNorm_PCConvReLU6_0.0eps_S2NoisyIYAsXZAs0_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_3K1S48C_0.25Dropout_18Layers_2Pool_scanGFI_2REP"
#MODEL_NAME="PCNetNoBatchNorm_PCConvReLU6_0.0eps_S2NoisyIYAsXZAs0_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_C100_3K1S80C_0.25Dropout_10Layers_2Pool_scanGFI_1REP"
#######################################################################################################################
# For QAT models, keep finetuning with full_param checkpoint, which keeps the original un-parametrized weights
MODEL_NAME="QAT5bNT0p1mulPCNetNoBatchNorm_PCConvReLU6_0.0eps_S2NoisyIYAsXZAs0_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_3K1S48C_0.25Dropout_18Layers_2Pool_scanGFI_5REP"
#######################################################################################################################
if [[ "$MODEL_NAME" == *C100* ]]; then
  _task="cifar100"
else
  _task="cifar10"
fi
# No 2
for nt in "${NOISE_TYPES[@]}"; do
  for nl in ${NOISE_LEVELS[$nt]}; do
    for n_bits in "${NBITS[@]}"; do
      echo "log dir: ${LOGDIR}/train_${EXP}_No_2_ReLU6_2State_${n_bits}_${nl}_${nt}.log"
      python train_ode_cifar.py \
        --task          "${_task}" \
        --ckpt          "full_param_best" \
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
        --R             "50e3" \
        --R_max         "90e3" \
        --C             "49e-15" \
        --v_dd          "1" \
        --w_bits        "${n_bits}" \
        --patch_node    "8" \
        --patch_stride  "8" \
        --patch_cycle   "1" \
        --patch_pad     "0" \
        --fold_scalar   "1" \
        --tie_cap       "false" \
        --one_over_q    "10" \
        --qat_cls       "SymQuantizeWeight" \
        --ode_wrapper   "QATWrapper2State" \
        --pc_conv       "PCConvReLU6" \
        --ode_block     "S2NoisyIYAsXZAs0" \
        --noise_level   "${nl}" \
        --noise_type    "${nt}" \
        2>&1 | tee "${LOGDIR}/train_${EXP}_No_2_ReLU6_2State_${n_bits}_${nl}_${nt}.log"
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
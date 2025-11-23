#!/usr/bin/env bash


declare -A NOISE_LEVELS=(
  [mul]="0.4"
  [add]="0.05 0.08 0.1 0.15 0.2"
)

NOISE_TYPES=(
  "mul"
#  "add"
)
EXP="NODE_1011_noise_inject_train"
LOGDIR="./logs/${EXP}"
mkdir -p "${LOGDIR}"

# No 2
for nt in "${NOISE_TYPES[@]}"; do
  for nl in ${NOISE_LEVELS[$nt]}; do
    python train_ode_cifar.py \
      --optim         "SGD" \
      --learning_rate 0.0001 \
      --eval_every    2 \
      --lr_reduce_on  "10,20" \
      --num_epochs    20 \
      --img_type      "scanGFI" \
      --model_name    "PCNetNoBatchNorm_PCConvReLU6_0.0eps_S2NoMinusZChgZNoisyI_dopri5Solver_1.5TEnd_0.0001Tol_0.001WD_128BS_0.01LR_0.25Dropout_5Layers_2Pool_scanGFI_3REP" \
      --offset_eps    0.0 \
      --dropout       0.25 \
      --tie_weights   "false" \
      --tie_bp        "false" \
      --bypass        "false" \
      --batch_size    128 \
      --method        "dopri5" \
      --tol           "1e-4" \
      --t_end         "1.5" \
      --pc_conv       "PCConvReLU6" \
      --ode_block     "S2NoMinusZChgZNoisyI" \
      --noise_level   "${nl}" \
      --noise_type    "${nt}" \
      2>&1 | tee "${LOGDIR}/train_${EXP}_No_2_ReLU6_2State_1p5_1e-4.log"
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
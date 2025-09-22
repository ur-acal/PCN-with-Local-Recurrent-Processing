#!/usr/bin/env bash

NBITS=4
EXP="NODE_0921_resume_training_S2NoMinusZChgZNoisyI_7Layers_2Pooling_${NBITS}bit"
LOGDIR="./logs/${EXP}"
mkdir -p "${LOGDIR}"

# No 2
python train_ode_cifar.py \
  --optim         "SGD" \
  --learning_rate 0.0001 \
  --lr_reduce_on  "10,20" \
  --num_epochs    10 \
  --model_name    "PCNetNoBatchNorm_PCConvReLU6_0.002eps_S2NoMinusZChgZNoisyI_dopri5Solver_1.5TEnd_0.0001Tol_0.001WD_128BS_0.01LR_0.25Dropout_7Layers_2Pool_2REP" \
  --offset_eps    0.0 \
  --dropout       0.25 \
  --tie_weights   "false" \
  --tie_bp        "false" \
  --bypass        "false" \
  --batch_size    128 \
  --method        "dopri5" \
  --tol           "0.0001" \
  --t_end         "1.5" \
  --R             "1e5" \
  --C             "49e-12" \
  --w_bits        "$NBITS" \
  --ode_wrapper   "QATWrapper2State" \
  --pc_conv       "PCConvReLU6" \
  --ode_block     "S2NoMinusZChgZNoisyI" \
  2>&1 | tee "${LOGDIR}/train_${EXP}_No_2_ReLU6_2State_1p5_1e-4.log"

echo "Completed."

############################################################
# launch in this way:
# nohup bash launch_scripts/run_ode_train.sh > ./logs/master_single.log 2>&1 &
# tail -f ./logs/master_single.log
# after train finished
# cat ./logs/master_single.log | grep "Train finished" -A 3
############################################################
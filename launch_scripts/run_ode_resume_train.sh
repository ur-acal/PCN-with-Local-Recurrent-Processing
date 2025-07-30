#!/usr/bin/env bash

EXP="NODE_0729_resume_training_ODEFixNoiseOffset_0.4eps"
LOGDIR="./logs/${EXP}"
mkdir -p "${LOGDIR}"

# No 2
python train_ode_cifar.py \
  --optim         "SGD" \
  --learning_rate 0.0001 \
  --lr_reduce_on  "10,20" \
  --num_epochs    10 \
  --model_name    "PCNetNoBatchNorm_PCConvReLU6_0.2eps_ODEFixNoiseOffset_dopri5Solver_0.75TEnd_0.0001Tol_0.001WD_noBPtied_noBP_128BS_0.01LR_0.25Dropout_7Layers_1REP" \
  --offset_eps    0.4 \
  --inp_channels  3  32 32 64 64  128 128 \
  --out_channels  32 32 64 64 128 128 128 \
  --max_pool      0  0  1  0  1   0   0   \
  --dropout       0.25 \
  --tie_weights   "false" \
  --tie_bp        "false" \
  --bypass        "false" \
  --batch_size    128 \
  --method        "dopri5" \
  --tol           "0.0001" \
  --t_end         "0.75" \
  --pcn           "PCNetNoBatchNorm" \
  --pc_conv       "PCConvReLU6" \
  --ode_block     "ODEFixNoiseOffset" \
  2>&1 | tee "${LOGDIR}/train_${EXP}_No_2_HardTanh_minus_y_0p75_1e-4.log"

echo "Completed."

## No 2
#python train_ode_cifar.py \
#  --optim         "SGD" \
#  --num_epochs    150 \
#  --offset_eps    0.25 \
#  --inp_channels  3  32 32 64 64  128 128 \
#  --out_channels  32 32 64 64 128 128 128 \
#  --max_pool      0  0  1  0  1   0   0   \
#  --dropout       0.25 \
#  --tie_weights   "false" \
#  --tie_bp        "false" \
#  --bypass        "false" \
#  --batch_size    128 \
#  --method        "dopri5" \
#  --tol           "0.0001" \
#  --t_end         "0.75" \
#  --pcn           "PCNetNoBatchNorm" \
#  --pc_conv       "PCConvReLU6" \
#  --ode_block     "ODEFixNoiseOffset" \
#  2>&1 | tee "${LOGDIR}/train_${EXP}_No_2_ReLU6_minus_y_0p75_1e-4.log"
#
#echo "Completed."

############################################################
# launch in this way:
# nohup bash launch_scripts/run_ode_train.sh > ./logs/master_single.log 2>&1 &
# tail -f ./logs/master_single.log
# after train finished
# cat ./logs/master_single.log | grep "Train finished" -A 3
############################################################
#!/usr/bin/env bash

EXP="no_bn_pcn_NODE_0718_HardTanh10_ReLU20_minus_y"
LOGDIR="./logs/${EXP}"
mkdir -p "${LOGDIR}"

# No 2 - 5 layer
#python train_ode_cifar.py \
#  --optim         "SGD" \
#  --num_epochs    120 \
#  --inp_channels  3  32 64 64  128 \
#  --out_channels  32 64 64 128 128 \
#  --max_pool      0  1  0  1   0 \
#  --dropout       0.25 \
#  --tie_weights   "false" \
#  --tie_bp        "false" \
#  --bypass        "false" \
#  --batch_size    128 \
#  --lr_reduce_on  "40,80" \
#  --method        "dopri5" \
#  --tol           "0.001" \
#  --t_end         "1.0" \
#  --pcn           "PCNetNoBatchNorm" \
#  --pc_conv       "PCConvHardTanhDyn" \
#  --ode_block     "ODEBlockPCLimitDyn" \
#  2>&1 | tee "${LOGDIR}/train_${EXP}_5l_baseline_hardTanhDyn.log"

# No 2
python train_ode_cifar.py \
  --optim         "SGD" \
  --num_epochs    150 \
  --max_g_norm    1.0 \
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
  --t_end         "1.75" \
  --pcn           "PCNetNoBatchNorm" \
  --pc_conv       "PCConvReLU20" \
  --ode_block     "ODEBlkProj" \
  2>&1 | tee "${LOGDIR}/train_${EXP}_No_2_Relu20_minus_y_1p75_1e-4.log"

echo "Completed."

# No 2
python train_ode_cifar.py \
  --optim         "SGD" \
  --num_epochs    150 \
  --max_g_norm    1.0 \
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
  --t_end         "1.75" \
  --pcn           "PCNetNoBatchNorm" \
  --pc_conv       "PCConvHardTanh10" \
  --ode_block     "ODEBlkProj" \
  2>&1 | tee "${LOGDIR}/train_${EXP}_No_2_HardTanh10_minus_y_1p75_1e-4.log"

echo "Completed."

############################################################
# launch in this way:
# nohup bash launch_scripts/run_ode_train.sh > ./logs/master_single.log 2>&1 &
# tail -f ./logs/master_single.log
# after train finished
# cat ./logs/master_single.log | grep "Train finished" -A 3
############################################################
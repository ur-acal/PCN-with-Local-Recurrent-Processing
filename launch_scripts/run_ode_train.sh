#!/usr/bin/env bash

EXP="no_bn_pcn_NODE_0722_ODEBlockPCMinusY_QAT_INT8_TEST"
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

# Task 1 - QAT Training with HardTanh (run in background) - QUICK TEST
python train_ode_cifar.py \
  --optim         "SGD" \
  --num_epochs    3 \
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
  --t_end         "0.1" \
  --pcn           "PCNetNoBatchNorm" \
  --pc_conv       "PCConvHardTanh" \
  --ode_block     "ODEBlockPCMinusY" \
  --qat           "true" \
  --qat_backend   "fbgemm" \
  --qat_start_epoch 1 \
  --subset_fraction 0.05 \
  2>&1 | tee "${LOGDIR}/train_${EXP}_No_2_HardTanh_minus_y_QAT_INT8_TEST.log" &

echo "Started Task 1 (HardTanh) in background with PID: $!"

# Task 2 - QAT Training with ReLU6 (run in background) - QUICK TEST
python train_ode_cifar.py \
  --optim         "SGD" \
  --num_epochs    3 \
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
  --t_end         "0.1" \
  --pcn           "PCNetNoBatchNorm" \
  --pc_conv       "PCConvReLU6" \
  --ode_block     "ODEBlockPCMinusY" \
  --qat           "true" \
  --qat_backend   "fbgemm" \
  --qat_start_epoch 1 \
  --subset_fraction 0.05 \
  2>&1 | tee "${LOGDIR}/train_${EXP}_No_2_ReLU6_minus_y_QAT_INT8_TEST.log" &

echo "Started Task 2 (ReLU6) in background with PID: $!"

# Wait for both tasks to complete
echo "Waiting for both training tasks to complete..."
wait

echo "Both tasks completed!"

############################################################
# launch in this way (both tasks run in parallel):
# nohup bash launch_scripts/run_ode_train.sh > ./logs/master_parallel.log 2>&1 &
#
# Monitor progress:
# tail -f ./logs/master_parallel.log
# tail -f ./logs/no_bn_pcn_NODE_0722_ODEBlockPCMinusY_QAT_INT8/train_*_HardTanh_*.log
# tail -f ./logs/no_bn_pcn_NODE_0722_ODEBlockPCMinusY_QAT_INT8/train_*_ReLU6_*.log
#
# Check when both tasks finish:
# cat ./logs/master_parallel.log | grep "Both tasks completed"
# cat ./logs/no_bn_pcn_NODE_0722_ODEBlockPCMinusY_QAT_INT8/train_*_*.log | grep "Train finished" -A 3
############################################################
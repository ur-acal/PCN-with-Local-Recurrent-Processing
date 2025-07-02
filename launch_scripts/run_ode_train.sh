#!/usr/bin/env bash

EXP="no_bn_pcn_NODE_0701_1TEnd_hardTanhDyn"
LOGDIR="./logs/${EXP}"
mkdir -p "${LOGDIR}"

# No 2 - 5 layer
python train_ode_cifar.py \
  --optim         "SGD" \
  --num_epochs    150 \
  --inp_channels  3  32 64 64  128 \
  --out_channels  32 64 64 128 128 \
  --max_pool      0  1  0  1   0 \
  --dropout       0.25 \
  --tie_weights   "false" \
  --tie_bp        "false" \
  --bypass        "false" \
  --batch_size    128 \
  --method        "dopri5" \
  --tol           "0.001" \
  --t_end         "1.0" \
  --pcn           "PCNetNoBatchNorm" \
  --pc_conv       "PCConvHardTanhDyn" \
  --ode_block     "ODEBlockPCLimitDyn" \
  2>&1 | tee "${LOGDIR}/train_${EXP}_5l_baseline_hardTanhDyn.log"

# No 2
python train_ode_cifar.py \
  --optim         "SGD" \
  --num_epochs    150 \
  --inp_channels  3  32 32 64 64  128 128 \
  --out_channels  32 32 64 64 128 128 128 \
  --max_pool      0  0  1  0  1   0   0   \
  --dropout       0.25 \
  --tie_weights   "false" \
  --tie_bp        "false" \
  --bypass        "false" \
  --batch_size    128 \
  --method        "dopri5" \
  --tol           "0.001" \
  --t_end         "1.0" \
  --pcn           "PCNetNoBatchNorm" \
  --pc_conv       "PCConvHardTanhDyn" \
  --ode_block     "ODEBlockPCLimitDyn" \
  2>&1 | tee "${LOGDIR}/train_${EXP}_No_2_hardTanhDyn.log"

# No 2
#python train_cifar.py \
#  --optim         "SGD" \
#  --num_epochs    150 \
#  --inp_channels  3  16 16 16 32 32 32 64 64 \
#  --out_channels  16 16 16 32 32 32 64 64 64 \
#  --max_pool      0  0  0  1  0  0  1  0  0  \
#  --dropout       0.25 \
#  --lr_pc         0.15 \
#  --cls           5 \
#  --tie_weights   "false" \
#  --tie_bp        "false" \
#  --relu_between  "true" \
#  --bypass        "false" \
#  --batch_size    128 \
#  --pcn           "PCNetNoBatchNorm" \
#  --pc_conv       "PCConvHardTanhLimit" \
#  2>&1 | tee "${LOGDIR}/train_${EXP}_No_2_hardtanh.log"

# No 2
#python train_cifar.py \
#  --optim         "SGD" \
#  --num_epochs    150 \
#  --inp_channels  3  32 64  128 \
#  --out_channels  32 64 128 128 \
#  --max_pool      0  1  1   0 \
#  --dropout       0.25 \
#  --lr_pc         0.15 \
#  --cls           5 \
#  --tie_weights   "false" \
#  --tie_bp        "false" \
#  --relu_between  "true" \
#  --bypass        "false" \
#  --batch_size    128 \
#  --pcn           "PCNetNoBatchNorm" \
#  --pc_conv       "PCConvHardTanhLimit" \
#  2>&1 | tee "${LOGDIR}/train_${EXP}_No_2_hardtanh_limit_128_chan.log"

## No 2
#python train_cifar.py \
#  --optim         "SGD" \
#  --num_epochs    150 \
#  --inp_channels  3  32 32 64 64  128 128 \
#  --out_channels  32 32 64 64 128 128 128 \
#  --max_pool      0  0  1  0  1   0   0   \
#  --dropout       0.25 \
#  --lr_pc         0.15 \
#  --cls           5 \
#  --tie_weights   "false" \
#  --tie_bp        "false" \
#  --relu_between  "true" \
#  --bypass        "false" \
#  --batch_size    128 \
#  --pcn           "PCNetNoBatchNorm" \
#  --pc_conv       "PCConvReLU6Limit" \
#  2>&1 | tee "${LOGDIR}/train_${EXP}_No_2_RELU6_limit.log"

#
## 7l-baseline
#python train_cifar.py \
#  --optim         "SGD" \
#  --num_epochs    150 \
#  --inp_channels  3  32 32 64 64  128 128 \
#  --out_channels  32 32 64 64 128 128 128 \
#  --max_pool      0  0  1  0  1   0   0   \
#  --dropout       0.25 \
#  --lr_pc         0.15 \
#  --cls           5 \
#  --tie_weights   "false" \
#  --tie_bp        "false" \
#  --relu_between  "true" \
#  --bypass        "true" \
#  --batch_size    128 \
#  --pcn           "PCNetNoBatchNorm" \
#  --pc_conv       "PCConvScaled" \
#  2>&1 | tee "${LOGDIR}/train_${EXP}_7l_baseline.log"
##
## 5l-baseline
#python train_cifar.py \
#  --optim         "SGD" \
#  --num_epochs    150 \
#  --inp_channels  3  32 64 64  128 \
#  --out_channels  32 64 64 128 128 \
#  --max_pool      0  1  0  1   0 \
#  --warmup_epoch  1 \
#  --dropout       0.25 \
#  --lr_pc         0.2 \
#  --cls           30 \
#  --tie_weights   "false" \
#  --tie_bp        "false" \
#  --relu_between  "true" \
#  --bypass        "true" \
#  --batch_size    128 \
#  --pcn           "PCNetNoBatchNorm" \
#  --pc_conv       "PCConvScaled" \
#  2>&1 | tee "${LOGDIR}/train_${EXP}_5l_baseline.log"
##
## No 6
#python train_cifar.py \
#  --optim         "SGD" \
#  --num_epochs    150 \
#  --warmup_epoch  1 \
#  --inp_channels  3  32 32 64 64  128 128 \
#  --out_channels  32 32 64 64 128 128 128 \
#  --max_pool      0  0  1  0  1   0   0   \
#  --dropout       0.25 \
#  --lr_pc         0.15 \
#  --cls           5 \
#  --tie_weights   "false" \
#  --tie_bp        "true" \
#  --relu_between  "true" \
#  --bypass        "true" \
#  --batch_size    128 \
#  --pcn           "PCNetNoBatchNorm" \
#  --pc_conv       "PCConvScaled" \
#  2>&1 | tee "${LOGDIR}/train_${EXP}_No_6.log"

#python train_cifar.py \
#  --optim         "SGD" \
#  --num_epochs    150 \
#  --inp_channels  3  32 32 64 64 64  128 128 128 \
#  --out_channels  32 32 64 64 64 128 128 128 256 \
#  --max_pool      0  0  1  0  0  1   0   0   0   \
#  --lr_pc         1 \
#  --cls           5 \
#  --tie_weights   "true" \
#  --tie_bp        "false" \
#  --relu_between  "false" \
#  --bypass        "false" \
#  --use_pc        "false" \
#2>&1 | tee "${LOGDIR}/train_yes_no_no_no_noPC.log"

#EXP="tw_false_tbp_false_relu_true_bp_true_9_layer" # ~10M params
#LOGDIR="./logs/${EXP}"
#mkdir -p "${LOGDIR}"
#
#python train_cifar.py \
#  --optim         "SGD" \
#  --num_epochs    300 \
#  --lr_pc         1 \
#  --cls           5 \
#  --tie_weights   "false" \
#  --tie_bp        "false" \
#  --relu_between  "true" \
#  --bypass        "true" \
#  2>&1 | tee "${LOGDIR}/train.log"

echo "Completed."

############################################################
# launch in this way:
# nohup bash run_single_act.sh > ./logs/master_single.log 2>&1 &
# tail -f ./logs/master_single.log
# after train finished
# cat ./logs/master_single.log | grep "Train finished" -A 3
############################################################
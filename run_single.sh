#!/usr/bin/env bash

EXP="bn_after_conv_0528"
LOGDIR="./logs/${EXP}"
mkdir -p "${LOGDIR}"

python train_cifar.py \
  --optim         "SGD" \
  --num_epochs    150 \
  --inp_channels  3  32 32 64 64  128 128 \
  --out_channels  32 32 64 64 128 128 128 \
  --max_pool      0  0  1  0  1   0   0   \
  --lr_pc         1 \
  --cls           5 \
  --tie_weights   "false" \
  --tie_bp        "false" \
  --relu_between  "true" \
  --bypass        "true" \
  --test_only     "true" \
  2>&1 | tee "${LOGDIR}/train_7l_baseline.log"

python train_cifar.py \
  --optim         "SGD" \
  --num_epochs    150 \
  --inp_channels  3  32 64 64  128 \
  --out_channels  32 64 64 128 128 \
  --max_pool      0  1  0  1   0 \
  --lr_pc         1 \
  --cls           5 \
  --tie_weights   "false" \
  --tie_bp        "false" \
  --relu_between  "true" \
  --bypass        "true" \
  2>&1 | tee "${LOGDIR}/train_5l_baseline.log"

python train_cifar.py \
  --optim         "SGD" \
  --num_epochs    150 \
  --inp_channels  3  32 32 64 64  128 128 \
  --out_channels  32 32 64 64 128 128 128 \
  --max_pool      0  0  1  0  1   0   0   \
  --lr_pc         1 \
  --cls           5 \
  --tie_weights   "false" \
  --tie_bp        "false" \
  --relu_between  "true" \
  --bypass        "false" \
  2>&1 | tee "${LOGDIR}/train_untie_No_2.log"

python train_cifar.py \
  --optim         "SGD" \
  --num_epochs    150 \
  --inp_channels  3  32 32 64 64  128 128 \
  --out_channels  32 32 64 64 128 128 128 \
  --max_pool      0  0  1  0  1   0   0   \
  --lr_pc         1 \
  --cls           5 \
  --tie_weights   "false" \
  --tie_bp        "true" \
  --relu_between  "true" \
  --bypass        "true" \
  2>&1 | tee "${LOGDIR}/train_untie_No_6.log"

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
# nohup bash run_single.sh > ./logs/master_single.log 2>&1 &
# tail -f ./logs/master_single.log
# after train finished
# cat ./logs/master_single.log | grep "Train finished" -A 3
############################################################
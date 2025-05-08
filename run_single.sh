#!/usr/bin/env bash

EXP="tw_false_tbp_false_relu_true_bp_true_left_exp_0508" # verified with 0.91 val acc
LOGDIR="./logs/${EXP}"
mkdir -p "${LOGDIR}"

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
  2>&1 | tee "${LOGDIR}/train_baseline_5l.log"

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
  --relu_between  "false" \
  --bypass        "true" \
  2>&1 | tee "${LOGDIR}/train_baseline_no_relu_5l.log"

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

# launch in this way:
# nohup bash run_single.sh > ./logs/master_single.log 2>&1 &
# tail -f ./logs/master_single.log

#!/usr/bin/env bash

#EXP="tw_false_tbp_false_relu_true_bp_false"
#LOGDIR=./logs/"${EXP}"
#mkdir -p "${LOGDIR}"
#
#python train_cifar.py \
#  --num_epochs    5 \
#  --tie_weights   "false" \
#  --tie_bp        "false" \
#  --relu_between  "true" \
#  --bypass        "false" \
#  2>&1 | tee "${LOGDIR}/train.log"


EXP="tw_false_tbp_false_relu_true_bp_true_sanity_check"
LOGDIR=./logs/"${EXP}"
mkdir -p "${LOGDIR}"

python train_cifar.py \
  --optim         "SGD" \
  --num_epochs    50 \
  --inp_channels  3  32 64 64  128 \
  --out_channels  32 64 64 128 128 \
  --max_pool      0  1  0  1   0 \
  --lr_pc         0.01 \
  --cls           30 \
  --tie_weights   "false" \
  --tie_bp        "false" \
  --relu_between  "true" \
  --bypass        "true" \
  2>&1 | tee "${LOGDIR}/train.log"

echo "Completed."

# launch in this way:
# nohup bash run_single.sh > ./logs/master_single.log 2>&1 &
# tail -f ./logs/master_single.log

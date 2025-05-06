#!/usr/bin/env bash

EXP="tw_${tie_w}_tb_${tie_bp}_r_${relu}_b_${bypass}"
LOGDIR=./logs/"${EXP}"
mkdir -p "${LOGDIR}"

python train_cifar.py \
  --num_epochs    2 \
  --tie_weights   "false" \
  --tie_bp        "false" \
  --relu_between  "true" \
  --bypass        "false" \
  2>&1 | tee "${LOGDIR}/train.log"

echo "Completed."

# launch in this way:
# nohup bash run_single.sh > /dev/null 2>&1 &
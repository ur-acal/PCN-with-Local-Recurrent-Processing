#!/usr/bin/env bash

# This script runs experiments with (ff/fb) weights decoupled: tie_weights=false
# 1. without bp and with/without relu in between
# 2. with bp but with bp weights tied with ff/fb and with/without relu in between

EPOCH=2

for tie_bp in true false; do
  for relu in true false; do
    for bypass in true false; do
      if [ "$tie_bp" = "false" ] && [ "$bypass" = "true" ] && [ "$relu" = "true" ]; then
        echo "----- Skipping tie_bp=false & bypass=true & relu=true -----"
        continue
      fi

      if [ "$tie_bp" = "true" ] && [ "$bypass" = "false" ]; then
        echo "----- Skipping tie_bp=true & bypass=false -----"
        continue
      fi
      EXP="tw_false_tbp_${tie_bp}_relu_${relu}_bp_${bypass}_${EPOCH}_epochs"
      LOGDIR="./logs/${EXP}"
      mkdir -p "${LOGDIR}"
      echo "=== Running ${EXP} at $(date) ==="
      python train_cifar.py \
        --optim         "SGD" \
        --num_epochs    "${EPOCH}" \
        --inp_channels  3  32 32 64 64  128 128 \
        --out_channels  32 32 64 64 128 128 128 \
        --max_pool      0  0  1  0  1   0   0   \
        --lr_pc         1 \
        --cls           5 \
        --tie_weights   "false" \
        --tie_bp        "${tie_bp}" \
        --relu_between  "${relu}" \
        --bypass        "${bypass}" \
        2>&1 | tee "${LOGDIR}/train.log"
      echo ">>> Finished ${EXP} at $(date) <<<"
    done
  done
done

echo "Completed."

# launch in this way:
# nohup bash run_single.sh > ./logs/master_single.log 2>&1 &
# tail -f ./logs/master_single.log

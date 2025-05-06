#!/usr/bin/env bash

for tie_w in true false; do
  for tie_bp in true false; do
    for relu in true false; do
      for bypass in true false; do

        if [ "$tie_bp" = "true" ] && [ "$bypass" = "false" ]; then
          echo "----- Skipping tie_bp=true & bypass=false -----"
          continue
        fi

        EXP="tw_${tie_w}_tb_${tie_bp}_r_${relu}_b_${bypass}"
        LOGDIR=./logs/"${EXP}"
        mkdir -p "${LOGDIR}"

        echo "=== Running ${EXP} at $(date) ==="
        python train_cifar.py \
          --tie_weights   "${tie_w}" \
          --tie_bp        "${tie_bp}" \
          --relu_between  "${relu}" \
          --bypass        "${bypass}" \
          2>&1 | tee "${LOGDIR}/train.log"

        echo ">>> Finished ${EXP} at $(date) <<<"
      done
    done
  done
done

echo "All runs completed."

# launch in this way:
# nohup bash run_multiple.sh > /dev/null 2>&1 &

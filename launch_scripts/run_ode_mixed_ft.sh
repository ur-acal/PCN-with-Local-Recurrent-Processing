#!/usr/bin/env bash

declare -A NOISE_LEVELS=(
  [mul]="0.4"
  [add]="0.05 0.08 0.1 0.15 0.2"
)
NOISE_TYPES=(
  "mul"
#  "add"
)
NBITS=(5)
EXP="NODE_1031_noise_inject_training"
LOGDIR="./logs/${EXP}"
mkdir -p "${LOGDIR}"

# Mixed finetune with hardware constraint and noise-inject trainig
# use ODEWrapperRC only
ODE_WRAPPER="ODEWrapperRC"
# Fill with a trained model
MODEL_NAME=""
for nt in "${NOISE_TYPES[@]}"; do
  for nl in ${NOISE_LEVELS[$nt]}; do
    for n_bits in "${NBITS[@]}"; do
      echo "log dir: ${LOGDIR}/train_${EXP}_No_2_ReLU6_2State_${n_bits}_${nl}_${nt}.log"
      python train_ode_cifar.py \
        --optim         "SGD" \
        --learning_rate 0.01 \
        --cosine_t0     10 \
        --eval_every    2 \
        --num_epochs    50 \
        --img_type      "scanGFI" \
        --model_name    "${MODEL_NAME}" \
        --offset_eps    0.0 \
        --dropout       0.25 \
        --tie_weights   "false" \
        --tie_bp        "false" \
        --bypass        "false" \
        --batch_size    128 \
        --method        "dopri5" \
        --tol           "1e-4" \
        --t_end         "1.5" \
        --R             "1e5" \
        --C             "49e-15" \
        --v_dd          "1" \
        --w_bits        "${n_bits}" \
        --ode_wrapper   "${ODE_WRAPPER}" \
        --pc_conv       "PCConvReLU6" \
        --ode_block     "ODEBlockXInit" \
        --noise_level   "${nl}" \
        --noise_type    "${nt}" \
        2>&1 | tee "${LOGDIR}/train_${EXP}_No_2_ReLU6_${n_bits}_${nl}_${nt}.log"
    done
  done
done
echo "Completed."

############################################################
# launch in this way:
# nohup bash launch_scripts/run_ode_train.sh > ./logs/master_single.log 2>&1 &
# tail -f ./logs/master_single.log
# after train finished
# cat ./logs/master_single.log | grep "Train finished" -A 3
############################################################
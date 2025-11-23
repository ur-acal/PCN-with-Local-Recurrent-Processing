#!/usr/bin/env bash

EXP="NODE_1104_KD"
LOGDIR="./logs/${EXP}"
mkdir -p "${LOGDIR}"

KD_WEIGHTS=(
  "1,0.5,0"
#  "0.8,0.2,0"
#  "0.5,0.5,0"
#  "0.1,0.9,0"
)
KD_T_LIST=(4)
#KD_T_LIST=( 3 4 5 6 )

# Teacher params
TEACHER="PCNetNoBatchNorm_PCConvReLU6_0.0eps_S2NoMinusZChgZNoisyI_dopri5Solver_1.5TEnd_0.0001Tol_0.001WD_128BS_0.01LR_3K1S128C_0.25Dropout_10Layers_1Pool_scanGFI_1REP"
__rest="${TEACHER#*_}"
t_pc_conv="${__rest%%_*}"
t_method="dopri5"
t_tol="1e-6"
t_n_steps="20"
#t_wrapper="ODEWrapper2State"
t_wrapper=""

STRIDE=(1)
KSZ=(3)
PADDING=1 # For ksz=5, padding=2; o.w. padding=1
for kd_w in "${KD_WEIGHTS[@]}"; do
  for kd_t in ${KD_T_LIST[$nt]}; do
    echo "log dir: ${LOGDIR}/train_${EXP}_No_2_ReLU6_2State_${kd_w}_${kd_t}.log"
    python train_ode_cifar.py \
    --optim         "SGD" \
    --img_type      "scanGFI" \
    --num_epochs    150 \
    --eval_every    2 \
    --offset_eps    0.0 \
    --inp_channels  4  16 16 32 64 \
    --out_channels  16 16 32 64 64 \
    --max_pool      0  0  1  0  0 \
    --stride        "${STRIDE[@]}" \
    --kernel_size   "${KSZ[@]}" \
    --padding       "${PADDING}" \
    --dropout       0.25 \
    --tie_weights   "false" \
    --tie_bp        "false" \
    --bypass        "false" \
    --batch_size    128 \
    --method        "dopri5" \
    --tol           "0.0001" \
    --t_end         "1.5" \
    --pcn           "PCNetNoBatchNorm" \
    --pc_conv       "PCConvReLU6" \
    --ode_block     "S2NoMinusZChgZNoisyI" \
    --teacher       "${TEACHER}" \
    --kd_type       "VanillaKD" \
    --t_pc_conv     "${t_pc_conv}" \
    --t_method      "${t_method}" \
    --t_tol         "${t_tol}" \
    --t_n_steps     "${t_n_steps}" \
    --t_wrapper     "${t_wrapper}" \
    --distill_T     "${kd_t}" \
    --distill_w     "${kd_w}" \
    2>&1 | tee "${LOGDIR}/train_${EXP}_No_2_ReLU6_2State_${kd_w}_${kd_t}.log"
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
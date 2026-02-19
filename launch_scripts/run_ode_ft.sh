#!/bin/bash
#SBATCH -p ds4ai
#SBATCH -c 16
#SBATCH -t 24:00:00
#SBATCH --mem=32gb
#SBATCH --gres=gpu:1
#SBATCH --output=logs/test_%j.out
#SBATCH --error=logs/test_%j.err
#SBATCH --job-name=PCN_Noise

export SCANGEN_DATA_ROOT=/scratch/tgeng_lab/sun/projs/ODE_CIFAR10/data
DATASET_NAME="${DATASET_NAME:-cifar10}"
declare -A NOISE_LEVELS=(
  [mul]="0.2"
  [add]="0.05 0.08 0.1 0.15 0.2"
)
NOISE_TYPES=(
  "mul"
#  "add"
)
NBITS=(5)
EXP="KD_CRD_PCN_Noise_Inject"
LOGDIR="./logs/${EXP}"
mkdir -p "${LOGDIR}"

# Mixed finetune with hardware constraint and noise-inject trainig
# use ODEWrapperRC only
ODE_WRAPPER="ODEWrapper2State"
# Fill with a trained model
# MODEL_NAME="PCNetNoBatchNorm_PCConvReLU6_0.0eps_S2NoisyIYAsXZAs0_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_3K1S48C_0.25Dropout_18Layers_2Pool_scanGFI_1REP"
MODEL_NAME="PCNetNoBatchNorm_PCConvReLU6_0.0eps_S2NoisyIYAsXZAs0_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_3K1S48C_0.25Dropout_18Layers_2Pool_kd_crdDistill_a0p3_t2p0_scanGFI_1REP"
for nt in "${NOISE_TYPES[@]}"; do
  for nl in ${NOISE_LEVELS[$nt]}; do
    for n_bits in "${NBITS[@]}"; do
      echo "log dir: ${LOGDIR}/train_${EXP}_No_2_ReLU6_2State_${n_bits}_${nl}_${nt}.log"
      python train_ode_cifar.py \
        --optim         "SGD" \
        --learning_rate 0.01 \
        --cosine_t0     10 \
        --eval_every    1 \
        --num_epochs    80 \
        --img_type      "scanGFI" \
        --dataset       "${DATASET_NAME}" \
        --model_name    "${MODEL_NAME}" \
        --offset_eps    0.0 \
        --dropout       0.25 \
        --tie_weights   "false" \
        --tie_bp        "false" \
        --bypass        "false" \
        --batch_size    128 \
        --method        "dopri5" \
        --tol           "1e-6" \
        --t_end         "1.75" \
        --R             "1e5" \
        --C             "49e-15" \
        --v_dd          "1" \
        --w_bits        "${n_bits}" \
        --ode_wrapper   "${ODE_WRAPPER}" \
        --pc_conv       "PCConvReLU6" \
        --ode_block     "S2NoisyIYAsXZAs0" \
        --noise_level   "${nl}" \
        --noise_type    "${nt}" \
        --teacher_ckpt checkpoint/ckpt_mismatch.pth \
        --teacher_arch efficientnet-b4 \
        --distill_method none \
        --distill_alpha 0.3 \
        --distill_temperature 2.0 \
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

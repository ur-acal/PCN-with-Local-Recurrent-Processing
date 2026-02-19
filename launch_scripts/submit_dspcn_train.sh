#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -G 1
#SBATCH -q regular
#SBATCH -J DS-PCN-cifar100
#SBATCH --mail-user=rsong10@ur.rochester.edu
#SBATCH --mail-type=ALL
#SBATCH -A m4243
#SBATCH -t 24:0:0
#SBATCH --cpus-per-task=32


# OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

conda activate one

export SCANGEN_DATA_ROOT=/pscratch/sd/r/rsong10/data
DATASET_NAME="${DATASET_NAME:-cifar100}"
EXP="DS_PCN"
LOGDIR="./logs/${EXP}"
mkdir -p "${LOGDIR}"
TEACHER_CKPT="${TEACHER_CKPT:-checkpoint/b4_100.pth}"
TEACHER_ARCH="${TEACHER_ARCH:-efficientnet_v2_l}"
TEACHER_ARCH_SOURCE="${TEACHER_ARCH_SOURCE:-auto}"
TEACHER_INPUT_SIZE="${TEACHER_INPUT_SIZE:-224}"
TEACHER_CENTER_CROP="${TEACHER_CENTER_CROP:-true}"

INP=(4 32 64 64 64)
OUT=(32 64 64 64 64)
POOL=(0  1  0  1  0)
# INP=(4  12 12 12 12 12 12 24 24 24 24 24 48 48 48 48 48 48)
# OUT=(12 12 12 12 12 12 24 24 24 24 24 48 48 48 48 48 48 48)
# POOL=(0 0  0  0  0  1  0  0  0  0  1  0  0  0  0  0  0  0)
STRIDE=(1)
KSZ=(3)
PADDING=1
NOISE_LEVEL=0.
# S model (0.26 M)
python train_ode_cifar.py \
  --optim         "SGD" \
  --img_type      "scanGFI" \
  --rggb_to_rgb   "true" \
  --dataset       "${DATASET_NAME}" \
  --num_epochs    150 \
  --eval_every    2 \
  --offset_eps    0.0 \
  --inp_channels  "${INP[@]}" \
  --out_channels  "${OUT[@]}" \
  --max_pool      "${POOL[@]}" \
  --kernel_size   "${KSZ[@]}" \
  --padding       "${PADDING}" \
  --stride        "${STRIDE[@]}" \
  --dropout       0.0 \
  --weight_decay  "1e-4" \
  --lr_reduce_on  "80,122" \
  --tie_weights   "false" \
  --tie_bp        "false" \
  --bypass        "false" \
  --batch_size    128 \
  --method        "dopri5" \
  --tol           "0.0001" \
  --t_end         "1.5" \
  --noise_level   "${NOISE_LEVEL}" \
  --noise_type    "mul" \
  --pcn           "PCNetNoBatchNorm" \
  --pc_conv       "PCConvReLU6" \
  --ode_block     "ODEBlockXInit" \
  --teacher_ckpt "${TEACHER_CKPT}" \
  --teacher_arch "${TEACHER_ARCH}" \
  --teacher_arch_source "${TEACHER_ARCH_SOURCE}" \
  --teacher_input_size "${TEACHER_INPUT_SIZE}" \
  --teacher_center_crop "${TEACHER_CENTER_CROP}" \
  --distill_method none \
  --distill_alpha 0.3 \
  --distill_temperature 2.0 \
  2>&1 | tee "${LOGDIR}/train_${EXP}_ODEBlockXInit.log"

# # M model (0.57 M)
# python train_ode_cifar.py \
#   --optim         "SGD" \
#   --img_type      "scanGFI" \
#   --dataset       "${DATASET_NAME}" \
#   --num_epochs    150 \
#   --eval_every    1 \
#   --offset_eps    0.0 \
#   --inp_channels  4  32 32 64 64  128 \
#   --out_channels  32 32 64 64 128 128 \
#   --max_pool      0  0  1  0  1   0  \
#   --kernel_size   "${KSZ[@]}" \
#   --padding       "${PADDING}" \
#   --stride        "${STRIDE[@]}" \
#   --dropout       0.0 \
#   --weight_decay  "1e-4" \
#   --lr_reduce_on  "80,122" \
#   --tie_weights   "false" \
#   --tie_bp        "false" \
#   --bypass        "false" \
#   --batch_size    128 \
#   --method        "dopri5" \
#   --tol           "0.0001" \
#   --t_end         "1.5" \
#   --pcn           "PCNetNoBatchNorm" \
#   --pc_conv       "PCConvReLU6" \
#   --ode_block     "ODEBlockXInit" \
#   2>&1 | tee "${LOGDIR}/train_${EXP}_ODEBlockXInit.log"

# # L model (0.86 M)
# python train_ode_cifar.py \
#   --optim         "SGD" \
#   --img_type      "scanGFI" \
#   --dataset       "${DATASET_NAME}" \
#   --num_epochs    150 \
#   --eval_every    1 \
#   --offset_eps    0.0 \
#   --inp_channels  4  32 32 64 64  128 128 \
#   --out_channels  32 32 64 64 128 128 128 \
#   --max_pool      0  0  1  0  1   0   0  \
#   --kernel_size   "${KSZ[@]}" \
#   --padding       "${PADDING}" \
#   --stride        "${STRIDE[@]}" \
#   --dropout       0.0 \
#   --weight_decay  "1e-4" \
#   --lr_reduce_on  "80,122" \
#   --tie_weights   "false" \
#   --tie_bp        "false" \
#   --bypass        "false" \
#   --batch_size    128 \
#   --method        "dopri5" \
#   --tol           "0.0001" \
#   --t_end         "1.5" \
#   --pcn           "PCNetNoBatchNorm" \
#   --pc_conv       "PCConvReLU6" \
#   --ode_block     "ODEBlockXInit" \
#   2>&1 | tee "${LOGDIR}/train_${EXP}_ODEBlockXInit.log"

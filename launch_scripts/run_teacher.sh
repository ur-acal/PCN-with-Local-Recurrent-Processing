#!/bin/bash
#SBATCH -p ds4ai
#SBATCH -c 16
#SBATCH -t 24:00:00
#SBATCH --mem=32gb
#SBATCH --gres=gpu:1
#SBATCH --output=logs/test_%j.out
#SBATCH --error=logs/test_%j.err
#SBATCH --job-name=Teacher

export SCANGEN_DATA_ROOT=/scratch/tgeng_lab/sun/projs/ODE_CIFAR10/data
DATASET_NAME="${DATASET_NAME:-cifar100}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-efficientnet_v2_l_in21k_cifar100.pth}"
ARCH_SOURCE="${ARCH_SOURCE:-auto}"

python train_teacher.py \
  --dataset "${DATASET_NAME}" \
  --arch efficientnet_v2_l \
  --arch_source "${ARCH_SOURCE}" \
  --init_checkpoint "${INIT_CHECKPOINT}" \
  --lr 0.002 \
  --gamma 0.1 \
  --wd 1e-6 \
  --ne 100 \
  --nsc 10 \
  --batch_split 1 \
  --batch 32 \
  --alpha 0 \
  --train_transform cifar \
  --train_size 224 \
  --test_size 224 \
  --test_center_crop \
  --mismatch_levels 0. \
  --mismatch_type mul \
  --root ${SCANGEN_DATA_ROOT} \

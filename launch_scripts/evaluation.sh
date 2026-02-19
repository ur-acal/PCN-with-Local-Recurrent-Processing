#!/bin/bash
#SBATCH -p ds4ai
#SBATCH -c 16
#SBATCH -t 24:00:00
#SBATCH --mem=32gb
#SBATCH --gres=gpu:1
#SBATCH --output=logs/test_%j.out
#SBATCH --error=logs/test_%j.err
#SBATCH --job-name=PCN_EVAL

export SCANGEN_DATA_ROOT=/scratch/tgeng_lab/sun/projs/ODE_CIFAR10/data
DATASET_NAME="${DATASET_NAME:-cifar10}"

MODEL=QAT5bNT0p4mulPCNetNoBatchNorm_PCConvReLU6_0.0eps_ODEBlockXInit_dopri5Solver_1.5TEnd_0.0001Tol_0.0001WD_128BS_0.01LR_3K1S64C_0.0Dropout_5Layers_2Pool_kd_crdDistill_a0p3_t2p0_scanGFI_1REP
python ode_inference.py \
  --model_dir saved_ckpt \
  --model_name "$MODEL" \
  --ckpt best \
  --ode_block ODEBlockXInit \
  --pc_conv PCConvReLU6Noisy \
  --img_type scanGFI \
  --dataset "${DATASET_NAME}" \
  --method dopri5 \
  --tol 1e-4 \
  --t_end 1.75 \
  --n_steps 15 \
  --ts_scale 1 \
  --d_start 0.2 \
  --d_end 0.4 \
  --n_sweep_left 1 \
  --n_sweep_right 2

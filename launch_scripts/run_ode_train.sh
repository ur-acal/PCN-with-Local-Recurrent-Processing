#!/bin/bash -l
#SBATCH -N 1
#SBATCH -p ising
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16            # modest CPU request so the node can be shared
#SBATCH --gres=gpu:1                  # exactly ONE GPU; allows packing on 4-GPU nodes
#SBATCH -t 60:10:00
#SBATCH --nodelist=bhgrb4x0081,bhgrb4x0082
#SBATCH -o /scratch/rzeng7/repos/PCN-with-Local-Recurrent-Processing/logs/slurm_jobs/slurm_%j.out

# OpenMP settings:
#export OMP_NUM_THREADS=1
#export OMP_PLACES=threads
#export OMP_PROC_BIND=spread

IS_SLURM="${IS_SLURM:-0}"

if [[ "${IS_SLURM}" == 1 ]]; then
  source activate base
  conda activate scanbase
fi

ORIG_T_INP="${ORIG_T_INP:-false}"
DATASET_NAME="${DATASET_NAME:-cifar100}"
EXP="DS_PCN"
LOGDIR="./logs/${EXP}"
mkdir -p "${LOGDIR}"

if [[ "${ORIG_T_INP}" == "true" ]]; then
  TEACHER_CKPT="${TEACHER_CKPT:-checkpoint/efficientnet_v2_l_${DATASET_NAME}.pth}"
  TEACHER_ARCH="${TEACHER_ARCH:-efficientnet_v2_l}"
else
  TEACHER_CKPT="${TEACHER_CKPT:-checkpoint/b4_100.pth}"
  TEACHER_ARCH="${TEACHER_ARCH:-efficientnet_v2_l}"
  if [[ "${DATASET_NAME}" == "cifar10" ]]; then
    TEACHER_CKPT="checkpoint/b4.pth"
    TEACHER_ARCH="efficientnet-b4"
  fi
fi
TEACHER_ARCH_SOURCE="${TEACHER_ARCH_SOURCE:-auto}"
TEACHER_INPUT_SIZE="${TEACHER_INPUT_SIZE:-224}"
TEACHER_CENTER_CROP="${TEACHER_CENTER_CROP:-true}"

NEG_SAMPLE="${NEG_SAMPLE:-index}"
CONTRAST_METHOD="${CONTRAST_METHOD:-memory}"
DISTILL_ALPHA="${DISTILL_ALPHA:-0.3}"
DISTILL_TEMPERATURE="${DISTILL_TEMPERATURE:-2.0}"

# Change exp name here
EXP_SUFFIX="16L96C_${DATASET_NAME}_${NEG_SAMPLE}_${CONTRAST_METHOD}_${DISTILL_ALPHA}_${DISTILL_TEMPERATURE}"

INP=(4  24 24 24 24 24 48 48 48 48 48 48 96 96 96 96)
OUT=(24 24 24 24 24 48 48 48 48 48 48 96 96 96 96 96)
POOL=(0  0  0  0  1  0  0  0  0  0  1  0  0  0  0  0 )
STRIDE=(1)
KSZ=(3)
PADDING=1
NOISE_LEVEL=0.

python train_ode_cifar.py \
  --optim         "SGD" \
  --img_type      "scanGFI" \
  --rggb_to_rgb   "false" \
  --dataset       "${DATASET_NAME}" \
  --num_epochs    300 \
  --eval_every    5 \
  --offset_eps    0.0 \
  --inp_channels  "${INP[@]}" \
  --out_channels  "${OUT[@]}" \
  --max_pool      "${POOL[@]}" \
  --kernel_size   "${KSZ[@]}" \
  --padding       "${PADDING}" \
  --stride        "${STRIDE[@]}" \
  --dropout       0.25 \
  --tie_weights   "false" \
  --tie_bp        "false" \
  --bypass        "false" \
  --batch_size    128 \
  --method        "dopri5" \
  --tol           "0.0001" \
  --t_end         "1.75" \
  --noise_type    "mul" \
  --pcn           "PCNetNoBatchNorm" \
  --pc_conv       "PCConvReLU6" \
  --ode_block     "ODEXInitFFFB" \
  --teacher_ckpt "${TEACHER_CKPT}" \
  --teacher_arch "${TEACHER_ARCH}" \
  --teacher_arch_source "${TEACHER_ARCH_SOURCE}" \
  --teacher_input_size "${TEACHER_INPUT_SIZE}" \
  --teacher_center_crop "${TEACHER_CENTER_CROP}" \
  --distill_method kd_crd \
  --contrast_method "${CONTRAST_METHOD}" \
  --neg_sample      "${NEG_SAMPLE}" \
  --orig_t_inp      "${ORIG_T_INP}" \
  --distill_alpha "${DISTILL_ALPHA}" \
  --distill_temperature "${DISTILL_TEMPERATURE}" \
  2>&1 | tee "${LOGDIR}/train_${EXP}_${EXP_SUFFIX}.log"

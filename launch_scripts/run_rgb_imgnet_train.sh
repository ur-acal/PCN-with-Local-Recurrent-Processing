#!/bin/bash -l
#SBATCH -N 1
#SBATCH -p ising
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16            # modest CPU request so the node can be shared
#SBATCH --gres=gpu:1                  # exactly ONE GPU; allows packing on 4-GPU nodes
#SBATCH -t 90000:10:00
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
DATASET_NAME="${DATASET_NAME:-imagenet}"
IMAGENET_ROOT="${IMAGENET_ROOT:-../data/imagenet/ILSVRC/Data/CLS-LOC}"
CASE="${CASE:-imagenet_custom}"
EXP="DS_PCN_IMAGENET"
LOGDIR="./logs/${EXP}"
mkdir -p "${LOGDIR}"

TEACHER_CKPT="${TEACHER_CKPT:-}"
TEACHER_ARCH="${TEACHER_ARCH:-}"
TEACHER_ARCH_SOURCE="${TEACHER_ARCH_SOURCE:-auto}"
TEACHER_INPUT_SIZE="${TEACHER_INPUT_SIZE:-224}"
TEACHER_CENTER_CROP="${TEACHER_CENTER_CROP:-true}"

NEG_SAMPLE="${NEG_SAMPLE:-index}"
CONTRAST_METHOD="${CONTRAST_METHOD:-memory}"
DISTILL_ALPHA="${DISTILL_ALPHA:-0.3}"
DISTILL_TEMPERATURE="${DISTILL_TEMPERATURE:-2.0}"

PCN="${PCN:-PCNetWith1stConv}" # Or PCNetWith1stConv, PCNetNoBatchNorm
T_END="${T_END:-1.75}"
WARMUP_EPOCH="${WARMUP_EPOCH:-0}"
IS_TIMM="${IS_TIMM:-true}"
TIMM_SCHED="${TIMM_SCHED:-cosine}"

# Change exp name here
EXP_SUFFIX="12L256C_${DATASET_NAME}_${NEG_SAMPLE}_${CONTRAST_METHOD}_${DISTILL_ALPHA}_${DISTILL_TEMPERATURE}"

INP=(3  64 64  128 128 128 128 256 256 256 512 512)
OUT=(64 64 128 128 128 128 256 256 256 512 512 512)
POOL=(0 0  1   0   1   0   1   0   0   1   0   0  )
#INP=(3  64 64  128 128 256 256 256)
#OUT=(64 64 128 128 256 256 256 256)
#POOL=(0 0  1   0   1   0   1   0)
STRIDE=(1)
KSZ=(3)
PADDING=1

## Same length as INP/OUT.
#STRIDE=()
#KSZ=()
#PADDING=()
#for _ in "${INP[@]}"; do
#  STRIDE+=(1)
#  KSZ+=(3)
#  PADDING+=(1)
#done
#
## First layer: 7x7 conv, stride 2, padding 3.
#STRIDE[0]=2
#KSZ[0]=7
## Todo: Need to change code in pc_conv for this
#PADDING[0]=3

NOISE_LEVEL=0.

# Only useful when PCN is PCNetWith1stConv
FIRST_KSZ="${FIRST_KSZ:-7}"
FIRST_STRIDE="${FIRST_STRIDE:-2}"
FIRST_PAD="${FIRST_PAD:-3}"
AVG_POOLING="${AVG_POOLING:-false}"

python train_ode_imagenet.py \
  --optim         "SGD" \
  --img_type      "rgb" \
  --dataset       "imagenet" \
  --learning_rate 0.01 \
  --weight_decay  "1e-4" \
  --warmup_epoch  0 \
  --imagenet_root "${IMAGENET_ROOT}" \
  --timm_trainer  "${IS_TIMM}" \
  --timm_sched    "${TIMM_SCHED}" \
  --num_epochs    300 \
  --warmup_epoch  "${WARMUP_EPOCH}" \
  --eval_every    5 \
  --offset_eps    0.0 \
  --inp_channels  "${INP[@]}" \
  --out_channels  "${OUT[@]}" \
  --max_pool      "${POOL[@]}" \
  --kernel_size   "${KSZ[@]}" \
  --padding       "${PADDING}" \
  --stride        "${STRIDE[@]}" \
  --first_ksz     "${FIRST_KSZ}" \
  --first_stride  "${FIRST_STRIDE}" \
  --first_pad     "${FIRST_PAD}" \
  --avg_pooling   "${AVG_POOLING}" \
  --dropout       0.25 \
  --tie_weights   "false" \
  --tie_bp        "false" \
  --bypass        "false" \
  --batch_size    32 \
  --method        "euler" \
  --n_steps       10 \
  --tol           "0.01" \
  --t_end         "${T_END}" \
  --noise_type    "mul" \
  --pcn           "${PCN}" \
  --pc_conv       "PCConvReLU6" \
  --ode_block     "ODEXInitFFFB" \
  --teacher_ckpt "${TEACHER_CKPT}" \
  --teacher_arch "${TEACHER_ARCH}" \
  --teacher_arch_source "${TEACHER_ARCH_SOURCE}" \
  --teacher_input_size "${TEACHER_INPUT_SIZE}" \
  --teacher_center_crop "${TEACHER_CENTER_CROP}" \
  --distill_method none \
  --contrast_method "${CONTRAST_METHOD}" \
  --neg_sample      "${NEG_SAMPLE}" \
  --orig_t_inp      "${ORIG_T_INP}" \
  --distill_alpha "${DISTILL_ALPHA}" \
  --distill_temperature "${DISTILL_TEMPERATURE}" \
  2>&1 | tee "${LOGDIR}/train_${EXP}_${EXP_SUFFIX}.log"

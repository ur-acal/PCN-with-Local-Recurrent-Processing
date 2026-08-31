#!/bin/bash -l
#SBATCH -p ising
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH -t 90:10:00
#SBATCH -o /scratch/rzeng7/repos/PCN-with-Local-Recurrent-Processing/logs/slurm_jobs/slurm_%j.out

IS_SLURM="${IS_SLURM:-0}"

if [[ "${IS_SLURM}" == 1 ]]; then
  source activate base
  conda activate scanbase
fi

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO_ROOT}"

source "${REPO_ROOT}/launch_scripts/tinyimagenet_pcn_arch.sh"

MODEL_ARCH="${MODEL_ARCH:-wrn_28_4}"
set_tinyimagenet_pcn_arch "${MODEL_ARCH}"

TINYIMAGENET_ROOT="${TINYIMAGENET_ROOT:-../data/tiny-imagenet-200}"
PCN="${PCN:-PCNetNoBatchNorm}"
PCCONV="${PCCONV:-PCConv}"
ODE_BLOCK="${ODE_BLOCK:-ODEXInitFFFB}"
T_END="${T_END:-1.75}"
SAVE_PATH="${SAVE_PATH:-${REPO_ROOT}/saved_ckpt}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/tinyimagenet/pcn_train}"
mkdir -p "${LOG_DIR}"

read -r -a INP <<< "${INP_CHANNELS}"
read -r -a OUT <<< "${OUT_CHANNELS}"
read -r -a POOL <<< "${MAX_POOL}"

python train_ode_cifar.py \
  --save_path "${SAVE_PATH}" \
  --dataset tinyimagenet \
  --task tinyimagenet \
  --tinyimagenet_root "${TINYIMAGENET_ROOT}" \
  --num_classes 200 \
  --seed 4096 \
  --img_type rgb \
  --timm_trainer true \
  --timm_sched cosine \
  --num_epochs 300 \
  --warmup_epoch 5 \
  --eval_every 5 \
  --batch_size 128 \
  --test_batch_size 512 \
  --learning_rate 0.01 \
  --weight_decay 0.001 \
  --inp_channels "${INP[@]}" \
  --out_channels "${OUT[@]}" \
  --max_pool "${POOL[@]}" \
  --kernel_size 3 \
  --stride 1 \
  --padding 1 \
  --dropout 0.25 \
  --pcn "${PCN}" \
  --pc_conv "${PCCONV}" \
  --ode_block "${ODE_BLOCK}" \
  --method dopri5 \
  --tol 0.0001 \
  --t_end "${T_END}" \
  --offset_eps 0.0 \
  --distill_method none \
  2>&1 | tee "${LOG_DIR}/${MODEL_ARCH}_${ODE_BLOCK}_${T_END}.log"

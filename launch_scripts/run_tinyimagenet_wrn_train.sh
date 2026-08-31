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

MODEL_NAME="${MODEL_NAME:-wrn_28_4_cifar}"
TINYIMAGENET_ROOT="${TINYIMAGENET_ROOT:-../data/tiny-imagenet-200}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/checkpoint/baselines}"
TRAIN_OVERRIDE="${TRAIN_OVERRIDE:-eval_every=5}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/tinyimagenet/wrn_train}"
mkdir -p "${LOG_DIR}"

python baseline/train_baseline_cifar.py \
  --model_name "${MODEL_NAME}" \
  --dataset tinyimagenet \
  --data_dir "${TINYIMAGENET_ROOT}" \
  --output_dir "${OUTPUT_DIR}" \
  --case custom_noresize \
  --pretrained false \
  --seed 4096 \
  --eval_every 5 \
  --override "${TRAIN_OVERRIDE}" \
  2>&1 | tee "${LOG_DIR}/${MODEL_NAME}.log"

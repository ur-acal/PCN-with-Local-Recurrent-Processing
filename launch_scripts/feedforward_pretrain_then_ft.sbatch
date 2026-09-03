#!/bin/bash -l
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH -t 48:00:00
#SBATCH -o logs/slurm_jobs/feedforward_%j.out

set -e
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO_ROOT}"
mkdir -p logs/slurm_jobs

if [[ -n "${CONDA_ENV:-scanbase}" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV:-scanbase}"
fi

./launch_scripts/run_feedforward_pretrain_then_ft.sh

#!/bin/bash -l
#SBATCH -p ising
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH -t 72:10:00
#SBATCH -o logs/slurm_jobs/slurm_%j.out

IS_SLURM="${IS_SLURM:-1}"
if [[ "${IS_SLURM}" == 1 ]]; then
  source activate base
  conda activate scanbase || exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
export ROWS="${ROW:?ROW is required}"
export DATASETS="${DATASET:?DATASET is required}"
export SIZES="${SIZE:?SIZE is required}"
export PARALLELISM=1
export PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
exec bash "${REPO_ROOT}/launch_scripts/run_wrn_controls.sh"

#!/bin/bash -l

#SBATCH -p ising
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH -t 72:10:00
#SBATCH -o logs/slurm_jobs/slurm_%j.out

source activate base
conda activate scanbase

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${REPO_ROOT}"

export PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
export DATA_ROOT="${DATA_ROOT:-/scratch/rzeng7/repos/data}"
export PARALLELISM="${PARALLELISM:-4}"

./launch_scripts/run_wrn28_2_nobn_pool_hparam_search.sh

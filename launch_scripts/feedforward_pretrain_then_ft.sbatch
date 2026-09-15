#!/bin/bash -l
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH -t 48:00:00
#SBATCH -o logs/slurm_jobs/feedforward_%j.out

# TC includes post-FT evaluation; stage/pipeline failures must fail this job.
set -eo pipefail
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO_ROOT}"
mkdir -p logs/slurm_jobs

source activate base
conda activate scanbase

source ./launch_scripts/run_feedforward_pretrain_then_ft.sh

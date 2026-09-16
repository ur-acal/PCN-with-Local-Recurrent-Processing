#!/bin/bash -l
#SBATCH -p ising
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH -t 90:10:00
#SBATCH -o /scratch/rzeng7/repos/PCN-with-Local-Recurrent-Processing/logs/slurm_jobs/slurm_%j.out

# TC includes post-FT evaluation; stage/pipeline failures must fail this job.
set -eo pipefail
REPO_ROOT="${REPO_ROOT:-/scratch/rzeng7/repos/PCN-with-Local-Recurrent-Processing}"
cd "${REPO_ROOT}"
mkdir -p logs/slurm_jobs

source activate base
conda activate scanbase

source ./launch_scripts/run_feedforward_pretrain_then_ft.sh

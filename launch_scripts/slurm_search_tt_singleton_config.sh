#!/bin/bash
# Source from the configured login shell, like slurm_search_feedforward_config.sh.
set -e
REPO_ROOT="${REPO_ROOT:-/scratch/rzeng7/repos/PCN-with-Local-Recurrent-Processing}"
SBATCH_SCRIPT="${REPO_ROOT}/launch_scripts/tt_singleton_study.sbatch"
cd "${REPO_ROOT}"
: "${MODEL_KIND:?pcn, wrn_c27 or resnet110}"
: "${TASK:?cifar10 or cifar100}"
: "${COUPLER_MONTE_TT_SINGLETON:?Pass the original TT singleton CSV}"
: "${COUPLER_MONTE_V2_TT_SINGLETON:?Pass the improved TT singleton CSV}"
export REPO_ROOT MODEL_KIND TASK COUPLER_MONTE_TT_SINGLETON COUPLER_MONTE_V2_TT_SINGLETON
mkdir -p logs/slurm_jobs
sbatch --parsable \
  --gres=gpu:${GPUS_PER_JOB:-1} \
  --export=ALL \
  "${SBATCH_SCRIPT}"

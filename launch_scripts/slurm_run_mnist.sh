#!/bin/bash
# Source from the user's module-configured shell, as with slurm_search_config.sh.
REPO_ROOT="${REPO_ROOT:-$PWD}"
export REPO_ROOT
cd "$REPO_ROOT"
mkdir -p logs/slurm_jobs logs/scheduler_slurm
if [[ "${DRY_RUN:-false}" == true ]]; then
  printf 'sbatch --parsable --gres=gpu:%s --export=ALL %s\n' \
    "${GPUS_PER_JOB:-1}" "${REPO_ROOT}/launch_scripts/mnist_pipeline.sbatch"
else
  sbatch --parsable --gres=gpu:${GPUS_PER_JOB:-1} --export=ALL \
    "${REPO_ROOT}/launch_scripts/mnist_pipeline.sbatch"
fi

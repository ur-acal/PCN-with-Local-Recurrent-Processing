#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-/scratch/rzeng7/repos/data}"
PARALLELISM="${PARALLELISM:-4}"
SBATCH_SCRIPT="${SBATCH_SCRIPT:-${REPO_ROOT}/launch_scripts/run_wrn28_2_nobn_pool_hparam_search.sbatch}"
SEARCH_BASE="${SEARCH_BASE:-${REPO_ROOT}/logs/wrn28_2_nobn_pool_hparam_search_v2}"

mkdir -p "${REPO_ROOT}/logs/slurm_jobs" "${SEARCH_BASE}"

submit_study() {
  local dataset_name="$1"
  local study_name="$2"
  local search_root="${SEARCH_BASE}/${dataset_name}_${study_name}"
  local job_id

  job_id=$(sbatch --parsable \
    --chdir="${REPO_ROOT}" \
    --output="${REPO_ROOT}/logs/slurm_jobs/slurm_%j.out" \
    --export=ALL,REPO_ROOT="${REPO_ROOT}",DATA_ROOT="${DATA_ROOT}",PARALLELISM="${PARALLELISM}",DATASET_NAME="${dataset_name}",STUDY_NAME="${study_name}",SEARCH_ROOT="${search_root}" \
    "${SBATCH_SCRIPT}")
  echo "submitted job ${job_id}: dataset=${dataset_name}, study=${study_name}, parallelism=${PARALLELISM}, output=${search_root}"
}

submit_study cifar100 avgpool_main
submit_study cifar10 stride2_main
submit_study cifar10 avgpool_main
submit_study cifar100 stride2_main

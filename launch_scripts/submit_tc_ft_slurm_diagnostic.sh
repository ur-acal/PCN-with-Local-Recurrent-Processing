#!/usr/bin/env bash
# Source this scheduler from the module-configured login shell, following the
# same convention as slurm_search_config.sh. The default submits only the
# observed state-1, timm_aug=none diagnostic.
REPO_ROOT=/scratch/rzeng7/repos/PCN-with-Local-Recurrent-Processing
SBATCH_SCRIPT="${REPO_ROOT}/launch_scripts/run_tc_ft_slurm_diagnostic.sbatch"
GPUS_PER_JOB="${GPUS_PER_JOB:-1}"

mkdir -p "${REPO_ROOT}/logs/slurm_jobs" "${REPO_ROOT}/results/tc_ft_slurm_gap"
read -r -a states <<< "${DIAG_STATES:-1}"
read -r -a augs <<< "${DIAG_AUGS:-none}"
tag="${DIAG_TAG:-$(date +%Y%m%d_%H%M%S)}"

for state in "${states[@]}"; do
  for aug in "${augs[@]}"; do
    job_id="$(sbatch --parsable \
      --gres="gpu:${GPUS_PER_JOB}" \
      --export="ALL,TC_STATE=${state},FT_TIMM_AUG_LEVEL=${aug},DIAG_TAG=${tag}" \
      "${SBATCH_SCRIPT}")" || return 2
    echo "submitted job=${job_id} state=${state} aug=${aug} tag=${tag}"
  done
done

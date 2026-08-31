#!/bin/bash

REPO_ROOT="/scratch/rzeng7/repos/PCN-with-Local-Recurrent-Processing"
SBATCH_SCRIPT="${REPO_ROOT}/launch_scripts/run_rgb_ode_mismatch_eval.sh"

###############################################################################################
# Existing six checkpoints: submits four one-GPU jobs.
# ( source ./launch_scripts/slurm_run_rgb_ode_mismatch_eval.sh ) \
#  > ./logs/scheduler_slurm/slurm_rgb_ode_mismatch_eval.log 2>&1 < /dev/null &
#
# Newly trained CIFAR-10 WRN-16-4/WRN-28-2 counterparts:
# MODEL_SET=pending source ./launch_scripts/slurm_run_rgb_ode_mismatch_eval.sh
###############################################################################################

MAX_TASKS_PER_GPU="${MAX_TASKS_PER_GPU:-1}"
GPUS_PER_JOB="${GPUS_PER_JOB:-1}"
MODEL_SET="${MODEL_SET:-existing}"
MODEL_NAME="${MODEL_NAME:-}"
MODEL_INDEX="${MODEL_INDEX:-}"
ARCHITECTURE="${ARCHITECTURE:-}"
RESULT_TAG="${RESULT_TAG:-}"
ODE_BLOCK="${ODE_BLOCK:-}"
PC_CONV="${PC_CONV:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-}"

if [[ -n "${MODEL_NAME}" ]]; then
  NUM_JOBS=1
elif [[ "${MODEL_SET}" == "pending" ]]; then
  NUM_JOBS=2
else
  NUM_JOBS=4
fi

for ((shard_id=0; shard_id<NUM_JOBS; shard_id++)); do
  jid=$(
    sbatch --parsable \
      --gres=gpu:${GPUS_PER_JOB} \
      --export=ALL,IS_SLURM=1,MODEL_SET="${MODEL_SET}",SHARD_ID="${shard_id}",MODEL_NAME="${MODEL_NAME}",MODEL_INDEX="${MODEL_INDEX}",ARCHITECTURE="${ARCHITECTURE}",RESULT_TAG="${RESULT_TAG}",ODE_BLOCK="${ODE_BLOCK}",PC_CONV="${PC_CONV}",OUTPUT_ROOT="${OUTPUT_ROOT}" \
      "${SBATCH_SCRIPT}"
  )
  echo "submitted job ${jid}: model_set=${MODEL_SET}, model_name=${MODEL_NAME:-mapped}, shard_id=${shard_id}"
done

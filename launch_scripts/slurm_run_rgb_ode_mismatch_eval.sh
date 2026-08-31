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
#
# MODEL_NAME selects dynamic mode and takes precedence over MODEL_SET.
# With MODEL_NAME unset, convenience sets new_x_minus_fb and legacy_no_x are available.
###############################################################################################

MAX_TASKS_PER_GPU="${MAX_TASKS_PER_GPU:-1}"
GPUS_PER_JOB="${GPUS_PER_JOB:-1}"
MODEL_SET="${MODEL_SET:-existing}"
EVAL_MODE="${EVAL_MODE:-mismatch}"
CONDITIONS="${CONDITIONS:-multiplicative,max_additive,rms_additive}"
DATASETS="${DATASETS:-all}"
MODEL_NAME="${MODEL_NAME:-}"
MODEL_INDEX="${MODEL_INDEX:-}"
ARCHITECTURE="${ARCHITECTURE:-}"
RESULT_TAG="${RESULT_TAG:-}"
ODE_BLOCK="${ODE_BLOCK:-}"
PC_CONV="${PC_CONV:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-}"
MUL_LEVELS="${MUL_LEVELS:-0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4}"
MAX_ADD_LEVELS="${MAX_ADD_LEVELS:-0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1}"
RMS_ADD_LEVELS="${RMS_ADD_LEVELS:-0.25,0.5,0.75,1.0,1.25}"
MAX_SQRT_LEVELS="${MAX_SQRT_LEVELS:-0,0.02,0.03,0.05,0.07,0.09,0.1}"
FF_GAIN_LIST="${FF_GAIN_LIST:-0.90,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99,1.00,1.01,1.02,1.03,1.04,1.05,1.06,1.07,1.08,1.09,1.10}"
NOISY_TRIALS="${NOISY_TRIALS:-10}"

export CONDITIONS DATASETS MUL_LEVELS MAX_ADD_LEVELS RMS_ADD_LEVELS MAX_SQRT_LEVELS FF_GAIN_LIST NOISY_TRIALS

if [[ -n "${MODEL_NAME}" ]]; then
  NUM_JOBS=1
elif [[ "${MODEL_SET}" == "pending" ]]; then
  NUM_JOBS=2
elif [[ "${MODEL_SET}" == "new_x_minus_fb" ]]; then
  NUM_JOBS=2
else
  NUM_JOBS=4
fi

for ((shard_id=0; shard_id<NUM_JOBS; shard_id++)); do
  jid=$(
    sbatch --parsable \
      --gres=gpu:${GPUS_PER_JOB} \
      --export=ALL,IS_SLURM=1,MODEL_SET="${MODEL_SET}",EVAL_MODE="${EVAL_MODE}",SHARD_ID="${shard_id}",MODEL_NAME="${MODEL_NAME}",MODEL_INDEX="${MODEL_INDEX}",ARCHITECTURE="${ARCHITECTURE}",RESULT_TAG="${RESULT_TAG}",ODE_BLOCK="${ODE_BLOCK}",PC_CONV="${PC_CONV}",OUTPUT_ROOT="${OUTPUT_ROOT}" \
      "${SBATCH_SCRIPT}"
  )
  echo "submitted job ${jid}: model_set=${MODEL_SET}, eval_mode=${EVAL_MODE}, conditions=${CONDITIONS}, model_name=${MODEL_NAME:-mapped}, shard_id=${shard_id}"
done

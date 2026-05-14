#!/bin/bash
#set -euo pipefail

REPO_ROOT="/scratch/rzeng7/repos/PCN-with-Local-Recurrent-Processing"
SBATCH_SCRIPT="${REPO_ROOT}/launch_scripts/run_rgb_ode_train.sh"

###############################################################################################
# running with
# module swap slurm slurm/24.05.0.b1
# ( source ./launch_scripts/slurm_run_rgb_ode_train.sh ) \
#  > ./logs/scheduler_slurm/slurm_rgb_ode_train.log 2>&1 < /dev/null &
###############################################################################################

MAX_TASKS_PER_GPU="${MAX_TASKS_PER_GPU:-1}"
GPUS_PER_JOB="${GPUS_PER_JOB:-1}"

PCNS=( "PCNetNoBatchNorm" "PCNetWith1stConv" )
T_END_LIST=( 1.75 )
WARMUP_EPOCH_LIST=( 0 )
TIMM_TRAINER=( "true" )
TIMM_SCHED_LIST=( "cosine" )
PCCONV_LIST=( "PCConv" "PCConvReLU6" )

for pc_conv in "${PCCONV_LIST[@]}"; do
  for timm_sched in "${TIMM_SCHED_LIST[@]}"; do
    for pcn in "${PCNS[@]}"; do
      for t_end in "${T_END_LIST[@]}"; do
        for warmup_epoch in "${WARMUP_EPOCH_LIST[@]}"; do
          for is_timm in "${TIMM_TRAINER[@]}"; do
            jid=$(
              sbatch --parsable \
                --gres=gpu:${GPUS_PER_JOB} \
                --export=ALL,IS_SLURM=1,PCN="${pcn}",T_END="${t_end}",WARMUP_EPOCH="${warmup_epoch}",IS_TIMM="${is_timm}",TIMM_SCHED="${timm_sched}",PCCONV="${pc_conv}" \
                "${SBATCH_SCRIPT}"
            )
            echo "submitted job ${jid}: pcn=${pcn}, t_end=${t_end}, warmup_epoch=${warmup_epoch}, is_timm=${is_timm}, timm_sched=${timm_sched}, pc_conv=${pc_conv}"
          done
        done
      done
    done
  done
done
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

PCNS=( "PCNetNoBatchNorm" )
ODE_BLOCK_LIST=( "ODEBlockPC" "ODEBlockXInit" )
T_END_LIST=( 1.75 )
WARMUP_EPOCH_LIST=( 5 )
TIMM_TRAINER=( "true" )
TIMM_SCHED_LIST=( "cosine" )
PCCONV_LIST=( "PCConv" )
DATASET_LIST=( "cifar10" "cifar100" )

COMB_LIST=(
#  "3 32 64 64 64|32 64 64 64 64|0 1 0 1 0"
#  "3 32 32 64 64 128|32 32 64 64 128 128|0 0 1 0 1 0"
#  "3 32 32 64 64 128 128|32 32 64 64 128 128 128|0 0 1 0 1 0 0"
#  "3 64 64 128 128 256 256 256|64 64 128 128 256 256 256 256|0 0 1 0 1 0 0 0"
  "3 16 32 32 64 64 128|16 32 32 64 64 128 128|0 0 0 1 0 1 0" # WRN-16-2 style, ~0.69M
  "3 16 64 64 128 128 256|16 64 64 128 128 256 256|0 0 0 1 0 1 0" # WRN-16-4 style, ~2.75M
#  "3 16 128 128 256 256 512|16 128 128 256 256 512 512|0 0 0 1 0 1 0" # WRN-16-8 style, ~11.0M
#  "3 16 160 160 320 320 640|16 160 160 320 320 640 640|0 0 0 1 0 1 0" # WRN-16-10 style, ~17.1M
  "3 16 32 32 32 32 64 64 64 64 128 128 128|16 32 32 32 32 64 64 64 64 128 128 128 128|0 0 0 0 0 1 0 0 0 1 0 0 0" # WRN-28-2 style, ~1.47M
  "3 16 64 64 64 64 128 128 128 128 256 256 256|16 64 64 64 64 128 128 128 128 256 256 256 256|0 0 0 0 0 1 0 0 0 1 0 0 0" # WRN-28-4 style, ~5.9M
#  "3 16 80 80 80 80 160 160 160 160 320 320 320|16 80 80 80 80 160 160 160 160 320 320 320 320|0 0 0 0 0 1 0 0 0 1 0 0 0" # WRN-28-5 style, ~9.2M
#  "3 16 32 32 32 32 32 32 64 64 64 64 64 64 128 128 128 128 128|16 32 32 32 32 32 32 64 64 64 64 64 64 128 128 128 128 128 128|0 0 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0" # WRN-40-2 style, ~2.2M
#  "3 16 64 64 64 64 64 64 128 128 128 128 128 128 256 256 256 256 256|16 64 64 64 64 64 64 128 128 128 128 128 128 256 256 256 256 256 256|0 0 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0" # WRN-40-4 style, ~8.9M
)

for dataset_name in "${DATASET_LIST[@]}"; do
  for comb in "${COMB_LIST[@]}"; do
    IFS='|' read -r inp_channels out_channels max_pool <<< "${comb}"

    for pc_conv in "${PCCONV_LIST[@]}"; do
      for timm_sched in "${TIMM_SCHED_LIST[@]}"; do
        for pcn in "${PCNS[@]}"; do
          for ode_block in "${ODE_BLOCK_LIST[@]}"; do
            for t_end in "${T_END_LIST[@]}"; do
              for warmup_epoch in "${WARMUP_EPOCH_LIST[@]}"; do
                for is_timm in "${TIMM_TRAINER[@]}"; do
                  jid=$(
                    sbatch --parsable \
                      --gres=gpu:${GPUS_PER_JOB} \
                      --export=ALL,IS_SLURM=1,DATASET_NAME="${dataset_name}",PCN="${pcn}",ODE_BLOCK="${ode_block}",T_END="${t_end}",WARMUP_EPOCH="${warmup_epoch}",IS_TIMM="${is_timm}",TIMM_SCHED="${timm_sched}",PCCONV="${pc_conv}",INP_CHANNELS="${inp_channels}",OUT_CHANNELS="${out_channels}",MAX_POOL="${max_pool}" \
                      "${SBATCH_SCRIPT}"
                  )
                  echo "submitted job ${jid}: dataset_name=${dataset_name}, pcn=${pcn}, ode_block=${ode_block}, t_end=${t_end}, warmup_epoch=${warmup_epoch}, is_timm=${is_timm}, timm_sched=${timm_sched}, pc_conv=${pc_conv}, inp=${inp_channels}, out=${out_channels}, pool=${max_pool}"
                done
              done
            done
          done
        done
      done
    done
  done
done

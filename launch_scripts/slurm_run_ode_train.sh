#!/bin/bash
#set -euo pipefail

REPO_ROOT="/scratch/rzeng7/repos/PCN-with-Local-Recurrent-Processing"
SBATCH_SCRIPT="${REPO_ROOT}/launch_scripts/run_ode_train.sh"

###############################################################################################
# running with
# module swap slurm slurm/24.05.0.b1
# ( source ./launch_scripts/slurm_run_ode_train.sh ) \
#  > ./logs/scheduler_slurm/slurm_ode_train.log 2>&1 < /dev/null &
###############################################################################################

MAX_TASKS_PER_GPU="${MAX_TASKS_PER_GPU:-1}"
GPUS_PER_JOB="${GPUS_PER_JOB:-1}"

CONTRAST_METHODS=(memory)
NEG_SAMPLES=(label)
DISTILL_ALPHAS=(0.3)
DISTILL_TEMPERATURES=(2.0)

for contrast_method in "${CONTRAST_METHODS[@]}"; do
  for neg_sample in "${NEG_SAMPLES[@]}"; do
    for distill_alpha in "${DISTILL_ALPHAS[@]}"; do
      for distill_temperature in "${DISTILL_TEMPERATURES[@]}"; do
        # Note: when using moco, the neg_samples arg is useless.
        if [[ "${contrast_method}" == "moco" && "${neg_sample}" == "label" ]]; then
          continue
        fi
        jid=$(
          sbatch --parsable \
            --gres=gpu:${GPUS_PER_JOB} \
            --export=ALL,IS_SLURM=1,CONTRAST_METHOD="${contrast_method}",NEG_SAMPLE="${neg_sample}",DISTILL_ALPHA="${distill_alpha}",DISTILL_TEMPERATURE="${distill_temperature}" \
            "${SBATCH_SCRIPT}"
        )
        echo "submitted job ${jid}: contrast_method=${contrast_method}, neg_sample=${neg_sample}, alpha=${distill_alpha}, temp=${distill_temperature}"
      done
    done
  done
done
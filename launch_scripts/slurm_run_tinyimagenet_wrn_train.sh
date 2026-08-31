#!/bin/bash

REPO_ROOT="${REPO_ROOT:-/scratch/rzeng7/repos/PCN-with-Local-Recurrent-Processing}"
SBATCH_SCRIPT="${REPO_ROOT}/launch_scripts/run_tinyimagenet_wrn_train.sh"
MODEL_LIST=(
  wrn_16_2_cifar
  wrn_16_4_cifar
  wrn_28_2_cifar
  wrn_28_4_cifar
  wrn_16_8_cifar
  wrn_40_2_cifar
)

for model_name in "${MODEL_LIST[@]}"; do
  jid=$(sbatch --parsable \
    --export=ALL,IS_SLURM=1,REPO_ROOT="${REPO_ROOT}",MODEL_NAME="${model_name}" \
    "${SBATCH_SCRIPT}")
  echo "submitted job ${jid}: dataset=tinyimagenet model=${model_name}"
done

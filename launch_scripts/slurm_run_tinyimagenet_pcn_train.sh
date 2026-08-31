#!/bin/bash

REPO_ROOT="${REPO_ROOT:-/scratch/rzeng7/repos/PCN-with-Local-Recurrent-Processing}"
SBATCH_SCRIPT="${REPO_ROOT}/launch_scripts/run_tinyimagenet_pcn_train.sh"
MODEL_LIST=(wrn_16_2 wrn_16_4 wrn_28_2 wrn_28_4 wrn_16_8 wrn_40_2)

for model_arch in "${MODEL_LIST[@]}"; do
  jid=$(sbatch --parsable \
    --export=ALL,IS_SLURM=1,REPO_ROOT="${REPO_ROOT}",MODEL_ARCH="${model_arch}" \
    "${SBATCH_SCRIPT}")
  echo "submitted job ${jid}: dataset=tinyimagenet model=${model_arch} pcn=PCNetNoBatchNorm"
done

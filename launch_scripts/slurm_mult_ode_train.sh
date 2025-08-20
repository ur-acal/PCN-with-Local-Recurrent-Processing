#!/usr/bin/env bash

#set -euo pipefail

for ARCH in A B; do
  for PCN in "PCNetNoBatchNorm" "PCNetWith1stConv"; do
    echo "Submitting ARCH=${ARCH} PCN=${PCN}"
    sbatch --export=ALL,ARCH_SET=${ARCH},PCN="${PCN}" ./launch_scripts/run_mult_ode_train.sbatch
  done
done

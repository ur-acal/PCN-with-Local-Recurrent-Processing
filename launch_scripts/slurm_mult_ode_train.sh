#!/usr/bin/env bash
set -euo pipefail

for ARCH in A B; do
  for PCN in "PCNetNoBatchNorm" "PCNetWith1stConv"; do
    echo "Submitting ARCH=${ARCH} PCN=${PCN}"
    sbatch --export=ALL,ARCH_SET=${ARCH},PCN="${PCN}" run_one_arch_pcn.sbatch
  done
done

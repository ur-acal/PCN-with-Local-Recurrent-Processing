#!/usr/bin/env bash
#set -euo pipefail

# How many blocks to launch together on ONE GPU per sbatch job
MAX_TASKS_PER_GPU=${MAX_TASKS_PER_GPU:-2}
# GPUs per job (kept at 1; override here if you need to)
GPUS_PER_JOB=${GPUS_PER_JOB:-1}

# Define blocks per ARCH (must match names used inside the sbatch script)
declare -A BLOCKS_BY_ARCH
# 6L3p
BLOCKS_BY_ARCH[A]="ODEState2FFFB State2InitYAsXZAs0 State2InitYAsXZAsX State2NoMinusZ State2NoMinusZYAsXZAs0 State2NoMinusZYAsXZAsX"
# Deep
BLOCKS_BY_ARCH[B]="ODEFixNoiseXInit ODEFixNoiseXInitFFFB ODEFixNoise0InitExpand"
# 7L2p
BLOCKS_BY_ARCH[C]="ODEState2FFFB State2InitYAsXZAs0 State2InitYAsXZAsX State2NoMinusZ State2NoMinusZYAsXZAs0 State2NoMinusZYAsXZAsX"

# Which ARCH/PCN combos to run
ARCHES=(A C)
PCNS=("PCNetNoBatchNorm")

# Paths
REPO_ROOT="/home/rzeng7/Desktop/research/repos/PCN-with-Local-Recurrent-Processing"
SBATCH_SCRIPT="${REPO_ROOT}/launch_scripts/run_blocks_ode_train.sbatch"

split_and_submit() {
  local arch="$1" pcn="$2"
  local blocks_str="${BLOCKS_BY_ARCH[$arch]:-}"
  if [[ -z "$blocks_str" ]]; then
    echo "No blocks defined for ARCH=$arch" >&2
    return 1
  fi

  read -r -a blocks <<< "$blocks_str"
  local n=${#blocks[@]}
  local i=0

  while (( i < n )); do
    local end=$(( i + MAX_TASKS_PER_GPU ))
    (( end > n )) && end=$n
    local chunk=( "${blocks[@]:i:end-i}" )

    # Comma-separated list for the sbatch script to consume
    local csv; csv=$(IFS=,; echo "${chunk[*]}")
    # Tag EXP with arch + joined block names + timestamp (unique per chunk)
    local tag; tag=$(echo "$csv" | tr ',' '+')
    local EXP="no_bn_${pcn}_NODE_0822_relu6_FFFB_DYN_${arch}_Exp_${tag}"

    echo "Submitting ARCH=${arch} PCN=${pcn} blocks=[${csv}] → EXP=${EXP}"
    sbatch \
      --gres=gpu:${GPUS_PER_JOB} \
      --export=ALL,ARCH_SET="${arch}",PCN="${pcn}",BLOCKS_LIST="${csv}",EXP="${EXP}" \
      "${SBATCH_SCRIPT}"

    i=$end
  done
}

for ARCH in "${ARCHES[@]}"; do
  for PCN in "${PCNS[@]}"; do
    split_and_submit "$ARCH" "$PCN"
  done
done

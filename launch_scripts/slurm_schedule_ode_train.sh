#!/usr/bin/env bash
#set -euo pipefail

# How many blocks to launch together on ONE GPU per sbatch job
MAX_TASKS_PER_GPU=${MAX_TASKS_PER_GPU:-2}
# GPUs per job (kept at 1; override here if you need to)
GPUS_PER_JOB=${GPUS_PER_JOB:-1}

###############################################################################################
# running with
# module swap slurm slurm/24.05.0.b1
# source ./launch_scripts/slurm_schedule_ode_train.sh > ./logs/scheduler_slurm/scheduler.log 2>&1 < /dev/null &
###############################################################################################
# Change exp parameter here
###############################################################################################
# Define blocks per ARCH (must match names used inside the sbatch script)
declare -A BLOCKS_BY_ARCH
# 6L3p
BLOCKS_BY_ARCH[A]="ODEState2FFFB State2InitYAsXZAsX State2NoMinusZ State2NoMinusZYAsXZAsX"
# Deep
BLOCKS_BY_ARCH[B]="ODEFixNoiseXInit ODEFixNoiseXInitFFFB ODEFixNoise0InitExpand"
# 7L2p
BLOCKS_BY_ARCH[C]="ODEState2FFFB State2InitYAsXZAsX State2NoMinusZ State2NoMinusZYAsXZAsX"

# Which ARCH/PCN combos to run
ARCHES=(A C)
PCNS=("PCNetNoBatchNorm")
IMG_TYPES=( "rggb" "cycleisp" )
###############################################################################################

# Paths
REPO_ROOT="/home/rzeng7/Desktop/research/repos/PCN-with-Local-Recurrent-Processing"
SBATCH_SCRIPT="${REPO_ROOT}/launch_scripts/run_blocks_ode_train.sbatch"

SLURM_LOG_DIR="${REPO_ROOT}/logs/slurm_jobs"
# track jobids per EXP
declare -A JOBS_BY_EXP
# where we’ll save merged CSVs (same subtree as your parser output)
MERGE_OUT_DIR="${REPO_ROOT}/shell_utils/parse_res/neural_ode_res"
MERGE_SCRIPT="${REPO_ROOT}/shell_utils/merge_csvs.py"

split_and_submit() {
  local arch="$1" pcn="$2" img_type="$3"
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
    local EXP="no_bn_${pcn}_NODE_0831_2State_RAWImg_${arch}_${img_type}_Exp"

    echo "Submitting ARCH=${arch} PCN=${pcn} blocks=[${csv}] img_type=[${img_type}] → EXP=${EXP}"
    jid=$( BLOCKS_LIST="${csv}" \
      sbatch --parsable \
             --gres=gpu:${GPUS_PER_JOB} \
             --export=ALL,ARCH_SET="${arch}",PCN="${pcn}",IMG_TYPE="${img_type}",EXP="${EXP}" \
             "${SBATCH_SCRIPT}" )
    echo "  -> job ${jid}"
    JOBS_BY_EXP["${EXP}"]+="${jid}:"

    i=$end
  done
}

wait_for_jobs() {
  local -a ids=("$@")
  while :; do
    local alive=0
    for j in "${ids[@]}"; do
      # if job is still known to scheduler
      if squeue -h -j "$j" >/dev/null 2>&1; then alive=1; break; fi
    done
    (( alive )) || break

    # status line
    local list; list="$(IFS=,; echo "${ids[*]}")"
    local r p
    r=$(squeue -h -j "$list" -t RUNNING -o '%i' | wc -l)
    p=$(squeue -h -j "$list" -t PENDING -o '%i' | wc -l)
    printf '\rWaiting… RUNNING=%d PENDING=%d  %s' "$r" "$p" "$(date +%H:%M:%S)"
    sleep 20
  done
}

merge_csvs_for_exp() {
  local exp="$1"
  local joblist="${JOBS_BY_EXP[$exp]}"
  joblist="${joblist%:}"          # trim trailing colon
  IFS=':' read -r -a ids <<< "$joblist"

  # Wait for these jobs
  wait_for_jobs "${ids[@]}"

  mkdir -p "${MERGE_OUT_DIR}"
  local merged_out="${MERGE_OUT_DIR}/${exp}_merged.csv"

  # Collect CSVs from slurm logs
  local csvs=()
  for j in "${ids[@]}"; do
    local slog="${SLURM_LOG_DIR}/slurm_${j}.out"
    [[ -s "$slog" ]] || { echo "[WARN] Missing slurm log $slog"; continue; }
    # read last line; expect: Wrote /path/to/file.csv
    local last; last=$(tail -n 1 "$slog" || true)
    # extract path after 'Wrote '
    local csv_path
    csv_path=$(sed -nE 's/^Wrote[[:space:]]+(.+\.csv)$/\1/p' <<< "$last")
    if [[ -n "$csv_path" && -f "$csv_path" ]]; then
      csvs+=("$csv_path")
    else
      echo "[WARN] No CSV path found in last line of $slog: $last"
    fi
  done

  if ((${#csvs[@]}==0)); then
    echo "[WARN] No CSVs to merge for ${exp}"
    return 0
  fi

  # Call merge script (assumed interface: first arg is output, rest are inputs)
  echo "[MERGE] ${exp}: ${#csvs[@]} files -> ${merged_out}"
  python -u "${MERGE_SCRIPT}" "${merged_out}" "${csvs[@]}"
  echo "[MERGED SAVED] ${merged_out}"
}

# Launch jobs
for IMG_TYPE in "${IMG_TYPES[@]}"; do
  for ARCH in "${ARCHES[@]}"; do
    for PCN in "${PCNS[@]}"; do
      split_and_submit "$ARCH" "$PCN" "$IMG_TYPE"
    done
  done
done

# Wait and merge CSVs
for exp in "${!JOBS_BY_EXP[@]}"; do
  merge_csvs_for_exp "$exp"
done

echo "Merged CSVs:"
for exp in "${!JOBS_BY_EXP[@]}"; do
  echo "  ${MERGE_OUT_DIR}/${exp}_merged.csv"
done
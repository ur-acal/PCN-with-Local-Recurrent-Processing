#!/usr/bin/env bash
#set -euo pipefail

# How many blocks to launch together on ONE GPU per sbatch job
MAX_TASKS_PER_GPU=${MAX_TASKS_PER_GPU:-2}
# GPUs per job (kept at 1; override here if you need to)
GPUS_PER_JOB=${GPUS_PER_JOB:-1}

###############################################################################################
# running with
# module swap slurm slurm/24.05.0.b1
# ( source ./launch_scripts/slurm_qat_ode_train.sh ) \
#  > ./logs/scheduler_slurm/qat_scheduler.log 2>&1 < /dev/null &
#
# sched_pid=$!
# disown -h "$sched_pid"
###############################################################################################
# Change exp parameter here
###############################################################################################
# Which Model to finetune
MODEL_NAMES=(
#  "PCNetNoBatchNorm_PCConvReLU6_0.2eps_S2NoMinusZChgZNoisyI_dopri5Solver_1.5TEnd_0.0001Tol_0.001WD_128BS_0.01LR_0.25Dropout_7Layers_2Pool_rggb_2REP" # RGGB
#  "PCNetNoBatchNorm_PCConvReLU6_0.2eps_S2NoMinusZChgZNoisyI_dopri5Solver_1.5TEnd_0.0001Tol_0.001WD_128BS_0.01LR_0.25Dropout_6Layers_2Pool_rggb_1REP"
#  "PCNetNoBatchNorm_PCConvReLU6_0.2eps_S2NoMinusZChgZNoisyI_dopri5Solver_1.5TEnd_0.0001Tol_0.001WD_128BS_0.01LR_0.25Dropout_4Layers_2Pool_rggb_1REP"

  "PCNetNoBatchNorm_PCConvReLU6_0.0eps_S2NoMinusZChgZNoisyI_dopri5Solver_1.5TEnd_0.0001Tol_0.001WD_128BS_0.01LR_3K1S128C_0.25Dropout_7Layers_2Pool_scanGFI_2REP"
  "PCNetNoBatchNorm_PCConvReLU6_0.0eps_S2NoMinusZChgZNoisyI_dopri5Solver_1.5TEnd_0.0001Tol_0.001WD_128BS_0.01LR_3K1S_0.25Dropout_6Layers_2Pool_scanGFI_1REP"
  "PCNetNoBatchNorm_PCConvReLU6_0.0eps_S2NoMinusZChgZNoisyI_dopri5Solver_1.5TEnd_0.0001Tol_0.001WD_128BS_0.01LR_3K1S64C_0.25Dropout_7Layers_2Pool_scanGFI_2REP"
  "PCNetNoBatchNorm_PCConvReLU6_0.0eps_S2NoMinusZChgZNoisyI_dopri5Solver_1.5TEnd_0.0001Tol_0.001WD_128BS_0.01LR_3K1S_0.25Dropout_4Layers_2Pool_scanGFI_1REP"
  "PCNetNoBatchNorm_PCConvReLU6_0.0eps_S2NoMinusZChgZNoisyI_dopri5Solver_1.5TEnd_0.0001Tol_0.001WD_128BS_0.01LR_3K1S_0.25Dropout_5Layers_2Pool_scanGFI_2REP"
)
NBITS=(5)
###############################################################################################

# Paths
REPO_ROOT="/scratch/rzeng7/repos/PCN-with-Local-Recurrent-Processing"
SBATCH_SCRIPT="${REPO_ROOT}/launch_scripts/run_qat_ode_train.sbatch"

SLURM_LOG_DIR="${REPO_ROOT}/logs/slurm_jobs"
# track jobids per EXP
declare -A JOBS_BY_EXP
# where we’ll save merged CSVs (same subtree as your parser output)
MERGE_OUT_DIR="${REPO_ROOT}/shell_utils/parse_res/neural_ode_res"
MERGE_SCRIPT="${REPO_ROOT}/shell_utils/merge_csvs.py"

split_and_submit() {
  local ft_model_name="$1" n_bits="$2"
  ###############################################################################################
  # Change EXP name here
  ###############################################################################################
  local EXP="no_bn_${ft_model_name}_NODE_QAT_noise_inject_1023_scanGFI_2State_Exp"
  ###############################################################################################
  echo "Submitting Model_Name=${ft_model_name} n_bits=${n_bits} → EXP=${EXP}"
  jid=$( sbatch --parsable \
           --gres=gpu:${GPUS_PER_JOB} \
           --export=ALL,MODEL_NAME="${ft_model_name}",NBITS="${n_bits}",EXP="${EXP}" \
           "${SBATCH_SCRIPT}" )
  echo "  -> job ${jid}"
  JOBS_BY_EXP["${EXP}"]+="${jid}:"
  i=$end
}

wait_for_jobs() {
  local -a ids=("$@")
  for i in "${!ids[@]}"; do
    ids[$i]="${ids[$i]%%;*}"
    ids[$i]="${ids[$i]%%.*}"
  done
  [[ ${#ids[@]} -gt 0 ]] || return 0

  while :; do
    local done=0
    for j in "${ids[@]}"; do
      # first line = job summary (not steps)
      local state
      state=$(sacct -X -n -j "$j" -o State 2>/dev/null | head -n1)
      case "$state" in
        *COMPLETED*|*FAILED*|*CANCELLED*|*TIMEOUT*|*OUT_OF_MEMORY*) ((done++)) ;;
        ""|RUNNING|PENDING|CONFIGURING|COMPLETING|SUSPENDED|REQUEUED|RESIZING|PREEMPTED|NODE_FAIL) : ;;
        *) : ;;
      esac
    done
    (( done == ${#ids[@]} )) && break
    sleep 20
  done
}

merge_csvs_for_exp() {
  local exp="$1"
  local joblist="${JOBS_BY_EXP[$exp]}"
  joblist="${joblist%:}"          # trim trailing colon
  IFS=':' read -r -a ids <<< "$joblist"
  echo "EXP: ${exp}, joblist: ${joblist}"

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
for MODEL_NAME in "${MODEL_NAMES[@]}"; do
  for NBIT in "${NBITS[@]}"; do
    split_and_submit "$MODEL_NAME" "$NBIT"
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
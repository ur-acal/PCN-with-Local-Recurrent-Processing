#!/usr/bin/env bash
#set -euo pipefail

# Pack how many combos to run within ONE sbatch job (one GPU)
MAX_TASKS_PER_GPU="${MAX_TASKS_PER_GPU:-1}"
GPUS_PER_JOB="${GPUS_PER_JOB:-1}"

###############################################################################################
# running with
# module swap slurm slurm/24.05.0.b1
# ( source ./launch_scripts/slurm_search_config.sh ) \
#  > ./logs/scheduler_slurm/search_scheduler.log 2>&1 < /dev/null &
#
# sched_pid=$!
# disown -h "$sched_pid"
# run with SHOW_COMB_ONLY model
# SHOW_COMB_ONLY=1 ./launch_scripts/slurm_search_config.sh
###############################################################################################

TASK="${TASK:-cifar100}"                 # for naming / future use
ODE_BLOCK="${ODE_BLOCK:-ODEXInitFFFB}"   # fixed block for now
NUM_COMB_PER_NUM_LAYER="${NUM_COMB_PER_NUM_LAYER:-6}"

PCNS=( "PCNetNoBatchNorm" )
IMG_TYPES=( "scanGFI" )
CIRC_CONFS=( "" )

# CHAN_0 options (order matters)
CHAN_0_LIST=( 28 30 32 )

# NUM_LAYERS dict: key=CHAN_0, value="layers..."
declare -A NUM_LAYERS_BY_CHAN0
NUM_LAYERS_BY_CHAN0[28]="13 14 15"
NUM_LAYERS_BY_CHAN0[30]="11 12 13"
NUM_LAYERS_BY_CHAN0[32]="7 8 9 10"

REPO_ROOT="/scratch/rzeng7/repos/PCN-with-Local-Recurrent-Processing"
SBATCH_SCRIPT="${REPO_ROOT}/launch_scripts/run_search_config.sbatch"
SLURM_LOG_DIR="${REPO_ROOT}/logs/slurm_jobs"

declare -A JOBS_BY_EXP
MERGE_OUT_DIR="${REPO_ROOT}/shell_utils/parse_res/neural_ode_res"
MERGE_SCRIPT="${REPO_ROOT}/shell_utils/merge_csvs.py"

SHOW_COMB_ONLY="${SHOW_COMB_ONLY:-0}"   # 1 => print comb_tag only and exit
COMB_MODE="${COMB_MODE:-balanced_A}"      # choose in (n_params | balanced_A)

SUMMARY_CSV_SCRIPT="${REPO_ROOT}/shell_utils/summary_csvs_as_dict.py"
# Change the saved pickle file name here
SUMMARY_PKL_OUT="${MERGE_OUT_DIR}/summary_dict_0204_balanced_A.pkl"

# ---------------------------
# Generate top-K combinations for a given (chan0, num_layers).
# Output format per line (TAB-delimited, 7 fields):
#   params \t comb_tag \t chan0 \t num_layers \t inp_str \t out_str \t pool_str
# ---------------------------
generate_combs() {
  local chan0="$1"
  local num_layers="$2"
  local k="$3"

  case "$COMB_MODE" in
    n_params)    generate_top_combs   "$chan0" "$num_layers" "$k" ;;
    balanced_A)  generate_ruleA_combs "$chan0" "$num_layers" "$k" ;;
    *)
      echo "ERROR: Unknown COMB_MODE='$COMB_MODE' (expected: n_params | balanced_A)" >&2
      return 2
      ;;
  esac
}

generate_top_combs() {
  local chan0="$1"
  local num_layers="$2"
  local topk="$3"

  if (( num_layers < 6 )); then
    echo "ERROR: NUM_LAYERS must be >= 6 (got ${num_layers})" >&2
    return 1
  fi

  local chan1=$((chan0 * 2))
  local chan2=$((chan0 * 4))
  local extra=$((num_layers - 6))
  local K2=9  # 3x3

  local lines=()
  local a0 a1 a2 n0 n1 n2 params
  local -a INP OUT POOL
  local i

  for ((a0=0; a0<=extra; a0++)); do
    for ((a1=0; a1<=extra-a0; a1++)); do
      a2=$((extra - a0 - a1))

      n0=$((1 + a0))
      n1=$((1 + a1))
      n2=$((1 + a2))

      INP=(); OUT=(); POOL=()

      # L1: 3 -> c0
      INP+=(3);         OUT+=("$chan0"); POOL+=(0)

      # Stage0: c0->c0 (n0), pool on last
      for ((i=1; i<=n0; i++)); do
        INP+=("$chan0"); OUT+=("$chan0")
        (( i == n0 )) && POOL+=(1) || POOL+=(0)
      done

      # Exp1: c0->c1
      INP+=("$chan0"); OUT+=("$chan1"); POOL+=(0)

      # Stage1: c1->c1 (n1), pool on last
      for ((i=1; i<=n1; i++)); do
        INP+=("$chan1"); OUT+=("$chan1")
        (( i == n1 )) && POOL+=(1) || POOL+=(0)
      done

      # Exp2: c1->c2
      INP+=("$chan1"); OUT+=("$chan2"); POOL+=(0)

      # Stage2: c2->c2 (n2), never pool
      for ((i=1; i<=n2; i++)); do
        INP+=("$chan2"); OUT+=("$chan2"); POOL+=(0)
      done

      params=$(( K2 * (
        3*chan0 +
        n0*chan0*chan0 +
        chan0*chan1 +
        n1*chan1*chan1 +
        chan1*chan2 +
        n2*chan2*chan2
      ) ))

      local comb_tag="N${num_layers}_C${chan0}_n0${n0}_n1${n1}_n2${n2}"
      lines+=( "${params}"$'\t'"${comb_tag}"$'\t'"${chan0}"$'\t'"${num_layers}"$'\t'"${INP[*]}"$'\t'"${OUT[*]}"$'\t'"${POOL[*]}" )
    done
  done

  printf '%s\n' "${lines[@]}" | sort -nr -k1,1 | head -n "${topk}"
}

generate_ruleA_combs() {
  local chan0="$1"
  local num_layers="$2"
  local k="$3"

  if (( num_layers < 6 )); then
    echo "ERROR: NUM_LAYERS must be >= 6 (got ${num_layers})" >&2
    return 1
  fi

  local chan1=$((chan0 * 2))
  local chan2=$((chan0 * 4))
  local extra=$((num_layers - 6))
  local q=$((extra / 3))
  local K2=9  # 3x3

  abs() { local x=$1; (( x < 0 )) && echo $(( -x )) || echo "$x"; }
  max3() { local a=$1 b=$2 c=$3; local m=$a; (( b>m )) && m=$b; (( c>m )) && m=$c; echo "$m"; }
  min3() { local a=$1 b=$2 c=$3; local m=$a; (( b<m )) && m=$b; (( c<m )) && m=$c; echo "$m"; }

  local lines=()
  local a0 a1 a2 n0 n1 n2 params
  local spread l1 maxv minv
  local -a INP OUT POOL
  local i

  for ((a0=0; a0<=extra; a0++)); do
    for ((a1=0; a1<=extra-a0; a1++)); do
      a2=$((extra - a0 - a1))

      n0=$((1 + a0))
      n1=$((1 + a1))
      n2=$((1 + a2))

      maxv="$(max3 "$a0" "$a1" "$a2")"
      minv="$(min3 "$a0" "$a1" "$a2")"
      spread=$((maxv - minv))
      l1=$(( $(abs $((a0-q))) + $(abs $((a1-q))) + $(abs $((a2-q))) ))

      INP=(); OUT=(); POOL=()

      INP+=(3);         OUT+=("$chan0"); POOL+=(0)

      for ((i=1; i<=n0; i++)); do
        INP+=("$chan0"); OUT+=("$chan0")
        (( i == n0 )) && POOL+=(1) || POOL+=(0)
      done

      INP+=("$chan0"); OUT+=("$chan1"); POOL+=(0)

      for ((i=1; i<=n1; i++)); do
        INP+=("$chan1"); OUT+=("$chan1")
        (( i == n1 )) && POOL+=(1) || POOL+=(0)
      done

      INP+=("$chan1"); OUT+=("$chan2"); POOL+=(0)

      for ((i=1; i<=n2; i++)); do
        INP+=("$chan2"); OUT+=("$chan2"); POOL+=(0)
      done

      params=$(( K2 * (
        3*chan0 +
        n0*chan0*chan0 +
        chan0*chan1 +
        n1*chan1*chan1 +
        chan1*chan2 +
        n2*chan2*chan2
      ) ))

      local comb_tag="N${num_layers}_C${chan0}_n0${n0}_n1${n1}_n2${n2}"

      # spread \t l1 \t params \t comb_tag \t chan0 \t num_layers \t inp \t out \t pool
      lines+=( "${spread}"$'\t'"${l1}"$'\t'"${params}"$'\t'"${comb_tag}"$'\t'"${chan0}"$'\t'"${num_layers}"$'\t'"${INP[*]}"$'\t'"${OUT[*]}"$'\t'"${POOL[*]}" )
    done
  done

  printf '%s\n' "${lines[@]}" \
    | sort -n -k1,1 -k2,2 -k3,3r \
    | head -n "${k}" \
    | awk -F'\t' 'BEGIN{OFS="\t"} {print $3,$4,$5,$6,$7,$8,$9}'
}

# Submit ONE sbatch job for the current chunk.
# Inputs:
#   $1 pcn, $2 img_type, $3 circ_conf, $4 chan0, $5 num_layers, $6 chunk_id, $7 chunk_tag, $8 comb_list
submit_chunk() {
  if (( ${SHOW_COMB_ONLY:-0} == 1 )); then
    # Do NOT submit in show comb only mode.
    return 0
  fi

  local pcn="$1" img_type="$2" circ_conf="$3"
  local chan0="$4" num_layers="$5"
  # chunk_tag and chunk_id are used for creating unique EXP for log files
  # If MAX_TASKS_PER_GPU > 1, chunk_tag is first_comb_tag__to__last_comb_tag
  local chunk_id="$6" chunk_tag="$7"
  # comb_list contains multiple configs used for training and is separated by new-lines
  # if MAX_TASKS_PER_GPU > 1.
  # Each string in comb_list is separated by tab.
  local comb_list="$8"

  # Human-readable + unique EXP:
  # - if MAX_TASKS_PER_GPU==1, chunk_tag will be the exact comb_tag
  # - else it is first__to__last
  local EXP="no_bn_${pcn}_NODE_search_${TASK}_${img_type}_${circ_conf:-NoCirc}_C${chan0}_N${num_layers}_${chunk_tag}_chunk${chunk_id}_Exp"

  # one fixed block; keep passing BLOCKS_LIST for sbatch compatibility
  local BLOCKS_LIST="${ODE_BLOCK}"

  echo "Submitting: C${chan0} N${num_layers} chunk=${chunk_id} tag=${chunk_tag} combos_in_job=${MAX_TASKS_PER_GPU}"
  jid=$(
    BLOCKS_LIST="${BLOCKS_LIST}" \
    sbatch --parsable \
      --gres=gpu:${GPUS_PER_JOB} \
      --export=ALL,PCN="${pcn}",IMG_TYPE="${img_type}",EXP="${EXP}",CIRC_CONF="${circ_conf}",TASK="${TASK}",ODE_BLOCK="${ODE_BLOCK}",CHAN_0="${chan0}",NUM_LAYERS="${num_layers}",CHUNK_ID="${chunk_id}",CHUNK_TAG="${chunk_tag}",COMB_LIST="${comb_list}" \
      "${SBATCH_SCRIPT}"
  )
  echo "  -> job ${jid}"
  JOBS_BY_EXP["${EXP}"]+="${jid}:"
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
  joblist="${joblist%:}"
  IFS=':' read -r -a ids <<< "$joblist"
  echo "EXP: ${exp}, joblist: ${joblist}"

  wait_for_jobs "${ids[@]}"

  mkdir -p "${MERGE_OUT_DIR}"
  local merged_out="${MERGE_OUT_DIR}/${exp}_merged.csv"

  local csvs=()
  for j in "${ids[@]}"; do
    local slog="${SLURM_LOG_DIR}/slurm_${j}.out"
    [[ -s "$slog" ]] || { echo "[WARN] Missing slurm log $slog"; continue; }
    local last; last=$(tail -n 1 "$slog" || true)
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

  echo "[MERGE] ${exp}: ${#csvs[@]} files -> ${merged_out}"
  python -u "${MERGE_SCRIPT}" "${merged_out}" "${csvs[@]}"
  echo "[MERGED SAVED] ${merged_out}"
}

# ---------------------------
# Main launch: pack combos into COMB_LIST chunks.
#
# COMB_LIST content (newline-delimited records):
#   comb_tag<TAB>params<TAB>inp_str<TAB>out_str<TAB>pool_str
#
# This avoids fragile delimiters like commas/semicolons.
# ---------------------------
for IMG_TYPE in "${IMG_TYPES[@]}"; do
  for PCN in "${PCNS[@]}"; do
    for CIRC_CONF in "${CIRC_CONFS[@]}"; do
      for CHAN_0 in "${CHAN_0_LIST[@]}"; do
        layers_str="${NUM_LAYERS_BY_CHAN0[${CHAN_0}]:-}"
        [[ -n "${layers_str}" ]] || { echo "[WARN] No layers for CHAN_0=${CHAN_0}, skip."; continue; }
        read -r -a LAYERS_ARR <<< "${layers_str}"

        for NUM_LAYERS in "${LAYERS_ARR[@]}"; do
          chunk_id=0
          chunk_count=0
          first_tag=""
          last_tag=""
          comb_list=""

          if (( SHOW_COMB_ONLY == 1 )); then
            echo "CHAN_0=${CHAN_0} NUM_LAYERS=${NUM_LAYERS}"
          fi

          while IFS=$'\t' read -r params comb_tag chan0 num_layers inp_str out_str pool_str; do
            if (( SHOW_COMB_ONLY == 1 )); then
              echo "${comb_tag}"
              continue
            fi

            # record: comb_tag<TAB>params<TAB>inp_str<TAB>out_str<TAB>pool_str
            rec="${comb_tag}"$'\t'"${params}"$'\t'"${inp_str}"$'\t'"${out_str}"$'\t'"${pool_str}"

            if (( chunk_count == 0 )); then
              comb_list="${rec}"
              first_tag="${comb_tag}"
              last_tag="${comb_tag}"
            else
              # If multiple combs, comb_list is separated by new lines.
              comb_list+=$'\n'"${rec}"
              last_tag="${comb_tag}"
            fi

            ((chunk_count++))

            if (( chunk_count >= MAX_TASKS_PER_GPU )); then
              # if only 1 combo in chunk, make EXP include exact comb tag
              if (( chunk_count == 1 )); then
                chunk_tag="${first_tag}"
              else
                chunk_tag="${first_tag}__to__${last_tag}"
              fi
              submit_chunk "${PCN}" "${IMG_TYPE}" "${CIRC_CONF}" "${CHAN_0}" "${NUM_LAYERS}" "${chunk_id}" "${chunk_tag}" "${comb_list}"
              ((chunk_id++))
              chunk_count=0
              first_tag=""
              last_tag=""
              comb_list=""
            fi
          done < <(generate_combs "${CHAN_0}" "${NUM_LAYERS}" "${NUM_COMB_PER_NUM_LAYER}")

          # flush remainder
          if (( chunk_count > 0 )); then
            if (( chunk_count == 1 )); then
              chunk_tag="${first_tag}"
            else
              chunk_tag="${first_tag}__to__${last_tag}"
            fi
            submit_chunk "${PCN}" "${IMG_TYPE}" "${CIRC_CONF}" "${CHAN_0}" "${NUM_LAYERS}" "${chunk_id}" "${chunk_tag}" "${comb_list}"
          fi
        done
      done
    done
  done
done

# In show-only mode: stop after printing all combs (CHAN_0, NUM_LAYERS)
if (( SHOW_COMB_ONLY == 1 )); then
  exit 0
fi

# Wait and merge
for exp in "${!JOBS_BY_EXP[@]}"; do
  merge_csvs_for_exp "$exp"
done

echo "Merged CSVs:"
CSV_PATHS=()
for exp in "${!JOBS_BY_EXP[@]}"; do
  csv_path="${MERGE_OUT_DIR}/${exp}_merged.csv"
  echo "  ${csv_path}"
  [[ -f "${csv_path}" ]] && CSV_PATHS+=( "${csv_path}" )
done


if ((${#CSV_PATHS[@]} > 0)); then
  echo "[SUMMARY] Writing dict pickle -> ${SUMMARY_PKL_OUT}"
  python -u "${SUMMARY_CSV_SCRIPT}" --out "${SUMMARY_PKL_OUT}" "${CSV_PATHS[@]}"
else
  echo "[SUMMARY] No merged CSVs found; skip pickle summary."
fi


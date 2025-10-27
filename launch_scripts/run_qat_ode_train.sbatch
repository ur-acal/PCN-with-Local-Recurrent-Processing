#!/bin/bash -l
#SBATCH -p ising
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16            # modest CPU request so the node can be shared
#SBATCH --gres=gpu:1             # exactly ONE GPU; allows packing on 4-GPU nodes
#SBATCH -t 60:10:00
#SBATCH -o /scratch/rzeng7/repos/PCN-with-Local-Recurrent-Processing/logs/slurm_jobs/slurm_%j.out

#set -euo pipefail

# ---- Conda activation (non-interactive safe) ----
source activate base
conda activate scanbase

##########################################################################
# Change exp name here.
##########################################################################
MODEL_NAME=${MODEL_NAME:-""}
NBITS=${NBITS:-""}

# Get PC_CONV, ODE_BLOCK, IMG_TYPE
__rest="${MODEL_NAME#*_}"
PC_CONV="${__rest%%_*}"

IFS='_' read -r -a parts <<< "$MODEL_NAME"
ODE_BLOCK="${parts[3]}"
prev="${parts[${#parts[@]}-2]}"
if [[ "$prev" == *Layers || "$prev" == *Pool ]]; then
  IMG_TYPE="rgb"
else
  IMG_TYPE="$prev"
fi

# EXP parameters
R_val="1e5"
C_val="49e-15"
t_end="1.5"
tie_cap="true"
ONE_OVER_Q="6"
TRAIN_TOL="1e-4"

EXP="${EXP:-no_bn_${PCN}_NODE_0822_relu6_FFFB_DYN_${ARCH_LABEL}_Exp}"
LOGDIR="/scratch/rzeng7/repos/PCN-with-Local-Recurrent-Processing/logs/${EXP}_${NBITS}"
mkdir -p "$LOGDIR"
##########################################################################

# On cancel/terminate, stop children too
trap 'echo "Signal caught; killing children..."; kill -- -$$ 2>/dev/null || true' TERM INT


# Split CPU threads evenly across the selected number of blocks
CPUS=${SLURM_CPUS_PER_TASK:-16}
PROCS=1
THREADS_PER_PROC=$(( CPUS / PROCS ))
if (( THREADS_PER_PROC < 1 )); then THREADS_PER_PROC=1; fi

# ---- Per-process GPU memory fraction ----------------------------------
MAX_MEM_FRAC=${MAX_MEM_FRAC:-0.9}
MEM_FRAC=$(awk -v m="$MAX_MEM_FRAC" -v n="$PROCS" 'BEGIN{printf "%.6f", m/n}')
echo "Per-proc --mem_frac=${MEM_FRAC}  (MAX_MEM_FRAC=${MAX_MEM_FRAC}, PROCS=${PROCS})"

echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}  (SLURM set)"
echo "CPUs per task: $CPUS  → OMP_NUM_THREADS per proc: $THREADS_PER_PROC"
echo "EXP=${EXP}  MODEL_NAME=${MODEL_NAME}  NBITS=${NBITS}"
echo "Logs in: $LOGDIR"

map_qat_block() {
  case "$1" in
    S2NoMinusZChgZNoisyI)          echo "QATWrapper2State" ;;
    S2NoMinusZChgZMinusNoisyI)     echo "QATWrapper2State" ;;
    *)                         echo "$1" ;;
  esac
}

run_one () {
  local logfile="${LOGDIR}/train_${EXP}_No_2_ReLU6_${NBITS}.log"
  echo "[$(date +%T)] starting ${NBITS} -> ${logfile}"

  local wrapper="$(map_qat_block "$ODE_BLOCK")"

  OMP_NUM_THREADS=${THREADS_PER_PROC} \
  python -u train_ode_cifar.py \
    --optim         "SGD" \
    --img_type      "${IMG_TYPE}" \
    --learning_rate 0.005 \
    --cosine_t0     20 \
    --num_epochs    80 \
    --model_name    "${MODEL_NAME}" \
    --eval_every    1 \
    --offset_eps    0.0 \
    --dropout       0.25 \
    --tie_weights   "false" \
    --tie_bp        "false" \
    --bypass        "false" \
    --batch_size    128 \
    --method        "dopri5" \
    --tol           "${TRAIN_TOL}" \
    --R             "${R_val}" \
    --C             "${C_val}" \
    --w_bits        "${NBITS}" \
    --tie_cap       "${tie_cap}" \
    --one_over_q    "${ONE_OVER_Q}" \
    --qat_cls       "SymQuantizeWeight" \
    --ode_wrapper   "${wrapper}" \
    --t_end         "${t_end}" \
    --pc_conv       "${PC_CONV}" \
    --ode_block     "${ODE_BLOCK}" \
    --mem_frac      "${MEM_FRAC}" \
    --noise_level   "0.4" \
    --noise_type    "mul" \
    2>&1 | tee "${logfile}"
}

# Launch the four ode_blocks concurrently on this ONE GPU
pids=()
run_one & pids+=($!)

ok=0
for p in "${pids[@]}"; do
  wait "$p" || ok=1
done
echo "Completed all NBITS for ${EXP} (job $SLURM_JOB_ID)."

################################################################################
#                           EVAL → FINETUNE → EVAL                             #
################################################################################

MODEL_DIR="./saved_ckpt"
METHOD="dopri5"
TOL="1e-6"
EVAL_BASE="${LOGDIR}/evals"
MASTER_LOG="${LOGDIR}/eval_master_merged.log"
mkdir -p "${EVAL_BASE}"
: > "${MASTER_LOG}"

# Train→Test block mapping
map_test_wrapper() {
  case "$1" in
    S2NoMinusZChgZNoisyI)          echo "QATTester2State" ;;
    S2NoMinusZChgZMinusNoisyI)     echo "QATTester2State" ;;
    *)                         echo "$1" ;;
  esac
}

# Helpers
extract_model_name() {
  local logfile="$1"
  sed -n 's/.*Model Name:[[:space:]]*\(.*\)[[:space:]]*-----.*$/\1/p' "$logfile" \
    | tail -n 1 \
    | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//'
}

eval_model() {
  # name, tag (pre_ft|post_ft), TEST_BLOCK
  local name="$1"
  local dir="${EVAL_BASE}/${NBITS}/${name}_method_${METHOD}_tol_${TOL}"
  mkdir -p "$dir"
  local ckpt="best"
  ################################
  local wrapper="$(map_test_wrapper "$ODE_BLOCK")"
  ################################
  echo "[EVAL ${NBITS}] ${name} (test_block=${ODE_BLOCK}, ckpt=${ckpt}) → $dir/job.log"
  OMP_NUM_THREADS=${THREADS_PER_PROC} \
  python -u ode_inference.py \
    --model_name      "$name" \
    --ckpt            "$ckpt" \
    --model_dir       "$MODEL_DIR" \
    --img_type        "${IMG_TYPE}" \
    --method          "$METHOD" \
    --tol             "$TOL" \
    --n_steps         15 \
    --ts_scale        1 \
    --d_start         0 \
    --d_end           1 \
    --n_sweep_left    0 \
    --n_sweep_right   1 \
    --R               "${R_val}" \
    --C               "${C_val}" \
    --w_bits          "${NBITS}" \
    --tie_cap         "${tie_cap}" \
    --one_over_q      "${ONE_OVER_Q}" \
    --pc_conv         "${PC_CONV}Noisy" \
    --ode_block       "${ODE_BLOCK}" \
    --ode_wrapper     "${wrapper}" \
    --mem_frac        "${MEM_FRAC}" \
    2>&1 | tee -a "$dir/job.log"
}

# Collect model names from training logs
declare -a MODEL_NAMES_PRE=()
logf="${LOGDIR}/train_${EXP}_No_2_ReLU6_${NBITS}.log"
mn="$(extract_model_name "$logf")"
if [[ -z "$mn" ]]; then
  echo "[WARN] Could not extract model name from $logf"
fi
MODEL_NAMES_PRE+=("$mn")

# EVAL in parallel
pids=()
for i in "${!MODEL_NAMES_PRE[@]}"; do
  eval_model "${MODEL_NAMES_PRE[$i]}" & pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done

# Merge summaries for both evals into one master log
echo "Merging summaries into ${MASTER_LOG}"
: > "${MASTER_LOG}"
for i in "${!MODEL_NAMES_PRE[@]}"; do
  pre="${MODEL_NAMES_PRE[$i]}"
  pre_log="${EVAL_BASE}/${NBITS}/${pre}_method_${METHOD}_tol_${TOL}/job.log"

  printf '========== %s (pre-finetune) ==========\n' "$pre" >>"$MASTER_LOG"
  if [[ -f "$pre_log" ]]; then
    if ! grep -A 1000 "Model name: ${pre} " "$pre_log" >>"$MASTER_LOG"; then
      echo "[Final Result not found]" >>"$MASTER_LOG"
    fi
  else
    echo "[Missing log: $pre_log]" >>"$MASTER_LOG"
  fi
  printf '\n' >>"$MASTER_LOG"
done

echo "All summaries written to ${MASTER_LOG}"

################################################################################
#                               RUN PARSER
################################################################################
PARSE_WRAPPER="shell_utils/pipeline_node_run_parse.py"
if [[ -f "$PARSE_WRAPPER" ]]; then
  echo "[PARSE] python $PARSE_WRAPPER ${MASTER_LOG}"
  python -u "$PARSE_WRAPPER" "${MASTER_LOG}"
else
  echo "[PARSE] Skipping parse: ${PARSE_WRAPPER} not found."
fi
exit $ok
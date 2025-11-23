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
IMG_TYPE=${IMG_TYPE:-rgb}
ARCH_SET=${ARCH_SET:-A}      # A or B
PCN=${PCN:-PCNetNoBatchNorm} # "PCNetNoBatchNorm" | "PCNetWith1stConv"
if [[ "$ARCH_SET" == "5L2PS" ]]; then
  ARCH_LABEL="5L2PSmall"
elif [[ "$ARCH_SET" == "5L1PS" ]]; then
  ARCH_LABEL="5L1PSmall"
elif [[ "$ARCH_SET" == "5L2PM" ]]; then
  ARCH_LABEL="5L2PMedium"
elif [[ "$ARCH_SET" == "5L1PM" ]]; then
  ARCH_LABEL="5L1PMedium"
elif [[ "$ARCH_SET" == "7L2PL" ]]; then
  ARCH_LABEL="7L2PLarge"
elif [[ "$ARCH_SET" == "7L1PL" ]]; then
  ARCH_LABEL="7L1PLarge"
elif [[ "$ARCH_SET" == "18L2PTi" ]]; then
  ARCH_LABEL="18L2PTiny"
elif [[ "$ARCH_SET" == "18L1PTi" ]]; then
  ARCH_LABEL="18L1PTiny"
else
  ARCH_LABEL="Invalid"
fi

SAN_BLOCKS="${BLOCKS_LIST:-ALL}"
SAN_BLOCKS="${SAN_BLOCKS//,/+}"
EXP="${EXP:-no_bn_${PCN}_ppcn_1121_relu6_FFFB_${ARCH_LABEL}_Exp}"
LOGDIR="/scratch/rzeng7/repos/PCN-with-Local-Recurrent-Processing/logs/${EXP}_${SAN_BLOCKS}"
mkdir -p "$LOGDIR"

TRAIN_BATCH_SIZE="${BATCH_SIZE:-128}"
##########################################################################

# On cancel/terminate, stop children too
trap 'echo "Signal caught; killing children..."; kill -- -$$ 2>/dev/null || true' TERM INT

KERNEL=(3)
padding=1
# ---- Arch definitions ----
if [[ "$ARCH_SET" == "5L2PM" ]]; then
  INP=(3  32 64 64  128)
  OUT=(32 64 64 128 128)
  POOL=(0  1  0  1   0)
  STRIDE=(1)
elif [[ "$ARCH_SET" == "5L1PM" ]]; then
  INP=(3  32 64 64  128)
  OUT=(32 64 64 128 128)
  POOL=(0  1  0  0   0)
  STRIDE=(1)
elif [[ "$ARCH_SET" == "18L2PTi" ]]; then
  # deep 0.14M model - 32 Chan 2 Pooling
  INP=(3  8 8 8 8 8 8  16 16 16 16 16 32 32 32 32 32 32)
  OUT=(8  8 8 8 8 8 16 16 16 16 16 32 32 32 32 32 32 32)
  POOL=(0 0 0 0 0 0 1  0  0  0  0  1  0  0  0  0  0  0)
  STRIDE=(1)
elif [[ "$ARCH_SET" == "18L1PTi" ]]; then
  # deep 0.14M model - 32 Chan 1 Pooling
  INP=(3  8 8 8 8 8 8  16 16 16 16 16 32 32 32 32 32 32)
  OUT=(8  8 8 8 8 8 16 16 16 16 16 32 32 32 32 32 32 32)
  POOL=(0 0 0 0 0 0 1  0  0  0  0  0  0  0  0  0  0  0)
  STRIDE=(1)
elif [[ "$ARCH_SET" == "7L2PL" ]]; then
  INP=(3  32 32 64 64  128 128)
  OUT=(32 32 64 64 128 128 128)
  POOL=(0 0  1  0  1   0   0)
  STRIDE=(1)
elif [[ "$ARCH_SET" == "7L1PL" ]]; then
  INP=(3  32 32 64 64  128 128)
  OUT=(32 32 64 64 128 128 128)
  POOL=(0 0  1  0  0   0   0)
  STRIDE=(1)
elif [[ "$ARCH_SET" == "5L2PS" ]]; then
  INP=(3  32 64 64 64)
  OUT=(32 64 64 64 64)
  POOL=(0  1  0  1  0)
  STRIDE=(1)
elif [[ "$ARCH_SET" == "5L1PS" ]]; then
  INP=(3  32 64 64 64)
  OUT=(32 64 64 64 64)
  POOL=(0  1  0  0  0)
  STRIDE=(1)
else
  echo "ARCH_SET must be A or B or C[2]"; exit 2
fi

if [[ "$IMG_TYPE" != "rgb" ]]; then
  INP[0]=4
fi


##########################################################################
# Blocks are passed in
##########################################################################
# Caller override: run a single BLOCK or a custom list (comma/space separated)
# The BLOCKS set above will be overwritten by BLOCKS passed in
if [[ -n "${BLOCKS_LIST:-}" ]]; then
  IFS=',' read -r -a BLOCKS <<< "${BLOCKS_LIST}"
fi
echo "BLOCKS_LIST=${BLOCKS_LIST}"
echo "Parsed ${#BLOCKS[@]} blocks: ${BLOCKS[*]}"
##########################################################################

is_finetunable() {
  local b="$1"
  if [[ "${FINETUNE:-0}" == "0" ]]; then
    return 0
  fi
  return 1
}

##########################################################################

# Split CPU threads evenly across the selected number of blocks
CPUS=${SLURM_CPUS_PER_TASK:-16}
PROCS=${#BLOCKS[@]}
THREADS_PER_PROC=$(( CPUS / PROCS ))
if (( THREADS_PER_PROC < 1 )); then THREADS_PER_PROC=1; fi

# ---- Per-process GPU memory fraction ----------------------------------
MAX_MEM_FRAC=${MAX_MEM_FRAC:-0.9}
PROCS=${#BLOCKS[@]}
MEM_FRAC=$(awk -v m="$MAX_MEM_FRAC" -v n="$PROCS" 'BEGIN{printf "%.6f", m/n}')
echo "Per-proc --mem_frac=${MEM_FRAC}  (MAX_MEM_FRAC=${MAX_MEM_FRAC}, PROCS=${PROCS})"

echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}  (SLURM set)"
echo "CPUs per task: $CPUS  → OMP_NUM_THREADS per proc: $THREADS_PER_PROC"
echo "EXP=${EXP}  ARCH_SET=${ARCH_SET}  PCN=${PCN}"
echo "Logs in: $LOGDIR"

run_one () {
  local block="$1"
  local logfile="${LOGDIR}/train_${EXP}_No_2_ReLU6_${block}.log"
  echo "[$(date +%T)] starting ${block} -> ${logfile}"


  OMP_NUM_THREADS=${THREADS_PER_PROC} \
  python -u train_cifar.py \
    --optim         "SGD" \
    --img_type      "${IMG_TYPE}" \
    --num_epochs    2 \
    --eval_every    1 \
    --inp_channels  "${INP[@]}" \
    --out_channels  "${OUT[@]}" \
    --max_pool      "${POOL[@]}" \
    --dropout       0.25 \
    --lr_pc         0.15 \
    --cls           10 \
    --tie_weights   "false" \
    --tie_bp        "false" \
    --relu_between  "true" \
    --bypass        "false" \
    --batch_size    128 \
    --pcn           "${PCN}" \
    --pc_conv       "${block}" \
    --mem_frac      "${MEM_FRAC}" \
    2>&1 | tee "${logfile}"
}

# Launch the four ode_blocks concurrently on this ONE GPU
pids=()
for block in "${BLOCKS[@]}"; do
  run_one "${block}" & pids+=($!)
done

ok=0
for p in "${pids[@]}"; do
  wait "$p" || ok=1
done
echo "Completed all ode_blocks for ${EXP} (job $SLURM_JOB_ID)."

################################################################################
#                           EVAL → FINETUNE → EVAL                             #
################################################################################

MODEL_DIR="./saved_ckpt"
EVAL_BASE="${LOGDIR}/evals"
MASTER_LOG="${LOGDIR}/eval_master_merged.log"
mkdir -p "${EVAL_BASE}"
: > "${MASTER_LOG}"

# Train→Test pc conv mapping
map_test_block() {
  case "$1" in
    *)       echo "$1" ;;
  esac
}
TEST_BLOCKS=()
for b in "${BLOCKS[@]}"; do
  TEST_BLOCKS+=( "$(map_test_block "$b")" )
done

# Helpers
extract_model_name() {
  local logfile="$1"
  sed -n 's/.*Model Name:[[:space:]]*\(.*\)[[:space:]]*-----.*$/\1/p' "$logfile" \
    | tail -n 1 \
    | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//'
}

eval_model() {
  # name, tag (pre_ft|post_ft), TEST_BLOCK
  local name="$1"; local tag="$2"; local test_block="$3"
  local __rest="${name#*_}"; local pc_conv="${__rest%%_*}"
  local dir="${EVAL_BASE}/${tag}/${name}"
  mkdir -p "$dir"
  local ckpt="best"
  ################################
  # Not used in ppcn
  local n_bits=5
  local R_val="1e5"
  local C_val="49e-15"
  local test_bs=128
  ################################
  if [[ "$tag" == "post_ft" ]]; then ckpt="last"; fi

  echo "[EVAL ${tag}] ${name} (test pc conv=${test_block}, ckpt=${ckpt}) → $dir/job.log"
  OMP_NUM_THREADS=${THREADS_PER_PROC} \
  python -u run_test.py \
    --model_name      "${name}" \
    --model_dir       "${MODEL_DIR}" \
    --img_type        "${IMG_TYPE}" \
    --noise_to_ff     "true" \
    --noise_to_bp     "true" \
    --tie_noise       "false" \
    --tie_noise_bp    "false" \
    --noise_to_bn     "true" \
    --noise_to_linear "true" \
    --fuse_bn         "false" \
    --noisy_test      "true" \
    --pc_conv         "${test_block}Noisy" \
    --mem_frac        "${MEM_FRAC}" \
    2>&1 | tee -a "$dir/job.log"
}

finetune_model() {
  # Finetune uses the SAME training block for that experiment
  local name="$1"; local train_block="$2"
  local logfile="${LOGDIR}/finetune_${train_block}.log"
  echo "[FINETUNE] ${name} (pc conv=${train_block}) → $logfile"

  OMP_NUM_THREADS=${THREADS_PER_PROC} \
  python -u train_cifar.py \
    --optim         "SGD" \
    --img_type      "${IMG_TYPE}" \
    --num_epochs    150 \
    --eval_every    1 \
    --inp_channels  "${INP[@]}" \
    --out_channels  "${OUT[@]}" \
    --max_pool      "${POOL[@]}" \
    --dropout       0.25 \
    --lr_pc         0.15 \
    --cls           10 \
    --tie_weights   "false" \
    --tie_bp        "false" \
    --relu_between  "true" \
    --bypass        "false" \
    --batch_size    128 \
    --pcn           "${PCN}" \
    --pc_conv       "${train_block}" \
    --mem_frac      "${MEM_FRAC}" \
    2>&1 | tee "${logfile}"
}

# Collect model names from the 4 training logs
declare -a MODEL_NAMES_PRE=()
for block in "${BLOCKS[@]}"; do
  logf="${LOGDIR}/train_${EXP}_No_2_ReLU6_${block}.log"
  mn="$(extract_model_name "$logf")"
  if [[ -z "$mn" ]]; then
    echo "[WARN] Could not extract model name from $logf"
  fi
  MODEL_NAMES_PRE+=("$mn")
done

# EVAL (pre-finetune) in parallel
pids=()
for i in "${!MODEL_NAMES_PRE[@]}"; do
  eval_model "${MODEL_NAMES_PRE[$i]}" "pre_ft" "${TEST_BLOCKS[$i]}" & pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done

# FINETUNE only selected blocks in parallel
pids=()
for i in "${!MODEL_NAMES_PRE[@]}"; do
  b="${BLOCKS[$i]}"
  if is_finetunable "$b"; then
    finetune_model "${MODEL_NAMES_PRE[$i]}" "$b" & pids+=($!)
  fi
done
for p in "${pids[@]}"; do wait "$p"; done

# Collect finetuned model names from finetune logs
declare -a MODEL_NAMES_FT=()
declare -a TEST_BLOCKS_FT=()
declare -A POST_BY_BLOCK=()
for idx in "${!BLOCKS[@]}"; do
  b="${BLOCKS[$idx]}"
  if is_finetunable "$b"; then
    ftlog="${LOGDIR}/finetune_${b}.log"
    mn="$(extract_model_name "$ftlog")"
    if [[ -z "$mn" ]]; then
      echo "[WARN] Could not extract finetuned model name from $ftlog"
    fi
    MODEL_NAMES_FT+=("$mn")
    TEST_BLOCKS_FT+=("${TEST_BLOCKS[$idx]}")
    POST_BY_BLOCK["$b"]="$mn"
  fi
done

# EVAL (post-finetune) in parallel
pids=()
for i in "${!MODEL_NAMES_FT[@]}"; do
  eval_model "${MODEL_NAMES_FT[$i]}" "post_ft" "${TEST_BLOCKS_FT[$i]}" & pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done

# Merge summaries for both evals into one master log
echo "Merging summaries into ${MASTER_LOG}"
: > "${MASTER_LOG}"
for i in "${!BLOCKS[@]}"; do
  pre="${MODEL_NAMES_PRE[$i]}"
  post="${MODEL_NAMES_FT[$i]}"
  pre_log="${EVAL_BASE}/pre_ft/${pre}/job.log"
  post_log="${EVAL_BASE}/post_ft/${post}/job.log"

  printf '========== %s (pre-finetune) ==========\n' "$pre" >>"$MASTER_LOG"
  if [[ -f "$pre_log" ]]; then
    if ! grep -A 1000 "Model name: ${pre} " "$pre_log" >>"$MASTER_LOG"; then
      echo "[Final Result not found]" >>"$MASTER_LOG"
    fi
  else
    echo "[Missing log: $pre_log]" >>"$MASTER_LOG"
  fi
  printf '\n' >>"$MASTER_LOG"

  if is_finetunable "${BLOCKS[$i]}"; then
    post="${POST_BY_BLOCK[${BLOCKS[$i]}]-}"
    post_log="${EVAL_BASE}/post_ft/${post}/job.log"

    printf '%s\n' "---------- ${post} (post-finetune) ----------" >>"$MASTER_LOG"
    if [[ -f "$post_log" ]]; then
      if ! grep -A 1000 "Model name: ${post} " "$post_log" >>"$MASTER_LOG"; then
        echo "[Final Result not found]" >>"$MASTER_LOG"
      fi
    else
      echo "[Missing log: $post_log]" >>"$MASTER_LOG"
    fi
    printf '\n\n' >>"$MASTER_LOG"
  fi
done

echo "All summaries written to ${MASTER_LOG}"

################################################################################
#                               RUN PARSER
################################################################################
PARSE_WRAPPER="shell_utils/parse_noise_logs.py"
if [[ -f "$PARSE_WRAPPER" ]]; then
  echo "[PARSE] python $PARSE_WRAPPER ${MASTER_LOG}"
  python -u "$PARSE_WRAPPER" "${MASTER_LOG}"
else
  echo "[PARSE] Skipping parse: ${PARSE_WRAPPER} not found."
fi
exit $ok
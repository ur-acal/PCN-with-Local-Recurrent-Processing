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
if [[ "$ARCH_SET" == "A" ]]; then
  ARCH_LABEL="6L3P"
elif [[ "$ARCH_SET" == "A2p" ]]; then
  ARCH_LABEL="6L2P"
elif [[ "$ARCH_SET" == "A2" ]]; then
  ARCH_LABEL="6LStride2"
elif [[ "$ARCH_SET" == "Tiny_5L1P" ]]; then
  ARCH_LABEL="5L1P64C"
elif [[ "$ARCH_SET" == "Tiny_5L2P" ]]; then
  ARCH_LABEL="5L2P64C"
elif [[ "$ARCH_SET" == "Tiny_7L1P" ]]; then
  ARCH_LABEL="7L1P64C"
elif [[ "$ARCH_SET" == "Tiny_7L2P" ]]; then
  ARCH_LABEL="7L2P64C"
elif [[ "$ARCH_SET" == "Tiny_8L" ]]; then
  ARCH_LABEL="8L1P32C"
elif [[ "$ARCH_SET" == "Tiny_11L1P" ]]; then
  ARCH_LABEL="11L1P32C"
elif [[ "$ARCH_SET" == "Tiny_11L2P" ]]; then
  ARCH_LABEL="11L2P32C"
elif [[ "$ARCH_SET" == "DeepTi" ]]; then
  ARCH_LABEL="18L2P32C"
elif [[ "$ARCH_SET" == "DeepTi1P" ]]; then
  ARCH_LABEL="18L1P32C"
elif [[ "$ARCH_SET" == "DeepS" ]]; then
  ARCH_LABEL="9L2P"
elif [[ "$ARCH_SET" == "B" || "$ARCH_SET" == "DeepM" ]]; then
  ARCH_LABEL="18L2P"
elif [[ "$ARCH_SET" == "DeepM48C" ]]; then
  ARCH_LABEL="18L1P48C"
elif [[ "$ARCH_SET" == "DeepM1P" ]]; then
  ARCH_LABEL="18L1P"
elif [[ "$ARCH_SET" == "C" ]]; then
  ARCH_LABEL="7L2P"
elif [[ "$ARCH_SET" == "Large8L" ]]; then
  ARCH_LABEL="Large8L"
elif [[ "$ARCH_SET" == "Large9L" ]]; then
  ARCH_LABEL="Large9L"
elif [[ "$ARCH_SET" == "Large10L" ]]; then
  ARCH_LABEL="Large10L"
elif [[ "$ARCH_SET" == "Large8L1P" ]]; then
  ARCH_LABEL="Large8L1P"
elif [[ "$ARCH_SET" == "Large9L1P" ]]; then
  ARCH_LABEL="Large9L1P"
elif [[ "$ARCH_SET" == "Large10L1P" ]]; then
  ARCH_LABEL="Large10L1P"
elif [[ "$ARCH_SET" == "C2" ]]; then
  ARCH_LABEL="7LStride2"
elif [[ "$ARCH_SET" == "D2p" ]]; then
  ARCH_LABEL="4L2P"
elif [[ "$ARCH_SET" == "D2" ]]; then
  ARCH_LABEL="4LStride2"
elif [[ "$ARCH_SET" == "Ker5_A" ]]; then
  ARCH_LABEL="5LKer5"
elif [[ "$ARCH_SET" == "Ker5_C" ]]; then
  ARCH_LABEL="6LKer5"
elif [[ "$ARCH_SET" == "Ker3_D" ]]; then
  ARCH_LABEL="5LKer3"
elif [[ "$ARCH_SET" == "7L64Chan" ]]; then
  ARCH_LABEL="7L64Chan"
elif [[ "$ARCH_SET" == "7L64Chan1P" ]]; then
  ARCH_LABEL="7L64Chan1P"
else
  ARCH_LABEL="Invalid"
fi

CIRC_CONF="${CIRC_CONF:-NoCirc}"
SAN_BLOCKS="${BLOCKS_LIST:-ALL}"
SAN_BLOCKS="${SAN_BLOCKS//,/+}"
EXP="${EXP:-no_bn_${PCN}_NODE_0822_relu6_FFFB_DYN_${ARCH_LABEL}_Exp}"
LOGDIR="/scratch/rzeng7/repos/PCN-with-Local-Recurrent-Processing/logs/${EXP}_${SAN_BLOCKS}"
mkdir -p "$LOGDIR"

if [[ "${CIRC_CONF}" == "NoCirc" ]]; then
  patch_node=""; patch_stride=""; patch_cycle=""
else
  IFS='|' read -r patch_node patch_stride patch_cycle <<< "${CIRC_CONF}"
fi
##########################################################################

# On cancel/terminate, stop children too
trap 'echo "Signal caught; killing children..."; kill -- -$$ 2>/dev/null || true' TERM INT

KERNEL=(3)
padding=1
# ---- Arch definitions ----
if [[ "$ARCH_SET" == "A" ]]; then
  INP=(3  32 32 64 64  128)
  OUT=(32 32 64 64 128 128)
  POOL=(0  1  1  0  1   0)
  STRIDE=(1)
elif [[ "$ARCH_SET" == "A2p" ]]; then
  INP=(3  32 32 64 64  128)
  OUT=(32 32 64 64 128 128)
  POOL=(0  0  1  0  1   0)
  STRIDE=(1)
elif [[ "$ARCH_SET" == "A2" ]]; then
  INP=(3  32 32 64 64  128)
  OUT=(32 32 64 64 128 128)
  POOL=(0  0  0  0  0   0)
  STRIDE=(1  1  2  1  2   1)
elif [[ "$ARCH_SET" == "Tiny_5L1P" ]]; then
  INP=(3  16 16 32 64)
  OUT=(16 16 32 64 64)
  POOL=(0  0  1  0  0)
  STRIDE=(1)
elif [[ "$ARCH_SET" == "Tiny_5L2P" ]]; then
  INP=(3  16 16 32 64)
  OUT=(16 16 32 64 64)
  POOL=(0  0  1  1  0)
  STRIDE=(1)
elif [[ "$ARCH_SET" == "Tiny_7L1P" ]]; then
  INP=( 3 8 8 8  16 32 64)
  OUT=( 8 8 8 16 32 64 64)
  POOL=(0 0 0 1  0  0  0)
  STRIDE=(1)
elif [[ "$ARCH_SET" == "Tiny_7L2P" ]]; then
  INP=( 3 8 8 8  16 32 64)
  OUT=( 8 8 8 16 32 64 64)
  POOL=(0 0 0 1  0  1  0)
  STRIDE=(1)
elif [[ "$ARCH_SET" == "Tiny_8L" ]]; then
  INP=(3  32 32 32 32 32 32 32)
  OUT=(32 32 32 32 32 32 32 32)
  POOL=(1  0  0  0  0  0  0  0)
  STRIDE=(1)
elif [[ "$ARCH_SET" == "Tiny_11L1P" ]]; then
  INP=(3 8 8  16 16 32 32 32 32 32 32)
  OUT=(8 8 16 16 32 32 32 32 32 32 32)
  POOL=(0 0 1  0  0  0  0  0  0  0  0)
  STRIDE=(1)
elif [[ "$ARCH_SET" == "Tiny_11L2P" ]]; then
  INP=(3 8 8  16 16 32 32 32 32 32 32)
  OUT=(8 8 16 16 32 32 32 32 32 32 32)
  POOL=(0 0 1  0  1  0  0  0  0  0  0)
  STRIDE=(1)
elif [[ "$ARCH_SET" == "DeepTi" ]]; then
  # deep 0.14M model - 32 Chan
  INP=(3  8 8 8 8 8 8  16 16 16 16 16 32 32 32 32 32 32)
  OUT=(8  8 8 8 8 8 16 16 16 16 16 32 32 32 32 32 32 32)
  POOL=(0 0 0 0 0 0 1  0  0  0  0  1  0  0  0  0  0  0)
  STRIDE=(1)
elif [[ "$ARCH_SET" == "DeepTi1P" ]]; then
  # deep 0.14M model - 32 Chan 1 Pooling
  INP=(3  8 8 8 8 8 8  16 16 16 16 16 32 32 32 32 32 32)
  OUT=(8  8 8 8 8 8 16 16 16 16 16 32 32 32 32 32 32 32)
  POOL=(0 0 0 0 0 0 1  0  0  0  0  0  0  0  0  0  0  0)
  STRIDE=(1)
elif [[ "$ARCH_SET" == "DeepS" ]]; then
  # deep 0.24M model
  INP=(3  16 16 16 32 32 32 64 64)
  OUT=(16 16 16 32 32 32 64 64 64)
  POOL=(0  0  0  1  0  0  1  0  0)
  STRIDE=(1)
elif [[ "$ARCH_SET" == "B" || "$ARCH_SET" == "DeepM" ]]; then
  # deep 0.58M model
  INP=(3  16 16 16 16 16 16 32 32 32 32 32 64 64 64 64 64 64)
  OUT=(16 16 16 16 16 16 32 32 32 32 32 64 64 64 64 64 64 64)
  POOL=(0  0  0  0  0  0  1  0  0  0  0  1  0  0  0  0  0  0)
  STRIDE=(1)
elif [[ "$ARCH_SET" == "DeepM48C" ]]; then
  # deep 48C model
  INP=(3  12 12 12 12 12 12 24 24 24 24 24 48 48 48 48 48 48)
  OUT=(12 12 12 12 12 12 24 24 24 24 24 48 48 48 48 48 48 48)
  POOL=(0  0  0  0  0  0  1  0  0  0  0  1  0  0  0  0  0  0)
  STRIDE=(1)
elif [[ "$ARCH_SET" == "DeepM1P" ]]; then
  # deep 0.58M model
  INP=(3  16 16 16 16 16 16 32 32 32 32 32 64 64 64 64 64 64)
  OUT=(16 16 16 16 16 16 32 32 32 32 32 64 64 64 64 64 64 64)
  POOL=(0  0  0  0  0  0  1  0  0  0  0  0  0  0  0  0  0  0)
  STRIDE=(1)
elif [[ "$ARCH_SET" == "C" ]]; then
  INP=(3  32 32 64 64  128 128)
  OUT=(32 32 64 64 128 128 128)
  POOL=(0  0  1  0  1   0  0)
  STRIDE=(1)
elif [[ "$ARCH_SET" == "Large8L" ]]; then
  INP=(3  32 32 64 64  128 128 128)
  OUT=(32 32 64 64 128 128 128 128)
  POOL=(0  0  1  0  1   0   0  0)
  STRIDE=(1)
elif [[ "$ARCH_SET" == "Large9L" ]]; then
  INP=(3  32 32 64 64 64  128 128 128)
  OUT=(32 32 64 64 64 128 128 128 128)
  POOL=(0  0  1  0  0  1   0   0  0)
  STRIDE=(1)
elif [[ "$ARCH_SET" == "Large10L" ]]; then
  INP=(3  32 32 32 64 64 64  128 128 128)
  OUT=(32 32 32 64 64 64 128 128 128 128)
  POOL=(0  0  0  1  0  0  1   0   0  0)
  STRIDE=(1)
elif [[ "$ARCH_SET" == "Large8L1P" ]]; then
  INP=(3  32 32 64 64  128 128 128)
  OUT=(32 32 64 64 128 128 128 128)
  POOL=(0  0  0  0  1   0   0  0)
  STRIDE=(1)
elif [[ "$ARCH_SET" == "Large9L1P" ]]; then
  INP=(3  32 32 64 64 64  128 128 128)
  OUT=(32 32 64 64 64 128 128 128 128)
  POOL=(0  0  0  0  0  1   0   0  0)
  STRIDE=(1)
elif [[ "$ARCH_SET" == "Large10L1P" ]]; then
  INP=(3  32 32 32 64 64 64  128 128 128)
  OUT=(32 32 32 64 64 64 128 128 128 128)
  POOL=(0  0  0  0  0  0  1   0   0  0)
  STRIDE=(1)
elif [[ "$ARCH_SET" == "C2" ]]; then
  INP=(   3  32 32 64 64  128 128)
  OUT=(   32 32 64 64 128 128 128)
  POOL=(  0  0  0  0  0   0   0)
  STRIDE=(1  1  2  1  2   1   1)
elif [[ "$ARCH_SET" == "D2p" ]]; then
  INP=(3  32 64 64)
  OUT=(32 64 64 128)
  POOL=(0 1  1  0)
  STRIDE=(1)
elif [[ "$ARCH_SET" == "D2" ]]; then
  INP=(3  32 64 64)
  OUT=(32 64 64 128)
  POOL=(0 0  0  0)
  STRIDE=(1 2  2  1)
elif [[ "$ARCH_SET" == "Ker5_A" ]]; then
  INP=(3  32 32 64 64)
  OUT=(32 32 64 64 64)
  POOL=(0  0  1  1  0)
  STRIDE=(1)
  KERNEL=(5)
  padding=2
elif [[ "$ARCH_SET" == "Ker5_C" ]]; then
  INP=(3  32 32 64 64 64)
  OUT=(32 32 64 64 64 64)
  POOL=(0  0  1  0  1  0)
  STRIDE=(1)
  KERNEL=(5)
  padding=2
elif [[ "$ARCH_SET" == "Ker3_D" ]]; then
  INP=(3  32 64 64 64)
  OUT=(32 64 64 64 64)
  POOL=(0  1  0  1  0)
  STRIDE=(1)
elif [[ "$ARCH_SET" == "7L64Chan" ]]; then
  INP=(3  64 64 64 64 64 64)
  OUT=(64 64 64 64 64 64 64)
  POOL=(0  1  0  1  0  0  0)
  STRIDE=(1)
elif [[ "$ARCH_SET" == "7L64Chan1P" ]]; then
  INP=(3  64 64 64 64 64 64)
  OUT=(64 64 64 64 64 64 64)
  POOL=(0  0  0  1  0  0  0)
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

##########################################################################
# Finetunable blocks list
##########################################################################
FINETUNE_BLOCKS=( "ODEFixNoiseXInit" "ODEFixNoiseXInitFFFB" "ODEFixNoise0InitExpand" "ODEFixNoiseOffset" "ODEFixNoiseFFFB" )
is_finetunable() {
  local b="$1"
  for x in "${FINETUNE_BLOCKS[@]}"; do
    [[ "$b" == "$x" ]] && return 0
  done
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

  local t_end="0.75"
  if [[ "$block" == *Circ* ]]; then
    t_end="1.25"
  elif [[ "$block" == *0Init* || "$block" == *As0* || "$block" == *YAsXZAsX* ]]; then
    t_end="1.75"
  elif [[ "$block" == *Charge* || "$block" == *Chg* ]]; then
    t_end="1.5"
  fi

  OMP_NUM_THREADS=${THREADS_PER_PROC} \
  python -u train_ode_cifar.py \
    --optim         "SGD" \
    --img_type      "${IMG_TYPE}" \
    --num_epochs    150 \
    --eval_every    10 \
    --offset_eps    0.0 \
    --inp_channels  "${INP[@]}" \
    --out_channels  "${OUT[@]}" \
    --max_pool      "${POOL[@]}" \
    --stride        "${STRIDE[@]}" \
    --kernel_size   "${KERNEL[@]}" \
    --padding       "${padding}" \
    --patch_node    "${patch_node}" \
    --patch_stride  "${patch_stride}" \
    --patch_cycle   "${patch_cycle}" \
    --dropout       0.25 \
    --tie_weights   "false" \
    --tie_bp        "false" \
    --bypass        "false" \
    --batch_size    128 \
    --method        "dopri5" \
    --tol           "0.0001" \
    --t_end         "${t_end}" \
    --pcn           "${PCN}" \
    --pc_conv       "PCConvReLU6" \
    --ode_block     "${block}" \
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
METHOD="dopri5"
TOL="1e-6"
EVAL_BASE="${LOGDIR}/evals"
MASTER_LOG="${LOGDIR}/eval_master_merged.log"
mkdir -p "${EVAL_BASE}"
: > "${MASTER_LOG}"

# Train→Test block mapping
map_test_block() {
  case "$1" in
    ODEFixNoiseXInit)          echo "ODESumAsBInitYAsX" ;;
    ODEFixNoiseXInitFFFB)      echo "SumAsBInitYAsXFFFB" ;;
    ODEFixNoise0InitExpand)    echo "ODESumAsBInitYAs0" ;;
    ODEFixNoiseOffset)         echo "ODESumAsBInitY" ;;
    ODEFixNoiseFFFB)           echo "ODESumAsBInitYFFFB" ;;
    SelfCUAbsSumFFFBInitB)     echo "SelfCUAbsSumFFFB" ;;
    *)                         echo "$1" ;;
  esac
}
TEST_BLOCKS=()
for b in "${BLOCKS[@]}"; do
  TEST_BLOCKS+=( "$(map_test_block "$b")" )
done

map_test_wrapper() {
  if [[ "$1" == *S2* || "$1" == *State2* ]]; then
    echo "ODEWrapper2State"
  else
    echo "WrapQuantizeW"
  fi
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
  local name="$1"; local tag="$2"; local test_block="$3"
  local __rest="${name#*_}"; local pc_conv="${__rest%%_*}"
  local dir="${EVAL_BASE}/${tag}/${name}_method_${METHOD}_tol_${TOL}"
  mkdir -p "$dir"
  local ckpt="best"
  ################################
  # Todo: Set n_bits, R, C value as fixed for now
  local n_bits=5
  local R_val="1e5"
  local C_val="49e-15"
  local wrapper="$(map_test_wrapper "$test_block")"
  ################################
  if [[ "$tag" == "post_ft" ]]; then ckpt="last"; fi

  echo "[EVAL ${tag}] ${name} (test_block=${test_block}, ckpt=${ckpt}) → $dir/job.log"
  OMP_NUM_THREADS=${THREADS_PER_PROC} \
  python -u ode_inference.py \
    --model_name      "$name" \
    --ckpt            "$ckpt" \
    --model_dir       "$MODEL_DIR" \
    --img_type        "${IMG_TYPE}" \
    --patch_node      "${patch_node}" \
    --patch_stride    "${patch_stride}" \
    --patch_cycle     "${patch_cycle}" \
    --method          "$METHOD" \
    --tol             "$TOL" \
    --n_steps         15 \
    --ts_scale        1 \
    --d_start         0 \
    --d_end           1 \
    --n_sweep_left    0 \
    --n_sweep_right   1 \
    --R               "$R_val" \
    --C               "$C_val" \
    --w_bits          "$n_bits" \
    --pc_conv         "${pc_conv}Noisy" \
    --ode_block       "$test_block" \
    --ode_wrapper     "$wrapper" \
    --mem_frac        "${MEM_FRAC}" \
    2>&1 | tee -a "$dir/job.log"
}

finetune_model() {
  # Finetune uses the SAME training block for that experiment
  local name="$1"; local train_block="$2"
  local logfile="${LOGDIR}/finetune_${train_block}.log"
  echo "[FINETUNE] ${name} (train_block=${train_block}) → $logfile"

  local t_end="0.75"
  if [[ "$block" == *0Init* || "$block" == *As0* ]]; then
    t_end="1.75"
  fi

  OMP_NUM_THREADS=${THREADS_PER_PROC} \
  python -u train_ode_cifar.py \
    --optim         "SGD" \
    --img_type      "${IMG_TYPE}" \
    --learning_rate 0.0001 \
    --lr_reduce_on  "10,20" \
    --num_epochs    10 \
    --model_name    "${name}" \
    --offset_eps    0.4 \
    --dropout       0.25 \
    --tie_weights   "false" \
    --tie_bp        "false" \
    --bypass        "false" \
    --batch_size    128 \
    --method        "dopri5" \
    --tol           "0.0001" \
    --t_end         "${t_end}" \
    --pcn           "${PCN}" \
    --pc_conv       "PCConvReLU6" \
    --ode_block     "${train_block}" \
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
  pre_log="${EVAL_BASE}/pre_ft/${pre}_method_${METHOD}_tol_${TOL}/job.log"
  post_log="${EVAL_BASE}/post_ft/${post}_method_${METHOD}_tol_${TOL}/job.log"

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
    post_log="${EVAL_BASE}/post_ft/${post}_method_${METHOD}_tol_${TOL}/job.log"

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
PARSE_WRAPPER="shell_utils/pipeline_node_run_parse.py"
if [[ -f "$PARSE_WRAPPER" ]]; then
  echo "[PARSE] python $PARSE_WRAPPER ${MASTER_LOG}"
  python -u "$PARSE_WRAPPER" "${MASTER_LOG}"
else
  echo "[PARSE] Skipping parse: ${PARSE_WRAPPER} not found."
fi
exit $ok
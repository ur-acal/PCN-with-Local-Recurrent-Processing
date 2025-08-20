#!/bin/bash -l
#SBATCH -p ising
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16            # modest CPU request so the node can be shared
#SBATCH --gres=gpu:1             # exactly ONE GPU; allows packing on 4-GPU nodes
#SBATCH -t 48:10:00
#SBATCH -o /home/rzeng7/Desktop/research/repos/PCN-with-Local-Recurrent-Processing/logs/slurm_jobs/slurm_%j.out

#set -euo pipefail

# ---- Conda activation (non-interactive safe) ----
source activate base
conda activate cross_sim

##########################################################################
# Change exp name here.
##########################################################################
ARCH_SET=${ARCH_SET:-A}      # A or B
PCN=${PCN:-PCNetNoBatchNorm} # "PCNetNoBatchNorm" | "PCNetWith1stConv"
if [[ "$ARCH_SET" == "B" ]]; then
  ARCH_LABEL=deep
else
  ARCH_LABEL=wide
fi
EXP="no_bn_${PCN}_NODE_0820_relu6_${ARCH_LABEL}_Exp"
LOGDIR="/home/rzeng7/Desktop/research/repos/PCN-with-Local-Recurrent-Processing/logs/${EXP}"
mkdir -p "$LOGDIR"
##########################################################################

# On cancel/terminate, stop children too
trap 'echo "Signal caught; killing children..."; kill -- -$$ 2>/dev/null || true' TERM INT

# ---- Arch definitions ----
if [[ "$ARCH_SET" == "A" ]]; then
  INP=(3  32 32 64 64  128)
  OUT=(32 32 64 64 128 128)
  POOL=(0  1  1  0  1   0)
elif [[ "$ARCH_SET" == "B" ]]; then
  INP=(3  16 16 16 16 16 16 32 32 32 32 32 64 64 64 64 64 64)
  OUT=(16 16 16 16 16 16 32 32 32 32 32 64 64 64 64 64 64 64)
  POOL=(0  0  0  0  0  0  1  0  0  0  0  1  0  0  0  0  0  0)
else
  echo "ARCH_SET must be A or B"; exit 2
fi

BLOCKS=( "ODEFixNoiseXInit" "ODEFixNoiseXInitFFFB" "ODEFixNoise0InitExpand" "ODEFixNoise0InitFFFB" )

# Split CPU threads evenly across 4 concurrent trainings
CPUS=${SLURM_CPUS_PER_TASK:-16}
THREADS_PER_PROC=$(( CPUS / 4 ))
if (( THREADS_PER_PROC < 1 )); then THREADS_PER_PROC=1; fi

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
  python -u train_ode_cifar.py \
    --optim         "SGD" \
    --num_epochs    150 \
    --eval_every    10 \
    --offset_eps    0.2 \
    --inp_channels  "${INP[@]}" \
    --out_channels  "${OUT[@]}" \
    --max_pool      "${POOL[@]}" \
    --dropout       0.25 \
    --tie_weights   "false" \
    --tie_bp        "false" \
    --bypass        "false" \
    --batch_size    128 \
    --method        "dopri5" \
    --tol           "0.0001" \
    --t_end         "0.75" \
    --pcn           "${PCN}" \
    --pc_conv       "PCConvReLU6" \
    --ode_block     "${block}" \
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
echo "Completed all four ode_blocks for ${EXP} (job $SLURM_JOB_ID)."
exit $ok

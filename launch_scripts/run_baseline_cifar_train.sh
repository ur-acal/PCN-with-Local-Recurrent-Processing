#!/bin/bash -l

#SBATCH -p ising
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16            # modest CPU request so the node can be shared
#SBATCH --gres=gpu:1                  # exactly ONE GPU; allows packing on 4-GPU nodes
#SBATCH -t 72:10:00
#SBATCH -o /scratch/rzeng7/repos/PCN-with-Local-Recurrent-Processing/logs/slurm_jobs/slurm_%j.out

#set -euo pipefail

# ---- Conda activation (non-interactive safe) ----
source activate base
conda activate scanbase

REPO_ROOT="${REPO_ROOT:-/scratch/rzeng7/repos/PCN-with-Local-Recurrent-Processing}"
cd "${REPO_ROOT}"

MODEL_NAME="${MODEL_NAME:?MODEL_NAME is required}"
DATASET_NAME="${DATASET_NAME:-cifar100}"
CASE_NAME="${CASE_NAME:-auto}"
PRETRAINED="${PRETRAINED:-false}"
PREFER_RESIZE="${PREFER_RESIZE:-false}"

DATA_DIR="${DATA_DIR:-/home/rongzeng/_workspce_old/repos/pcn/collaboration/data}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/logs/baseline_cifar}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-baseline/train_baseline_cifar.py}"
EXTRA_OVERRIDE="${EXTRA_OVERRIDE:-}"
PRINT_ONLY="${PRINT_ONLY:-false}"

EXP="BASELINE_CIFAR"
LOGDIR="${LOGDIR:-./logs/${EXP}}"
mkdir -p "${LOGDIR}"

EXP_SUFFIX="${DATASET_NAME}_${CASE_NAME}_${MODEL_NAME}_pretrained_${PRETRAINED}"
LOG_FILE="${LOGDIR}/train_${EXP}_${EXP_SUFFIX}.log"

echo "REPO_ROOT=${REPO_ROOT}"
echo "TRAIN_SCRIPT=${TRAIN_SCRIPT}"
echo "MODEL_NAME=${MODEL_NAME}"
echo "DATASET_NAME=${DATASET_NAME}"
echo "CASE_NAME=${CASE_NAME}"
echo "PRETRAINED=${PRETRAINED}"
echo "PREFER_RESIZE=${PREFER_RESIZE}"
echo "DATA_DIR=${DATA_DIR}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "EXTRA_OVERRIDE=${EXTRA_OVERRIDE}"
echo "PRINT_ONLY=${PRINT_ONLY}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
which python
python --version

CMD=(
  python "${TRAIN_SCRIPT}"
  --model_name "${MODEL_NAME}"
  --dataset "${DATASET_NAME}"
  --data_dir "${DATA_DIR}"
  --output_dir "${OUTPUT_DIR}"
  --case "${CASE_NAME}"
  --pretrained "${PRETRAINED}"
  --prefer_resize "${PREFER_RESIZE}"
  --print_only "${PRINT_ONLY}"
)

if [[ -n "${EXTRA_OVERRIDE}" ]]; then
  CMD+=(--override "${EXTRA_OVERRIDE}")
fi

printf 'Running command:'
printf ' %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}" 2>&1 | tee "${LOG_FILE}"
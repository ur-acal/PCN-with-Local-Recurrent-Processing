#!/bin/bash -l
set -eo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate scanbase
fi

OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/logs/wrn_bn_recalibration_preflight}"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/../data}"
CALIBRATION_NUM_SAMPLES="${CALIBRATION_NUM_SAMPLES:-5000}"
CALIBRATION_BATCH_SIZE="${CALIBRATION_BATCH_SIZE:-128}"
CALIBRATION_SUBSET_SEED="${CALIBRATION_SUBSET_SEED:-20240618}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-2}"
MAX_EVAL_BATCHES="${MAX_EVAL_BATCHES:-none}"
PREFLIGHT_ARCHITECTURE="${PREFLIGHT_ARCHITECTURE:-WRN_16_2}"
PREFLIGHT_DATASET="${PREFLIGHT_DATASET:-cifar10}"

python baseline/run_wrn_bn_recalibration_experiment.py \
  --mode preflight \
  --compare_dir logs/wrn_like_compare_target_models_by_arch \
  --output_dir "${OUTPUT_DIR}" \
  --data_dir "${DATA_DIR}" \
  --checkpoint_root checkpoint/baselines \
  --preflight_architecture "${PREFLIGHT_ARCHITECTURE}" \
  --preflight_dataset "${PREFLIGHT_DATASET}" \
  --preflight_additive_level 0.01 \
  --preflight_multiplicative_level 0.05 \
  --preflight_trials 2 \
  --calibration_num_samples "${CALIBRATION_NUM_SAMPLES}" \
  --calibration_batch_size "${CALIBRATION_BATCH_SIZE}" \
  --calibration_subset_seed "${CALIBRATION_SUBSET_SEED}" \
  --calibration_num_workers "${NUM_WORKERS}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --max_eval_batches "${MAX_EVAL_BATCHES}"

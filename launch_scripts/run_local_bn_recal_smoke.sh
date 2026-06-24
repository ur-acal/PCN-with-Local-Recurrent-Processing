#!/bin/bash -l
set -eo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate scanbase
fi

MODEL_NAME="${MODEL_NAME:-wrn_16_2_cifar}"
DATASET_NAME="${DATASET_NAME:-cifar10}"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/../data}"
CASE_NAME="${CASE_NAME:-custom_noresize}"
CKPT_PATH="${CKPT_PATH:-${REPO_ROOT}/checkpoint/baselines/${DATASET_NAME}/${CASE_NAME}/${MODEL_NAME}/${CASE_NAME}_${DATASET_NAME}_${MODEL_NAME}/${CASE_NAME}_${DATASET_NAME}_${MODEL_NAME}_best_ckpt.pth}"
RESULTS_ROOT="${RESULTS_ROOT:-${REPO_ROOT}/logs/baselines/bn_recal_smoke}"
CALIB_SAMPLES="${CALIB_SAMPLES:-256}"
CALIB_BATCH_SIZE="${CALIB_BATCH_SIZE:-128}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-2}"
SEED="${SEED:-123}"

if [[ ! -f "${CKPT_PATH}" ]]; then
  echo "Checkpoint not found: ${CKPT_PATH}" >&2
  exit 1
fi

mkdir -p "${RESULTS_ROOT}/results" "${RESULTS_ROOT}/diagnostics" "${RESULTS_ROOT}/logs"

run_smoke() {
  local label="$1"
  local noise_type="$2"
  local noise_levels="$3"

  echo "======================================================================"
  echo "BN recalibration smoke: ${label} (${noise_type}, levels=${noise_levels})"
  echo "======================================================================"

  python baseline/run_baseline.py \
    --model_list "${MODEL_NAME}" \
    --dataset "${DATASET_NAME}" \
    --data_dir "${DATA_DIR}" \
    --checkpoint_map "${MODEL_NAME}=${CKPT_PATH}" \
    --case "${CASE_NAME}" \
    --pretrained false \
    --prefer_resize false \
    --noise_level_list "${noise_levels}" \
    --noisy_trials 1 \
    --noise_type "${noise_type}" \
    --noise_to_norm false \
    --batch_size "${EVAL_BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --seed "${SEED}" \
    --max_eval_batches 2 \
    --bn_recalibration_enabled true \
    --bn_recalibration_num_samples "${CALIB_SAMPLES}" \
    --bn_recalibration_batch_size "${CALIB_BATCH_SIZE}" \
    --bn_recalibration_subset_seed "${SEED}" \
    --bn_recalibration_num_workers "${NUM_WORKERS}" \
    --bn_recalibration_diagnostics_dir "${RESULTS_ROOT}/diagnostics/${label}" \
    --results_dir "${RESULTS_ROOT}/results/${label}" \
    2>&1 | tee "${RESULTS_ROOT}/logs/${label}.log"
}

run_smoke "sigma0" "multiplicative" "0"
run_smoke "additive_nonzero" "additive" "0.01"
run_smoke "multiplicative_nonzero" "multiplicative" "0.05"

echo "Smoke results written under ${RESULTS_ROOT}"

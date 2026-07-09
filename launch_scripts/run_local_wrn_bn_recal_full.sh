#!/bin/bash -l
set -eo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate scanbase
fi

OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/logs/wrn_bn_recalibration_full}"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/../data}"
CALIBRATION_NUM_SAMPLES="${CALIBRATION_NUM_SAMPLES:-5000}"
CALIBRATION_BATCH_SIZE="${CALIBRATION_BATCH_SIZE:-128}"
CALIBRATION_SUBSET_SEED="${CALIBRATION_SUBSET_SEED:-20240618}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-4}"
NOISY_TRIALS="${NOISY_TRIALS:-10}"
DATASETS="${DATASETS:-all}"
ARCHITECTURES="${ARCHITECTURES:-all}"
MISMATCH_TYPES="${MISMATCH_TYPES:-all}"
NOISE_LEVELS="${NOISE_LEVELS:-csv}"
MISMATCH_PARAMETER_POLICY="${MISMATCH_PARAMETER_POLICY:-existing}"
INCLUDE_ALL_BN_RECAL_BEFORE_FOLD_COLUMN="${INCLUDE_ALL_BN_RECAL_BEFORE_FOLD_COLUMN:-false}"

python baseline/run_wrn_bn_recalibration_experiment.py \
  --mode full \
  --compare_dir logs/wrn_like_compare_target_models_by_arch \
  --output_dir "${OUTPUT_DIR}" \
  --data_dir "${DATA_DIR}" \
  --checkpoint_root checkpoint/baselines \
  --datasets "${DATASETS}" \
  --architectures "${ARCHITECTURES}" \
  --mismatch_types "${MISMATCH_TYPES}" \
  --noise_levels "${NOISE_LEVELS}" \
  --noisy_trials "${NOISY_TRIALS}" \
  --mismatch_parameter_policy "${MISMATCH_PARAMETER_POLICY}" \
  $(if [[ "${INCLUDE_ALL_BN_RECAL_BEFORE_FOLD_COLUMN}" == "true" ]]; then echo --include_all_bn_recal_before_fold_column; fi) \
  --calibration_num_samples "${CALIBRATION_NUM_SAMPLES}" \
  --calibration_batch_size "${CALIBRATION_BATCH_SIZE}" \
  --calibration_subset_seed "${CALIBRATION_SUBSET_SEED}" \
  --calibration_num_workers "${NUM_WORKERS}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}"

#!/bin/bash -l
set -eo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate scanbase
fi

DATA_DIR="${DATA_DIR:-${REPO_ROOT}/../data}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${REPO_ROOT}/checkpoint/baselines_nobn}"
RESULT_DIR="${RESULT_DIR:-${REPO_ROOT}/logs/wrn_nobn_mismatch_full}"
SOURCE_TABLE_DIR="${SOURCE_TABLE_DIR:-${REPO_ROOT}/../ScAN-PCN/logs/wrn_bn_recalibration_full_5120_bn_fold_no_mismatch_bias}"
FINAL_TABLE_DIR="${FINAL_TABLE_DIR:-${REPO_ROOT}/logs/wrn_bn_recalibration_full_5120_bn_fold_no_mismatch_bias_with_nobn}"
TRAIN_LOG_DIR="${TRAIN_LOG_DIR:-${REPO_ROOT}/logs/wrn_nobn_train}"
TRAIN_OVERRIDE="${TRAIN_OVERRIDE:-max_norm=1.0,eval_every=5}"
DATASETS="${DATASETS:-cifar10,cifar100}"
ARCHITECTURES="${ARCHITECTURES:-WRN_16_2,WRN_16_4,WRN_28_2,WRN_28_4}"
MISMATCH_TYPES="${MISMATCH_TYPES:-additive,multiplicative}"
NOISY_TRIALS="${NOISY_TRIALS:-10}"
FORCE_TRAIN="${FORCE_TRAIN:-false}"

mkdir -p "${CHECKPOINT_ROOT}" "${RESULT_DIR}" "${FINAL_TABLE_DIR}" "${TRAIN_LOG_DIR}"

model_name() {
  printf 'wrn_%s_cifar_nobn' "$(printf '%s' "$1" | sed -E 's/^WRN_//; s/[A-Z]/\L&/g')"
}

checkpoint_path() {
  local dataset="$1" name="$2"
  local run="custom_noresize_${dataset}_${name}"
  printf '%s/%s/custom_noresize/%s/%s/%s_best_ckpt.pth' \
    "${CHECKPOINT_ROOT}" "${dataset}" "${name}" "${run}" "${run}"
}

IFS=',' read -r -a dataset_list <<< "${DATASETS}"
IFS=',' read -r -a architecture_list <<< "${ARCHITECTURES}"
for dataset in "${dataset_list[@]}"; do
  for architecture in "${architecture_list[@]}"; do
    name="$(model_name "${architecture}")"
    checkpoint="$(checkpoint_path "${dataset}" "${name}")"
    if [[ "${FORCE_TRAIN}" == "true" || ! -f "${checkpoint}" ]]; then
      echo "Training ${dataset} ${name}"
      python baseline/train_baseline_cifar.py \
        --model_name "${name}" \
        --dataset "${dataset}" \
        --data_dir "${DATA_DIR}" \
        --output_dir "${CHECKPOINT_ROOT}" \
        --case custom_noresize \
        --pretrained false \
        --override "${TRAIN_OVERRIDE}" \
        2>&1 | tee "${TRAIN_LOG_DIR}/${dataset}_${name}.log"
    else
      echo "Using existing checkpoint ${checkpoint}"
    fi
    [[ -f "${checkpoint}" ]] || { echo "Missing checkpoint ${checkpoint}" >&2; exit 1; }
  done
done

python baseline/run_wrn_nobn_mismatch_experiment.py \
  --output_dir "${RESULT_DIR}" \
  --data_dir "${DATA_DIR}" \
  --checkpoint_root "${CHECKPOINT_ROOT}" \
  --datasets "${DATASETS}" \
  --architectures "${ARCHITECTURES}" \
  --mismatch_types "${MISMATCH_TYPES}" \
  --noisy_trials "${NOISY_TRIALS}"

python baseline/add_wrn_nobn_to_presentation_table.py \
  --markdown "${SOURCE_TABLE_DIR}/full_presentation_table.md" \
  --presentation_csv "${SOURCE_TABLE_DIR}/full_presentation_table.csv" \
  --bn_free_aggregate_csv "${RESULT_DIR}/full_aggregate.csv" \
  --output_markdown "${FINAL_TABLE_DIR}/full_presentation_table.md" \
  --output_csv "${FINAL_TABLE_DIR}/full_presentation_table.csv"

echo "BN-free WRN table: ${FINAL_TABLE_DIR}/full_presentation_table.md"

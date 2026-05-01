#!/usr/bin/env bash
#set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO_ROOT}"

DATA_DIR="${DATA_DIR:-/home/rongzeng/_workspce_old/repos/pcn/collaboration/data}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/logs/baselines}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-baseline/train_baseline_cifar.py}"
EVAL_SCRIPT="${EVAL_SCRIPT:-baseline/run_baseline.py}"

TRAIN_LOG_DIR="${TRAIN_LOG_DIR:-${OUTPUT_DIR}/train_logs}"
EVAL_LOG_DIR="${EVAL_LOG_DIR:-${OUTPUT_DIR}/eval_logs}"
EVAL_RESULTS_DIR="${EVAL_RESULTS_DIR:-${OUTPUT_DIR}/mismatch_eval}"

mkdir -p "${TRAIN_LOG_DIR}" "${EVAL_LOG_DIR}" "${EVAL_RESULTS_DIR}"
SAVED_CKPT_PATHS=()

CUSTOM_CIFAR_MODELS=(
  resnet20_cifar
  resnet32_cifar
  resnet44_cifar
  resnet56_cifar
  preact_resnet164_cifar
  wrn_28_10_cifar
)

ADAPT_NORESIZE_TIMM_MODELS=(
  resnet18
  resnet34
  resnet50
  resnext26ts
  resnext50_32x4d
  seresnet18
  seresnet34
  seresnet50
  mobilenetv2_100
#  vgg19
)

RESIZE_FINETUNE_TIMM_MODELS=(
  efficientnet_b0
  mobilenetv3_small_100
  vit_tiny_patch16_224
  deit_tiny_patch16_224
  mixer_b16_224
  convnext_tiny
)

DATASETS=(cifar10 cifar100)

make_train_log_path() {
  local model_name="$1"
  local dataset_name="$2"
  local case_name="$3"
  local pretrained="$4"

  local tag="${dataset_name}_${case_name}_${model_name}_pretrained_${pretrained}"
  echo "${TRAIN_LOG_DIR}/train_${tag}.log"
}

extract_ckpt_path() {
  local log_file="$1"

  grep -- "----- Model path:" "${log_file}" \
    | tail -n 1 \
    | sed -E 's/^----- Model path: (.*) -----$/\1/'
}

run_train_one() {
  local model_name="$1"
  local dataset_name="$2"
  local case_name="$3"
  local pretrained="$4"
  local prefer_resize="$5"

  local train_log
  train_log="$(make_train_log_path "${model_name}" "${dataset_name}" "${case_name}" "${pretrained}")"

  echo "======================================================================"
  echo "TRAIN: dataset=${dataset_name}, model=${model_name}, case=${case_name}, pretrained=${pretrained}"
  echo "LOG:   ${train_log}"
  echo "======================================================================"

  python "${TRAIN_SCRIPT}" \
    --model_name "${model_name}" \
    --dataset "${dataset_name}" \
    --data_dir "${DATA_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --case "${case_name}" \
    --pretrained "${pretrained}" \
    --prefer_resize "${prefer_resize}" \
    --override "num_epochs=2,eval_every=2" \
    --print_only false \
    2>&1 | tee "${train_log}"
}

run_eval_one() {
  local model_name="$1"
  local dataset_name="$2"
  local case_name="$3"
  local ckpt_path="$4"
  local prefer_resize="$5"

  local tag="${dataset_name}_${case_name}_${model_name}"
  local eval_log="${EVAL_LOG_DIR}/eval_${tag}.log"

  echo "======================================================================"
  echo "EVAL: dataset=${dataset_name}, model=${model_name}"
  echo "CKPT: ${ckpt_path}"
  echo "LOG:  ${eval_log}"
  echo "======================================================================"

  python "${EVAL_SCRIPT}" \
    --model_list "${model_name}" \
    --dataset "${dataset_name}" \
    --data_dir "${DATA_DIR}" \
    --checkpoint_map "${model_name}=${ckpt_path}" \
    --case "${case_name}" \
    --pretrained false \
    --prefer_resize "${prefer_resize}" \
    --noise_level_list "0,0.2" \
    --noisy_trials 2 \
    --noise_type multiplicative \
    --noise_to_norm false \
    --results_dir "${EVAL_RESULTS_DIR}/${dataset_name}/${case_name}/${model_name}" \
    2>&1 | tee "${eval_log}"
}

run_one() {
  local model_name="$1"
  local dataset_name="$2"
  local case_name="$3"
  local pretrained="$4"
  local prefer_resize="$5"

  local train_log
  local ckpt_path

  train_log="$(make_train_log_path "${model_name}" "${dataset_name}" "${case_name}" "${pretrained}")"

  run_train_one "${model_name}" "${dataset_name}" "${case_name}" "${pretrained}" "${prefer_resize}"

  ckpt_path="$(extract_ckpt_path "${train_log}")"

  if [[ -z "${ckpt_path}" ]]; then
    echo "ERROR: failed to extract checkpoint path from ${train_log}" >&2
    exit 1
  fi

  if [[ ! -f "${ckpt_path}" ]]; then
    echo "ERROR: extracted checkpoint does not exist: ${ckpt_path}" >&2
    exit 1
  fi

  SAVED_CKPT_PATHS+=("${ckpt_path}")

  run_eval_one "${model_name}" "${dataset_name}" "${case_name}" "${ckpt_path}" "${prefer_resize}"
}

for dataset_name in "${DATASETS[@]}"; do
  for model_name in "${CUSTOM_CIFAR_MODELS[@]}"; do
    run_one "${model_name}" "${dataset_name}" "custom_noresize" "false" "false"
  done

  for model_name in "${ADAPT_NORESIZE_TIMM_MODELS[@]}"; do
    run_one "${model_name}" "${dataset_name}" "adapt_noresize_scratch" "false" "false"
  done

  for model_name in "${RESIZE_FINETUNE_TIMM_MODELS[@]}"; do
    run_one "${model_name}" "${dataset_name}" "resize_finetune" "true" "true"
  done
done

echo "======================================================================"
echo "All saved checkpoint paths:"
echo "======================================================================"

for item in "${SAVED_CKPT_PATHS[@]}"; do
  read -r dataset_name case_name model_name ckpt_path <<< "${item}"
  echo "dataset=${dataset_name} case=${case_name} model=${model_name}"
  echo "  ${ckpt_path}"
done

echo "======================================================================"
echo "All 2-epoch train + mismatch-eval tests finished."

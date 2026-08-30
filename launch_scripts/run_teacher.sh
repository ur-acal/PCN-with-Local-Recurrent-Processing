#!/bin/bash
#SBATCH -p ds4ai
#SBATCH -c 16
#SBATCH -t 24:00:00
#SBATCH --mem=32gb
#SBATCH --gres=gpu:1
#SBATCH --output=logs/test_%j.out
#SBATCH --error=logs/test_%j.err
#SBATCH --job-name=Teacher

SCANGEN_DATA_ROOT="${SCANGEN_DATA_ROOT:-../cifar-10-data/scanGFI}"
DATASET_NAME="${DATASET_NAME:-cifar100}"
IMG_TYPE="${IMG_TYPE:-${img_type:-scanGFI}}"
USE_TIMM="${USE_TIMM:-${use_timm:-false}}"
USE_OLD_AUGS_FOR_TIMM="${USE_OLD_AUGS_FOR_TIMM:-${use_old_augs_for_timm:-false}}"
MATCH_DISTILL_AUG_ORDER="${MATCH_DISTILL_AUG_ORDER:-${match_distill_aug_order:-false}}"
USE_DIRECT_RESIZE_FOR_TIMM_AUGS="${USE_DIRECT_RESIZE_FOR_TIMM_AUGS:-false}"
MATCH_DISTILL_PREPROCESS="${MATCH_DISTILL_PREPROCESS:-${match_distill_preprocess:-false}}"
INPUT_QUANT_BITS="${INPUT_QUANT_BITS:-${input_quant_bits:-}}"
case "${IMG_TYPE,,}" in
  cifair)
    IMG_TYPE="CiFAIR"
    TEACHER_IMG_LABEL="CiFAIR"
    ;;
  scangfi|raw|_raw)
    IMG_TYPE="scanGFI"
    TEACHER_IMG_LABEL="raw"
    ;;
esac
TIMM_RECIPE_LABEL="timm"
if [[ "${USE_TIMM,,}" == "true" && "${USE_OLD_AUGS_FOR_TIMM,,}" == "true" ]]; then
  TIMM_RECIPE_LABEL="timm_old_augs"
fi
if [[ "${USE_TIMM,,}" == "true" && "${MATCH_DISTILL_AUG_ORDER,,}" == "true" ]]; then
  TIMM_RECIPE_LABEL="timm_distill_order"
fi
TEACHER_CHECKPOINT="${TEACHER_CHECKPOINT:-checkpoint/efficientnet_v2_l_${DATASET_NAME}_${TEACHER_IMG_LABEL}_${TIMM_RECIPE_LABEL}.pth}"

if [[ "${USE_TIMM,,}" == "true" ]]; then
  TIMM_RE_ARGS=()
  if [[ -n "${TIMM_RE_PROB:-}" && "${TIMM_RE_PROB}" != "none" ]]; then
    TIMM_RE_ARGS+=(--re_prob "${TIMM_RE_PROB}")
  fi
  TIMM_CONFIG_ARGS=()
  if [[ -n "${TIMM_EPOCHS:-}" ]]; then TIMM_CONFIG_ARGS+=(--epochs "${TIMM_EPOCHS}"); fi
  if [[ -n "${TIMM_TEST_BATCH_SIZE:-}" ]]; then TIMM_CONFIG_ARGS+=(--test_batch_size "${TIMM_TEST_BATCH_SIZE}"); fi
  if [[ -n "${TIMM_LR:-}" ]]; then TIMM_CONFIG_ARGS+=(--lr "${TIMM_LR}"); fi
  if [[ -n "${TIMM_WEIGHT_DECAY:-}" ]]; then TIMM_CONFIG_ARGS+=(--weight_decay "${TIMM_WEIGHT_DECAY}"); fi
  if [[ -n "${TIMM_WARMUP_EPOCHS:-}" ]]; then TIMM_CONFIG_ARGS+=(--warmup_epochs "${TIMM_WARMUP_EPOCHS}"); fi
  if [[ -n "${TIMM_OPT:-}" ]]; then TIMM_CONFIG_ARGS+=(--timm_opt "${TIMM_OPT}"); fi
  if [[ -n "${TIMM_SCHED:-}" ]]; then TIMM_CONFIG_ARGS+=(--timm_sched "${TIMM_SCHED}"); fi
  if [[ -n "${TIMM_LR_REDUCE_ON:-}" ]]; then TIMM_CONFIG_ARGS+=(--lr_reduce_on "${TIMM_LR_REDUCE_ON}"); fi
  if [[ -n "${TIMM_DECAY_RATE:-}" ]]; then TIMM_CONFIG_ARGS+=(--decay_rate "${TIMM_DECAY_RATE}"); fi
  if [[ -n "${TIMM_FIRST_EVAL_EPOCH:-}" ]]; then TIMM_CONFIG_ARGS+=(--first_eval_epoch "${TIMM_FIRST_EVAL_EPOCH}"); fi
  TEACHER_LOG_DIR="${TEACHER_LOG_DIR:-logs/teacher_timm}"
  TEACHER_LOG="${TEACHER_LOG:-${TEACHER_LOG_DIR}/${DATASET_NAME}_${TEACHER_IMG_LABEL}_${TIMM_RECIPE_LABEL}.log}"
  mkdir -p "${TEACHER_LOG_DIR}"
  set -o pipefail
  python -u train_teacher_timm.py \
    --dataset "${DATASET_NAME}" \
    --img_type "${IMG_TYPE}" \
    --data_root "${SCANGEN_DATA_ROOT}" \
    --checkpoint "${TEACHER_CHECKPOINT}" \
    --pretrained "${TIMM_PRETRAINED:-true}" \
    --arch_source "${TIMM_ARCH_SOURCE:-hankyul2}" \
    --use_old_augs_for_timm "${USE_OLD_AUGS_FOR_TIMM}" \
    --use_direct_resize_for_timm_augs "${USE_DIRECT_RESIZE_FOR_TIMM_AUGS}" \
    --match_distill_aug_order "${MATCH_DISTILL_AUG_ORDER}" \
    --batch_size "${TIMM_BATCH_SIZE:-32}" \
    --num_workers "${NUM_WORKERS:-4}" \
    --eval_every "${EVAL_EVERY:-5}" \
    "${TIMM_CONFIG_ARGS[@]}" \
    "${TIMM_RE_ARGS[@]}" \
    2>&1 | tee "${TEACHER_LOG}"
else
  INIT_CHECKPOINT="${INIT_CHECKPOINT:-}"
  ARCH_SOURCE="${ARCH_SOURCE:-auto}"
  INIT_CHECKPOINT_ARGS=()
  PREPROCESS_ARGS=()
  if [[ -n "${INIT_CHECKPOINT}" ]]; then
    INIT_CHECKPOINT_ARGS+=(--init_checkpoint "${INIT_CHECKPOINT}")
  fi
  if [[ "${MATCH_DISTILL_PREPROCESS,,}" == "true" ]]; then
    PREPROCESS_ARGS+=(--match_distill_preprocess)
  fi
  if [[ -n "${INPUT_QUANT_BITS}" ]]; then
    PREPROCESS_ARGS+=(--input_quant_bits "${INPUT_QUANT_BITS}")
  fi

  python -u train_teacher.py \
    --dataset "${DATASET_NAME}" \
    --img_type "${IMG_TYPE}" \
    --arch efficientnet_v2_l \
    --arch_source "${ARCH_SOURCE}" \
    "${INIT_CHECKPOINT_ARGS[@]}" \
    "${PREPROCESS_ARGS[@]}" \
    --checkpoint "${TEACHER_CHECKPOINT}" \
    --lr 0.002 \
    --gamma 0.1 \
    --wd 1e-6 \
    --ne 100 \
    --nsc 10 \
    --batch_split 1 \
    --batch 32 \
    --alpha 0 \
    --train_transform cifar \
    --train_size 224 \
    --test_size 224 \
    --test_center_crop \
    --mismatch_levels 0. \
    --mismatch_type mul \
    --root "${SCANGEN_DATA_ROOT}"
fi

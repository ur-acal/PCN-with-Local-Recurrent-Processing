#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
stage="${1:-ft}"
export TC_NONIDEALITIES=true
: "${MODEL_NAME:?Set MODEL_NAME to the saved model name}"
# Match the legacy PCN naming convention; explicit selections take precedence.
model_lower="${MODEL_NAME,,}"
if [[ -z "${TASK:-}" ]]; then
  if [[ "$model_lower" == *c100* || "$model_lower" == *cifar100* ]]; then
    TASK=cifar100
  else
    TASK=cifar10
  fi
fi
if [[ -z "${IMG_TYPE:-}" ]]; then
  case "$model_lower" in
    *cifair*) IMG_TYPE=CiFAIR ;;
    *scangfi*|*_raw_*) IMG_TYPE=scanGFI ;;
    *) IMG_TYPE=rgb ;;
  esac
fi
case "${IMG_TYPE,,}" in
  cifair) IMG_TYPE=CiFAIR ;;
  scangfi|raw|_raw) IMG_TYPE=scanGFI ;;
  rgb) IMG_TYPE=rgb ;;
  *) echo 'TC launcher expects IMG_TYPE=rgb, CiFAIR or scanGFI.' >&2; exit 2 ;;
esac
if [[ -z "${CKPT:-}" ]]; then
  CKPT=best
  if [[ "$stage" == ft && "$model_lower" == *qat* ]]; then
    CKPT=full_param_best
  fi
fi
if [[ -z "${TC_STATE:-}" ]]; then
  TC_STATE=1
  if [[ "$MODEL_NAME" == *S2* || "$MODEL_NAME" == *State2* ]]; then TC_STATE=2; fi
fi
source ./launch_scripts/tc_nonideality_args.sh "$stage"
source ./launch_scripts/rgb_teacher_defaults.sh
if [[ "$stage" == ft ]]; then
  default_teacher_arch=efficientnet_v2_l
  if [[ "$IMG_TYPE" == CiFAIR ]]; then
    default_teacher="./checkpoint/efficientnet_v2_l_${TASK}_CiFAIR_OldNoTimm_MatchDistill.pth"
  elif [[ "$IMG_TYPE" == scanGFI && "$TASK" == cifar100 ]]; then
    default_teacher=./checkpoint/b4_100.pth
  elif [[ "$IMG_TYPE" == scanGFI && "$TASK" == cifar10 ]]; then
    default_teacher=./checkpoint/b4.pth
    default_teacher_arch=efficientnet-b4
  else
    # RGB defaults were selected by rgb_teacher_defaults.sh above.
    default_teacher="$TEACHER_CKPT"
  fi
  TEACHER_CKPT="${TEACHER_CKPT:-${default_teacher}}"
  cmd=(python -u train_ode_cifar.py --model_name "$MODEL_NAME"
    --save_path "${MODEL_DIR:-./saved_ckpt_runs}" --output_save_path "${OUTPUT_DIR:-./saved_ckpt_runs/tc_nonidealities_ft}"
    --dataset "${TASK:-cifar100}" --img_type "${IMG_TYPE:-CiFAIR}" --ckpt "${CKPT:-best}"
    --input_quant_bits "${INPUT_QUANT_BITS:-none}" --center_student_input "${CENTER_STUDENT_INPUT:-false}"
    --timm_trainer true --optim SGD --learning_rate "${FT_LEARNING_RATE:-0.005}"
    --num_epochs "${FT_NUM_EPOCHS:-140}" --eval_every 2 --warmup_epoch 0
    --timm_sched cosine --timm_aug_level no_aug --timm_re_prob "${TIMM_RE_PROB:-0}"
    --pcn PCNetNoBatchNorm --pc_conv PCConvReLU6 --avg_pooling true --dropout .25
    --t_end 1.75 --tol "${TOL:-1e-6}" --batch_size 128
    --distill_method "${DISTILL_METHOD:-srrl}" --teacher_ckpt "${TEACHER_CKPT:?Set TEACHER_CKPT}"
    --distill_alpha "${DISTILL_ALPHA:-0.3}" --distill_temperature "${DISTILL_TEMPERATURE:-2.0}"
    --teacher_arch "${TEACHER_ARCH:-${default_teacher_arch}}"
    --teacher_arch_source "${TEACHER_ARCH_SOURCE:-auto}"
    --teacher_input_size "${TEACHER_INPUT_SIZE:-224}" --teacher_center_crop "${TEACHER_CENTER_CROP:-true}"
    --adapt_PIL_teacher "${ADAPT_PIL_TEACHER:-false}")
else
  cmd=(python -u ode_inference.py --model_name "$MODEL_NAME" --model_dir "${MODEL_DIR:-./saved_ckpt_runs}"
    --task "${TASK:-cifar100}" --img_type "${IMG_TYPE:-CiFAIR}" --ckpt "${CKPT:-best}"
    --input_quant_bits "${INPUT_QUANT_BITS:-none}" --center_student_input "${CENTER_STUDENT_INPUT:-false}"
    --pc_conv PCConvReLU6Noisy --conv_only true --test_bs "${TEST_BS:-128}" --tol "${TOL:-1e-6}")
fi
cmd+=("${TC_ARGS[@]}")
if [[ "${TC_DRY_RUN:-false}" == true ]]; then
  printf '%q ' "${cmd[@]}"; printf '\n'
else
  exec "${cmd[@]}"
fi

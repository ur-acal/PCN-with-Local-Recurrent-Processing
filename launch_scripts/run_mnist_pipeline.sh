#!/bin/bash
# Source locally, or from the Slurm worker. No environment/module reinitialization.
set -eo pipefail
REPO_ROOT="${REPO_ROOT:-$PWD}"
cd "$REPO_ROOT"

FAMILY="${FAMILY:-cnn}"
VARIANT="${VARIANT:-small}"
TC_STATE="${TC_STATE:-1}"
STAGE="${STAGE:-all}"
if [[ -n "${mode:-}" ]]; then
  case "$mode" in
    default) STAGE=all ;;
    pretrain_only) STAGE=pretrain ;;
    ft_only) STAGE=ft ;;
    ft_and_eval) STAGE=ft_and_eval ;;
    *) echo 'Invalid mode: use default, pretrain_only, ft_only, or ft_and_eval' >&2; return 2 2>/dev/null || exit 2 ;;
  esac
fi
case "$FAMILY:$VARIANT" in
  cnn:small) mnist_name=mnist_cnn3_avgpool ;;
  cnn:deep) mnist_name=mnist_cnn5_avgpool ;;
  pcn:small) mnist_name="mnist_pcn2_state${TC_STATE}" ;;
  pcn:deep) mnist_name="mnist_pcn3_state${TC_STATE}" ;;
  *) echo 'FAMILY must be cnn/pcn and VARIANT small/deep' >&2; return 2 2>/dev/null || exit 2 ;;
esac
case "$STAGE" in all|pretrain|ft|eval|ft_and_eval) ;; *) echo 'Invalid STAGE' >&2; return 2 2>/dev/null || exit 2 ;; esac
if [[ ( "$STAGE" == all || "$STAGE" == ft_and_eval ) && -n "${MODEL_CKPT:-}" ]]; then
  echo 'MODEL_CKPT is for standalone ft/eval; unset it for the complete pipeline.' >&2
  return 2 2>/dev/null || exit 2
fi
OUTPUT_DIR="${OUTPUT_DIR:-./saved_ckpt_runs/${EXP_PREFIX:-mnist_tc_${mnist_name}}}"
RESULT_DIR="${RESULT_DIR:-${OUTPUT_DIR}/results}"
mnist_pretrain="${OUTPUT_DIR}/${mnist_name}_pretrain/${mnist_name}_pretrain_last_ckpt.pth"
mnist_ft="${OUTPUT_DIR}/${mnist_name}_ft/${mnist_name}_ft_last_ckpt.pth"
if [[ "${mode:-}" == ft_only || "$STAGE" == ft_and_eval ]]; then
  if [[ ! -s "${MODEL_CKPT:-$mnist_pretrain}" ]]; then
    echo "Missing or empty pretrained checkpoint: ${MODEL_CKPT:-$mnist_pretrain}" >&2
    return 2 2>/dev/null || exit 2
  fi
fi

mnist_run_stage() {
  local stage="$1"
  local policy=ft
  [[ "$stage" != eval ]] || policy=eval
  source ./launch_scripts/tc_hardware_defaults.sh "$policy"
  local -a cmd=(python -u -m mnist_train_eval.mnist_train)
  [[ "$stage" != eval ]] || cmd=(python -u -m mnist_train_eval.mnist_evaluate)
  cmd+=(--family "$FAMILY" --variant "$VARIANT" --tc_state "$TC_STATE" --stage "$stage"
        --data_dir "${DATA_DIR:-../data}" --output_dir "$OUTPUT_DIR"
        --seed "${SEED:-4096}" --num_workers "${NUM_WORKERS:-4}"
        --dropout "${DROPOUT:-0.25}" --t_end "${T_END:-1.75}"
        --one_shot_conv "${ONE_SHOT_CONV:-false}" --mem_frac "${MEM_FRAC:-0.9}"
        --device "${DEVICE:-auto}"
        --limit_train_samples "${LIMIT_TRAIN_SAMPLES:-0}"
        --limit_test_samples "${LIMIT_TEST_SAMPLES:-0}"
        --download "${DOWNLOAD:-false}" --dry_run "${DRY_RUN:-false}")
  local prefix=PRETRAIN lr=1.0 tol=1e-4
  if [[ "$stage" != pretrain ]]; then prefix=FT; lr=0.01; tol=1e-6; fi
  local name value spec key fallback
  for spec in 'EPOCHS:epochs:14' 'BATCH_SIZE:batch_size:64' 'OPTIMIZER:optimizer:adadelta' \
              'RHO:rho:0.9' 'EPS:eps:1e-6' 'WEIGHT_DECAY:weight_decay:0' \
              'MOMENTUM:momentum:0.9' 'SCHEDULER:scheduler:step' \
              'STEP_SIZE:step_size:1' 'GAMMA:gamma:0.7' 'MIN_LR:min_lr:0' \
              'HEALTH_CHECK_EPOCHS:health_check_epochs:5,10' 'HEALTH_CHECK_BATCHES:health_check_batches:4'; do
    IFS=: read -r key name fallback <<< "$spec"
    value="${!key:-$fallback}"
    key="${prefix}_${key}"
    cmd+=("--$name" "${!key:-$value}")
  done
  name="${prefix}_LR"; cmd+=(--lr "${!name:-$lr}")
  name="${prefix}_TOL"; cmd+=(--tol "${!name:-$tol}")
  cmd+=(--test_batch_size "${TEST_BATCH_SIZE:-64}")
  if [[ "$stage" == ft ]]; then
    cmd+=(--checkpoint "${MODEL_CKPT:-$mnist_pretrain}")
  elif [[ "$stage" == eval ]]; then
    cmd+=(--checkpoint "${MODEL_CKPT:-$mnist_ft}" --output_dir "$RESULT_DIR"
          --n_trials "${N_TRIALS:-10}" --expanded_weight_dir "${EXPANDED_WEIGHT_DIR:-./expanded_weights/mnist}")
  fi
  printf 'Running:'; printf ' %q' "${cmd[@]}"; printf '\n'
  "${cmd[@]}"
}

if [[ "$STAGE" == all || "$STAGE" == pretrain ]]; then mnist_run_stage pretrain; fi
if [[ "$STAGE" == all || "$STAGE" == ft || "$STAGE" == ft_and_eval ]]; then mnist_run_stage ft; fi
if [[ "$STAGE" == all || "$STAGE" == eval || "$STAGE" == ft_and_eval ]]; then mnist_run_stage eval; fi

#!/bin/bash
#set -euo pipefail

REPO_ROOT="/scratch/rzeng7/repos/PCN-with-Local-Recurrent-Processing"
SBATCH_SCRIPT="${REPO_ROOT}/launch_scripts/run_baseline_cifar_train.sh"

###############################################################################################
# running with
# module swap slurm slurm/24.05.0.b1
# ( source ./launch_scripts/slurm_run_baseline.sh ) \
#  > ./logs/scheduler_slurm/slurm_baseline_cifar.log 2>&1 < /dev/null &
# sched_pid=$!
# disown -h "$sched_pid"
###############################################################################################

#EXTRA_OVERRIDE="num_epochs=2,eval_every=2,skip_eval_epochs=0" # For test purpose
EXTRA_OVERRIDE="${EXTRA_OVERRIDE:-eval_every=5}"
MULT_NOISE_LEVEL_LIST="${MULT_NOISE_LEVEL_LIST:-0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4}"
ADD_NOISE_LEVEL_LIST="${ADD_NOISE_LEVEL_LIST:-0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1}"
EVAL_NOISY_TRIALS="${EVAL_NOISY_TRIALS:-10}"

export EXTRA_OVERRIDE MULT_NOISE_LEVEL_LIST ADD_NOISE_LEVEL_LIST EVAL_NOISY_TRIALS

MAX_TASKS_PER_GPU="${MAX_TASKS_PER_GPU:-1}"
GPUS_PER_JOB="${GPUS_PER_JOB:-1}"
SLURM_LOG_DIR="${SLURM_LOG_DIR:-/scratch/rzeng7/repos/PCN-with-Local-Recurrent-Processing/logs/slurm_jobs}"

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

JOB_IDS=()
JOB_DESCS=()

wait_for_jobs() {
  local -a ids=("$@")
  for i in "${!ids[@]}"; do
    ids[$i]="${ids[$i]%%;*}"
    ids[$i]="${ids[$i]%%.*}"
  done
  [[ ${#ids[@]} -gt 0 ]] || return 0

  while :; do
    local done=0
    for j in "${ids[@]}"; do
      local state
      state=$(sacct -X -n -j "$j" -o State 2>/dev/null | head -n1 | xargs)
      case "$state" in
        *COMPLETED*|*FAILED*|*CANCELLED*|*TIMEOUT*|*OUT_OF_MEMORY*) ((done++)) ;;
        ""|RUNNING|PENDING|CONFIGURING|COMPLETING|SUSPENDED|REQUEUED|RESIZING|PREEMPTED|NODE_FAIL) : ;;
        *) : ;;
      esac
    done
    (( done == ${#ids[@]} )) && break
    sleep 20
  done
}

submit_one() {
  local model_name="$1"
  local dataset_name="$2"
  local case_name="$3"
  local pretrained="$4"
  local prefer_resize="$5"

  local tag="${dataset_name}_${case_name}_${model_name}_pretrained_${pretrained}"

  jid=$(
    sbatch --parsable \
      --gres=gpu:${GPUS_PER_JOB} \
      --job-name="${tag}" \
      --export=ALL,IS_SLURM=1,MODEL_NAME="${model_name}",DATASET_NAME="${dataset_name}",CASE_NAME="${case_name}",PRETRAINED="${pretrained}",PREFER_RESIZE="${prefer_resize}" \
      "${SBATCH_SCRIPT}"
  )

  JOB_IDS+=("${jid}")
  JOB_DESCS+=("${dataset_name}|${case_name}|${model_name}|${pretrained}|${prefer_resize}")

  echo "submitted job ${jid}: dataset=${dataset_name}, model=${model_name}, case=${case_name}, pretrained=${pretrained}"
}

for dataset_name in "${DATASETS[@]}"; do
  for model_name in "${CUSTOM_CIFAR_MODELS[@]}"; do
    submit_one "${model_name}" "${dataset_name}" "custom_noresize" "false" "false"
  done

  for model_name in "${ADAPT_NORESIZE_TIMM_MODELS[@]}"; do
    submit_one "${model_name}" "${dataset_name}" "adapt_noresize_scratch" "false" "false"
  done

  for model_name in "${RESIZE_FINETUNE_TIMM_MODELS[@]}"; do
    submit_one "${model_name}" "${dataset_name}" "resize_finetune" "true" "true"
  done
done

echo "======================================================================"
echo "Submitted ${#JOB_IDS[@]} jobs. Waiting for completion..."
echo "======================================================================"

wait_for_jobs "${JOB_IDS[@]}"

echo "======================================================================"
echo "All submitted jobs finished or reached terminal states."
echo "All saved checkpoint paths:"
echo "======================================================================"

for i in "${!JOB_IDS[@]}"; do
  jid="${JOB_IDS[$i]}"
  jid="${jid%%;*}"
  jid="${jid%%.*}"

  desc="${JOB_DESCS[$i]}"
  IFS='|' read -r dataset_name case_name model_name pretrained prefer_resize <<< "${desc}"

  slog="${SLURM_LOG_DIR}/slurm_${jid}.out"

  if [[ ! -s "${slog}" ]]; then
    echo "[WARN] Missing slurm log ${slog}"
    continue
  fi

  # Extract model path
  ckpt_path="$(
  sed -nE 's/^----- Model path: (.*) -----$/\1/p' "${slog}" \
      | tail -n 1
  )"

  # Extract evaluation result path for additive and multiplicative mismatch eval
  csv_paths="$(
    sed -nE 's/^Saved CSV to:[[:space:]]*(.*\.csv)$/\1/p' "${slog}"
  )"

  echo "dataset=${dataset_name} case=${case_name} model=${model_name} pretrained=${pretrained} prefer_resize=${prefer_resize}"

  if [[ -n "${ckpt_path}" && -f "${ckpt_path}" ]]; then
    echo "  ckpt: ${ckpt_path}"
  else
    echo "  [WARN] No valid checkpoint path found"
    echo "  slurm_log=${slog}"
  fi

  if [[ -n "${csv_paths}" ]]; then
    while IFS= read -r csv_path; do
      if [[ -f "${csv_path}" ]]; then
        echo "  csv:  ${csv_path}"
      else
        echo "  [WARN] CSV path found but file missing: ${csv_path}"
      fi
    done <<< "${csv_paths}"
  else
    echo "  [WARN] No CSV path found"
  fi
done

echo "======================================================================"
echo "Scheduler finished."
echo "======================================================================"
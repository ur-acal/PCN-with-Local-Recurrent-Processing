#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${REPO_ROOT}" || exit 1
export OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/logs/wrn_controls_slurm}"
export DATA_DIR="${DATA_DIR:-${REPO_ROOT}/../data}"
export STAGE="${STAGE:-train-test}"
export EVAL_PARALLELISM="${EVAL_PARALLELISM:-8}"
export PARALLELISM="${PARALLELISM:-4}"
export SIZES="${SIZES:-16_2,16_4,28_2,28_4}"
export DATASETS="${DATASETS:-cifar10,cifar100}"
export SIMULATE="${SIMULATE:-0}"
export CONDITIONS="${CONDITIONS:-max_additive,multiplicative,rms_additive}"
export SEED="${SEED:-4096}"
export BASE_SEED="${BASE_SEED:-123}"

IFS=',' read -ra rows <<< "${ROWS:-2,4,5,6,7,8,9,10}"
IFS=',' read -ra datasets <<< "${DATASETS:-cifar10,cifar100}"
IFS=',' read -ra sizes <<< "${SIZES:-16_2,16_4,28_2,28_4}"
case "${STAGE}" in train|test|train-test) ;; *) echo "Invalid STAGE" >&2; exit 2 ;; esac
for row in "${rows[@]}"; do
  case "$row" in 2|4|5|6|7|8|9|10) ;; *) echo "SLURM excludes rows 1 and 3; invalid row: $row" >&2; exit 2 ;; esac
done
for dataset in "${datasets[@]}"; do
  case "$dataset" in cifar10|cifar100) ;; *) echo "Invalid dataset: $dataset" >&2; exit 2 ;; esac
done
for size in "${sizes[@]}"; do
  case "$size" in 16_2|16_4|28_2|28_4) ;; *) echo "Invalid size: $size" >&2; exit 2 ;; esac
done
if [[ "${DRY_RUN:-0}" != 1 ]]; then
  mkdir -p "${REPO_ROOT}/logs/slurm_jobs" || exit 1
  plan_args=(plan --output-root "${OUTPUT_ROOT}" --rows "${ROWS:-2,4,5,6,7,8,9,10}"
    --datasets "${DATASETS:-cifar10,cifar100}" --sizes "${SIZES:-16_2,16_4,28_2,28_4}"
    --conditions "${CONDITIONS}" --stage "${STAGE}")
  if [[ "${SIMULATE}" == 1 ]]; then plan_args+=(--simulate); fi
  PLAN_FILE=$("${PYTHON_BIN:-python}" -m baseline.wrn_control_artifacts "${plan_args[@]}") || exit 1
  echo "Submission inventory: ${PLAN_FILE}"
fi
failures=0
for ROW in "${rows[@]}"; do
      export ROW
      command=(sbatch --parsable --chdir="${REPO_ROOT}"
        --output="${REPO_ROOT}/logs/slurm_jobs/slurm_%j.out"
        --job-name="wrn-r${ROW}-${STAGE}"
        --gres=gpu:1 --export=ALL "${REPO_ROOT}/launch_scripts/run_wrn_controls.sbatch")
      if [[ "${DRY_RUN:-0}" == 1 ]]; then
        printf 'ROW=%s DATASETS=%s SIZES=%s STAGE=%s ' "$ROW" "$DATASETS" "$SIZES" "$STAGE"
        printf '%q ' "${command[@]}"
        printf '\n'
      else
        if [[ "${SIMULATE}" == 1 ]]; then
          jid="SIMULATED-${ROW}"
        else
          jid=$("${command[@]}") || exit 1
        fi
        for DATASET in "${datasets[@]}"; do
          for SIZE in "${sizes[@]}"; do
          "${PYTHON_BIN:-python}" -m baseline.wrn_control_artifacts submitted --plan "${PLAN_FILE}" \
          --row "${ROW}" --dataset "${DATASET}" --size "${SIZE}" --job-id "${jid}" || exit 1
          done
        done
        echo "submitted job ${jid}: row=${ROW}, datasets=${DATASETS}, sizes=${SIZES}, stage=${STAGE}, output=${OUTPUT_ROOT}"
        if [[ "${SIMULATE}" == 1 ]]; then
          IS_SLURM=0 SLURM_JOB_ID="${jid}" bash "${REPO_ROOT}/launch_scripts/run_wrn_controls.sbatch" || failures=$((failures + 1))
        fi
      fi
done
[[ "${failures}" == 0 ]]

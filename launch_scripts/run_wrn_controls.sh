#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${REPO_ROOT}" || exit 1

args=(--rows "${ROWS:-3}" --datasets "${DATASETS:-cifar10,cifar100}"
      --sizes "${SIZES:-16_2,16_4,28_2,28_4}" --stage "${STAGE:-train}"
      --parallelism "${PARALLELISM:-4}" --seed "${SEED:-4096}"
      --eval-parallelism "${EVAL_PARALLELISM:-3}"
      --mismatch-seed "${BASE_SEED:-123}"
      --conditions "${CONDITIONS:-max_additive,multiplicative}"
      --data-dir "${DATA_DIR:-${REPO_ROOT}/../data}"
      --output-root "${OUTPUT_ROOT:-${REPO_ROOT}/logs/wrn_controls_local}")
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  args+=(--dry-run)
fi
if [[ "${SIMULATE:-0}" == 1 ]]; then
  args+=(--simulate --simulate-failure "${SIMULATE_FAILURE:-}")
fi
exec "${PYTHON_BIN:-python}" -u -m baseline.run_wrn_controls "${args[@]}"

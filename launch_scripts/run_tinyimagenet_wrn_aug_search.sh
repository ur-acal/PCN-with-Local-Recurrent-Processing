#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/home/rongzeng/anaconda3/envs/scanbase/bin/python}"
SEARCH_ROOT="${SEARCH_ROOT:-${ROOT}/logs/tinyimagenet_wrn28_4_aug_search}"
DATA_ROOT="${DATA_ROOT:-${ROOT}/../data/tiny-imagenet-200}"
PARALLELISM="${PARALLELISM:-3}"

exec "${PYTHON_BIN}" -u baseline/run_tinyimagenet_wrn_aug_search.py \
  --search_root "${SEARCH_ROOT}" \
  --data_root "${DATA_ROOT}" \
  --parallelism "${PARALLELISM}"

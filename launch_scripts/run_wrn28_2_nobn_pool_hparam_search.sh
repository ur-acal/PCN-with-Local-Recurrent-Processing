#!/usr/bin/env bash

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/home/rongzeng/anaconda3/envs/scanbase/bin/python}"
SEARCH_ROOT="${SEARCH_ROOT:-$ROOT/logs/wrn28_2_nobn_pool_hparam_search_v2}"
DATA_ROOT="${DATA_ROOT:-$ROOT/../data}"
PARALLELISM="${PARALLELISM:-4}"
DATASET_NAME="${DATASET_NAME:?DATASET_NAME is required}"
STUDY_NAME="${STUDY_NAME:?STUDY_NAME is required}"

exec "$PYTHON_BIN" -u baseline/run_wrn28_2_nobn_pool_hparam_search.py \
  --dataset "$DATASET_NAME" \
  --study "$STUDY_NAME" \
  --search_root "$SEARCH_ROOT" \
  --data_root "$DATA_ROOT" \
  --parallelism "$PARALLELISM" \
  "$@"

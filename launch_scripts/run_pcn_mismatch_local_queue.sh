#!/usr/bin/env bash

set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-/home/rongzeng/_workspce_old/repos/pcn/collaboration/ScAN-PCN-mismatch_analysis}"
LEGACY_MODEL_DIR="${LEGACY_MODEL_DIR:-${REPO_ROOT}/../ScAN-PCN/saved_ckpt}"
NEW_MODEL_DIR="${NEW_MODEL_DIR:-${REPO_ROOT}/saved_ckpt}"
STATE_DIR="${STATE_DIR:-${REPO_ROOT}/logs/pcn_mismatch_local_queue}"
WORKER="${REPO_ROOT}/launch_scripts/run_rgb_ode_mismatch_eval.sh"

export PATH="/home/rongzeng/anaconda3/envs/scanbase/bin:${PATH}"
mkdir -p "${STATE_DIR}"
cd "${REPO_ROOT}"

write_state() {
  printf 'phase=%s\nstatus=%s\nupdated=%s\n' \
    "$1" "$2" "$(date --iso-8601=seconds)" > "${STATE_DIR}/state.txt"
}

wait_for_group() {
  local phase="$1"
  shift
  local status=0
  local pid

  for pid in "$@"; do
    if ! wait "${pid}"; then
      status=1
    fi
  done

  if (( status != 0 )); then
    write_state "${phase}" "failed"
    return 1
  fi
}

write_state "legacy_cifar100_max_sqrt" "running"
legacy_pids=()
for shard_id in 0 1 2 3; do
  env \
    REPO_ROOT="${REPO_ROOT}" \
    MODEL_DIR="${LEGACY_MODEL_DIR}" \
    OUTPUT_ROOT="${REPO_ROOT}/logs/pcn_legacy_no_x_cifar100_max_sqrt_local" \
    MODEL_SET="legacy_no_x" \
    DATASETS="cifar100" \
    EVAL_MODE="mismatch" \
    CONDITIONS="max_sqrt_additive" \
    MAX_SQRT_LEVELS="0,0.02,0.03,0.05,0.07,0.09,0.1" \
    NOISY_TRIALS="10" \
    SHARD_ID="${shard_id}" \
    bash "${WORKER}" &
  legacy_pids+=("$!")
done
wait_for_group "legacy_cifar100_max_sqrt" "${legacy_pids[@]}"

write_state "new_x_minus_fb_all_mismatch" "running"
new_pids=()
for shard_id in 0 1; do
  env \
    REPO_ROOT="${REPO_ROOT}" \
    MODEL_DIR="${NEW_MODEL_DIR}" \
    OUTPUT_ROOT="${REPO_ROOT}/logs/pcn_x_minus_fb_mismatch_local" \
    MODEL_SET="new_x_minus_fb" \
    EVAL_MODE="mismatch" \
    CONDITIONS="multiplicative,max_additive,rms_additive,max_sqrt_additive" \
    RMS_ADD_LEVELS="0.25,0.5,0.75,1.0,1.25" \
    MAX_SQRT_LEVELS="0,0.02,0.03,0.05,0.07,0.09,0.1" \
    NOISY_TRIALS="10" \
    SHARD_ID="${shard_id}" \
    bash "${WORKER}" &
  new_pids+=("$!")
done
wait_for_group "new_x_minus_fb_all_mismatch" "${new_pids[@]}"

write_state "complete" "complete"

#!/usr/bin/env bash
# Wait for both corrected WRN fine-tuning runs, validate their terminal
# artifacts, then run the four one-trial MC45 evaluations sequentially.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO_ROOT}"

LOG_DIR="logs/local_runs/tt_singleton_50ep"
WRN_NAME="wrn_flexible_cifar_avgpool_shortcut"
RUNNER="./launch_scripts/run_cifair100_tt_singleton_ft_eval.sh"
POLL_SECONDS="${POLL_SECONDS:-60}"
COUPLER_MONTE_TT_SINGLETON="${COUPLER_MONTE_TT_SINGLETON:?Specify the coupler_monte training singleton CSV}"
COUPLER_MONTE_V2_TT_SINGLETON="${COUPLER_MONTE_V2_TT_SINGLETON:?Specify the coupler_monte_v2 training singleton CSV}"
mkdir -p "${LOG_DIR}"

on_exit() {
  local status=$?
  trap - EXIT
  printf '%s QUEUE_EXIT_STATUS=%s\n' "$(date --iso-8601=seconds)" "${status}"
  exit "${status}"
}
trap on_exit EXIT

wrn_pidfile() {
  printf '%s/wrn_c27_%s_fixed15ns_vdd0p1_qinv5.pid' "${LOG_DIR}" "$1"
}

wrn_log() {
  printf '%s/wrn_c27_%s_fixed15ns_vdd0p1_qinv5.ft.log' "${LOG_DIR}" "$1"
}

run_tag() {
  local model_kind=$1 family=$2
  if [[ "${model_kind}" == wrn_c27 ]]; then
    printf 'tt_singleton_50ep_CiFAIR100_wrn_c27_%s_fixedTiming15ns_vdd0p1_qinv5' "${family}"
  else
    printf 'tt_singleton_50ep_CiFAIR100_pcn_%s' "${family}"
  fi
}

wrn_checkpoint_dir() {
  local family=$1
  local tag="$(run_tag wrn_c27 "${family}")"
  local model_tag="custom_noresize_cifar100_${WRN_NAME}"
  printf 'saved_ckpt_runs/%s/cifar100/custom_noresize/%s/%s' \
    "${tag}_ft_iq12" "${WRN_NAME}" "${model_tag}"
}

wait_for_wrns() {
  local family pidfile pid state alive
  while true; do
    alive=0
    for family in coupler_monte coupler_monte_v2; do
      pidfile="$(wrn_pidfile "${family}")"
      [[ -s "${pidfile}" ]] || {
        echo "Missing WRN supervisor pidfile: ${pidfile}" >&2
        return 1
      }
      pid="$(<"${pidfile}")"
      state="$(ps -o stat= -p "${pid}" 2>/dev/null || true)"
      if kill -0 "${pid}" 2>/dev/null && [[ "${state}" != Z* ]]; then
        alive=1
      fi
    done
    (( alive == 0 )) && break
    printf '%s WAITING_FOR_BOTH_WRNS\n' "$(date --iso-8601=seconds)"
    sleep "${POLL_SECONDS}"
  done
}

validate_wrn_completion() {
  local family=$1 log checkpoint_dir model_tag
  log="$(wrn_log "${family}")"
  checkpoint_dir="$(wrn_checkpoint_dir "${family}")"
  model_tag="custom_noresize_cifar100_${WRN_NAME}"

  [[ -f "${log}" ]] || { echo "Missing WRN log: ${log}" >&2; return 1; }
  ! grep -q 'Traceback (most recent call last)' "${log}" || {
    echo "WRN ${family} ended with a traceback: ${log}" >&2
    return 1
  }
  grep -q -- '----- Train finished' "${log}" || {
    echo "WRN ${family} did not reach Train finished: ${log}" >&2
    return 1
  }
  [[ -f "${checkpoint_dir}/${model_tag}_full_param_best_ckpt.pth" ]] || {
    echo "Missing WRN ${family} full-parameter best checkpoint" >&2
    return 1
  }
  [[ -f "${checkpoint_dir}/${model_tag}_last_ckpt.pth" ]] || {
    echo "Missing WRN ${family} terminal last checkpoint" >&2
    return 1
  }
  [[ ! -e "${checkpoint_dir}/${model_tag}_latest_ckpt.pth" ]] || {
    echo "WRN ${family} still has latest_ckpt; training is not terminal" >&2
    return 1
  }
  printf '%s VERIFIED_WRN_COMPLETE family=%s checkpoint=%s\n' \
    "$(date --iso-8601=seconds)" "${family}" \
    "${checkpoint_dir}/${model_tag}_full_param_best_ckpt.pth"
}

run_eval() {
  local model_kind=$1 family=$2
  local singleton tag
  case "${family}" in
    coupler_monte) singleton="${COUPLER_MONTE_TT_SINGLETON}" ;;
    coupler_monte_v2) singleton="${COUPLER_MONTE_V2_TT_SINGLETON}" ;;
    *) echo "Unsupported coupler family: ${family}" >&2; return 2 ;;
  esac
  tag="$(run_tag "${model_kind}" "${family}")"
  local log="${LOG_DIR}/${model_kind}_${family}.mc45.log"
  printf '%s START_MC45 model=%s family=%s\n' \
    "$(date --iso-8601=seconds)" "${model_kind}" "${family}" | tee -a "${log}"

  # Keep the historical namespace used by completion validation, rather than
  # the runner's newer default containing the singleton basename.
  env MODEL_KIND="${model_kind}" COUPLER_FAMILY="${family}" \
    TT_SINGLETON="${singleton}" RUN_TAG="${tag}" \
    STAGE=eval WRN_ONE_OVER_Q=5 N_TRIALS=1 CORNER_IDS="" \
    bash "${RUNNER}" 2>&1 | tee -a "${log}"

  printf '%s COMPLETE_MC45 model=%s family=%s\n' \
    "$(date --iso-8601=seconds)" "${model_kind}" "${family}" | tee -a "${log}"
}

echo "$(date --iso-8601=seconds) QUEUE_STARTED"
wait_for_wrns
validate_wrn_completion coupler_monte
validate_wrn_completion coupler_monte_v2

# Keep only one full MC45 evaluation active at a time.  A resumed queue may
# retain already completed PCN results and continue directly with WRN.
if [[ "${SKIP_COMPLETED_PCN:-false}" != true ]]; then
  run_eval pcn coupler_monte
  run_eval pcn coupler_monte_v2
fi
run_eval wrn_c27 coupler_monte
run_eval wrn_c27 coupler_monte_v2

echo "$(date --iso-8601=seconds) ALL_FOUR_MC45_COMPLETE"

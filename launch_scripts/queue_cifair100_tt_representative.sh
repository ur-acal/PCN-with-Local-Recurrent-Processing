#!/usr/bin/env bash
# Wait for the existing BN-recalibration queue, then run four FT -> MC45
# pipelines sequentially. --preflight exercises the same run_job function.
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO_ROOT}"
PYTHON="${PYTHON:-/home/rongzeng/anaconda3/envs/scanbase/bin/python}"
export PYTHON PATH="$(dirname "${PYTHON}"):${PATH}"
COUPLER_MONTE_TT_SINGLETON="${COUPLER_MONTE_TT_SINGLETON:?Specify the coupler_monte singleton CSV}"
COUPLER_MONTE_V2_TT_SINGLETON="${COUPLER_MONTE_V2_TT_SINGLETON:?Specify the coupler_monte_v2 singleton CSV}"
LOG_DIR="${LOG_DIR:-logs/local_runs/tt_representative_50ep}"
mkdir -p "${LOG_DIR}"
preflight=false
rehearsal=false
case "${1:-}" in
  --preflight) preflight=true ;;
  --rehearse)
    rehearsal=true
    export REHEARSAL_ARTIFACT_ROOT="${REHEARSAL_ARTIFACT_ROOT:-${REPO_ROOT}/results/tt_representative_launch_rehearsal}"
    ;;
  "") ;;
  *) echo "Usage: $0 [--preflight|--rehearse]" >&2; exit 2 ;;
esac
on_exit() {
  status=$?
  printf '%s QUEUE_EXIT_STATUS=%s\n' "$(date --iso-8601=seconds)" "${status}"
}
trap on_exit EXIT

run_job() {
  local kind="$1" family="$2" singleton="$3"
  local log="${LOG_DIR}/${kind}_${family}.${preflight}.pipeline.log"
  [[ "${rehearsal}" == true ]] && log="${LOG_DIR}/${kind}_${family}.rehearsal.pipeline.log"
  printf '%s START model=%s family=%s preflight=%s\n' \
    "$(date --iso-8601=seconds)" "${kind}" "${family}" "${preflight}"
  MODEL_KIND="${kind}" COUPLER_FAMILY="${family}" \
  TT_SINGLETON="${singleton}" WRN_ONE_OVER_Q=5 \
  STAGE=all N_TRIALS=1 CORNER_IDS="" FUSE_MEASURED_ACTIVATION=true \
  PREFLIGHT_ONLY="${preflight}" \
  REHEARSAL_ONLY="${rehearsal}" \
    bash ./launch_scripts/run_cifair100_tt_singleton_ft_eval.sh \
    2>&1 | tee "${log}"
  if [[ "${preflight}" == true ]]; then
    [[ "$(grep -c '^PREFLIGHT_PASS ' "${log}")" == 2 ]] || {
      echo "Preflight did not validate both FT and eval: ${log}" >&2
      return 1
    }
    "${PYTHON}" diagnostic_scripts/verify_tt_singleton_launch.py --check-handoff "${log}"
  fi
  printf '%s COMPLETE model=%s family=%s preflight=%s\n' \
    "$(date --iso-8601=seconds)" "${kind}" "${family}" "${preflight}"
}

if [[ "${preflight}" != true && "${rehearsal}" != true ]]; then
  printf '%s\n' "$$" > "${LOG_DIR}/queue.pid"
  WAIT_PID="${WAIT_PID:?Specify the current BN-recalibration queue PID}"
  dependency="${DEPENDENCY_QUEUE_SCRIPT:?Specify the current BN-recalibration queue script}"
  printf '%s WAITING_FOR_BN_RECAL_QUEUE pid=%s\n' "$(date --iso-8601=seconds)" "${WAIT_PID}"
  while [[ -r "/proc/${WAIT_PID}/cmdline" ]] && \
        tr '\0' ' ' < "/proc/${WAIT_PID}/cmdline" | grep -Fq -- "${dependency}"; do
    sleep 30
  done
  # The predecessor is not our child, so validate both its explicit status and
  # all completed corner records rather than guessing success from PID exit.
  for family in coupler_monte coupler_monte_v2; do
    log="logs/local_runs/tt_singleton_50ep/wrn_c27_${family}.mc45.bnrecal5120.resume.log"
    grep -q 'continuation exit status 0\.' "${log}" || {
      echo "Dependency failed or lacks terminal success: ${log}" >&2; exit 1;
    }
    output="results/tt_singleton_50ep_CiFAIR100_wrn_c27_${family}_fixedTiming15ns_vdd0p1_qinv5_mc45_bnrecal5120"
    for process in TT FF SS SF FS; do
      for voltage in 0 1 2; do
        for temperature in 0 1 2; do
          grep -q '^accuracy=' "${output}/${process}_V${voltage}_T${temperature}.log" || {
            echo "Dependency corner unfinished: ${process}_V${voltage}_T${temperature} (${family})" >&2
            exit 1
          }
        done
      done
    done
  done
  printf '%s BOTH_BN_RECAL_RUNS_VERIFIED_COMPLETE\n' "$(date --iso-8601=seconds)"
fi

model_kinds=(pcn wrn_c27)
if [[ "${rehearsal}" == true && -n "${REHEARSAL_MODEL_KINDS:-}" ]]; then
  read -r -a model_kinds <<< "${REHEARSAL_MODEL_KINDS}"
fi
for kind in "${model_kinds[@]}"; do
  run_job "${kind}" coupler_monte "${COUPLER_MONTE_TT_SINGLETON}"
  run_job "${kind}" coupler_monte_v2 "${COUPLER_MONTE_V2_TT_SINGLETON}"
done
printf '%s ALL_FOUR_PIPELINES_COMPLETE preflight=%s\n' "$(date --iso-8601=seconds)" "${preflight}"

#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RMS_ROOT="$ROOT/logs/rms_additive_full"
JOBLOG="$RMS_ROOT/pcn_rms.joblog"
STATE="$RMS_ROOT/corrected_report_state.json"
LOG="$RMS_ROOT/corrected_report_generation.log"
EXPECTED_JOBS=8
PYTHON_BIN="/home/rongzeng/anaconda3/envs/scanbase/bin/python"
write_state() {
  local status="$1" detail="$2"
  printf '{\n  "status": "%s",\n  "stage": "corrected_report",\n  "detail": "%s",\n  "controller_pid": %d,\n  "updated_epoch_seconds": %d\n}\n' "$status" "$detail" "$$" "$(date +%s)" >"$STATE.tmp"
  mv "$STATE.tmp" "$STATE"
}
on_error() {
  local code=$?
  write_state failed "exit_code_${code}"
  exit "$code"
}
trap on_error ERR
write_state waiting "pcn_rms_0_of_${EXPECTED_JOBS}"
while true; do
  completed=0
  failed=0
  if [[ -f "$JOBLOG" ]]; then
    completed="$(awk 'NR > 1 {count++} END {print count+0}' "$JOBLOG")"
    failed="$(awk 'NR > 1 && $7 != 0 {count++} END {print count+0}' "$JOBLOG")"
  fi
  if ((failed > 0)); then
    write_state failed "pcn_rms_failed_${failed}"
    exit 1
  fi
  write_state waiting "pcn_rms_${completed}_of_${EXPECTED_JOBS}"
  if ((completed == EXPECTED_JOBS)); then break; fi
  if ((completed > EXPECTED_JOBS)); then
    write_state failed "unexpected_job_count_${completed}"
    exit 1
  fi
  sleep 60
done
result_count="$(find "$RMS_ROOT/pcn/rms" -type f -name result.pkl | wc -l)"
if ((result_count != EXPECTED_JOBS)); then
  write_state failed "result_pickle_count_${result_count}"
  exit 1
fi
write_state running "repair_validate_generate"
cd "$ROOT"
"$PYTHON_BIN" -u -m baseline.generate_combined_rms_mismatch_report --repair_rms_wrn_derived >"$LOG" 2>&1
write_state completed "validated_reports_ready"

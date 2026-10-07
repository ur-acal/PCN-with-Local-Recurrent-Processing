# Optional evaluation-only total-current clamp. Empty summary disables it.
RHS_CURRENT_ARGS=()
if [[ -n "${RHS_CURRENT_SUMMARY:-}" ]]; then
  RHS_CURRENT_ARGS+=(--rhs_current_summary "${RHS_CURRENT_SUMMARY}"
    --rhs_current_bound_percentile "${RHS_CURRENT_BOUND_PERCENTILE:-99}")
  if [[ -n "${RHS_CURRENT_AUDIT_PATH:-}" ]]; then
    RHS_CURRENT_ARGS+=(--rhs_current_audit_path "${RHS_CURRENT_AUDIT_PATH}")
  fi
fi

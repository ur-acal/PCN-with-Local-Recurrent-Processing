# Normalize the public validation-mode environment flag once at each launcher
# boundary so shell naming/selection agrees with Python's boolean parser.
case "${VALIDATION_MODE:-false}" in
  [Tt][Rr][Uu][Ee]|[Tt]|1|[Yy][Ee][Ss]|[Yy]|[Oo][Nn])
    export VALIDATION_MODE=true
    ;;
  [Ff][Aa][Ll][Ss][Ee]|[Ff]|0|[Nn][Oo]|[Nn]|[Oo][Ff][Ff]|"")
    export VALIDATION_MODE=false
    ;;
  *)
    echo "VALIDATION_MODE must be true/false, 1/0, yes/no, or on/off; got '${VALIDATION_MODE}'." >&2
    return 2
    ;;
esac

# Keep validation checkpoints separate even when callers provide an explicit
# output root. The check is idempotent so parent and standalone launchers can
# both normalize the same path safely.
validation_output_path() {
  local path="$1"
  if [[ "${VALIDATION_MODE}" == true && "${path}" != *"_val5k"* ]]; then
    path+="_val5k"
  fi
  printf '%s\n' "${path}"
}

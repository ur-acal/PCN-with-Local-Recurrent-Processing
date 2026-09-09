#!/usr/bin/env bash
# Compatibility alias for the reusable local PCN pipeline.
exec bash "$(dirname "${BASH_SOURCE[0]}")/run_local_toggle_pretrain_then_ft.sh" "$@"

#!/usr/bin/env bash
# Run the four controlled FT -> MC45 pipelines sequentially. A failed stage
# stops the queue with a nonzero status; completed stages are marked explicitly.
# Supply COUPLER_MONTE_TT_SINGLETON and COUPLER_MONTE_V2_TT_SINGLETON CSVs.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO_ROOT}"
LOG_DIR="${LOG_DIR:-logs/local_runs/tt_singleton_50ep_dynamic}"
mkdir -p "${LOG_DIR}"
COUPLER_MONTE_TT_SINGLETON="${COUPLER_MONTE_TT_SINGLETON:?Set the coupler_monte training singleton CSV}"
COUPLER_MONTE_V2_TT_SINGLETON="${COUPLER_MONTE_V2_TT_SINGLETON:?Set the coupler_monte_v2 training singleton CSV}"

requested_stage="${STAGE:-all}"
case "${requested_stage}" in
  ft) phases=(ft) ;;
  eval) phases=(eval) ;;
  all) phases=(ft eval) ;;
  *) echo "STAGE must be ft, eval, or all; got ${requested_stage}" >&2; exit 2 ;;
esac

# Finish every 50-epoch FT job before beginning the expensive MC45 phase.
for phase in "${phases[@]}"; do
  for model_kind in pcn wrn_c27; do
    for coupler_family in coupler_monte coupler_monte_v2; do
      case "${coupler_family}" in
        coupler_monte) singleton="${COUPLER_MONTE_TT_SINGLETON}" ;;
        coupler_monte_v2) singleton="${COUPLER_MONTE_V2_TT_SINGLETON}" ;;
      esac
      tag="${model_kind}_${coupler_family}"
      log="${LOG_DIR}/${tag}.${phase}.log"
      echo "$(date --iso-8601=seconds) START ${phase} ${tag}" | tee -a "${log}"
      MODEL_KIND="${model_kind}" \
      COUPLER_FAMILY="${coupler_family}" \
      TT_SINGLETON="${singleton}" \
      WRN_ONE_OVER_Q="${WRN_ONE_OVER_Q:-5}" \
      STAGE="${phase}" \
      N_TRIALS="${N_TRIALS:-1}" \
        bash ./launch_scripts/run_cifair100_tt_singleton_ft_eval.sh \
        2>&1 | tee -a "${log}"
      echo "$(date --iso-8601=seconds) COMPLETE ${phase} ${tag}" | tee -a "${log}"
    done
  done
done

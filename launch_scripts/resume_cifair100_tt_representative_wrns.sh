#!/usr/bin/env bash
# Continue the interrupted representative-curve queue without repeating PCNs.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHON=/home/rongzeng/anaconda3/envs/scanbase/bin/python
export PATH="$(dirname "${PYTHON}"):${PATH}"
LOG_DIR=logs/local_runs/tt_representative_50ep
mkdir -p "${LOG_DIR}"
printf '%s\n' "$$" > "${LOG_DIR}/resume_wrns.pid"
trap 'printf "%s RESUME_QUEUE_EXIT_STATUS=%s\n" "$(date --iso-8601=seconds)" "$?"' EXIT
name=custom_noresize_cifar100_wrn_flexible_cifar_avgpool_shortcut
tag=tt_singleton_50ep_CiFAIR100_wrn_c27_coupler_monte_fixedTiming15ns_vdd0p1_qinv5_coupler_monte_tt_25_1_mc73_singleton
recovery="saved_ckpt_runs/${tag}_ft_iq12/cifar100/custom_noresize/wrn_flexible_cifar_avgpool_shortcut/${name}/${name}_latest_ckpt.pth"
[[ -f "${recovery}" ]] || { echo "Missing recovery checkpoint: ${recovery}" >&2; exit 1; }
unset REHEARSAL_ONLY REHEARSAL_MODEL_KINDS PYTHONPATH RUN_TAG
for family in coupler_monte coupler_monte_v2; do
  if [[ "${family}" == coupler_monte ]]; then
    singleton=coupler_monte_tt_25_1_mc73_singleton.csv
    export WRN_RECOVERY_CKPT="${recovery}"
  else
    singleton=coupler_monte_v2_tt_25_1_mc34_singleton.csv
    unset WRN_RECOVERY_CKPT
  fi
  printf '%s RESUME_QUEUE_START model=wrn_c27 family=%s\n' "$(date --iso-8601=seconds)" "${family}"
  MODEL_KIND=wrn_c27 COUPLER_FAMILY="${family}" TT_SINGLETON="${singleton}" \
    WRN_ONE_OVER_Q=5 STAGE=all N_TRIALS=1 CORNER_IDS="" \
    FUSE_MEASURED_ACTIVATION=true PREFLIGHT_ONLY=false REHEARSAL_ONLY=false \
    bash ./launch_scripts/run_cifair100_tt_singleton_ft_eval.sh \
    2>&1 | tee -a "${LOG_DIR}/wrn_c27_${family}.resume.pipeline.log"
  printf '%s RESUME_QUEUE_COMPLETE model=wrn_c27 family=%s\n' "$(date --iso-8601=seconds)" "${family}"
done

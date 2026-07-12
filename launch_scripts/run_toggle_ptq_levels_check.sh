#!/usr/bin/env bash
set -euo pipefail

# Compare a Level-1 unitless toggle checkpoint against PTQ Level-2/3 toggle models.
# Required:
#   MODEL_NAME=<checkpoint directory name> bash launch_scripts/run_toggle_ptq_levels_check.sh

: "${MODEL_NAME:?Set MODEL_NAME to the trained Level-1 toggle checkpoint name.}"

RESET_MODE="${RESET_MODE:-reset}" # reset or persistent
case "${RESET_MODE}" in
  reset)
    LEVEL1_BLOCK="ToggleUnitlessResetZFFFB"
    LEVEL2_BLOCK="ToggleResetZ"
    LEVEL3_BLOCK="TogglePulseResetZ"
    ;;
  persistent)
    LEVEL1_BLOCK="ToggleUnitlessPersistentZFFFB"
    LEVEL2_BLOCK="ToggleKeepZ"
    LEVEL3_BLOCK="TogglePulseKeepZ"
    ;;
  *) echo "RESET_MODE must be reset or persistent, got: ${RESET_MODE}" >&2; exit 2 ;;
esac

MODEL_DIR="${MODEL_DIR:-saved_ckpt}"
CKPT="${CKPT:-best}"
TASK="${TASK:-${DATASET_NAME:-cifar10}}"
IMG_TYPE="${IMG_TYPE:-rgb}"
PC_CONV="${PC_CONV:-PCConvReLU6Noisy}"
METHOD="${METHOD:-euler}"
TOL="${TOL:-0.0001}"
N_STEPS="${N_STEPS:-10}"
TOGGLE_N_CYCLES="${TOGGLE_N_CYCLES:-${N_STEPS}}"
TOGGLE_TIME_SPLIT="${TOGGLE_TIME_SPLIT:-0.5}"
W_BITS="${W_BITS:-5}"
ENOB="${ENOB:-${W_BITS}}"
R_VAL="${R_VAL:-1e5}"
R_MAX="${R_MAX:-none}"
C_VAL="${C_VAL:-49e-15}"
K_VAL="${K_VAL:-1e3}"
V_DD="${V_DD:-1.0}"
TIE_CAP="${TIE_CAP:-false}"
ONE_OVER_Q="${ONE_OVER_Q:-10}"
TEST_BS="${TEST_BS:-128}"
LOGDIR="${LOGDIR:-logs/toggle_ptq_check/${MODEL_NAME}}"
mkdir -p "${LOGDIR}" logs/ode_noisy_acc

summary="${LOGDIR}/summary.tsv"
printf "class\tlevel\tode_block\taccuracy_percent\tlog\n" > "${summary}"

run_eval() {
  local class_name="$1"
  local level="$2"
  local block="$3"
  local wrapper="$4"
  local log_path="${LOGDIR}/${class_name}_level${level}_${block}.log"

  local cmd=(python -u ode_inference.py
    --model_name "${MODEL_NAME}"
    --ckpt "${CKPT}"
    --model_dir "${MODEL_DIR}"
    --task "${TASK}"
    --img_type "${IMG_TYPE}"
    --test_bs "${TEST_BS}"
    --pc_conv "${PC_CONV}"
    --ode_block "${block}"
    --method "${METHOD}"
    --tol "${TOL}"
    --n_steps "${N_STEPS}"
    --ts_scale "1"
    --d_start "0"
    --d_end "1"
    --n_sweep_left "0"
    --n_sweep_right "1"
    --noise_level_list "0.0"
    --noisy_trials "1"
    --thermal_noise "false"
    --sweep_eps "false"
    --diff_mismatch "${DIFF_MISMATCH:-false}"
    --test_expanded "${TEST_EXPANDED:-false}"
    --nonlinear_R "false"
    --R "${R_VAL}"
    --R_max "${R_MAX}"
    --C "${C_VAL}"
    --k "${K_VAL}"
    --v_dd "${V_DD}"
    --w_bits "${W_BITS}"
    --enob "${ENOB}"
    --tie_cap "${TIE_CAP}"
    --one_over_q "${ONE_OVER_Q}"
    --toggle_n_cycles "${TOGGLE_N_CYCLES}"
    --toggle_time_split "${TOGGLE_TIME_SPLIT}"
    --toggle_fast_path "${TOGGLE_FAST_PATH:-true}"
    --conv_only "false"
    --return_init "0"
    --test_only "false")

  if [[ "${wrapper}" != "none" ]]; then
    cmd+=(--ode_wrapper "${wrapper}")
  else
    cmd+=(--ode_wrapper "none")
  fi

  printf 'Running: %q ' "${cmd[@]}" | tee "${log_path}"
  printf '\n' | tee -a "${log_path}"
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf "%s\t%s\t%s\t%s\t%s\n" "${class_name}" "${level}" "${block}" "DRY_RUN" "${log_path}" >> "${summary}"
    return 0
  fi

  "${cmd[@]}" 2>&1 | tee -a "${log_path}"
  local acc
  acc="$(sed -n 's/.*Acc:\([0-9.][0-9.]*\).*/\1/p' "${log_path}" | tail -1)"
  if [[ -z "${acc}" ]]; then
    acc="NA"
  fi
  printf "%s\t%s\t%s\t%s\t%s\n" "${class_name}" "${level}" "${block}" "${acc}" "${log_path}" >> "${summary}"
}

run_eval "${RESET_MODE}" "1" "${LEVEL1_BLOCK}" "none"
run_eval "${RESET_MODE}" "2" "${LEVEL2_BLOCK}" "ToggleWrapper1State"
run_eval "${RESET_MODE}" "3" "${LEVEL3_BLOCK}" "ToggleWrapper1State"

cat "${summary}"

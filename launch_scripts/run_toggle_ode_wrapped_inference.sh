#!/usr/bin/env bash
set -eu
trap '' HUP   # ignore hangup so the children survive

# ─────────────── fixed params ───────────────
export MODEL_DIR="${MODEL_DIR:-./saved_ckpt}"
export tol="1e-6"
export R_VAL="${R_VAL:-10e3}"
export R_MAX="${R_MAX:-150e3}"
#export CKPT="full_param_best"
export CKPT="best"

# ─────────────── noise toggles ───────────────
#N_BITS_VALS=(4 5 6 7 8)
N_BITS_VALS=(${N_BITS_OVERRIDE:-5})
#for ((i=0; i<20; i++)); do N_BITS_VALS+=(5); done
#N_BITS_VALS=(15 15 15 15 15 15 15 15 15 15)
METHOD_VALS=("${METHOD:-dopri5}")
CAP_VALS=("49e-15")
#METHOD_VALS=("euler")
#METHOD_VALS=("rk4")

export BASE_LOGDIR="${BASE_LOGDIR:-./logs/test_toggle_ode_noisy}"
export K_VAL="${K_VAL:-1e3}"
export RESET_MODE="${RESET_MODE:-odexinit}" # reset, persistent, odexinit, or odexinit_keep
export TOGGLE_LEVEL="${TOGGLE_LEVEL:-3}" # 1, 2, or 3
export TOGGLE_N_CYCLES="${TOGGLE_N_CYCLES:-${N_STEPS:-5}}"
export TOGGLE_TIME_SPLIT="${TOGGLE_TIME_SPLIT:-0.5}"
export TOGGLE_FAST_PATH="${TOGGLE_FAST_PATH:-true}"
export ODEXINIT_SCALING_MODE="${ODEXINIT_SCALING_MODE:-approx}"
case "${RESET_MODE}_${TOGGLE_LEVEL}" in
  reset_1) export TOGGLE_ODE_BLOCK="ToggleResetZ" ;;
  reset_2) export TOGGLE_ODE_BLOCK="ToggleResetZ" ;;
  reset_3) export TOGGLE_ODE_BLOCK="TogglePulseResetZ" ;;
  persistent_1) export TOGGLE_ODE_BLOCK="ToggleKeepZ" ;;
  persistent_2) export TOGGLE_ODE_BLOCK="ToggleKeepZ" ;;
  persistent_3) export TOGGLE_ODE_BLOCK="TogglePulseKeepZ" ;;
  odexinit_1) export TOGGLE_ODE_BLOCK="ODEXInitFFFB" ;;
  odexinit_2) export TOGGLE_ODE_BLOCK="ToggleODEXInitFFFB" ;;
  odexinit_3) export TOGGLE_ODE_BLOCK="TogglePulseODEXInitFFFB" ;;
  odexinit_keep_1) export TOGGLE_ODE_BLOCK="ToggleODEXInitKeep" ;;
  odexinit_keep_2) export TOGGLE_ODE_BLOCK="ToggleODEXInitKeep" ;;
  odexinit_keep_3) export TOGGLE_ODE_BLOCK="TogglePulseODEXInitKeep" ;;
  *) echo "Invalid RESET_MODE/TOGGLE_LEVEL combination" >&2; exit 2 ;;
esac
# MASTER_LOG and JOB_LOG will be set per noise combination

# ─────────────── model list ───────────────
MODEL_NAMES=(
  # The same 16L baseline; trained with srrl + timm augs; ft with srrl + QAT + MT.
#  "TIMMQAT5bNT0p25mulTIMMPCNetNoBatchNorm_PCConvReLU6_0.0eps_ODEXInitFFFB_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_C100_3K1S96C_0.25Dropout_16Layers4l5l4_2Pool_srrlDistill_a0p3_t2p0_scanGFI_11REP" # srrl + timm trained model, then QAT + MT
#  "TIMMQAT5b2aNT0p25mulTIMMPCNetNoBatchNorm_PCConvReLU6_0.0eps_ODEXInitFFFB_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_C100_3K1S96C_0.25Dropout_16Layers4l5l4_2Pool_srrlDistill_a0p3_t2p0_scanGFI_1REP" # same, but 10e3 - 150e3 Ohm
#  "TIMMQAT5b3aNT0p25mulTIMMPCNetNoBatchNorm_PCConvReLU6_0.0eps_ODEXInitFFFB_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_C100_3K1S96C_0.25Dropout_16Layers4l5l4_2Pool_srrlDistill_a0p3_t2p0_scanGFI_1REP"
#  "TIMMQAT5b4aNT0p25mulTIMMPCNetNoBatchNorm_PCConvReLU6_0.0eps_ODEXInitFFFB_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_C100_3K1S96C_0.25Dropout_16Layers4l5l4_2Pool_srrlDistill_a0p3_t2p0_scanGFI_1REP"
#  "TIMMQAT5b5aNT0p25mulTIMMPCNetNoBatchNorm_PCConvReLU6_0.0eps_ODEXInitFFFB_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_C100_3K1S96C_0.25Dropout_16Layers4l5l4_2Pool_srrlDistill_a0p3_t2p0_scanGFI_1REP"
#  "TIMMQAT5b6aNT0p25mulTIMMPCNetNoBatchNorm_PCConvReLU6_0.0eps_ODEXInitFFFB_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_C100_3K1S96C_0.25Dropout_16Layers4l5l4_2Pool_srrlDistill_a0p3_t2p0_scanGFI_1REP"
#  "TIMMQAT5b7aNT0p25mulTIMMPCNetNoBatchNorm_PCConvReLU6_0.0eps_ODEXInitFFFB_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_C100_3K1S96C_0.25Dropout_16Layers4l5l4_2Pool_srrlDistill_a0p3_t2p0_scanGFI_1REP"
#  "TIMMQAT5b8aNT0p25mulTIMMPCNetNoBatchNorm_PCConvReLU6_0.0eps_ODEXInitFFFB_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_C100_3K1S96C_0.25Dropout_16Layers4l5l4_2Pool_srrlDistill_a0p3_t2p0_scanGFI_1REP"

  # "TIMMPCNetNoBatchNorm_PCConvReLU6_0.0eps_ODEXInitFFFB_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_C100_3K1S96C_0.25Dropout_16Layers4l5l4_2Pool_srrlDistill_a0p3_t2p0_scanGFI_6REP" # The model used as ft base for the model below.
  "TIMMQAT5b8aNT0p25mulTIMMPCNetNoBatchNorm_PCConvReLU6_0.0eps_ToggleODEXInitFFFB_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_C100_3K1S96C_0.25Dropout_16Layers4l5l4_2Pool_srrlDistill_a0p3_t2p0_scanGFI_1REP" # ft with ToggleODEXInitFFFB 5-steps based on a pretrained model on dopri45
#  "TIMMQAT5b8aNT0p25mulTIMMPCNetNoBatchNorm_PCConvReLU6_0.0eps_ToggleODEXInitFFFB_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_C100_3K1S96C_0.25Dropout_16Layers4l5l4_2Pool_srrlDistill_a0p3_t2p0_scanGFI_3REP" # ft with ToggleODEXInitFFFB and measured act fn, based on a pretrained model on dopri45
#  "PCNetNoBatchNorm_PCConvReLU6_0.0eps_ODEBlockXInit_dopri5Solver_1.5TEnd_0.0001Tol_0.001WD_128BS_0.01LR_C100_3K1S96C_0.25Dropout_16Layers4l5l4_2Pool_kd_crdDistill_a0p3_t2p0_scanGFI_3REP"
#  "PCNetNoBatchNorm_PCConvReLU6_0.0eps_ODEBlockPC_dopri5Solver_1.5TEnd_0.0001Tol_0.001WD_128BS_0.01LR_C100_3K1S96C_0.25Dropout_16Layers4l5l4_2Pool_kd_crdDistill_a0p3_t2p0_scanGFI_3REP"
#  "QAT5bNT0p25mulPCNetNoBatchNorm_PCConvReLU6_0.0eps_ODEBlockXInit_dopri5Solver_1.5TEnd_0.0001Tol_0.001WD_128BS_0.01LR_C100_3K1S96C_0.25Dropout_16Layers4l5l4_2Pool_kd_crdDistill_a0p3_t2p0_scanGFI_2REP"

#  "QAT5bNT0p25mulPCNetNoBatchNorm_PCConvReLU6_0.0eps_ODEBlockXInit_dopri5Solver_1.5TEnd_0.0001Tol_0.001WD_128BS_0.01LR_C100_3K1S96C_0.25Dropout_16Layers4l5l4_2Pool_kd_crdDistill_a0p3_t2p0_scanGFI_3REP" # With X QAT ft
)

if [[ -n "${MODEL_NAME_OVERRIDE:-}" ]]; then
  MODEL_NAMES=("${MODEL_NAME_OVERRIDE}")
fi

# ─────────────── prepare logs ───────────────
mkdir -p "$BASE_LOGDIR"
for n_bits in "${N_BITS_VALS[@]}"; do
  for cap_val in "${CAP_VALS[@]}"; do
    for method in "${METHOD_VALS[@]}"; do
      for name in "${MODEL_NAMES[@]}"; do
        job_dir_name="${JOB_DIR_NAME_OVERRIDE:-${name}_method_${method}_tol_${tol}_nbits_${n_bits}_cap_${cap_val}}"
        mkdir -p "$BASE_LOGDIR/${job_dir_name}"
        > "$BASE_LOGDIR/${job_dir_name}/job.log"
      done
    done
  done
done

# ─────────────── helper function ───────────────
run_model(){
  local name="$1"
  local method="$2"
  local n_bits="$3"
  local cap_val="$4"
  local job_dir_name="${JOB_DIR_NAME_OVERRIDE:-${name}_method_${method}_tol_${tol}_nbits_${n_bits}_cap_${cap_val}}"

  if [[ "$name" == *C100* ]]; then
    local _task="cifar100"
  else
    local _task="cifar10"
  fi

  local __rest="${name#*_}"
  local _pc_conv="${__rest%%_*}"

  IFS='_' read -r -a parts <<< "$name"
  local _ode_block="${TOGGLE_ODE_BLOCK}"
  local pulse_trained="false"
  if [[ "$name" == *TogglePulseBlkXInitFFFB* ]]; then
    _ode_block="TogglePulseBlkXInitFFFB"
    pulse_trained="true"
  elif [[ "$name" == *TogglePulseBlk* ]]; then
    _ode_block="TogglePulseBlk"
    pulse_trained="true"
  fi
  prev="${parts[${#parts[@]}-2]}"
  if [[ "$prev" == *Layers || "$prev" == *Pool ]]; then
    local _img_type="rgb"
  else
    local _img_type="$prev"
  fi

  if [[ "${pulse_trained}" == "true" ]]; then
    _ode_wrapper="TogglePulseQATTester1State"
  elif [[ "${TOGGLE_LEVEL}" == "1" ]]; then
    if [[ "${RESET_MODE}" == odexinit* ]]; then
      _ode_wrapper="QATTester1State"
    else
      _ode_wrapper="none"
    fi
  elif [[ "$name" == *QAT* && "$CKPT" != *full_param* ]]; then
    _ode_wrapper="ToggleQATTester1State"
  else
    _ode_wrapper="ToggleWrapper1State"
  fi

  local enob=8
  if [[ "$name" =~ TIMMQAT[0-9]+b([0-9]+)a ]]; then
    enob="${BASH_REMATCH[1]}"
  fi
  echo "extracted ENOB: ${enob}"

  local test_bs=128
  if [[ "${_ode_block}" == *Circ* ]]; then test_bs=128; fi
  local test_expanded="${TEST_EXPANDED:-true}"
  local nonlinear_r="${NONLINEAR_R:-true}"
  ########################################
  # only fuse_bn when noise is added to bn
  ########################################
  set -o pipefail
  python -u ode_inference.py \
    --model_name      "$name" \
    --ckpt            "${CKPT}" \
    --task            "${_task}" \
    --model_dir       "$MODEL_DIR" \
    --test_bs         "${test_bs}" \
    --method          "$method" \
    --tol             "$tol" \
    --n_steps         "${N_STEPS:-100}" \
    --ts_scale        1 \
    --d_start         0 \
    --d_end           1 \
    --n_sweep_left    0 \
    --n_sweep_right   1 \
    --thermal_noise   "true" \
    --sde_noise_type  "mul" \
    --mismatch_type   "mul" \
    --sweep_eps       "false" \
    --R               "$R_VAL" \
    --R_max           "$R_MAX" \
    --C               "$cap_val" \
    --k               "${K_VAL}" \
    --v_dd            "0.1" \
    --enob            "${enob}" \
    --w_bits          "$n_bits" \
    --patch_node      "8" \
    --patch_stride    "8" \
    --patch_cycle     "1" \
    --patch_pad       "0" \
    --fold_scalar     "1" \
    --tie_cap         "false" \
    --one_over_q      "${ONE_OVER_Q:-1}" \
    --toggle_n_cycles "${TOGGLE_N_CYCLES}" \
    --toggle_time_split "${TOGGLE_TIME_SPLIT}" \
    --toggle_fast_path "${TOGGLE_FAST_PATH}" \
    --odexinit_scaling_mode "${ODEXINIT_SCALING_MODE}" \
    --enable_spin_variation "${ENABLE_SPIN_VARIATION:-true}" \
    --sigma_spin "${SIGMA_SPIN:-0.10}" \
    --spin_variation_seed "${SPIN_VARIATION_SEED:-none}" \
    --enable_measured_activation "${ENABLE_MEASURED_ACTIVATION:-true}" \
    --activation_curve_path "${ACTIVATION_CURVE_PATH:-${PWD}/hardware_data/relu_current_0p2uA_finer.csv}" \
    --activation_corner "${ACTIVATION_CORNER:-TT}" \
    --activation_interpolation "${ACTIVATION_INTERPOLATION:-piecewise_linear}" \
    --activation_spline_parameters "${ACTIVATION_SPLINE_PARAMETERS:-10}" \
    --activation_fit_constraint "${ACTIVATION_FIT_CONSTRAINT:-auto}" \
    --activation_normalize_positive_endpoint "${SCALE_MEASURED_ACTIVATION:-false}" \
    --enable_summing_current_noise "${ENABLE_SUMMING_CURRENT_NOISE:-false}" \
    --summing_current_p "${SUMMING_CURRENT_P:-18.5e-12}" \
    --summing_noise_seed "${SUMMING_NOISE_SEED:-none}" \
    --enable_coupler_noise "${ENABLE_COUPLER_NOISE:-true}" \
    --coupler_noise_p "${COUPLER_NOISE_P:-0.6e-12}" \
    --coupler_noise_seed "${COUPLER_NOISE_SEED:-none}" \
    --enable_dtc_nonideality "${ENABLE_DTC_NONIDEALITY:-false}" \
    --dtc_leading_edge_variation_std "${DTC_LEADING_EDGE_VARIATION_STD:-0.0}" \
    --dtc_width_variation_std "${DTC_WIDTH_VARIATION_STD:-0.018}" \
    --dtc_leading_edge_jitter_std "${DTC_LEADING_EDGE_JITTER_STD:-0.005}" \
    --dtc_falling_edge_jitter_std "${DTC_FALLING_EDGE_JITTER_STD:-0.005}" \
    --dtc_timing_seed "${DTC_TIMING_SEED:-none}" \
    --pc_conv         "${_pc_conv}Noisy" \
    --ode_block       "${_ode_block}" \
    --ode_wrapper     "${_ode_wrapper}" \
    --img_type        "${_img_type}" \
    --noisy_trials    "${NOISY_TRIALS:-2}" \
    --conv_only       "true" \
    --test_expanded   "${test_expanded}" \
    --diff_mismatch   "${DIFF_MISMATCH:-true}" \
    --nonlinear_R     "${nonlinear_r}" \
    --nonlinear_R_table "${NONLINEAR_R_TABLE:-none}" \
    --mul_mismatch_mode "${MUL_MISMATCH_MODE:-scale_mismatch}" \
    --test_only_nl    "0.0" \
    --return_init     "${RETURN_INIT:-0}" \
    --test_only       "${TEST_ONLY:-false}" \
    2>&1 | tee -a "$BASE_LOGDIR/${job_dir_name}/job.log"
}
export -f run_model

# ─────────────── loop over methods ───────────────
#for n_bits in "${N_BITS_VALS[@]}"; do
for method in "${METHOD_VALS[@]}"; do
  ##########################################################################################
  # Modify log name here before each run
  ##########################################################################################
#  EXP_NAME="0818_3pooling_wrapped_${n_bits}bits_ODESumAsBInitY_${method}Method_${tol}Tol.log"
  EXP_NAME="0609_scanGFI_multiplicative_mismatch_${method}Method_${tol}Tol.log"
  MASTER_LOG="$BASE_LOGDIR/master_${EXP_NAME}"
  JOB_LOG="$BASE_LOGDIR/parallel_master_${EXP_NAME}"
  > "$MASTER_LOG"
  > "$JOB_LOG"
  echo "Tail master with: tail -f $MASTER_LOG"
  # ─────────────── run in parallel ───────────────
  parallel \
    --jobs 1 \
    --joblog "$JOB_LOG" \
    --keep-order \
    run_model {1} "${method}" {2} {3} \
    ::: "${MODEL_NAMES[@]}" \
    ::: "${N_BITS_VALS[@]}" \
    ::: "${CAP_VALS[@]}"
  echo "All jobs finished — merging logs into $MASTER_LOG"
  # ─────────────── merge logs sequentially ───────────────
  : >"$MASTER_LOG"
  for n_bits in "${N_BITS_VALS[@]}"; do
    for cap_val in "${CAP_VALS[@]}"; do
      for name in "${MODEL_NAMES[@]}"; do
        printf '========== %s | method=%s | n_bits=%s | cap=%s ==========\n' \
               "$name" "$method" "$n_bits" "$cap_val" >>"$MASTER_LOG"
        job_dir_name="${JOB_DIR_NAME_OVERRIDE:-${name}_method_${method}_tol_${tol}_nbits_${n_bits}_cap_${cap_val}}"
        logfile="$BASE_LOGDIR/${job_dir_name}/job.log"
        if ! grep -A 100 "Model name: ${name} " "$logfile" >>"$MASTER_LOG"; then
          echo "[Final Result not found] $logfile" >>"$MASTER_LOG"
        fi
        printf '\n\n' >>"$MASTER_LOG"
      done
    done
  done
  echo "All summaries written to $MASTER_LOG"
done
#done

#######################################################
# running
# nohup bash ./launch_scripts/run_ode_wrapped_inference.sh > logs/run_script_output/launcher.out 2>&1 &
# tail -f logs/run_script_output/launcher.out
# After the run is finished, the master_log file will be printed out
# then cat master_log
# then use shell_utils/parse_noise_logs.py to convert the log into csv
#######################################################

#!/usr/bin/env bash
set -eu
trap '' HUP   # ignore hangup so the children survive

# ─────────────── fixed params ───────────────
export MODEL_DIR="./saved_ckpt"
export tol="1e-6"
export R_VAL="20e3"
export R_MAX="300e3"
#export CKPT="full_param_best"
export CKPT="best"

# ─────────────── noise toggles ───────────────
#N_BITS_VALS=(4 5 6 7 8)
N_BITS_VALS=()
N_BITS_VALS=(5)
#for ((i=0; i<20; i++)); do N_BITS_VALS+=(5); done
#N_BITS_VALS=(15 15 15 15 15 15 15 15 15 15)
METHOD_VALS=("dopri5")
CAP_VALS=("49e-12")
I_LEAK_VALS=("0.5e-9" "1e-9" "2e-9" "3e-9" "4e-9" "5e-9" "6e-9" "8e-9" "10e-9")
#METHOD_VALS=("euler")
#METHOD_VALS=("rk4")

export BASE_LOGDIR="./logs/test_ode_noisy"
# MASTER_LOG and JOB_LOG will be set per noise combination

# ─────────────── model list ───────────────
MODEL_NAMES=(
#  "QAT5bNT0p25mulPCNetNoBatchNorm_PCConvReLU6_0.0eps_ODEXInitFFFB_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_C100_3K1S96C_0.25Dropout_16Layers4l5l4_2Pool_kd_crdDistill_a0p3_t2p0_scanGFI_1REP" # clamp with beta_c
#  "QAT5bNT0p25mulPCNetNoBatchNorm_PCConvReLU6_0.0eps_ODEXInitFFFB_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_C100_3K1S96C_0.25Dropout_16Layers4l5l4_2Pool_kd_crdDistill_a0p3_t2p0_scanGFI_2REP" # one_over_q = 1, clamp 0-0.2
#  "QAT5bNT0p25mulPCNetNoBatchNorm_PCConvReLU6_0.0eps_ODEXInitFFFB_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_C100_3K1S96C_0.25Dropout_16Layers4l5l4_2Pool_kd_crdDistill_a0p3_t2p0_scanGFI_6REP" # one_over_q = 1, qat with output quantization 8 enob, clamp 0-0.2
#  "QAT5bNT0p25mulPCNetNoBatchNorm_PCConvReLU6_0.0eps_ODEXInitFFFB_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_C100_3K1S96C_0.25Dropout_16Layers4l5l4_2Pool_kd_crdDistill_a0p3_t2p0_scanGFI_10REP" # one_over_q = 1, qat with output quantization 8 enob, clamp 0-0.2, 160 epochs (effectively 140 epochs)
#  "QAT5bNT0p25mulPCNetNoBatchNorm_PCConvReLU6_0.0eps_ODEXInitFFFB_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_C100_3K1S96C_0.25Dropout_16Layers4l5l4_2Pool_kd_crdDistill_a0p3_t2p0_scanGFI_14REP" # one_over_q = 1, qat + kd_crd with output quantization 8 enob, clamp 0-0.2, 140 epochs
#  "QAT5bNT0p25mulPCNetNoBatchNorm_PCConvReLU6_0.0eps_ODEXInitFFFB_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_3K1S96C_0.25Dropout_16Layers4l5l4_2Pool_kd_crdDistill_a0p3_t2p0_scanGFI_1REP"

#  "QAT5bNT0p0mulPCNetNoBatchNorm_PCConvReLU6_0.0eps_ODEXInitFFFB_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_C100_3K1S96C_0.25Dropout_16Layers4l5l4_2Pool_kd_crdDistill_a0p3_t2p0_scanGFI_1REP"  # ft with 5 step Euler
#  "QAT5bNT0p25mulPCNetNoBatchNorm_PCConvReLU6_0.0eps_ODEXInitFFFB_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_C100_3K1S96C_0.25Dropout_16Layers4l5l4_2Pool_kd_crdDistill_a0p3_t2p0_scanGFI_15REP" # ft with 5 step Euler, 0.25 MT
#  "QAT5bNT0p0mulPCNetNoBatchNorm_PCConvReLU6_0.0eps_PerturbODEXInitFFFB_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_C100_3K1S96C_0.25Dropout_16Layers4l5l4_2Pool_kd_crdDistill_a0p3_t2p0_scanGFI_1REP" # ft with 5 step Euler, perturbed
#  "QAT5bNT0p0mulPCNetNoBatchNorm_PCConvReLU6_0.0eps_ODEXInitFFFB_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_C100_3K1S96C_0.25Dropout_16Layers4l5l4_2Pool_kd_crdDistill_a0p3_t2p0_scanGFI_2REP" # ft with 5 step RK4

#  "TIMMQAT5bNT0p25mulTIMMPCNetNoBatchNorm_PCConvReLU6_0.0eps_ODEXInitFFFB_eulerSolver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_C100_3K1S72C_0.25Dropout_12Layers11l0l0_1Pool5_srrlDistill_a0p3_t2p0_scanGFI_3REP"
  "TIMMQAT5bNT0p25mulTIMMPCNetNoBatchNorm_PCConvReLU6_0.0eps_PerturbODEXInitFFFB_eulerSolver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_C100_3K1S72C_0.25Dropout_12Layers11l0l0_1Pool5_srrlDistill_a0p3_t2p0_scanGFI_1REP"
)

# ─────────────── prepare logs ───────────────
mkdir -p "$BASE_LOGDIR"
for n_bits in "${N_BITS_VALS[@]}"; do
  for cap_val in "${CAP_VALS[@]}"; do
    for method in "${METHOD_VALS[@]}"; do
      for name in "${MODEL_NAMES[@]}"; do
        for i_leak in "${I_LEAK_VALS[@]}"; do
          mkdir -p "$BASE_LOGDIR/${name}_method_${method}_tol_${tol}_nbits_${n_bits}_cap_${cap_val}_ileak_${i_leak}"
          > "$BASE_LOGDIR/${name}_method_${method}_tol_${tol}_nbits_${n_bits}_cap_${cap_val}_ileak_${i_leak}/job.log"
        done
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
  local i_leak="$5"

  if [[ "$name" == *C100* ]]; then
    local _task="cifar100"
  else
    local _task="cifar10"
  fi

  local __rest="${name#*_}"
  local _pc_conv="${__rest%%_*}"

  IFS='_' read -r -a parts <<< "$name"
  local _ode_block="${parts[3]}"
  prev="${parts[${#parts[@]}-2]}"
  if [[ "$prev" == *Layers || "$prev" == *Pool ]]; then
    local _img_type="rgb"
  else
    local _img_type="$prev"
  fi

#  local _ode_wrapper="ODEWrapperRC"
#  local _ode_wrapper="WrapQuantizeW"
  local _ode_wrapper="ODEWrapper1State"
  if [[ "$name" == *S2* || "$name" == *State2* ]]; then
    if [[ "$name" == *QAT* && "$CKPT" != *full_param* ]]; then
      _ode_wrapper="QATTester2State"
    else
      _ode_wrapper="ODEWrapper2State"
    fi
  else
    if [[ "$name" == *QAT* && "$CKPT" != *full_param* ]]; then
      _ode_wrapper="QATTester1State"
    fi
  fi

  local test_bs=128
  if [[ "${_ode_block}" == *Circ* ]]; then test_bs=128; fi
#  _ode_block="ODEXInitFFFBPixelSwitchEfficient"
#  _ode_block="ODEXInitFFFBPixelSwitchStretchT"
  _ode_block="ODEXInitFFFBPixelSwitchStretchTDecay"
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
    --n_steps         2 \
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
    --v_dd            "0.1" \
    --enob            "8" \
    --w_bits          "$n_bits" \
    --patch_node      "8" \
    --patch_stride    "8" \
    --patch_cycle     "1" \
    --patch_pad       "0" \
    --fold_scalar     "1" \
    --tie_cap         "false" \
    --one_over_q      "1" \
    --pc_conv         "${_pc_conv}Noisy" \
    --ode_block       "${_ode_block}" \
    --ode_wrapper     "${_ode_wrapper}" \
    --img_type        "${_img_type}" \
    --noisy_trials    "5" \
    --switch_period   "" \
    --switch_iter     "5" \
    --i_leak          "${i_leak}" \
    --conv_only       "true" \
    --test_expanded   "false" \
    --diff_mismatch   "true" \
    --nonlinear_R     "false" \
    --test_only       "false" \
    2>&1 | tee -a "$BASE_LOGDIR/${name}_method_${method}_tol_${tol}_nbits_${n_bits}_cap_${cap_val}_ileak_${i_leak}/job.log"
}
export -f run_model

# ─────────────── loop over methods ───────────────
for method in "${METHOD_VALS[@]}"; do
  ##########################################################################################
  # Modify log name here before each run
  ##########################################################################################
  EXP_NAME="0317_scanGFI_multiplicative_mismatch_${method}Method_${tol}Tol.log"
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
    run_model {1} "${method}" {2} {3} {4} \
    ::: "${MODEL_NAMES[@]}" \
    ::: "${N_BITS_VALS[@]}" \
    ::: "${CAP_VALS[@]}" \
    ::: "${I_LEAK_VALS[@]}"
  echo "All jobs finished — merging logs into $MASTER_LOG"
  # ─────────────── merge logs sequentially ───────────────
  : >"$MASTER_LOG"
  for n_bits in "${N_BITS_VALS[@]}"; do
    for cap_val in "${CAP_VALS[@]}"; do
      for name in "${MODEL_NAMES[@]}"; do
        for i_leak in "${I_LEAK_VALS[@]}"; do
          printf '========== %s | method=%s | n_bits=%s | cap=%s | i_leak=%s ==========\n' \
                 "$name" "$method" "$n_bits" "$cap_val" "$i_leak" >>"$MASTER_LOG"
          logfile="$BASE_LOGDIR/${name}_method_${method}_tol_${tol}_nbits_${n_bits}_cap_${cap_val}_ileak_${i_leak}/job.log"
          if ! grep -A 100 "Model name: ${name} " "$logfile" >>"$MASTER_LOG"; then
            echo "[Final Result not found] $logfile" >>"$MASTER_LOG"
          fi
          printf '\n\n' >>"$MASTER_LOG"
        done
      done
    done
  done
  echo "All summaries written to $MASTER_LOG"
done

#######################################################
# running
# nohup bash ./launch_scripts/run_ode_wrapped_inference.sh > logs/run_script_output/launcher.out 2>&1 &
# tail -f logs/run_script_output/launcher.out
# After the run is finished, the master_log file will be printed out
# then cat master_log
# then use shell_utils/parse_noise_logs.py to convert the log into csv
#######################################################

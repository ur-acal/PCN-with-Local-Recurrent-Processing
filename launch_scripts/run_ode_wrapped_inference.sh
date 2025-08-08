#!/usr/bin/env bash
set -eu
trap '' HUP   # ignore hangup so the children survive

# ─────────────── fixed params ───────────────
export MODEL_DIR="./saved_ckpt"
export tol="1e-6"

# ─────────────── noise toggles ───────────────
METHOD_VALS=("dopri5")
#METHOD_VALS=("euler")

export BASE_LOGDIR="./logs/test_ode_noisy"
# MASTER_LOG and JOB_LOG will be set per noise combination

# ─────────────── model list ───────────────
MODEL_NAMES=(
#  "PCNetNoBatchNorm_PCConvHardTanh_dopri5Solver_0.75TEnd_0.0001Tol_0.001WD_noBPtied_noBP_withPC_128BS_0.01LR_0.25Dropout_7Layers_1REP" # NODE baseline
#  "PCNetNoBatchNorm_PCConvHardTanh_dopri5Solver_1.0TEnd_0.0001Tol_0.001WD_noBPtied_noBP_withPC_128BS_0.01LR_0.25Dropout_7Layers_1REP" # NODE baseline

#  "PCNetNoBatchNorm_PCConvReLU6_ODEActInpNoMinus_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_noBPtied_noBP_128BS_0.01LR_0.25Dropout_7Layers_1REP" # zero init
#  "PCNetNoBatchNorm_PCConvHardTanh2_ODEActInpNoMinus_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_noBPtied_noBP_128BS_0.01LR_0.25Dropout_7Layers_1REP" # zero init

#  "PCNetNoBatchNorm_PCConvHardTanh_ODEBlockPCMinusY_dopri5Solver_0.75TEnd_0.001Tol_0.001WD_noBPtied_noBP_128BS_0.01LR_0.25Dropout_7Layers_2REP" # no max_g_norm
#  "PCNetNoBatchNorm_PCConvReLU6_ODEBlockPCMinusY_dopri5Solver_0.75TEnd_0.001Tol_0.001WD_noBPtied_noBP_128BS_0.01LR_0.25Dropout_7Layers_2REP" # no max_g_norm

#  "PCNetNoBatchNorm_PCConvHardTanh_0.2eps_ODEFixNoiseOffset_dopri5Solver_0.75TEnd_0.0001Tol_0.001WD_noBPtied_noBP_128BS_0.01LR_0.25Dropout_7Layers_1REP"
#  "ftPCNetNoBatchNorm_PCConvHardTanh_0.4eps_ODEFixNoiseOffset_dopri5Solver_0.75TEnd_0.0001Tol_0.001WD_noBPtied_noBP_128BS_0.0001LR_0.25Dropout_7Layers_1REP"
#  "PCNetNoBatchNorm_PCConvReLU6_0.2eps_ODEFixNoiseOffset_dopri5Solver_0.75TEnd_0.0001Tol_0.001WD_noBPtied_noBP_128BS_0.01LR_0.25Dropout_7Layers_1REP"
#  "ftPCNetNoBatchNorm_PCConvReLU6_0.4eps_ODEFixNoiseOffset_dopri5Solver_0.75TEnd_0.0001Tol_0.001WD_noBPtied_noBP_128BS_0.0001LR_0.25Dropout_7Layers_1REP" # passive baseline

  "PCNetNoBatchNorm_PCConvReLU6_0.2eps_ODEFixNoiseOffset_dopri5Solver_0.75TEnd_0.0001Tol_0.001WD_noBPtied_noBP_128BS_0.01LR_0.25Dropout_5Layers_2Pool_1REP" # 2 max pooling
)

# ─────────────── prepare logs ───────────────
mkdir -p "$BASE_LOGDIR"
for method in "${METHOD_VALS[@]}"; do
  for name in "${MODEL_NAMES[@]}"; do
    mkdir -p "$BASE_LOGDIR/${name}_method_${method}_tol_${tol}"
    > "$BASE_LOGDIR/${name}_method_${method}_tol_${tol}/job.log"
  done
done

# ─────────────── helper function ───────────────
run_model(){
  local name="$1"
  local method="$2"

  local __rest="${name#*_}"
  local _pc_conv="${__rest%%_*}"
  ########################################
  # only fuse_bn when noise is added to bn
  ########################################
  set -o pipefail
  python -u ode_inference.py \
    --model_name      "$name" \
    --model_dir       "$MODEL_DIR" \
    --method          "$method" \
    --tol             "$tol" \
    --n_steps         15 \
    --ts_scale        1 \
    --d_start         0 \
    --d_end           1 \
    --n_sweep_left    0 \
    --n_sweep_right   1 \
    --pc_conv         "${_pc_conv}Noisy" \
    --ode_block       "ODESumAsBInitY" \
    --ode_wrapper     "ODEWrapperRC" \
    --test_only       "true" \
    2>&1 | tee -a "$BASE_LOGDIR/${name}_method_${method}_tol_${tol}/job.log"
}
export -f run_model

# ─────────────── loop over methods ───────────────
for method in "${METHOD_VALS[@]}"; do
  ##########################################################################################
  # Modify log name here before each run
  ##########################################################################################
  EXP_NAME="0808_2pooling_ODESumAsBInitY_${method}Method_${tol}Tol.log"
  MASTER_LOG="$BASE_LOGDIR/master_${EXP_NAME}"
  JOB_LOG="$BASE_LOGDIR/parallel_master_${EXP_NAME}"
  > "$MASTER_LOG"
  > "$JOB_LOG"
  echo "Tail master with: tail -f $MASTER_LOG"
  # ─────────────── run in parallel ───────────────
  parallel \
    --jobs 4 \
    --joblog "$JOB_LOG" \
    --keep-order \
    run_model {1} "${method}" \
    ::: "${MODEL_NAMES[@]}"
  echo "All jobs finished — merging logs into $MASTER_LOG"
  # ─────────────── merge logs sequentially ───────────────
  : >"$MASTER_LOG"
  for name in "${MODEL_NAMES[@]}"; do
    printf '========== %s ==========\n' "$name" >>"$MASTER_LOG"
    if ! grep -A 11 "Model name: ${name} " \
              "$BASE_LOGDIR/${name}_method_${method}_tol_${tol}/job.log" >>"$MASTER_LOG"; then
      echo "[Final Result not found]" >>"$MASTER_LOG"
    fi
    printf '\n\n' >>"$MASTER_LOG"
  done
  echo "All summaries written to $MASTER_LOG"
done

#######################################################
# running
# nohup bash run_ode_inference.sh > logs/run_script_output/launcher.out 2>&1 &
# tail -f logs/run_script_output/launcher.out
# After the run is finished, the master_log file will be printed out
# then cat master_log
# then use shell_utils/parse_noise_logs.py to convert the log into csv
#######################################################

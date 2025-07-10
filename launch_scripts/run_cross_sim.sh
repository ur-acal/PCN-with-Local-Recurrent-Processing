#!/usr/bin/env bash
set -eu
trap '' HUP   # ignore hangup so the children survive

# ─────────────── fixed params ───────────────
export MODEL_DIR="./saved_ckpt"
export CALIB_DIR="./cross_sim/calibrated_config/pcn_input_calib"

#######################################################
# change the log names here to identify each run
#######################################################
export BASE_LOGDIR="./logs/test_cross_sim"
EXP_NAME="0709_ppcn_hardtanh_calib"
MASTER_LOG="$BASE_LOGDIR/master_${EXP_NAME}.log"
JOB_LOG="$BASE_LOGDIR/parallel_job_master_${EXP_NAME}.log"

# ─────────────── model list ───────────────
MODEL_NAMES=(
#  "PCNetNoBatchNorm_PCConvHardTanhLimit_5CLS_0.15LRPC_0.001WD_noTied_noBPtied_withRelu_noBP_noReluBP_withPC_128BS_0.01LR_0.25Dropout_7Layers_1REP"
#  "PCNetNoBatchNorm_PCConvReLU6Limit_5CLS_0.15LRPC_0.001WD_noTied_noBPtied_withRelu_noBP_noReluBP_withPC_128BS_0.01LR_0.25Dropout_7Layers_1REP"
#  "PCNetNoBatchNorm_PCConvHardTanh_5CLS_0.15LRPC_0.001WD_noTied_noBPtied_withRelu_noBP_noReluBP_withPC_128BS_0.01LR_0.25Dropout_7Layers_1REP"
  "PCNetNoBatchNorm_PCConvHardTanh_dopri5Solver_0.75TEnd_0.001Tol_0.001WD_noBPtied_noBP_withPC_128BS_0.01LR_0.25Dropout_7Layers_2REP"
#  "PCNetNoBatchNorm_PCConvHardTanh_dopri5Solver_1.0TEnd_0.0001Tol_0.001WD_noBPtied_noBP_withPC_128BS_0.01LR_0.25Dropout_7Layers_1REP"
#  "PCNetNoBatchNorm_PCConvHardTanhDyn_30CLS_0.06LRPC_0.001WD_noTied_noBPtied_withRelu_noBP_noReluBP_withPC_128BS_0.01LR_0.25Dropout_7Layers_1REP"
)

# ─────────────── prepare logs ───────────────
mkdir -p "$BASE_LOGDIR"
> "$MASTER_LOG"
> "$JOB_LOG"
for name in "${MODEL_NAMES[@]}"; do
  mkdir -p "$BASE_LOGDIR/$name"
  > "$BASE_LOGDIR/$name/job.log"
done

# ─────────────── helper function ───────────────
run_model(){
  local name="$1"
  set -o pipefail
  python -u cross_sim_inference.py \
    --model_name      "$name" \
    --model_dir       "$MODEL_DIR" \
    --calib_type      "perc_hi_lo" \
    --calib_samples   256 \
    --calib_perc      "0.995" \
    --calib_path      "$CALIB_DIR" \
    --calib_only      "false" \
    --prop_error      "true" \
    --weight_bits     4 \
    --input_bits      4 \
    --adc_bits        0 \
    --bias_rows       0 \
    --pc_conv         "PCConvHardTanh" \
    --ode_block       "ODEBlockPCMinusY" \
    --test_only       "true" \
    2>&1 | tee -a "$BASE_LOGDIR/$name/job.log"
}
export -f run_model

echo "Tail master with: tail -f $MASTER_LOG"

# ─────────────── run in parallel ───────────────
parallel \
  --jobs 1 \
  --joblog "$JOB_LOG" \
  --keep-order \
  run_model {} \
  ::: "${MODEL_NAMES[@]}"

echo "All jobs finished — merging logs into $MASTER_LOG"

# ─────────────── merge logs sequentially ───────────────
: >"$MASTER_LOG"
for name in "${MODEL_NAMES[@]}"; do
  printf '========== %s ==========\n' "$name" >>"$MASTER_LOG"
  if ! grep -A 21 "Final Result " \
             "$BASE_LOGDIR/$name/job.log" >>"$MASTER_LOG"; then
     echo "[Final Result not found]" >>"$MASTER_LOG"
  fi
  printf '\n\n' >>"$MASTER_LOG"
done

echo "All summaries written to $MASTER_LOG"
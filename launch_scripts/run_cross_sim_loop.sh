#!/usr/bin/env bash
set -eu
trap '' HUP   # ignore hangup so the children survive

# ─────────────── fixed params ───────────────
export MODEL_DIR="./saved_ckpt"
export CALIB_DIR="./cross_sim/calibrated_config/pcn_input_calib"

# ─────────────── noise toggles ───────────────
N_BITS_VALS=(4 8)
PROP_ERR_VALS=("true" "false")
declare -A PREC_MAP=( ["4"]="0.995" ["8"]="0.99999" )
export PREC_MAP_DEF="$(declare -p PREC_MAP)"

export BASE_LOGDIR="./logs/test_cross_sim"
# MASTER_LOG and JOB_LOG will be set per noise combination

# ─────────────── model list ───────────────
MODEL_NAMES=(
#  "PCNetNoBatchNorm_PCConvHardTanh_5CLS_0.15LRPC_0.001WD_noTied_noBPtied_withRelu_noBP_noReluBP_withPC_128BS_0.01LR_0.25Dropout_7Layers_1REP"
#  "PCNetNoBatchNorm_PCConvHardTanhLimit_5CLS_0.15LRPC_0.001WD_noTied_noBPtied_withRelu_noBP_noReluBP_withPC_128BS_0.01LR_0.25Dropout_7Layers_1REP"
#  "PCNetNoBatchNorm_PCConvReLU6Limit_5CLS_0.15LRPC_0.001WD_noTied_noBPtied_withRelu_noBP_noReluBP_withPC_128BS_0.01LR_0.25Dropout_7Layers_1REP"
#  "PCNetNoBatchNorm_PCConvHardTanhDyn_30CLS_0.06LRPC_0.001WD_noTied_noBPtied_withRelu_noBP_noReluBP_withPC_128BS_0.01LR_0.25Dropout_7Layers_1REP"
  "PCNetNoBatchNorm_PCConvHardTanh_5CLS_0.15LRPC_0.001WD_noTied_noBPtied_withRelu_noBP_noReluBP_withPC_128BS_0.01LR_0.25Dropout_7Layers_1REP"
  "PCNetNoBatchNorm_PCConvHardTanh2Dyn_5CLS_0.15LRPC_0.001WD_noTied_noBPtied_withRelu_noBP_noReluBP_withPC_128BS_0.01LR_0.25Dropout_7Layers_1REP"
  "PCNetNoBatchNorm_PCConvHardTanh2Dyn_30CLS_0.06LRPC_0.001WD_noTied_noBPtied_withRelu_noBP_noReluBP_withPC_128BS_0.01LR_0.25Dropout_7Layers_1REP"
)

# ─────────────── prepare logs ───────────────
mkdir -p "$BASE_LOGDIR"
for n_bits in "${N_BITS_VALS[@]}"; do
  for prop_err in "${PROP_ERR_VALS[@]}"; do
    for name in "${MODEL_NAMES[@]}"; do
      mkdir -p "$BASE_LOGDIR/${name}_n_bits_${n_bits}_prop_err_${prop_err}"
      > "$BASE_LOGDIR/${name}_n_bits_${n_bits}_prop_err_${prop_err}/job.log"
    done
  done
done

# ─────────────── helper function ───────────────
run_model(){
  local name="$1"
  local n_bits="$2"
  local prop_err="$3"

  # inferred params
  eval "$PREC_MAP_DEF"
  local prec="${PREC_MAP[$n_bits]}"
  local __rest="${name#*_}"
  local _pc_conv="${__rest%%_*}"
  ########################################
  # only fuse_bn when noise is added to bn
  ########################################
  set -o pipefail
  python -u cross_sim_inference.py \
    --model_name      "$name" \
    --model_dir       "$MODEL_DIR" \
    --calib_type      "perc_hi_lo" \
    --calib_perc      "$prec" \
    --calib_samples   256 \
    --calib_path      "$CALIB_DIR" \
    --calib_only      "false" \
    --prop_error      "$prop_err" \
    --weight_bits     "$n_bits" \
    --input_bits      "$n_bits" \
    --bias_rows       0 \
    --pc_conv         "$_pc_conv" \
    2>&1 | tee -a "$BASE_LOGDIR/${name}_n_bits_${n_bits}_prop_err_${prop_err}/job.log"
}
export -f run_model

# ─────────────── loop over n_bits_vals and prop_error_vals ───────────────
for n_bits in "${N_BITS_VALS[@]}"; do
  for prop_err in "${PROP_ERR_VALS[@]}"; do
    ##########################################################################################
    # Modify log name here before each run
    ##########################################################################################
    EXP_NAME="0706_ppcn_hardtanh_and_Dyn_${n_bits}bit_prop_err_${prop_err}"
    MASTER_LOG="$BASE_LOGDIR/master_${EXP_NAME}.log"
    JOB_LOG="$BASE_LOGDIR/parallel_master_${EXP_NAME}.log"

    > "$MASTER_LOG"
    > "$JOB_LOG"

    echo "Tail master with: tail -f $MASTER_LOG"

    # ─────────────── run in parallel ───────────────
    parallel \
      --jobs 1 \
      --joblog "$JOB_LOG" \
      --keep-order \
      run_model {1} "$n_bits" "$prop_err" \
      ::: "${MODEL_NAMES[@]}"

    echo "All jobs finished — merging logs into $MASTER_LOG"

    # ─────────────── merge logs sequentially ───────────────
    : >"$MASTER_LOG"
    for name in "${MODEL_NAMES[@]}"; do
      printf '========== %s ==========\n' "$name" >>"$MASTER_LOG"
      if ! grep -A 11 "Model name: ${name} " \
                "$BASE_LOGDIR/${name}_n_bits_${n_bits}_prop_err_${prop_err}/job.log" >>"$MASTER_LOG"; then
        echo "[Final Result not found]" >>"$MASTER_LOG"
      fi
      printf '\n\n' >>"$MASTER_LOG"
    done

    echo "All summaries written to $MASTER_LOG"
  done
done

#######################################################
# running
# nohup bash run_cross_sim_loop.sh > logs/run_script_output/launcher.out 2>&1 &
# tail -f logs/run_script_output/launcher.out
# After the run is finished, the master_log file will be printed out
# then cat master_log
# then use shell_utils/parse_noise_logs.py to convert the log into csv
#######################################################
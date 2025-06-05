#!/usr/bin/env bash
set -eu
trap '' HUP   # ignore hangup so the children survive

# ─────────────── fixed params ───────────────
export MODEL_DIR="./saved_ckpt"
export WEIGHT_PATH="./expanded_weights"
export PLOT_PATH="./loss_plot"
export W_TYPE="fb_flip"
export NOISE_TO_FF=true
export NOISE_TO_BP=true

# ─────────────── noise toggles ───────────────
NOISE_TO_BN_VALUES=("true" "false")
NOISE_TO_LINEAR_VALUES=("true" "false")

export BASE_LOGDIR="./logs/noisy_test"
# MASTER_LOG and JOB_LOG will be set per noise combination

# ─────────────── model list ───────────────
MODEL_NAMES=(
  "PPCN_5CLS_0.06LRPC_0.001WD_noTied_noBPtied_withRelu_withBP_noReluBP_withPC_5Layers_1REP"
  "PPCN_5CLS_0.06LRPC_0.001WD_noTied_noBPtied_withRelu_noBP_noReluBP_withPC_7Layers_1REP"
  "PPCN_5CLS_0.06LRPC_0.001WD_noTied_withBPtied_withRelu_withBP_noReluBP_withPC_7Layers_1REP"
  "PPCN_5CLS_0.06LRPC_0.001WD_noTied_noBPtied_withRelu_withBP_noReluBP_withPC_7Layers_1REP"
)

# ─────────────── prepare logs ───────────────
mkdir -p "$BASE_LOGDIR"
for noise_to_bn in "${NOISE_TO_BN_VALUES[@]}"; do
  for noise_to_linear in "${NOISE_TO_LINEAR_VALUES[@]}"; do
    for name in "${MODEL_NAMES[@]}"; do
      mkdir -p "$BASE_LOGDIR/${name}_bn_${noise_to_bn}_linear_${noise_to_linear}"
      > "$BASE_LOGDIR/${name}_bn_${noise_to_bn}_linear_${noise_to_linear}/job.log"
    done
  done
done

# ─────────────── helper function ───────────────
run_model(){
  local name="$1"
  local noise_to_bn="$2"
  local noise_to_linear="$3"
  ########################################
  # only fuse_bn when noise is added to bn
  ########################################
  set -o pipefail
  python -u run_test.py \
    --model_name      "$name" \
    --model_dir       "$MODEL_DIR" \
    --weight          "$WEIGHT_PATH" \
    --plot_path       "$PLOT_PATH" \
    --w_type          "$W_TYPE" \
    --noise_to_ff     "$NOISE_TO_FF" \
    --noise_to_bp     "$NOISE_TO_BP" \
    --tie_noise       "false" \
    --tie_noise_bp    "false" \
    --noise_to_bn     "$noise_to_bn" \
    --noise_to_linear "$noise_to_linear" \
    --fuse_bn         "$noise_to_bn" \
    --diff_noise      "true" \
    2>&1 | tee -a "$BASE_LOGDIR/${name}_bn_${noise_to_bn}_linear_${noise_to_linear}/job.log"
}
export -f run_model

# ─────────────── loop over BN / Linear noise toggles ───────────────
for noise_to_bn in "${NOISE_TO_BN_VALUES[@]}"; do
  for noise_to_linear in "${NOISE_TO_LINEAR_VALUES[@]}"; do
#    if [[ "$noise_to_bn" == "false" && "$noise_to_linear" == "false" ]]; then
#      continue
#    fi

    ##########################################################################################
    # Modify log name here before each run
    ##########################################################################################
    MASTER_LOG="$BASE_LOGDIR/master_no_bn_0531_bn_${noise_to_bn}_linear_${noise_to_linear}.log"
    JOB_LOG="$BASE_LOGDIR/parallel_no_bn_0531_job_bn_${noise_to_bn}_linear_${noise_to_linear}.log"

    > "$MASTER_LOG"
    > "$JOB_LOG"

    echo "Tail master with: tail -f $MASTER_LOG"

    # ─────────────── run in parallel ───────────────
    parallel \
      --jobs 2 \
      --joblog "$JOB_LOG" \
      --keep-order \
      run_model {1} "$noise_to_bn" "$noise_to_linear" \
      ::: "${MODEL_NAMES[@]}"

    echo "All jobs finished — merging logs into $MASTER_LOG"

    # ─────────────── merge logs sequentially ───────────────
    : >"$MASTER_LOG"
    for name in "${MODEL_NAMES[@]}"; do
      printf '========== %s ==========\n' "$name" >>"$MASTER_LOG"
      if ! grep -A 11 "Final Result " \
                "$BASE_LOGDIR/${name}_bn_${noise_to_bn}_linear_${noise_to_linear}/job.log" >>"$MASTER_LOG"; then
        echo "[Final Result not found]" >>"$MASTER_LOG"
      fi
      printf '\n\n' >>"$MASTER_LOG"
    done

    echo "All summaries written to $MASTER_LOG"
  done
done

#######################################################
# running
# nohup bash run_loop_noisy_test_parallel.sh > logs/run_script_output/launcher.out 2>&1 &
# tail -f logs/run_script_output/launcher.out
# After the run is finished, the master_log file will be printed out
# then cat master_log
# then use shell_utils/parse_noise_logs.py to convert the log into csv
#######################################################

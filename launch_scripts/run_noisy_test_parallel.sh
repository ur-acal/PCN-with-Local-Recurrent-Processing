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

#######################################################
# change the log names here to identify each run
#######################################################
export BASE_LOGDIR="./logs/noisy_test"
MASTER_LOG="$BASE_LOGDIR/master_0609_ppcn_no_bn_0.06LRPC_128BS_noise_to_all.log"
JOB_LOG="$BASE_LOGDIR/parallel_job_master_0609_ppcn_no_bn_0.06LRPC_128BS_noise_to_all.log"

# ─────────────── model list ───────────────
MODEL_NAMES=(
  "PCNetNoBatchNorm_5CLS_0.06LRPC_0.001WD_noTied_noBPtied_withRelu_withBP_noReluBP_withPC_128BS_0.01LR_5Layers_1REP" # retrained baseline
  "PCNetNoBatchNorm_5CLS_0.06LRPC_0.001WD_noTied_noBPtied_withRelu_noBP_noReluBP_withPC_128BS_0.01LR_7Layers_1REP" # No2
  "PCNetNoBatchNorm_5CLS_0.06LRPC_0.001WD_noTied_withBPtied_withRelu_withBP_noReluBP_withPC_128BS_0.01LR_7Layers_1REP" # No6
  "PCNetNoBatchNorm_5CLS_0.06LRPC_0.001WD_noTied_noBPtied_withRelu_withBP_noReluBP_withPC_128BS_0.01LR_7Layers_1REP" # 7 layer baseline
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
    --noise_to_bn     "true" \
    --noise_to_linear "true" \
    --fuse_bn         "false" \
    --noisy_test      "true" \
    --pc_conv         "PCConvNoisy" \
    2>&1 | tee -a "$BASE_LOGDIR/$name/job.log"
}
export -f run_model

echo "Tail master with: tail -f $MASTER_LOG"

# ─────────────── run in parallel ───────────────
parallel \
  --jobs 2 \
  --joblog "$JOB_LOG" \
  --keep-order \
  run_model {} \
  ::: "${MODEL_NAMES[@]}"

echo "All jobs finished — merging logs into $MASTER_LOG"

# ─────────────── merge logs sequentially ───────────────
: >"$MASTER_LOG"
for name in "${MODEL_NAMES[@]}"; do
  printf '========== %s ==========\n' "$name" >>"$MASTER_LOG"
  if ! grep -A 11 "Final Result " \
             "$BASE_LOGDIR/$name/job.log" >>"$MASTER_LOG"; then
     echo "[Final Result not found]" >>"$MASTER_LOG"
  fi
  printf '\n\n' >>"$MASTER_LOG"
done

echo "All summaries written to $MASTER_LOG"

#######################################################
# running
# nohup bash run_noisy_test_parallel.sh > logs/run_script_output/launcher.out 2>&1 &
# tail -f logs/run_script_output/launcher.out
# After the run is finished, the master_log file will be printed out
# then cat master_log
# then use shell_utils/parse_noise_logs.py to convert the log into csv
#######################################################
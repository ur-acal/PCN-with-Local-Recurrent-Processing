#!/usr/bin/env bash
# run_test_parallel_grouped.sh

trap '' HUP

# fixed parameters
MODEL_DIR="./saved_ckpt"
WEIGHT_PATH="./expanded_weights"
PLOT_PATH="./loss_plot"
W_TYPE="fb_flip"
NOISE_TO_FF=true
NOISE_TO_BP=true

# list of model names
MODEL_NAMES=(
  "PPCN_5CLS_1.0LRPC_0.001WD_noTied_withBPtied_withBP_withRelu_7Layers_2REP"
  "PPCN_5CLS_1.0LRPC_0.001WD_noTied_withBPtied_withBP_noRelu_7Layers_2REP"
  "PPCN_5CLS_1.0LRPC_0.001WD_noTied_noBPtied_noBP_withRelu_7Layers_2REP"
  "PPCN_5CLS_1.0LRPC_0.001WD_noTied_noBPtied_withBP_noRelu_7Layers_2REP"
  "PPCN_5CLS_1.0LRPC_0.001WD_noTied_noBPtied_noBP_noRelu_7Layers_2REP"
  "PPCN_5CLS_1.0LRPC_0.001WD_withTied_withBPtied_withBP_noRelu_9Layers_2REP"
  "PPCN_5CLS_1.0LRPC_0.001WD_withTied_withBPtied_withBP_withRelu_9Layers_2REP"
  "PPCN_5CLS_1.0LRPC_0.001WD_withTied_noBPtied_withBP_noRelu_9Layers_2REP"
  "PPCN_5CLS_1.0LRPC_0.001WD_withTied_noBPtied_noBP_noRelu_9Layers_2REP"
  "PPCN_5CLS_1.0LRPC_0.001WD_withTied_noBPtied_noBP_withRelu_9Layers_2REP"
)

BASE_LOGDIR="./logs/noisy_test"
#############################
# modify the log name here:
#############################
MASTER_LOG="$BASE_LOGDIR/master_0508_noisy_exp.log"

mkdir -p "$BASE_LOGDIR"
> "$MASTER_LOG"

# ─────────────────────────────────────────────────────────────────
# GNU parallel invocation:
parallel \
  --jobs 4               \
  --keep-order           \
  "python run_test.py --model_name {} --model_dir '$MODEL_DIR' \
     --weight '$WEIGHT_PATH' --plot_path '$PLOT_PATH' --w_type '$W_TYPE' \
     --noise_to_ff $NOISE_TO_FF --noise_to_bp $NOISE_TO_BP \
    2>&1 | tee '$BASE_LOGDIR/{}/job.log'" \
  ::: "${MODEL_NAMES[@]}" \
  >> "$MASTER_LOG" 2>&1 &
# ─────────────────────────────────────────────────────────────────

echo "Launched all jobs (max 4)."
echo "Grouped master log at: $MASTER_LOG"
echo "Run: tail -f $MASTER_LOG"

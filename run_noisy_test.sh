#!/usr/bin/env bash
# run_test.sh
##########################################
# set model_name here
MODEL_NAME="PPCN_5CLS_1.0LRPC_0.001WD_withTied_withBPtied_withBP_noRelu_9Layers_2REP"
##########################################

# noisy parameters
MODEL_DIR="./saved_ckpt"
WEIGHT_PATH="./expanded_weights"
PLOT_PATH="./loss_plot"
W_TYPE="fb_flip"
NOISE_TO_FF=true
NOISE_TO_BP=true

LOGDIR="./logs/noisy_test/${MODEL_NAME}"
mkdir -p "${LOGDIR}"
echo "Launching run_test.py for model '$MODEL_NAME' at $(date)…"
nohup python run_test.py \
  --model_dir   "$MODEL_DIR"    \
  --model_name  "$MODEL_NAME"   \
  --weight      "$WEIGHT_PATH"  \
  --plot_path   "$PLOT_PATH"    \
  --w_type      "$W_TYPE"       \
  --noise_to_ff $NOISE_TO_FF    \
  --noise_to_bp $NOISE_TO_BP    \
  > "${LOGDIR}/master_test.log" 2>&1 &

echo "Started (PID $!) – logging to ${LOGDIR}/master_test.log"
tail -f "${LOGDIR}/master_test.log"

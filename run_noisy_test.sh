#!/usr/bin/env bash
# run_test.sh
#
# Usage: ./run_test.sh <model_name>
# (or export MODEL_DIR before running to override the default)

DEFAULT_MODEL_DIR="./checkpoints"
MODEL_DIR="${MODEL_DIR:-$DEFAULT_MODEL_DIR}"

if [ -z "$1" ]; then
  echo "Usage: $0 <model_name>"
  exit 1
fi
MODEL_NAME="$1"

# noisy parameters (edit as needed)
WEIGHT_PATH=""       # e.g. "/path/to/weights.pth"
LAYER_IDX=2
PLOT_PATH="./plots/conv2.png"
W_TYPE="fb_flip"
NOISE_TO_FF=true
NOISE_TO_BP=false

echo "Launching run_test.py for model '$MODEL_NAME' at $(date)…"
nohup python run_test.py \
  --model_dir   "$MODEL_DIR"    \
  --model_name  "$MODEL_NAME"   \
  --weight      "$WEIGHT_PATH"  \
  --layer_idx   $LAYER_IDX      \
  --plot_path   "$PLOT_PATH"    \
  --w_type      "$W_TYPE"       \
  --noise_to_ff $NOISE_TO_FF    \
  --noise_to_bp $NOISE_TO_BP    \
  > master_test.log 2>&1 &

echo "Started (PID $!) – logging to master_test.log"
tail -f master_test.log

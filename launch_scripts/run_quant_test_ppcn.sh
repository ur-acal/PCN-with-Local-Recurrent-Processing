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
EXP_NAME="1126_ppcn_scanGFI_quant"
MASTER_LOG="$BASE_LOGDIR/master_${EXP_NAME}.log"
JOB_LOG="$BASE_LOGDIR/parallel_job_master_${EXP_NAME}.log"

# ─────────────── model list ───────────────
MODEL_NAMES=(
#  "PCNetNoBatchNorm_FFFBReLU6NoLastConvYasX_5CLS_0.15LRPC_0.001WD_noBPtied_noBP_128BS_0.25Dropout_5Layers_64Chan_2Pool_scanGFI_1REP"  # S
#  "PCNetNoBatchNorm_FFFBReLU6NoLastConvYasX_5CLS_0.15LRPC_0.001WD_noBPtied_noBP_128BS_0.25Dropout_5Layers_128Chan_2Pool_scanGFI_1REP" # M
  "PCNetNoBatchNorm_FFFBReLU6NoLastConvYasX_5CLS_0.15LRPC_0.001WD_noBPtied_noBP_128BS_0.25Dropout_7Layers_128Chan_2Pool_scanGFI_1REP"  # L
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

  set -o pipefail
  python -u run_test.py \
    --model_name      "$name" \
    --model_dir       "$MODEL_DIR" \
    --img_type        "$_img_type" \
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
    --noisy_test      "false" \
    --quant_test      "true" \
    --quant_cls       "QuantHelper" \
    --agg_bits        "8" \
    --w_quant_type    "per_tenspr" \
    --w_bits          "4" \
    --act_bits        "4" \
    --pc_conv         "${_pc_conv}" \
    --test_only       "true" \
    2>&1 | tee -a "$BASE_LOGDIR/$name/job.log"
}
export -f run_model

echo "Tail master with: tail -f $MASTER_LOG"

# ─────────────── run in parallel ───────────────
parallel \
  --jobs 3 \
  --joblog "$JOB_LOG" \
  --keep-order \
  run_model {} \
  ::: "${MODEL_NAMES[@]}"

echo "All jobs finished — merging logs into $MASTER_LOG"

# ─────────────── merge logs sequentially ───────────────
: >"$MASTER_LOG"
for name in "${MODEL_NAMES[@]}"; do
  printf '========== %s ==========\n' "$name" >>"$MASTER_LOG"
  if ! grep -A 20 "Final Result " \
             "$BASE_LOGDIR/$name/job.log" >>"$MASTER_LOG"; then
     echo "[Final Result not found]" >>"$MASTER_LOG"
  fi
  printf '\n\n' >>"$MASTER_LOG"
done

echo "All summaries written to $MASTER_LOG"

#######################################################
# running
# nohup bash ./launch_scripts/run_noisy_test_dropout.sh > logs/run_script_output/launcher.out 2>&1 &
# tail -f logs/run_script_output/launcher.out
# After the run is finished, the master_log file will be printed out
# then cat master_log
# then use shell_utils/parse_noise_logs.py to convert the log into csv
#######################################################
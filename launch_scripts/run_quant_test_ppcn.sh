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
MAX_INPS=(144)
N_BITS_VALS=(4)
export AGG_BITS=8
export BASE_LOGDIR="./logs/noisy_test"
# MASTER_LOG and JOB_LOG will be set per noise combination

# ─────────────── model list ───────────────
MODEL_NAMES=(
#  "PCNetNoBatchNorm_FFFBReLU6NoLastConvYasX_5CLS_0.15LRPC_0.001WD_noBPtied_noBP_128BS_0.25Dropout_5Layers_64Chan_2Pool_scanGFI_1REP"  # S
#  "PCNetNoBatchNorm_FFFBReLU6NoLastConvYasX_5CLS_0.15LRPC_0.001WD_noBPtied_noBP_128BS_0.25Dropout_5Layers_128Chan_2Pool_scanGFI_1REP" # M
#  "PCNetNoBatchNorm_FFFBReLU6NoLastConvYasX_5CLS_0.15LRPC_0.001WD_noBPtied_noBP_128BS_0.25Dropout_7Layers_128Chan_2Pool_scanGFI_1REP"  # L

#  "QAT4w4aPCNetNoBatchNorm_FFFBReLU6NoLastConvYasX_5CLS_0.15LRPC_0.001WD_noBPtied_noBP_128BS_0.25Dropout_5Layers_128Chan_2Pool_scanGFI_1REP"
#  "QAT4w4aPCNetNoBatchNorm_FFFBReLU6NoLastConvYasX_5CLS_0.15LRPC_0.001WD_noBPtied_noBP_128BS_0.25Dropout_5Layers_64Chan_2Pool_scanGFI_4REP"
#  "QAT4w4aPCNetNoBatchNorm_FFFBReLU6NoLastConvYasX_5CLS_0.15LRPC_0.001WD_noBPtied_noBP_128BS_0.25Dropout_5Layers_128Chan_2Pool_scanGFI_3REP"
#  "QAT4w4aPCNetNoBatchNorm_FFFBReLU6NoLastConvYasX_5CLS_0.15LRPC_0.001WD_noBPtied_noBP_128BS_0.25Dropout_7Layers_128Chan_2Pool_scanGFI_1REP"

  "PCNetNoBatchNorm_FFFBReLU6NoLastConvYasX_5CLS_0.15LRPC_0.001WD_noBPtied_noBP_128BS_0.25Dropout_5Layers_C100_64Chan_2Pool_scanGFI_1REP"
)

# ─────────────── prepare logs ───────────────
mkdir -p "$BASE_LOGDIR"
for n_bits in "${N_BITS_VALS[@]}"; do
  for max_inp in "${MAX_INPS[@]}"; do
    for name in "${MODEL_NAMES[@]}"; do
      mkdir -p "$BASE_LOGDIR/${name}_maxInp_${max_inp}_nbits_${n_bits}"
      > "$BASE_LOGDIR/${name}_maxInp_${max_inp}_nbits_${n_bits}/job.log"
    done
  done
done

# ─────────────── helper function ───────────────
run_model(){
  local name="$1"
  local max_inp="$2"
  local n_bits="$3"
  local __rest="${name#*_}"
  local _pc_conv="${__rest%%_*}"

  if [[ "$name" == *C100* ]]; then
    local _task="cifar100"
  else
    local _task="cifar10"
  fi

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
    --task            "${_task}" \
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
    --act_quant_cls   "PercQuantHelper" \
    --act_perc        "0.999" \
    --agg_bits        "$AGG_BITS" \
    --w_quant_type    "per_channel" \
    --w_bits          "${n_bits}" \
    --act_bits        "${n_bits}" \
    --max_inp         "${max_inp}" \
    --pc_conv         "${_pc_conv}" \
    --test_only       "true" \
    2>&1 | tee -a "$BASE_LOGDIR/${name}_maxInp_${max_inp}_nbits_${n_bits}/job.log"
}
export -f run_model

# ─────────────── loop over methods ───────────────
for n_bits in "${N_BITS_VALS[@]}"; do
  ##########################################################################################
  # Modify log name here before each run
  ##########################################################################################
  EXP_NAME="1228_scanGFI_PPCN_split_${n_bits}NBit.log"
  MASTER_LOG="$BASE_LOGDIR/master_${EXP_NAME}"
  JOB_LOG="$BASE_LOGDIR/parallel_master_${EXP_NAME}"
  > "$MASTER_LOG"
  > "$JOB_LOG"
  echo "Tail master with: tail -f $MASTER_LOG"
  # ─────────────── run in parallel ───────────────
  parallel \
    --jobs 3 \
    --joblog "$JOB_LOG" \
    --keep-order \
    run_model {1} {2} "${n_bits}" \
    ::: "${MODEL_NAMES[@]}" \
    ::: "${MAX_INPS[@]}"
  echo "All jobs finished — merging logs into $MASTER_LOG"
  # ─────────────── merge logs sequentially ───────────────
  : >"$MASTER_LOG"
  for max_inp in "${MAX_INPS[@]}"; do
    for name in "${MODEL_NAMES[@]}"; do
      printf '========== %s | max_inp=%s | n_bits=%s ==========\n' \
             "$name" "$max_inp" "$n_bits" >>"$MASTER_LOG"
      logfile="$BASE_LOGDIR/${name}_maxInp_${max_inp}_nbits_${n_bits}/job.log"
      if ! grep -A 100 "Model name: ${name} " "$logfile" >>"$MASTER_LOG"; then
        echo "[Final Result not found] $logfile" >>"$MASTER_LOG"
      fi
      printf '\n\n' >>"$MASTER_LOG"
    done
  done
  echo "All summaries written to $MASTER_LOG"
done

#######################################################
# running
# nohup bash ./launch_scripts/run_quant_test_ppcn.sh > logs/run_script_output/launcher.out 2>&1 &
# tail -f logs/run_script_output/launcher.out
# After the run is finished, the master_log file will be printed out
# then cat master_log
# then use shell_utils/parse_noise_logs.py to convert the log into csv
#######################################################
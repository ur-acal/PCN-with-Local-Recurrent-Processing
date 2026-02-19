#!/bin/bash
#SBATCH -p ds4ai
#SBATCH -c 16
#SBATCH -t 24:00:00
#SBATCH --mem=32gb
#SBATCH --gres=gpu:1
#SBATCH --output=logs/test_%j.out
#SBATCH --error=logs/test_%j.err
#SBATCH --job-name=PCN_EVAL

# ─────────────── fixed params ───────────────
export SCANGEN_DATA_ROOT=/scratch/tgeng_lab/sun/projs/ODE_CIFAR10/data
export MODEL_DIR="./saved_ckpt"
export WEIGHT_PATH="./expanded_weights"
export PLOT_PATH="./loss_plot"
export W_TYPE="fb_flip"
export NOISE_TO_FF=true
export NOISE_TO_BP=true
DATASET_NAME="${DATASET_NAME:-cifar10}"

#######################################################
# change the log names here to identify each run
#######################################################
export BASE_LOGDIR="./logs/noisy_test"
EXP_NAME="1126_ppcn_scanGFI_quant"
MASTER_LOG="$BASE_LOGDIR/master_${EXP_NAME}.log"
JOB_LOG="$BASE_LOGDIR/parallel_job_master_${EXP_NAME}.log"

# ─────────────── model list ───────────────
MODEL_NAMES=(
 "PCNetNoBatchNorm_FFFBReLU6NoLastConvYasX_5CLS_0.15LRPC_0.001WD_noBPtied_noBP_128BS_0.25Dropout_5Layers_64Chan_2Pool_scanGFI_1REP"  # S
# "PCNetNoBatchNorm_FFFBReLU6NoLastConvYasX_5CLS_0.15LRPC_0.001WD_noBPtied_noBP_128BS_0.25Dropout_5Layers_64Chan_2Pool_scanGFI_2REP"
"PCNetNoBatchNorm_FFFBReLU6NoLastConvYasX_5CLS_0.15LRPC_0.001WD_noBPtied_noBP_128BS_0.25Dropout_5Layers_64Chan_2Pool_kd_crdDistill_a0p3_t1p5_scanGFI_2REP"
#  "PCNetNoBatchNorm_FFFBReLU6NoLastConvYasX_5CLS_0.15LRPC_0.001WD_noBPtied_noBP_128BS_0.25Dropout_5Layers_64Chan_2Pool_kd_crdDistill_a0p3_t1p5_scanGFI_2REP" # M
#  "PCNetNoBatchNorm_FFFBReLU6NoLastConvYasX_5CLS_0.15LRPC_0.001WD_noBPtied_noBP_128BS_0.25Dropout_7Layers_128Chan_2Pool_scanGFI_1REP"  # L

"QAT4w4aPCNetNoBatchNorm_FFFBReLU6NoLastConvYasX_5CLS_0.15LRPC_0.001WD_noBPtied_noBP_128BS_0.25Dropout_5Layers_64Chan_2Pool_kd_crdDistill_a0p3_t1p5_scanGFI_3REP"
# "FTNT1p0add8b_PCNetNoBatchNorm_FFFBReLU6NoLastConvYasX_5CLS_0.15LRPC_0.001WD_noBPtied_noBP_128BS_0.25Dropout_5Layers_64Chan_2Pool_scanGFI_1REP"

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
    --dataset         "${DATASET_NAME}" \
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
    --agg_bits        "8" \
    --w_quant_type    "per_channel" \
    --w_bits          "4" \
    --act_bits        "4" \
    --pc_conv         "${_pc_conv}" \
    --test_only       "false" \
    2>&1 | tee -a "$BASE_LOGDIR/$name/job.log"
}
export -f run_model

echo "Tail master with: tail -f $MASTER_LOG"
# ─────────────── run in parallel ───────────────
MAX_JOBS=3
pids=()

for name in "${MODEL_NAMES[@]}"; do
  # Wait if we've reached max jobs
  while [ ${#pids[@]} -ge $MAX_JOBS ]; do
    # Check which jobs are still running
    new_pids=()
    for pid in "${pids[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then
        new_pids+=("$pid")
      fi
    done
    pids=("${new_pids[@]}")
    sleep 0.1
  done
  
  # Start job in background
  (
    start_time=$(date +%s)
    run_model "$name"
    exit_code=$?
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    echo "$(date +%Y-%m-%d\ %H:%M:%S)	$duration	$exit_code	run_model	$name" >> "$JOB_LOG"
  ) &
  pids+=("$!")
done

# Wait for all remaining jobs to finish
for pid in "${pids[@]}"; do
  wait "$pid" 2>/dev/null || true
done
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
# nohup bash ./launch_scripts/run_quant_test_ppcn.sh > logs/run_script_output/launcher.out 2>&1 &
# tail -f logs/run_script_output/launcher.out
# After the run is finished, the master_log file will be printed out
# then cat master_log
# then use shell_utils/parse_noise_logs.py to convert the log into csv
#######################################################

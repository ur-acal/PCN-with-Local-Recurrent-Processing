#!/usr/bin/env bash
set -eu
trap '' HUP   # ignore hangup so the children survive

# ─────────────── fixed params ───────────────
export MODEL_DIR="./saved_ckpt"
export tol="1e-6"

# ─────────────── noise toggles ───────────────
METHOD_VALS=("dopri5")
CAP_VALS=("49e-12")
#METHOD_VALS=("euler")
#METHOD_VALS=("rk4")

export BASE_LOGDIR="./logs/test_ode_noisy"
# MASTER_LOG and JOB_LOG will be set per noise combination

# ─────────────── model list ───────────────
MODEL_NAMES=(
#  "QAT4bPCNetNoBatchNorm_PCConvReLU6_0.0eps_S2NoMinusZChgZNoisyI_dopri5Solver_1.5TEnd_0.0001Tol_0.001WD_128BS_0.01LR_0.25Dropout_7Layers_2Pool_3REP"
#  "QAT4bPCNetNoBatchNorm_PCConvReLU6_0.0eps_S2NoMinusZChgZNoisyI_dopri5Solver_1.5TEnd_0.0001Tol_0.001WD_128BS_0.01LR_0.25Dropout_7Layers_2Pool_5REP" # Use relu6 scaled
#  "QAT5bPCNetNoBatchNorm_PCConvReLU6_0.0eps_S2NoMinusZChgZNoisyI_dopri5Solver_1.5TEnd_0.0001Tol_0.001WD_128BS_0.01LR_0.25Dropout_7Layers_2Pool_1REP"
#  "QAT6bPCNetNoBatchNorm_PCConvReLU6_0.0eps_S2NoMinusZChgZNoisyI_dopri5Solver_1.5TEnd_0.0001Tol_0.001WD_128BS_0.01LR_0.25Dropout_7Layers_2Pool_1REP"

  "QAT4bPCNetNoBatchNorm_PCConvReLU6_0.0eps_S2NoMinusZChgZNoisyI_dopri5Solver_1.5TEnd_0.0001Tol_0.001WD_128BS_0.01LR_0.25Dropout_7Layers_2Pool_6REP"
  "QAT5bPCNetNoBatchNorm_PCConvReLU6_0.0eps_S2NoMinusZChgZNoisyI_dopri5Solver_1.5TEnd_0.0001Tol_0.001WD_128BS_0.01LR_0.25Dropout_7Layers_2Pool_2REP"
)

# ─────────────── prepare logs ───────────────
mkdir -p "$BASE_LOGDIR"
for cap_val in "${CAP_VALS[@]}"; do
  for method in "${METHOD_VALS[@]}"; do
    for name in "${MODEL_NAMES[@]}"; do
      mkdir -p "$BASE_LOGDIR/${name}_method_${method}_tol_${tol}_cap_${cap_val}"
      > "$BASE_LOGDIR/${name}_method_${method}_tol_${tol}_cap_${cap_val}/job.log"
    done
  done
done


# ─────────────── helper function ───────────────
run_model(){
  local name="$1"
  local method="$2"
  local cap_val="$3"

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

  local __first="${name%%_*}"
  local __num="${__first#QAT}"
  local n_bits="${__num%%b*}"
  ########################################
  # only fuse_bn when noise is added to bn
  ########################################
  set -o pipefail
  python -u ode_inference.py \
    --model_name      "$name" \
    --ckpt            "best" \
    --model_dir       "$MODEL_DIR" \
    --method          "$method" \
    --tol             "$tol" \
    --n_steps         15 \
    --ts_scale        1 \
    --d_start         0 \
    --d_end           1 \
    --n_sweep_left    0 \
    --n_sweep_right   1 \
    --R               1e5 \
    --C               "$cap_val" \
    --w_bits          "${n_bits}" \
    --pc_conv         "${_pc_conv}Noisy" \
    --ode_block       "${_ode_block}" \
    --ode_wrapper     "QATTester2State" \
    --img_type        "${_img_type}" \
    --test_only       "true" \
    2>&1 | tee -a "$BASE_LOGDIR/${name}_method_${method}_tol_${tol}_cap_${cap_val}/job.log"
}
export -f run_model

# ─────────────── loop over methods ───────────────
for method in "${METHOD_VALS[@]}"; do
  ##########################################################################################
  # Modify log name here before each run
  ##########################################################################################
  EXP_NAME="0923_2State_QAT_EXP_rgb_${method}Method_${tol}Tol_test_only.log"
  MASTER_LOG="$BASE_LOGDIR/master_${EXP_NAME}"
  JOB_LOG="$BASE_LOGDIR/parallel_master_${EXP_NAME}"
  > "$MASTER_LOG"
  > "$JOB_LOG"
  echo "Tail master with: tail -f $MASTER_LOG"
  # ─────────────── run in parallel ───────────────
  parallel \
    --jobs 4 \
    --joblog "$JOB_LOG" \
    --keep-order \
    run_model {1} "${method}" {2} \
    ::: "${MODEL_NAMES[@]}" \
    ::: "${CAP_VALS[@]}"
  echo "All jobs finished — merging logs into $MASTER_LOG"
  # ─────────────── merge logs sequentially ───────────────
  : >"$MASTER_LOG"
  for cap_val in "${CAP_VALS[@]}"; do
    for name in "${MODEL_NAMES[@]}"; do
      printf '========== %s | method=%s | cap=%s ==========\n' \
             "$name" "$method" "$cap_val" >>"$MASTER_LOG"
      logfile="$BASE_LOGDIR/${name}_method_${method}_tol_${tol}_cap_${cap_val}/job.log"
      if ! grep -A 11 "Model name: ${name} " "$logfile" >>"$MASTER_LOG"; then
        echo "[Final Result not found] $logfile" >>"$MASTER_LOG"
      fi
      printf '\n\n' >>"$MASTER_LOG"
    done
  done
  echo "All summaries written to $MASTER_LOG"
done

#######################################################
# running
# nohup bash run_ode_inference.sh > logs/run_script_output/launcher.out 2>&1 &
# tail -f logs/run_script_output/launcher.out
# After the run is finished, the master_log file will be printed out
# then cat master_log
# then use shell_utils/parse_noise_logs.py to convert the log into csv
#######################################################

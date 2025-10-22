#!/usr/bin/env bash
set -eu
trap '' HUP   # ignore hangup so the children survive

# ─────────────── fixed params ───────────────
export MODEL_DIR="./saved_ckpt"
export tol="1e-6"

# ─────────────── noise toggles ───────────────
#N_BITS_VALS=(4 5 6 7 8)
N_BITS_VALS=(5 5 5 5 5 5)
#N_BITS_VALS=(5 5 5 5 5 5 5 5 5 5)
METHOD_VALS=("dopri5")
CAP_VALS=("49e-15")
#METHOD_VALS=("euler")
#METHOD_VALS=("rk4")

export BASE_LOGDIR="./logs/test_ode_noisy"
# MASTER_LOG and JOB_LOG will be set per noise combination

# ─────────────── model list ───────────────
MODEL_NAMES=(
#  "PCNetNoBatchNorm_PCConvHardTanh_dopri5Solver_0.75TEnd_0.0001Tol_0.001WD_noBPtied_noBP_withPC_128BS_0.01LR_0.25Dropout_7Layers_1REP" # NODE baseline
#  "PCNetNoBatchNorm_PCConvHardTanh_dopri5Solver_1.0TEnd_0.0001Tol_0.001WD_noBPtied_noBP_withPC_128BS_0.01LR_0.25Dropout_7Layers_1REP" # NODE baseline

#  "PCNetNoBatchNorm_PCConvReLU6_ODEActInpNoMinus_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_noBPtied_noBP_128BS_0.01LR_0.25Dropout_7Layers_1REP" # zero init
#  "PCNetNoBatchNorm_PCConvHardTanh2_ODEActInpNoMinus_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_noBPtied_noBP_128BS_0.01LR_0.25Dropout_7Layers_1REP" # zero init

#  "PCNetNoBatchNorm_PCConvHardTanh_ODEBlockPCMinusY_dopri5Solver_0.75TEnd_0.001Tol_0.001WD_noBPtied_noBP_128BS_0.01LR_0.25Dropout_7Layers_2REP" # no max_g_norm
#  "PCNetNoBatchNorm_PCConvReLU6_ODEBlockPCMinusY_dopri5Solver_0.75TEnd_0.001Tol_0.001WD_noBPtied_noBP_128BS_0.01LR_0.25Dropout_7Layers_2REP" # no max_g_norm

#  "PCNetNoBatchNorm_PCConvHardTanh_0.2eps_ODEFixNoiseOffset_dopri5Solver_0.75TEnd_0.0001Tol_0.001WD_noBPtied_noBP_128BS_0.01LR_0.25Dropout_7Layers_1REP"
#  "ftPCNetNoBatchNorm_PCConvHardTanh_0.4eps_ODEFixNoiseOffset_dopri5Solver_0.75TEnd_0.0001Tol_0.001WD_noBPtied_noBP_128BS_0.0001LR_0.25Dropout_7Layers_1REP"
#  "PCNetNoBatchNorm_PCConvReLU6_0.2eps_ODEFixNoiseOffset_dopri5Solver_0.75TEnd_0.0001Tol_0.001WD_noBPtied_noBP_128BS_0.01LR_0.25Dropout_7Layers_1REP"
#  "ftPCNetNoBatchNorm_PCConvReLU6_0.4eps_ODEFixNoiseOffset_dopri5Solver_0.75TEnd_0.0001Tol_0.001WD_noBPtied_noBP_128BS_0.0001LR_0.25Dropout_7Layers_1REP" # passive baseline

#  "PCNetNoBatchNorm_PCConvReLU6_0.2eps_ODEFixNoiseOffset_dopri5Solver_0.75TEnd_0.0001Tol_0.001WD_noBPtied_noBP_128BS_0.01LR_0.25Dropout_5Layers_2Pool_1REP" # 2 max pooling

#  "ftPCNetNoBatchNorm_PCConvReLU6_0.4eps_ODEFixNoiseOffset_dopri5Solver_0.75TEnd_0.0001Tol_0.001WD_noBPtied_noBP_128BS_0.0001LR_0.25Dropout_6Layers_1REP" # 3 max pooling
#  "PCNetNoBatchNorm_PCConvReLU6_0.002eps_S2NoMinusZChgZNoisyI_dopri5Solver_1.5TEnd_0.0001Tol_0.001WD_128BS_0.01LR_0.25Dropout_7Layers_2Pool_1REP"
#  "PCNetNoBatchNorm_PCConvReLU6_0.002eps_S2NoMinusZChgZNoisyI_dopri5Solver_1.5TEnd_0.0001Tol_0.001WD_128BS_0.01LR_0.25Dropout_7Layers_2Pool_2REP" # S2NoMinusZChgZNoisyI baseline

#  "PCNetNoBatchNorm_PCConvReLU6_0.2eps_S2NoMinusZChgZMinusNoisyI_dopri5Solver_1.5TEnd_0.0001Tol_0.001WD_128BS_0.01LR_0.25Dropout_7Layers_2Pool_rggb_2REP"
#  "PCNetNoBatchNorm_PCConvReLU6_0.2eps_S2NoMinusZChgZMinusNoisyI_dopri5Solver_1.5TEnd_0.0001Tol_0.001WD_128BS_0.01LR_0.25Dropout_7Layers_2Pool_cycleisp_2REP"
#  "PCNetNoBatchNorm_PCConvReLU6_0.2eps_S2NoMinusZChgZNoisyI_dopri5Solver_1.5TEnd_0.0001Tol_0.001WD_128BS_0.01LR_0.25Dropout_7Layers_2Pool_cycleisp_2REP"

#  "QAT4bPCNetNoBatchNorm_PCConvReLU6_0.0eps_S2NoMinusZChgZNoisyI_dopri5Solver_1.5TEnd_0.0001Tol_0.001WD_128BS_0.01LR_0.25Dropout_7Layers_2Pool_3REP"
#  "QAT4bPCNetNoBatchNorm_PCConvReLU6_0.0eps_S2NoMinusZChgZNoisyI_dopri5Solver_1.5TEnd_0.0001Tol_0.001WD_128BS_0.01LR_0.25Dropout_7Layers_2Pool_5REP" # Use relu6 scaled
#  "QAT6bPCNetNoBatchNorm_PCConvReLU6_0.0eps_S2NoMinusZChgZNoisyI_dopri5Solver_1.5TEnd_0.0001Tol_0.001WD_128BS_0.01LR_0.25Dropout_7Layers_2Pool_1REP"

#  "QAT5bPCNetNoBatchNorm_PCConvReLU6_0.0eps_S2NoMinusZChgZNoisyI_dopri5Solver_1.5TEnd_0.0001Tol_0.001WD_128BS_0.01LR_0.25Dropout_7Layers_2Pool_3REP" # QAT that uses W^Q in [-1,1]
#  "QAT5bPCNetNoBatchNorm_PCConvReLU6_0.0eps_S2NoMinusZChgZNoisyI_dopri5Solver_1.5TEnd_0.0001Tol_0.001WD_128BS_0.01LR_0.25Dropout_7Layers_2Pool_4REP" # Trained with R=C=1
#  "QAT5bPCNetNoBatchNorm_PCConvReLU6_0.0eps_S2NoMinusZChgZNoisyI_dopri5Solver_1.5TEnd_0.0001Tol_0.001WD_128BS_0.01LR_0.25Dropout_7Layers_2Pool_5REP" # R=C=1 and tol=1e-6
#  "QAT5bPCNetNoBatchNorm_PCConvReLU6_0.0eps_S2NoMinusZChgZNoisyI_dopri5Solver_1.5TEnd_0.0001Tol_0.001WD_128BS_0.01LR_0.25Dropout_7Layers_2Pool_6REP" # R=C=1 and v_dd=10
#  "QAT5bPCNetNoBatchNorm_PCConvReLU6_0.0eps_S2NoMinusZChgZNoisyI_dopri5Solver_1.5TEnd_0.0001Tol_0.001WD_128BS_0.01LR_0.25Dropout_7Layers_2Pool_7REP" # R=C=1, lr=1e-3
#  "QAT5bPCNetNoBatchNorm_PCConvReLU6_0.0eps_S2NoMinusZChgZNoisyI_dopri5Solver_1.5TEnd_0.0001Tol_0.001WD_128BS_0.01LR_0.25Dropout_7Layers_2Pool_8REP" # R=1e5, C=49e-15 lr=1e-3
#  "QAT5bPCNetNoBatchNorm_PCConvReLU6_0.0eps_S2NoMinusZChgZNoisyI_dopri5Solver_1.5TEnd_0.0001Tol_0.001WD_128BS_0.01LR_0.25Dropout_7Layers_2Pool_10REP" # LR=1e-2 Cosine Annealing, R=1e5, C=49e-15
#  "QAT4bPCNetNoBatchNorm_PCConvReLU6_0.0eps_S2NoMinusZChgZNoisyI_dopri5Solver_1.5TEnd_0.0001Tol_0.001WD_128BS_0.01LR_0.25Dropout_7Layers_2Pool_7REP" # Same as above, but 4-bit
#  "QAT4bLSQWeightPCNetNoBatchNorm_PCConvReLU6_0.0eps_S2NoMinusZChgZNoisyI_dopri5Solver_1.5TEnd_0.0001Tol_0.001WD_128BS_0.01LR_0.25Dropout_7Layers_2Pool_1REP"

#  "QAT5bLSQWeightPCNetNoBatchNorm_PCConvReLU6_0.0eps_S2NoMinusZChgZNoisyI_dopri5Solver_1.5TEnd_0.0001Tol_0.001WD_128BS_0.01LR_0.25Dropout_7Layers_2Pool_3REP" # LSQ
#  "QAT5bLSQWeightPCNetNoBatchNorm_PCConvReLU6_0.0eps_S2NoMinusZChgZNoisyI_dopri5Solver_1.5TEnd_0.0001Tol_0.001WD_128BS_0.01LR_0.25Dropout_7Layers_2Pool_7REP" # R=C=1, lr=1e-3
#  "QAT5bLSQWeightPCNetNoBatchNorm_PCConvReLU6_0.0eps_S2NoMinusZChgZNoisyI_dopri5Solver_1.5TEnd_0.0001Tol_0.001WD_128BS_0.01LR_0.25Dropout_7Layers_2Pool_5REP" # R=1e5, C=49e-15
#  "QAT5bLSQWeightPCNetNoBatchNorm_PCConvReLU6_0.0eps_S2NoMinusZChgZNoisyI_dopri5Solver_1.5TEnd_0.0001Tol_0.001WD_128BS_0.01LR_0.25Dropout_7Layers_2Pool_6REP" # R=1e5, C=49e-15 lr=1e-3

#  "PCNetNoBatchNorm_PCConvReLU6_0.2eps_S2NoMinusZChgZNoisyI_dopri5Solver_1.5TEnd_0.0001Tol_0.001WD_128BS_0.01LR_0.25Dropout_7Layers_2Pool_rggb_2REP" # RGGB
#  "PCNetNoBatchNorm_PCConvReLU6_0.2eps_S2NoMinusZChgZNoisyI_dopri5Solver_1.5TEnd_0.0001Tol_0.001WD_128BS_0.01LR_0.25Dropout_6Layers_2Pool_rggb_1REP"
#  "PCNetNoBatchNorm_PCConvReLU6_0.2eps_S2NoMinusZChgZNoisyI_dopri5Solver_1.5TEnd_0.0001Tol_0.001WD_128BS_0.01LR_0.25Dropout_4Layers_2Pool_rggb_1REP"

#  "PCNetNoBatchNorm_PCConvReLU6_0.0eps_S2NoMinusZChgZNoisyI_dopri5Solver_1.5TEnd_0.0001Tol_0.001WD_128BS_0.01LR_0.25Dropout_5Layers_2Pool_scanGFI_1REP"
  "PCNetNoBatchNorm_PCConvReLU6_0.0eps_S2NoMinusZChgZNoisyI_dopri5Solver_1.5TEnd_0.0001Tol_0.001WD_128BS_0.01LR_0.25Dropout_5Layers_2Pool_scanGFI_3REP"
  "ftNT0p4mulPCNetNoBatchNorm_PCConvReLU6_0.0eps_S2NoMinusZChgZNoisyI_dopri5Solver_1.5TEnd_0.0001Tol_0.001WD_128BS_0.01LR_0.25Dropout_5Layers_2Pool_scanGFI_1REP"
  "QAT5bNT0p4mulPCNetNoBatchNorm_PCConvReLU6_0.0eps_S2NoMinusZChgZNoisyI_dopri5Solver_1.5TEnd_0.0001Tol_0.001WD_128BS_0.01LR_0.25Dropout_5Layers_2Pool_scanGFI_2REP"
)

# ─────────────── prepare logs ───────────────
mkdir -p "$BASE_LOGDIR"
for n_bits in "${N_BITS_VALS[@]}"; do
  for cap_val in "${CAP_VALS[@]}"; do
    for method in "${METHOD_VALS[@]}"; do
      for name in "${MODEL_NAMES[@]}"; do
        mkdir -p "$BASE_LOGDIR/${name}_method_${method}_tol_${tol}_nbits_${n_bits}_cap_${cap_val}"
        > "$BASE_LOGDIR/${name}_method_${method}_tol_${tol}_nbits_${n_bits}_cap_${cap_val}/job.log"
        done
      done
  done
done

# ─────────────── helper function ───────────────
run_model(){
  local name="$1"
  local method="$2"
  local n_bits="$3"
  local cap_val="$4"

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

  local _ode_wrapper="WrapQuantizeW"
  if [[ "$name" == *S2* || "$name" == *State2* ]]; then
    if [[ "$name" == *QAT* ]]; then
      _ode_wrapper="QATTester2State"
    else
      _ode_wrapper="ODEWrapper2State"
    fi
  fi
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
    --d_start         0.2 \
    --d_end           0.2 \
    --n_sweep_left    5 \
    --n_sweep_right   0 \
    --thermal_noise   "true" \
    --R               1e5 \
    --C               "$cap_val" \
    --v_dd            "1" \
    --w_bits          "$n_bits" \
    --pc_conv         "${_pc_conv}Noisy" \
    --ode_block       "${_ode_block}" \
    --ode_wrapper     "${_ode_wrapper}" \
    --img_type        "${_img_type}" \
    --test_only       "true" \
    2>&1 | tee -a "$BASE_LOGDIR/${name}_method_${method}_tol_${tol}_nbits_${n_bits}_cap_${cap_val}/job.log"
}
export -f run_model

# ─────────────── loop over methods ───────────────
#for n_bits in "${N_BITS_VALS[@]}"; do
for method in "${METHOD_VALS[@]}"; do
  ##########################################################################################
  # Modify log name here before each run
  ##########################################################################################
#  EXP_NAME="0818_3pooling_wrapped_${n_bits}bits_ODESumAsBInitY_${method}Method_${tol}Tol.log"
  EXP_NAME="1021_scanGPI_noise_inject_test_${method}Method_${tol}Tol.log"
  MASTER_LOG="$BASE_LOGDIR/master_${EXP_NAME}"
  JOB_LOG="$BASE_LOGDIR/parallel_master_${EXP_NAME}"
  > "$MASTER_LOG"
  > "$JOB_LOG"
  echo "Tail master with: tail -f $MASTER_LOG"
  # ─────────────── run in parallel ───────────────
  parallel \
    --jobs 2 \
    --joblog "$JOB_LOG" \
    --keep-order \
    run_model {1} "${method}" {2} {3} \
    ::: "${MODEL_NAMES[@]}" \
    ::: "${N_BITS_VALS[@]}" \
    ::: "${CAP_VALS[@]}"
  echo "All jobs finished — merging logs into $MASTER_LOG"
  # ─────────────── merge logs sequentially ───────────────
  : >"$MASTER_LOG"
  for n_bits in "${N_BITS_VALS[@]}"; do
    for cap_val in "${CAP_VALS[@]}"; do
      for name in "${MODEL_NAMES[@]}"; do
        printf '========== %s | method=%s | n_bits=%s | cap=%s ==========\n' \
               "$name" "$method" "$n_bits" "$cap_val" >>"$MASTER_LOG"
        logfile="$BASE_LOGDIR/${name}_method_${method}_tol_${tol}_nbits_${n_bits}_cap_${cap_val}/job.log"
        if ! grep -A 11 "Model name: ${name} " "$logfile" >>"$MASTER_LOG"; then
          echo "[Final Result not found] $logfile" >>"$MASTER_LOG"
        fi
        printf '\n\n' >>"$MASTER_LOG"
      done
    done
  done
  echo "All summaries written to $MASTER_LOG"
done
#done

#######################################################
# running
# nohup bash run_ode_inference.sh > logs/run_script_output/launcher.out 2>&1 &
# tail -f logs/run_script_output/launcher.out
# After the run is finished, the master_log file will be printed out
# then cat master_log
# then use shell_utils/parse_noise_logs.py to convert the log into csv
#######################################################

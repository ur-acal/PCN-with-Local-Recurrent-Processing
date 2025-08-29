#!/usr/bin/env bash
set -eu
trap '' HUP   # ignore hangup so the children survive

# ─────────────── fixed params ───────────────
export MODEL_DIR="./saved_ckpt"
export tol="1e-4"

# ─────────────── noise toggles ───────────────
METHOD_VALS=("dopri5")
#METHOD_VALS=("euler")

export BASE_LOGDIR="./logs/test_ode_noisy"
# MASTER_LOG and JOB_LOG will be set per noise combination

# ─────────────── model list ───────────────
MODEL_NAMES=(
#  "PCNetNoBatchNorm_PCConvHardTanh_5CLS_0.15LRPC_0.001WD_noTied_noBPtied_withRelu_noBP_noReluBP_withPC_128BS_0.01LR_0.25Dropout_7Layers_1REP"
#  "PCNetNoBatchNorm_PCConvHardTanhDyn_30CLS_0.06LRPC_0.001WD_noTied_noBPtied_withRelu_noBP_noReluBP_withPC_128BS_0.01LR_0.25Dropout_5Layers_1REP"
#  "PCNetNoBatchNorm_PCConvHardTanhDyn_dopri5Solver_1.0TEnd_0.001Tol_0.001WD_noBPtied_noBP_withPC_128BS_0.01LR_0.25Dropout_5Layers_1REP"
#  "PCNetNoBatchNorm_PCConvHardTanhDyn_dopri5Solver_1.0TEnd_0.001Tol_0.001WD_noBPtied_noBP_withPC_128BS_0.01LR_0.25Dropout_7Layers_1REP"
#  "PCNetNoBatchNorm_PCConvHardTanh2Dyn_dopri5Solver_0.75TEnd_0.0001Tol_0.001WD_noBPtied_noBP_withPC_128BS_0.01LR_0.25Dropout_7Layers_1REP"
#  "PCNetNoBatchNorm_PCConvHardTanh2Dyn_dopri5Solver_1.0TEnd_0.0001Tol_0.001WD_noBPtied_noBP_withPC_128BS_0.01LR_0.25Dropout_7Layers_1REP"

#  "PCNetNoBatchNorm_PCConvHardTanh_dopri5Solver_0.75TEnd_0.0001Tol_0.001WD_noBPtied_noBP_withPC_128BS_0.01LR_0.25Dropout_7Layers_1REP" # NODE baseline
#  "PCNetNoBatchNorm_PCConvHardTanh_dopri5Solver_1.0TEnd_0.0001Tol_0.001WD_noBPtied_noBP_withPC_128BS_0.01LR_0.25Dropout_7Layers_1REP" # NODE baseline

#  "PCNetNoBatchNorm_PCConvHardTanhWSFF_dopri5Solver_0.75TEnd_0.0001Tol_0.001WD_noBPtied_noBP_withPC_128BS_0.01LR_0.25Dropout_7Layers_1REP"
#  "PCNetNoBatchNorm_PCConvHardTanhWSFFFB_ODEBlockPCMinusY_dopri5Solver_0.75TEnd_0.0001Tol_0.001WD_noBPtied_noBP_128BS_0.01LR_0.25Dropout_7Layers_1REP"

#  "PCNetNoBatchNorm_PCConvHardTanh_dopri5Solver_0.75TEnd_0.001Tol_0.001WD_noBPtied_noBP_withPC_128BS_0.01LR_0.25Dropout_7Layers_2REP" # MinusY
#  "PCNetNoBatchNorm_PCConvHardTanh_ODEBlockPCMinusY_dopri5Solver_0.75TEnd_0.0001Tol_0.001WD_noBPtied_noBP_128BS_0.01LR_0.25Dropout_7Layers_1REP" # MinusY
#  "PCNetNoBatchNorm_PCConvReLU6_ODEBlockPCMinusY_dopri5Solver_0.75TEnd_0.0001Tol_0.001WD_noBPtied_noBP_128BS_0.01LR_0.25Dropout_7Layers_1REP" # MinusY
#  "PCNetNoBatchNorm_PCConvHardTanh_ODEBlockPCMinusY_dopri5Solver_0.75TEnd_0.001Tol_0.001WD_noBPtied_noBP_128BS_0.01LR_0.25Dropout_7Layers_1REP" # MinusY
#  "PCNetNoBatchNorm_PCConvReLU6_ODEBlockPCMinusY_dopri5Solver_0.75TEnd_0.001Tol_0.001WD_noBPtied_noBP_128BS_0.01LR_0.25Dropout_7Layers_1REP" # MinusY

#  "PCNetNoBatchNorm_PCConvHardTanh2_ODEBlkActInp_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_noBPtied_noBP_128BS_0.01LR_0.25Dropout_7Layers_1REP"
#  "PCNetNoBatchNorm_PCConvHardTanh_ODEBlkActInp_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_noBPtied_noBP_128BS_0.01LR_0.25Dropout_7Layers_1REP"
#  "PCNetNoBatchNorm_PCConvHardTanh2_ODEBlkActDyn_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_noBPtied_noBP_128BS_0.01LR_0.25Dropout_7Layers_1REP"
#  "PCNetNoBatchNorm_PCConvHardTanh2_ODEBlkActInp_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_noBPtied_noBP_128BS_0.01LR_0.25Dropout_7Layers_2REP"
#  "PCNetNoBatchNorm_PCConvHardTanh2_ODEBlkProj_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_noBPtied_noBP_128BS_0.01LR_0.25Dropout_7Layers_1REP"
#  "PCNetNoBatchNorm_PCConvReLU6_ODEBlkProj_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_noBPtied_noBP_128BS_0.01LR_0.25Dropout_7Layers_1REP"
#  "PCNetNoBatchNorm_PCConvReLU6_ODEActInpNoMinus_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_noBPtied_noBP_128BS_0.01LR_0.25Dropout_7Layers_1REP"
#  "PCNetNoBatchNorm_PCConvHardTanh_ODEActInpNoMinus_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_noBPtied_noBP_128BS_0.01LR_0.25Dropout_7Layers_1REP"
#  "PCNetNoBatchNorm_PCConvReLU20_ODEBlkProj_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_noBPtied_noBP_128BS_0.01LR_0.25Dropout_7Layers_1REP"
#  "PCNetNoBatchNorm_PCConvReLU6_ODEBlkProjInitY_eulerSolver_0.75TEnd_0.0001Tol_0.001WD_noBPtied_noBP_128BS_0.01LR_0.25Dropout_7Layers_1REP"
#  "PCNetNoBatchNorm_PCConvHardTanh_ODEBlkProjInitY_eulerSolver_0.75TEnd_0.0001Tol_0.001WD_noBPtied_noBP_128BS_0.01LR_0.25Dropout_7Layers_1REP"

#  "PCNetNoBatchNorm_PCConvReLU6_ODEActInpNoMinus_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_noBPtied_noBP_128BS_0.01LR_0.25Dropout_7Layers_1REP" # zero init
#  "PCNetNoBatchNorm_PCConvHardTanh2_ODEActInpNoMinus_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_noBPtied_noBP_128BS_0.01LR_0.25Dropout_7Layers_1REP" # zero init

#  "PCNetNoBatchNorm_PCConvReLU6_ODEActDynNoMinus_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_noBPtied_noBP_128BS_0.01LR_0.25Dropout_7Layers_1REP"

#  "PCNetNoBatchNorm_PCConvHardTanh_ODEBlockPCMinusY_dopri5Solver_0.75TEnd_0.001Tol_0.001WD_noBPtied_noBP_128BS_0.01LR_0.25Dropout_7Layers_2REP" # no max_g_norm
#  "PCNetNoBatchNorm_PCConvReLU6_ODEBlockPCMinusY_dopri5Solver_0.75TEnd_0.001Tol_0.001WD_noBPtied_noBP_128BS_0.01LR_0.25Dropout_7Layers_2REP" # no max_g_norm

#  "PCNetNoBatchNorm_PCConvHardTanh_ODESelfCoupleInitY_dopri5Solver_0.75TEnd_0.0001Tol_0.001WD_noBPtied_noBP_128BS_0.01LR_0.25Dropout_7Layers_1REP"
#  "PCNetNoBatchNorm_PCConvReLU6_ODESelfCoupleInitY_dopri5Solver_0.75TEnd_0.0001Tol_0.001WD_noBPtied_noBP_128BS_0.01LR_0.25Dropout_7Layers_1REP"

#  "PCNetNoBatchNorm_PCConvHardTanh_0.2eps_ODEFixNoiseOffset_dopri5Solver_0.75TEnd_0.0001Tol_0.001WD_noBPtied_noBP_128BS_0.01LR_0.25Dropout_7Layers_1REP"
#  "ftPCNetNoBatchNorm_PCConvHardTanh_0.4eps_ODEFixNoiseOffset_dopri5Solver_0.75TEnd_0.0001Tol_0.001WD_noBPtied_noBP_128BS_0.0001LR_0.25Dropout_7Layers_1REP"
#  "PCNetNoBatchNorm_PCConvReLU6_0.2eps_ODEFixNoiseOffset_dopri5Solver_0.75TEnd_0.0001Tol_0.001WD_noBPtied_noBP_128BS_0.01LR_0.25Dropout_7Layers_1REP"
#  "ftPCNetNoBatchNorm_PCConvReLU6_0.4eps_ODEFixNoiseOffset_dopri5Solver_0.75TEnd_0.0001Tol_0.001WD_noBPtied_noBP_128BS_0.0001LR_0.25Dropout_7Layers_1REP" # passive baseline
#  "PCNetNoBatchNorm_PCConvReLU6_0.2eps_ODEFixNoiseOffset_dopri5Solver_0.75TEnd_0.0001Tol_0.001WD_noBPtied_noBP_128BS_0.01LR_0.25Dropout_7Layers_2pool_2REP"
#  "ftPCNetNoBatchNorm_PCConvReLU6_0.4eps_ODEFixNoiseOffset_dopri5Solver_0.75TEnd_0.0001Tol_0.001WD_noBPtied_noBP_128BS_0.0001LR_0.25Dropout_7Layers_2REP"
#  "ftPCNetNoBatchNorm_PCConvReLU6_0.4eps_ODEFixNoiseOffset_dopri5Solver_0.75TEnd_0.0001Tol_0.001WD_noBPtied_noBP_128BS_0.0001LR_0.0Dropout_7Layers_2REP"
#  "ftPCNetNoBatchNorm_PCConvReLU6_0.4eps_ODEFixNoiseOffset_dopri5Solver_0.75TEnd_0.0001Tol_0.001WD_noBPtied_noBP_128BS_0.0001LR_0.25Dropout_7Layers_20epoch_2REP"

#  "PCNetNoBatchNorm_PCConvReLU6_0.2eps_ODEFixNoiseOffset_dopri5Solver_0.75TEnd_0.0001Tol_0.001WD_noBPtied_noBP_128BS_0.01LR_0.25Dropout_5Layers_1REP" # 5Layer - 64 chan
#  "ftPCNetNoBatchNorm_PCConvReLU6_0.4eps_ODEFixNoiseOffset_dopri5Solver_0.75TEnd_0.0001Tol_0.001WD_noBPtied_noBP_128BS_0.0001LR_0.25Dropout_5Layers_1REP" # 5Layer - 64 chan ft
#  "PCNetNoBatchNorm_PCConvReLU6_0.2eps_ODEFixNoiseOffset_dopri5Solver_0.75TEnd_0.0001Tol_0.001WD_noBPtied_noBP_128BS_0.01LR_0.25Dropout_4Layers_1REP" # 4Layer - 128 chan
#  "ftPCNetNoBatchNorm_PCConvReLU6_0.4eps_ODEFixNoiseOffset_dopri5Solver_0.75TEnd_0.0001Tol_0.001WD_noBPtied_noBP_128BS_0.0001LR_0.25Dropout_4Layers_1REP" # 4Layer - 128 chan ft

#  "PCNetNoBatchNorm_PCConv_0.2eps_ODEFixNoiseOffset_dopri5Solver_0.75TEnd_0.0001Tol_0.001WD_noBPtied_noBP_128BS_0.01LR_0.25Dropout_5Layers_1REP" # 5Layer - 128 chan - 3 pooling
#  "PCNetNoBatchNorm_PCConv_0.2eps_ODEFixNoiseOffset_dopri5Solver_0.75TEnd_0.0001Tol_0.001WD_noBPtied_noBP_128BS_0.01LR_0.25Dropout_7Layers_3Pool_1REP"
#  "PCNetNoBatchNorm_PCConvReLU6_0.2eps_ODEFixNoiseOffset_dopri5Solver_0.75TEnd_0.0001Tol_0.001WD_noBPtied_noBP_128BS_0.01LR_0.25Dropout_5Layers_3Pool_1REP"
#  "PCNetNoBatchNorm_PCConvReLU6_0.2eps_ODEFixNoiseOffset_dopri5Solver_0.75TEnd_0.0001Tol_0.001WD_noBPtied_noBP_128BS_0.01LR_0.25Dropout_7Layers_3Pool_1REP"

#  "PCNetNoBatchNorm_PCConvReLU6_0.2eps_ODEFixNoiseOffset_dopri5Solver_0.75TEnd_0.0001Tol_0.001WD_noBPtied_noBP_128BS_0.01LR_0.25Dropout_5Layers_2REP" # 3 avg pooling
#  "PCNetNoBatchNorm_PCConvReLU6_0.2eps_ODEFixNoiseOffset_dopri5Solver_0.75TEnd_0.0001Tol_0.001WD_noBPtied_noBP_128BS_0.01LR_0.25Dropout_5Layers_2Pool_1REP" # 2 max pooling
#  "PCNetNoBatchNorm_PCConvReLU6_0.2eps_ODEFixNoiseOffset_dopri5Solver_0.75TEnd_0.0001Tol_0.001WD_noBPtied_noBP_128BS_0.01LR_0.25Dropout_5Layers_3REP"
#  "PCNetNoBatchNorm_PCConvReLU6_0.2eps_ODEFixNoiseOffset_dopri5Solver_0.75TEnd_0.0001Tol_0.001WD_noBPtied_noBP_128BS_0.01LR_0.25Dropout_5Layers_4REP"

#  "PCNetNoBatchNorm_PCConvReLU6_0.2eps_ODEFixNoiseOffset_dopri5Solver_0.75TEnd_0.0001Tol_0.001WD_noBPtied_noBP_128BS_0.01LR_0.25Dropout_6Layers_1REP" # 3 max pooling
#  "ftPCNetNoBatchNorm_PCConvReLU6_0.4eps_ODEFixNoiseOffset_dopri5Solver_0.75TEnd_0.0001Tol_0.001WD_noBPtied_noBP_128BS_0.0001LR_0.25Dropout_6Layers_1REP" # 3 max pooling BASELINE
#  "ftPCNetNoBatchNorm_PCConvReLU6_0.4eps_ODEFixNoiseOffset_dopri5Solver_0.75TEnd_0.0001Tol_0.001WD_noBPtied_noBP_128BS_0.0001LR_0.0Dropout_6Layers_1REP" # 3 max pooling
#  "ftPCNetNoBatchNorm_PCConvReLU6_0.4eps_ODEFixNoiseOffset_dopri5Solver_0.75TEnd_0.0001Tol_0.001WD_noBPtied_noBP_128BS_0.0001LR_0.25Dropout_6Layers_2REP"
#  "ftPCNetNoBatchNorm_PCConvReLU6_0.4eps_ODEFixNoiseOffset_dopri5Solver_0.75TEnd_0.0001Tol_0.001WD_noBPtied_noBP_128BS_0.0001LR_0.25Dropout_5Layers_3REP" # 2 max pooling
#  "PCNetNoBatchNorm_PCConvReLU6_0.2eps_ODEFixNoiseOffset_dopri5Solver_0.75TEnd_0.0001Tol_0.001WD_noBPtied_noBP_128BS_0.01LR_0.25Dropout_7Layers_2REP" # 3 max pooling
#  "ftPCNetNoBatchNorm_PCConvReLU6_0.4eps_ODEFixNoiseOffset_dopri5Solver_0.75TEnd_0.0001Tol_0.001WD_noBPtied_noBP_128BS_0.0001LR_0.0Dropout_7Layers_1REP"

#  "PCNetWith1stConv_PCConvReLU6_0.1eps_ODEFixNoiseXInit_dopri5Solver_0.75TEnd_0.0001Tol_0.001WD_noBPtied_noBP_128BS_0.01LR_0.25Dropout_6Layers_1REP" # 3p 6l
#  "PCNetWith1stConv_PCConvReLU6_0.2eps_ODEFixNoiseXInitFFFB_dopri5Solver_0.75TEnd_0.0001Tol_0.001WD_noBPtied_noBP_128BS_0.01LR_0.25Dropout_16Layers_1REP" # 2p
#  "ftPCNetWith1stConv_PCConvReLU6_0.4eps_FixNoiseXInitFFFBNoExpand_dopri5Solver_0.75TEnd_0.0001Tol_0.001WD_noBPtied_noBP_128BS_0.0001LR_0.25Dropout_8Layers_1REP"
#  "PCNetWith1stConv_PCConvReLU6_0.1eps_ODEFixNoiseXInit_dopri5Solver_0.75TEnd_0.0001Tol_0.001WD_noBPtied_noBP_128BS_0.01LR_0.25Dropout_6Layers_2REP" # 3 first_ksz, 3p

#  "PCNetNoBatchNorm_PCConvReLU6_0.2eps_ODEFixNoiseXInitFFFB_dopri5Solver_0.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_0.25Dropout_6Layers_1REP" # 3p-wide
#  "PCNetNoBatchNorm_PCConvReLU6_0.2eps_ODEFixNoiseXInitFFFB_dopri5Solver_0.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_0.25Dropout_6Layers_2REP" # 2p-wide
#  "PCNetNoBatchNorm_PCConvReLU6_0.2eps_ODEFixNoiseXInitFFFB_dopri5Solver_0.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_0.25Dropout_18Layers_1REP" # 2p-deep

#  "ftPCNetNoBatchNorm_PCConvReLU6_0.4eps_FixNoiseXInitFFFBNoExpand_dopri5Solver_0.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_0.25Dropout_6Layers_2REP" # 2p-wide-ft

#  "PCNetNoBatchNorm_PCConvReLU6_0.2eps_ODEState2FFFB_dopri5Solver_0.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_0.25Dropout_6Layers_3Pool_1REP"
  "PCNetNoBatchNorm_PCConvReLU6_0.2eps_State2InitYAsXZAsX_dopri5Solver_0.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_0.25Dropout_6Layers_3Pool_1REP"
)

# ─────────────── prepare logs ───────────────
mkdir -p "$BASE_LOGDIR"
for method in "${METHOD_VALS[@]}"; do
  for name in "${MODEL_NAMES[@]}"; do
    mkdir -p "$BASE_LOGDIR/${name}_method_${method}_tol_${tol}"
    > "$BASE_LOGDIR/${name}_method_${method}_tol_${tol}/job.log"
  done
done

# ─────────────── helper function ───────────────
run_model(){
  local name="$1"
  local method="$2"

  local __rest="${name#*_}"
  local _pc_conv="${__rest%%_*}"
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
    --pc_conv         "${_pc_conv}Noisy" \
    --ode_block       "State2InitYAsXZAsX" \
    --test_only       "true" \
    2>&1 | tee -a "$BASE_LOGDIR/${name}_method_${method}_tol_${tol}/job.log"
}
export -f run_model

# ─────────────── loop over methods ───────────────
for method in "${METHOD_VALS[@]}"; do
  ##########################################################################################
  # Modify log name here before each run
  ##########################################################################################
  EXP_NAME="0820_2pooling_wide_SumAsBInitYAsXFFFB_${method}Method_${tol}Tol.log"
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
    run_model {1} "${method}" \
    ::: "${MODEL_NAMES[@]}"
  echo "All jobs finished — merging logs into $MASTER_LOG"
  # ─────────────── merge logs sequentially ───────────────
  : >"$MASTER_LOG"
  for name in "${MODEL_NAMES[@]}"; do
    printf '========== %s ==========\n' "$name" >>"$MASTER_LOG"
    if ! grep -A 11 "Model name: ${name} " \
              "$BASE_LOGDIR/${name}_method_${method}_tol_${tol}/job.log" >>"$MASTER_LOG"; then
      echo "[Final Result not found]" >>"$MASTER_LOG"
    fi
    printf '\n\n' >>"$MASTER_LOG"
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

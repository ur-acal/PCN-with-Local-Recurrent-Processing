#!/bin/bash -l
#SBATCH -N 1
#SBATCH -p ising
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16            # modest CPU request so the node can be shared
#SBATCH --gres=gpu:1                  # exactly ONE GPU; allows packing on 4-GPU nodes
#SBATCH -t 90:10:00
#SBATCH -o /scratch/rzeng7/repos/PCN-with-Local-Recurrent-Processing/logs/slurm_jobs/slurm_%j.out

#set -eu
#trap '' HUP   # ignore hangup so the children survive

IS_SLURM="${IS_SLURM:-0}"

if [[ "${IS_SLURM}" == 1 ]]; then
  source activate base
  conda activate scanbase
fi

# ─────────────── fixed params ───────────────
export MODEL_DIR="./saved_ckpt"
export tol="1e-4"

# ─────────────── noise toggles ───────────────
#METHOD_VALS=("dopri5")
METHOD_VALS=("euler")

export BASE_LOGDIR="./logs/test_ode_noisy"
# MASTER_LOG and JOB_LOG will be set per noise combination

# ─────────────── model list ───────────────
MODEL_NAMES_STR="${MODEL_NAMES_STR:-}"
JOB_TAG="${JOB_TAG:-0}"

if [[ -n "${MODEL_NAMES_STR}" ]]; then
  readarray -t MODEL_NAMES <<< "${MODEL_NAMES_STR}"
else
  MODEL_NAMES=(
#  "TIMMPCNetWith1stConv_PCConvReLU6_0.0eps_ODEXInitFFFB_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_C100_3K1S256C_0.25Dropout_8Layers1l1l3_2Pool_1REP"
#  "TIMMPCNetNoBatchNorm_PCConvReLU6_0.0eps_ODEXInitFFFB_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_C100_3K1S256C_0.25Dropout_8Layers1l1l3_2Pool_1REP"
#
#  "TIMMPCNetNoBatchNorm_PCConv_0.0eps_ODEXInitFFFB_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_C100_3K1S256C_0.25Dropout_8Layers1l1l3_2Pool_1REP"
#  "TIMMPCNetWith1stConv_PCConv_0.0eps_ODEXInitFFFB_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_C100_3K1S256C_0.25Dropout_8Layers1l1l3_2Pool_1REP"
#  "TIMMPCNetNoBatchNorm_PCConvReLU6_0.0eps_ODEXInitFFFB_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_C100_3K1S256C_0.25Dropout_8Layers1l1l3_2Pool_6REP"
#  "TIMMPCNetWith1stConv_PCConvReLU6_0.0eps_ODEXInitFFFB_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_C100_3K1S256C_0.25Dropout_8Layers1l1l3_2Pool_6REP"

#  "TIMMPCNetNoBatchNorm_PCConv_0.0eps_ODEXInitFFFB_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_3K1S64C_0.25Dropout_5Layers3l0l0_2Pool_1REP"
#  "TIMMPCNetWith1stConv_PCConv_0.0eps_ODEXInitFFFB_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_3K1S64C_0.25Dropout_5Layers3l0l0_2Pool_1REP"
#  "TIMMPCNetNoBatchNorm_PCConv_0.0eps_ODEXInitFFFB_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_3K1S128C_0.25Dropout_6Layers1l1l1_2Pool_1REP"
#  "TIMMPCNetWith1stConv_PCConv_0.0eps_ODEXInitFFFB_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_3K1S128C_0.25Dropout_6Layers1l1l1_2Pool_1REP"
  "TIMMPCNetNoBatchNorm_PCConv_0.0eps_ODEXInitFFFB_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_3K1S128C_0.25Dropout_7Layers1l1l2_2Pool_1REP"
#  "TIMMPCNetWith1stConv_PCConv_0.0eps_ODEXInitFFFB_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_3K1S128C_0.25Dropout_7Layers1l1l2_2Pool_1REP"
#  "TIMMPCNetNoBatchNorm_PCConv_0.0eps_ODEXInitFFFB_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_C100_3K1S64C_0.25Dropout_5Layers3l0l0_2Pool_1REP"
#  "TIMMPCNetWith1stConv_PCConv_0.0eps_ODEXInitFFFB_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_C100_3K1S64C_0.25Dropout_5Layers3l0l0_2Pool_1REP"
#  "TIMMPCNetNoBatchNorm_PCConv_0.0eps_ODEXInitFFFB_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_C100_3K1S128C_0.25Dropout_6Layers1l1l1_2Pool_1REP"
#  "TIMMPCNetWith1stConv_PCConv_0.0eps_ODEXInitFFFB_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_C100_3K1S128C_0.25Dropout_6Layers1l1l1_2Pool_1REP"
#  "TIMMPCNetNoBatchNorm_PCConv_0.0eps_ODEXInitFFFB_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_C100_3K1S128C_0.25Dropout_7Layers1l1l2_2Pool_1REP"
#  "TIMMPCNetWith1stConv_PCConv_0.0eps_ODEXInitFFFB_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_C100_3K1S128C_0.25Dropout_7Layers1l1l2_2Pool_1REP"

#  "PCNetNoBatchNorm_PCConvReLU6_0.0eps_ODEXInitFFFB_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_C100_3K1S256C_0.25Dropout_8Layers1l1l3_2Pool_1REP"
  )
fi

declare -A MISMATCH_DICT
MISMATCH_DICT["mul"]="0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4"
#MISMATCH_DICT["add"]="0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1"

# ─────────────── prepare logs ───────────────
mkdir -p "$BASE_LOGDIR"
for method in "${METHOD_VALS[@]}"; do
  for mismatch_type in "${!MISMATCH_DICT[@]}"; do
    for name in "${MODEL_NAMES[@]}"; do
      mkdir -p "$BASE_LOGDIR/${name}_method_${method}_mismatch_${mismatch_type}_tol_${tol}"
      > "$BASE_LOGDIR/${name}_method_${method}_mismatch_${mismatch_type}_tol_${tol}/job.log"
    done
  done
done

# ─────────────── helper function ───────────────
run_model(){
  local name="$1"
  local method="$2"
  local mismatch_type="$3"
  local noise_level_list="$4"

  if [[ "$name" == *C100* ]]; then
    local _task="cifar100"
  else
    local _task="cifar10"
  fi

  local __rest="${name#*_}"
  local _pc_conv="${__rest%%_*}"
  ########################################
  # only fuse_bn when noise is added to bn
  ########################################
  set -o pipefail
  python -u ode_inference.py \
    --model_name        "$name" \
    --ckpt              "best" \
    --task            "${_task}" \
    --img_type          "rgb" \
    --model_dir         "$MODEL_DIR" \
    --method            "$method" \
    --tol               "$tol" \
    --n_steps           15 \
    --ts_scale          1 \
    --d_start           0 \
    --d_end             1 \
    --n_sweep_left      0 \
    --n_sweep_right     1 \
    --thermal_noise     "false" \
    --mismatch_type     "${mismatch_type}" \
    --noise_level_list  "${noise_level_list}" \
    --pc_conv           "${_pc_conv}Noisy" \
    --ode_block         "ODEXInitFFFB" \
    --test_only         "true" \
    --analyze_mm        "false" \
    2>&1 | tee -a "$BASE_LOGDIR/${name}_method_${method}_mismatch_${mismatch_type}_tol_${tol}/job.log"
}
export -f run_model

# ─────────────── loop over methods ───────────────
for method in "${METHOD_VALS[@]}"; do
  for mismatch_type in "${!MISMATCH_DICT[@]}"; do
    noise_level_list="${MISMATCH_DICT[$mismatch_type]}"
    ##########################################################################################
    # Modify log name here before each run
    ##########################################################################################
    EXP_NAME="0520_rgb_SML_${JOB_TAG}_${method}Method_${mismatch_type}Mismatch_${tol}Tol.log"
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
      run_model {1} "${method}" "${mismatch_type}" "${noise_level_list}" \
      ::: "${MODEL_NAMES[@]}"
    echo "All jobs finished — merging logs into $MASTER_LOG"
    # ─────────────── merge logs sequentially ───────────────
    : >"$MASTER_LOG"
    for name in "${MODEL_NAMES[@]}"; do
      printf '========== %s ==========\n' "$name" >>"$MASTER_LOG"
      if ! grep -A 50 "Model name: ${name} " \
                "$BASE_LOGDIR/${name}_method_${method}_mismatch_${mismatch_type}_tol_${tol}/job.log" >>"$MASTER_LOG"; then
        echo "[Final Result not found]" >>"$MASTER_LOG"
      fi
      printf '\n\n' >>"$MASTER_LOG"
    done
    echo "All summaries written to $MASTER_LOG"
  done
done

#######################################################
# running
# nohup bash run_ode_inference.sh > logs/run_script_output/launcher.out 2>&1 &
# tail -f logs/run_script_output/launcher.out
# After the run is finished, the master_log file will be printed out
# then cat master_log
# then use shell_utils/parse_noise_logs.py to convert the log into csv
#######################################################

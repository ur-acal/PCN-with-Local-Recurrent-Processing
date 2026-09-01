#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PATH="/home/rongzeng/anaconda3/envs/scanbase/bin:$PATH"

MODEL_DIR="$ROOT/saved_ckpt"
LEGACY_MODEL_DIR="$ROOT/../ScAN-PCN/saved_ckpt"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/logs/pcn_wrn_ff_gain_extended_cifar100_wrn28_2}"
RAW_ROOT="$OUTPUT_ROOT/raw"
GAINS="${GAINS:-0.5,0.6,0.7,0.8,1.2,1.4,1.6,1.8,2.0}"
REPORT_GAINS="0.5,0.6,0.7,0.8,1.0,1.2,1.4,1.6,1.8,2.0"

LEGACY="TIMMPCNetNoBatchNorm_PCConv_0.0eps_ODEXInitFFFB_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_C100_3K1S128C_0.25Dropout_13Layers0l3l6_2Pool_1REP"
ODEBLOCK_PC="TIMMPCNetNoBatchNorm_PCConv_0.0eps_ODEBlockPC_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_C100_3K1S128C_0.25Dropout_13Layers0l3l6_2Pool_1REP"
ODEBLOCK_XINIT="TIMMPCNetNoBatchNorm_PCConv_0.0eps_ODEBlockXInit_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_C100_3K1S128C_0.25Dropout_13Layers0l3l6_2Pool_1REP"

run_model() {
  local model_name="$1" model_dir="$2" result_tag="$3" ode_block="$4"
  MODEL_NAME="$model_name" \
  REPO_ROOT="$ROOT" \
  MODEL_DIR="$model_dir" \
  MODEL_INDEX=6 \
  ARCHITECTURE=WRN_28_2 \
  RESULT_TAG="$result_tag" \
  ODE_BLOCK="$ode_block" \
  PC_CONV=PCConvNoisy \
  OUTPUT_ROOT="$RAW_ROOT" \
  EVAL_MODE=gain \
  FF_GAIN_LIST="$GAINS" \
  bash launch_scripts/run_rgb_ode_mismatch_eval.sh
}

mkdir -p "$OUTPUT_ROOT"
run_model "$LEGACY" "$LEGACY_MODEL_DIR" PCN_legacy_no_x ODEXInitFFFB &
legacy_pid=$!
run_model "$ODEBLOCK_PC" "$MODEL_DIR" PCN_ODEBlockPC_with_x ODEBlockPC &
pc_pid=$!
run_model "$ODEBLOCK_XINIT" "$MODEL_DIR" PCN_ODEBlockXInit_with_x ODEBlockXInit &
xinit_pid=$!

wait "$legacy_pid"
wait "$pc_pid"
wait "$xinit_pid"

python_bin="${PYTHON_BIN:-/home/rongzeng/anaconda3/envs/scanbase/bin/python}"
"$python_bin" baseline/generate_pcn_wrn_ff_gain_summary.py \
  --legacy_root "$RAW_ROOT/cifar100/PCN_legacy_no_x/ff_gain" \
  --legacy_reference_root "$ROOT/logs/pcn_legacy_no_x_mismatch_slurm/cifar100/WRN_28_2_legacy_no_x/ff_gain" \
  --odeblockpc_root "$RAW_ROOT/cifar100/PCN_ODEBlockPC_with_x/ff_gain" \
  --odeblockpc_reference_root "$ROOT/logs/pcn_x_minus_fb_mismatch_slurm/cifar100/WRN_28_2_ODEBlockPC/ff_gain" \
  --odeblockxinit_root "$RAW_ROOT/cifar100/PCN_ODEBlockXInit_with_x/ff_gain" \
  --odeblockxinit_reference_root "$ROOT/logs/pcn_x_minus_fb_mismatch_slurm/cifar100/WRN_28_2_ODEBlockXInit/ff_gain" \
  --wrn_csv "$ROOT/logs/wrn_wd1e3_finaldrop025_ff_gain_cifar100_wrn28_2_extended/full_aggregate.csv" \
  --wrn_reference_csv "$ROOT/logs/wrn_wd1e3_finaldrop025_ff_gain_cifar100_wrn28_2/full_aggregate.csv" \
  --gains "$REPORT_GAINS" \
  --output_csv "$OUTPUT_ROOT/full_aggregate.csv" \
  --output_md "$OUTPUT_ROOT/summary.md"

echo "Completed extended PCN/WRN FF-gain comparison: $OUTPUT_ROOT"

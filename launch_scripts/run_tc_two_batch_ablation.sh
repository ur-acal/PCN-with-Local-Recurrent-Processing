#!/usr/bin/env bash
# Seven independent unrolled TC cases, identical data seed and first two batches.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PATH=/home/rongzeng/anaconda3/envs/scanbase/bin:$PATH
export MODEL_NAME=TIMMQAT5b8aNT0p25mulTIMMPCNetNoBatchNorm_PCConvReLU6_0.0eps_ODEXInitFFFB_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_C100_3K1S96C_0.25Dropout_16Layers4l5l4_2Pool_srrlDistill_a0p3_t2p0_scanGFI_1REP
export MODEL_DIR=./saved_ckpt CKPT=best TASK=cifar100 IMG_TYPE=scanGFI
export TC_MAX_EVAL_BATCHES=2 TEST_BS=128 N_TRIALS=1 OMP_NUM_THREADS=1
export TC_DRY_RUN=false TOGGLE_MODE=none SWITCH_INF=false TC_STATE=1
export DATA_SEED=4096 HARDWARE_SEED=4096 SPIN_VARIATION_SEED=4096
export NONLINEAR_R_CURVE_SEED=4096 SUMMING_NOISE_SEED=4096 COUPLER_NOISE_SEED=4096 ACTIVATION_CURVE_SEED=4096
export R_VAL=10e3 R_MAX=150e3 C_VAL=49e-15 V_DD=0.1 TOGGLE_ONE_OVER_Q=1
export SUMMING_CURRENT_P=0.6e-12 COUPLER_NOISE_P=0.6e-12 SIGMA_SPIN=0.1
export ACTIVATION_CURVE_PATH=./hardware_data/mc_45_corners/0906_RELU_Voltage ACTIVATION_CORNER=TT_25_1_MC18
root=./results/tc_two_batch_ablation_8a_0906
mkdir -p "$root"
for case_name in clean relu pooling nonlinear_R spin summing_noise coupler_noise; do
  export ENABLE_SPIN_VARIATION=false ENABLE_SUMMING_CURRENT_NOISE=false ENABLE_COUPLER_NOISE=false
  export ENABLE_NONLINEAR_R=false ENABLE_MEASURED_POOLING=false ENABLE_MEASURED_ACTIVATION=false
  case "$case_name" in
    relu) export ENABLE_MEASURED_ACTIVATION=true ;;
    pooling) export ENABLE_MEASURED_POOLING=true ;;
    nonlinear_R) export ENABLE_NONLINEAR_R=true ;;
    spin) export ENABLE_SPIN_VARIATION=true ;;
    summing_noise) export ENABLE_SUMMING_CURRENT_NOISE=true ;;
    coupler_noise) export ENABLE_COUPLER_NOISE=true ;;
  esac
  export TC_METADATA_PATH="$root/${case_name}.jsonl"
  echo "Starting $case_name"
  bash ./launch_scripts/run_tc_nonidealities.sh eval > "$root/${case_name}.log" 2>&1
  echo "Finished $case_name"
done
echo 'All seven two-batch cases completed.'

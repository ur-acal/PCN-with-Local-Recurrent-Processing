#!/usr/bin/env bash
# Two-batch unrolled TC nonlinear-R-only comparison. No model/default changes.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PATH=/home/rongzeng/anaconda3/envs/scanbase/bin:$PATH
export MODEL_NAME=TIMMQAT5b8aNT0p25mulTIMMPCNetNoBatchNorm_PCConvReLU6_0.0eps_ODEXInitFFFB_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_C100_3K1S96C_0.25Dropout_16Layers4l5l4_2Pool_srrlDistill_a0p3_t2p0_scanGFI_1REP
export MODEL_DIR=./saved_ckpt CKPT=best TASK=cifar100 IMG_TYPE=scanGFI
export TC_MAX_EVAL_BATCHES=2 TEST_BS=128 N_TRIALS=1 OMP_NUM_THREADS=1
export TC_DRY_RUN=false TOGGLE_MODE=none SWITCH_INF=false TC_STATE=1 TC_CONV_METHOD=loop
export DATA_SEED=4096 HARDWARE_SEED=4096 NONLINEAR_R_CURVE_SEED=4096
export SPIN_VARIATION_SEED=4096 SUMMING_NOISE_SEED=4096 COUPLER_NOISE_SEED=4096 ACTIVATION_CURVE_SEED=4096
export R_VAL=10e3 R_MAX=150e3 C_VAL=49e-15 V_DD=0.1 TOGGLE_ONE_OVER_Q=1
export SUMMING_CURRENT_P=0.6e-12 COUPLER_NOISE_P=0.6e-12 SIGMA_SPIN=0.1
export ACTIVATION_CURVE_PATH=./hardware_data/mc_45_corners/0906_RELU_Voltage ACTIVATION_CORNER=TT_25_1_MC18
export ENABLE_SPIN_VARIATION=false ENABLE_SUMMING_CURRENT_NOISE=false ENABLE_COUPLER_NOISE=false
export ENABLE_NONLINEAR_R=true ENABLE_MEASURED_POOLING=false ENABLE_MEASURED_ACTIVATION=false
export TC_MEAN_TABLE=./hardware_data/res_vs_vin_10k_150k.csv
root=./results/tc_covariance_v2_comparison
mkdir -p "$root"
for name in current v2; do
  if [[ "$name" == current ]]; then
    export TC_COVARIANCE_TABLE="$root/current_all4500_common_grid.csv"
  else
    export TC_COVARIANCE_TABLE="$root/coupler_monte_v2_all4500_resistance.csv"
  fi
  export TC_METADATA_PATH="$root/${name}.jsonl"
  echo "Starting $name covariance (all 4500 curves; common voltage grid)"
  bash launch_scripts/run_tc_nonidealities.sh eval > "$root/${name}.log" 2>&1
  echo "Finished $name covariance"
done

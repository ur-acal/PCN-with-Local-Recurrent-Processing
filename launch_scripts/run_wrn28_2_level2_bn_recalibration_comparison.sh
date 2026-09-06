#!/bin/bash

# Five paired Level-2 hardware realizations without/with BN recalibration.
# Both groups use identical base seeds; trial i uses BASE_SEED + i.
set -e

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO_ROOT}"

if [[ "${CONDA_DEFAULT_ENV:-}" != "scanbase" ]]; then
  eval "$(conda shell.bash hook)"
  conda activate scanbase
fi

MODEL_NAME="wrn_28_2_cifar_avgpool"
MODEL_CKPT="saved_ckpt_runs/coupler_full_range_CiFAIR100_wrn_28_2_cifar_avgpool_qf1_noENOB_fixedTiming_scaledRecipe1_ft_fixed_retry_iq12/cifar100/custom_noresize/wrn_28_2_cifar_avgpool/custom_noresize_cifar100_wrn_28_2_cifar_avgpool/custom_noresize_cifar100_wrn_28_2_cifar_avgpool_full_param_best_ckpt.pth"
BASE_SEED="${BASE_SEED:-20260904}"

run_group() (
  local recalibrate="$1"
  local result_path="$2"
  export MODEL_NAME MODEL_CKPT BASE_SEED \
         TASK=cifar100 \
         IMG_TYPE=CiFAIR \
         RESULT_PATH="${result_path}" \
         N_TRIALS=5 \
         PHYSICAL_LEVEL=2 \
         R_VAL=50e3 \
         C_VAL=500e-15 \
         V_DD=0.5 \
         ONE_OVER_Q=5 \
         TOGGLE_TIMING_MODE=fixed \
         TOGGLE_Y_TIME=5e-9 \
         Z_OVER_Y_TIME=1 \
         WEIGHT_QUANT_FACTOR_BITS=1 \
         ENOB=none \
         INPUT_QUANT_BITS=12 \
         CENTER_STUDENT_INPUT=false \
         USE_EXPANDED_WEIGHTS=false \
         ENABLE_NONLINEAR_R=true \
         NONLINEAR_R_TABLE=hardware_data/mc_45_corners/coupler_full_range \
         NONLINEAR_R_TRAIN_MODE=exact_curve \
         NONLINEAR_R_CORNER_RANGE=all \
         MC_COUPLER_NONLINEAR_VARIATION_QUANTITY=conductance \
         ENABLE_MEASURED_ACTIVATION=true \
         ACTIVATION_CORNER=TT \
         ACTIVATION_CURVE_SHARING=per_model \
         ENABLE_MEASURED_POOLING=true \
         ENABLE_SPIN_VARIATION=true \
         SIGMA_SPIN=0.10 \
         SPIN_VARIATION_MEAN=1.0 \
         ENABLE_SUMMING_CURRENT_NOISE=true \
         SUMMING_CURRENT_P=0.6e-12 \
         ENABLE_COUPLER_NOISE=true \
         COUPLER_NOISE_P=0.6e-12 \
         ENABLE_SLOW_SUMMING_CURRENT=false \
         ENABLE_SLOW_COUPLER_NOISE=false \
         ENABLE_DTC_NONIDEALITY=false \
         ENABLE_BN_RECALIBRATION="${recalibrate}" \
         SPIN_VARIATION_SEED="${BASE_SEED}" \
         SUMMING_NOISE_SEED="${BASE_SEED}" \
         COUPLER_NOISE_SEED="${BASE_SEED}" \
         ACTIVATION_CURVE_SEED="${BASE_SEED}" \
         NONLINEAR_R_CURVE_SEED="${BASE_SEED}" \
         DATA_SEED="${BASE_SEED}"
  ./launch_scripts/run_feedforward_physical_eval.sh
)

echo "[1/2] Five trials without BN recalibration"
run_group false results/wrn28_2_cifair100_level2_bn_paired5/no_recalibration

echo "[2/2] The same five seeds with BN recalibration"
run_group true results/wrn28_2_cifair100_level2_bn_paired5/with_recalibration

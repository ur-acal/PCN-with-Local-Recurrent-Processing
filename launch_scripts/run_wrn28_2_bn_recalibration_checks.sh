#!/bin/bash

# Reproduce the WRN training-time Level-2 evaluation once, then run the
# PCN-comparable SS_V2_T2 Level-3 corner twice. BN is recalibrated from
# augmentation-free CiFAIR-100 training data after trial-fixed nonidealities
# are sampled, with dynamic hardware noise enabled throughout calibration.
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

echo "[1/2] Level-2 training-protocol evaluation with BN recalibration"
(
  export MODEL_NAME MODEL_CKPT BASE_SEED \
         TASK=cifar100 \
         IMG_TYPE=CiFAIR \
         RESULT_PATH=results/wrn28_2_cifair100_bn_recal_level2_training_protocol \
         N_TRIALS=1 \
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
         ENABLE_BN_RECALIBRATION=true \
         SPIN_VARIATION_SEED="${BASE_SEED}" \
         SUMMING_NOISE_SEED="${BASE_SEED}" \
         COUPLER_NOISE_SEED="${BASE_SEED}" \
         ACTIVATION_CURVE_SEED="${BASE_SEED}" \
         NONLINEAR_R_CURVE_SEED="${BASE_SEED}" \
         DATA_SEED="${BASE_SEED}"
  ./launch_scripts/run_feedforward_physical_eval.sh
)

echo "[2/2] Level-3 SS_V2_T2 evaluation (2 trials) with BN recalibration"
(
  export MODEL_NAME MODEL_CKPT BASE_SEED \
         TASK=cifar100 \
         IMG_TYPE=CiFAIR \
         OUTPUT_DIR=results/wrn28_2_cifair100_bn_recal_level3_SS_V2_T2 \
         CORNER_IDS=SS_V2_T2 \
         N_TRIALS=2 \
         JOBS=1 \
         PHYSICAL_LEVEL=3 \
         FULL_45_CORNER_C=500e-15 \
         V_DD=0.5 \
         ONE_OVER_Q=5 \
         TOGGLE_TIMING_MODE=fixed \
         TOGGLE_Y_TIME=5e-9 \
         Z_OVER_Y_TIME=1 \
         WEIGHT_QUANT_FACTOR_BITS=1 \
         ENOB=none \
         INPUT_QUANT_BITS=12 \
         CENTER_STUDENT_INPUT=false \
         MC_COUPLER_NONLINEAR_VARIATION_SOURCE=coupler_monte_v2 \
         MC_COUPLER_NONLINEAR_VARIATION_QUANTITY=conductance \
         MC_COUPLER_NOMINAL_R=50e3 \
         NONLINEAR_R_CURVE_SHARING=per_coupler \
         NONLINEAR_R_CURVE_SAMPLING=empirical_with_replacement \
         ACTIVATION_CURVE_SHARING=per_spin \
         ENABLE_MEASURED_POOLING=true \
         ENABLE_SLOW_SUMMING_CURRENT=false \
         ENABLE_SLOW_COUPLER_NOISE=false \
         ENABLE_BN_RECALIBRATION=true
  ./launch_scripts/run_feedforward_mc45_ablation.sh
)

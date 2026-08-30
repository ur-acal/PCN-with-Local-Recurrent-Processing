#!/bin/bash
#SBATCH -p ising
#SBATCH -c 16
#SBATCH --gres=gpu:1
#SBATCH -t 90:10:00
#SBATCH -o /scratch/rzeng7/repos/PCN-with-Local-Recurrent-Processing/logs/slurm_jobs/slurm_%j.out

################################################################################################
#sbatch -N 1 \
#  --export=ALL,RUN_TAG=nonlinearR_mean_curve_MT,ACTIVATION_CORNER_MODE=fixed,R_VAL=67e3,MISMATCH_LEVEL=0.25,NONLINEAR_R_TRAIN_MODE=mean,ENABLE_NONLINEAR_R=true \
#  ./launch_scripts/run_slurm_toggle_ode_mixed_ft.sh

# Or launch via constructing a full export string
#sbatch_exports="ALL"
#sbatch_exports+=",RUN_TAG=coupler_v2_cifar10"
#sbatch_exports+=",NONLINEAR_R_TABLE=coupler_monte_v2"
#sbatch_exports+=",R_VAL=50e3"
#sbatch_exports+=",ENABLE_MEASURED_POOLING=true"
#sbatch_exports+=",MODEL_DIR=./saved_ckpt_runs/0804_kd_crd_ft_ToggleODEXInitFFFB_toggle_odexinit/"
#sbatch_exports+=",MODEL_NAME_OVERRIDE=TIMMPCNetNoBatchNorm_PCConvReLU6_0.0eps_ToggleODEXInitFFFB_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_3K1S96C_0.25Dropout_16Layers4l5l4_2Pool_srrlDistill_a0p3_t2p0_scanGFI_1REP"
#sbatch -N 1 \
#  --export="${sbatch_exports}" \
#  ./launch_scripts/run_slurm_toggle_ode_mixed_ft.sh
################################################################################################

source activate base
conda activate scanbase

export MODEL_NAME_OVERRIDE="${MODEL_NAME_OVERRIDE:-TIMMPCNetNoBatchNorm_PCConvReLU6_0.0eps_ODEXInitFFFB_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_C100_3K1S96C_0.25Dropout_16Layers4l5l4_2Pool_srrlDistill_a0p3_t2p0_scanGFI_6REP}"
if [[ -z "${IMG_TYPE:-}" ]]; then
  if [[ "${MODEL_NAME_OVERRIDE,,}" == *cifair* ]]; then
    export IMG_TYPE="CiFAIR"
  else
    export IMG_TYPE="scanGFI"
  fi
else
  case "${IMG_TYPE,,}" in
    cifair) IMG_TYPE="CiFAIR" ;;
    scangfi|raw|_raw) IMG_TYPE="scanGFI" ;;
  esac
  export IMG_TYPE
fi
export MODEL_DIR="${MODEL_DIR:-./saved_ckpt}"
export PCN="${PCN:-PCNetNoBatchNorm}"
export PULSE_MISMATCH_TRAINING_MODE="${PULSE_MISMATCH_TRAINING_MODE:-post_quant_amplitude}"
export WEIGHT_QUANT_FACTOR_BITS="${WEIGHT_QUANT_FACTOR_BITS:-none}"
export MISMATCH_LEVEL="${MISMATCH_LEVEL:-0.0}"
export RUN_TAG="${RUN_TAG:?RUN_TAG must be set}"
export EXP_OVERRIDE="pulse_qat_mt_${RUN_TAG}"
export OUTPUT_SAVE_PATH="./saved_ckpt_runs/${RUN_TAG}"

export ODE_BLOCK_OVERRIDE="ToggleODEXInitFFFB"
export ODEXINIT_SCALING_MODE="${ODEXINIT_SCALING_MODE:-direct}"
export TOGGLE_N_CYCLES="${TOGGLE_N_CYCLES:-5}"
export TOGGLE_TIME_SPLIT="${TOGGLE_TIME_SPLIT:-0.5}"
export TOGGLE_FAST_PATH="${TOGGLE_FAST_PATH:-true}"
export TOGGLE_TIMING_MODE="${TOGGLE_TIMING_MODE:-derived}"
export TOGGLE_Y_TIME="${TOGGLE_Y_TIME:-5e-9}"
export Z_OVER_Y_TIME="${Z_OVER_Y_TIME:-3}"
export SCALE_TRAIN_RECIPE="${SCALE_TRAIN_RECIPE:-false}"
export TOGGLE_ONE_OVER_Q="${TOGGLE_ONE_OVER_Q:-1}"
export ACTIVATION_CORNER_MODE="${ACTIVATION_CORNER_MODE:-fixed}" # or random_per_forward
export ENABLE_MEASURED_ACTIVATION="${ENABLE_MEASURED_ACTIVATION:-true}"
export ENABLE_MEASURED_POOLING="${ENABLE_MEASURED_POOLING:-false}"
export SCALE_MEASURED_ACTIVATION="${SCALE_MEASURED_ACTIVATION:-false}"
export WARMUP_FT="${WARMUP_FT:-0}"
export FT_LEARNING_RATE="${FT_LEARNING_RATE:-0.005}"
export FT_NUM_EPOCHS="${FT_NUM_EPOCHS:-${NUM_EPOCHS:-140}}"
export DISTILL_METHOD="${DISTILL_METHOD:-srrl}"
export SRRL_WEIGHT="${SRRL_WEIGHT:-1.0}"
export TIMM_RE_PROB="${TIMM_RE_PROB:-0.0}"
export TRAIN_CONV_EXPANDED="${TRAIN_CONV_EXPANDED:-false}"
export NONLINEAR_R_TRAIN_MODE="${NONLINEAR_R_TRAIN_MODE:-exact_curve}" # exact_curve or mean
export NONLINEAR_R_CORNER_RANGE="${NONLINEAR_R_CORNER_RANGE:-all}" # or passing in corners "FF_V0_T0,FF_V1_T0"
export ENABLE_NONLINEAR_R="${ENABLE_NONLINEAR_R:-true}"
export R_VAL="${R_VAL:-67e3}"
export R_MAX="${R_MAX:-none}"
export C_VAL="${C_VAL:-282e-15}"
export NONLINEAR_R_TABLE="${NONLINEAR_R_TABLE:-coupler_monte}"
export NONLINEAR_R_MC_QUANTITY="${NONLINEAR_R_MC_QUANTITY:-conductance}"
export NONLINEAR_R_CURVE_SHARING="${NONLINEAR_R_CURVE_SHARING:-shared}"
export NONLINEAR_R_CURVE_SEED="${NONLINEAR_R_CURVE_SEED:-none}"
export ENABLE_SPIN_VARIATION="${ENABLE_SPIN_VARIATION:-true}"
export SIGMA_SPIN="${SIGMA_SPIN:-0.10}"
export ENABLE_SUMMING_CURRENT_NOISE="${ENABLE_SUMMING_CURRENT_NOISE:-true}"
export SUMMING_CURRENT_P="${SUMMING_CURRENT_P:-0.6e-12}"
export ENABLE_COUPLER_NOISE="${ENABLE_COUPLER_NOISE:-true}"
export COUPLER_NOISE_P="${COUPLER_NOISE_P:-0.6e-12}"
export ENABLE_SLOW_SUMMING_CURRENT="${ENABLE_SLOW_SUMMING_CURRENT:-false}"
export SLOW_SUMMING_CURRENT="${SLOW_SUMMING_CURRENT:-2.47e-9}"
export ENABLE_SLOW_COUPLER_NOISE="${ENABLE_SLOW_COUPLER_NOISE:-false}"
export SLOW_COUPLER_NOISE="${SLOW_COUPLER_NOISE:-2.47e-9}"


echo "model=${MODEL_NAME_OVERRIDE}"
echo "img_type=${IMG_TYPE}"
echo "model_dir=${MODEL_DIR}"
echo "pcn=${PCN}"
echo "pulse_mismatch_training_mode=${PULSE_MISMATCH_TRAINING_MODE}"
echo "weight_quant_factor_bits=${WEIGHT_QUANT_FACTOR_BITS}"
echo "R=${R_VAL}"
echo "R_max=${R_MAX}"
echo "C=${C_VAL}"
echo "mismatch_level=${MISMATCH_LEVEL}"
echo "ode_block_override=${ODE_BLOCK_OVERRIDE}"
echo "odexinit_scaling_mode=${ODEXINIT_SCALING_MODE}"
echo "toggle_timing_mode=${TOGGLE_TIMING_MODE}"
echo "toggle_y_time=${TOGGLE_Y_TIME}"
echo "z_over_y_time=${Z_OVER_Y_TIME}"
echo "scale_train_recipe=${SCALE_TRAIN_RECIPE}"
echo "activation_corner_mode=${ACTIVATION_CORNER_MODE}"
echo "distill_method=${DISTILL_METHOD}"
echo "srrl_weight=${SRRL_WEIGHT}"
echo "timm_re_prob=${TIMM_RE_PROB}"
echo "ft_learning_rate=${FT_LEARNING_RATE}"
echo "ft_num_epochs=${FT_NUM_EPOCHS}"
echo "warmup_ft=${WARMUP_FT}"
echo "enable_measured_pooling=${ENABLE_MEASURED_POOLING}"
echo "train_conv_expanded=${TRAIN_CONV_EXPANDED}"
echo "nonlinear_R_train_mode=${NONLINEAR_R_TRAIN_MODE}"
echo "nonlinear_R_corner_range=${NONLINEAR_R_CORNER_RANGE}"
echo "nonlinear_R_table=${NONLINEAR_R_TABLE}"
echo "nonlinear_R_mc_quantity=${NONLINEAR_R_MC_QUANTITY}"
echo "nonlinear_R_curve_sharing=${NONLINEAR_R_CURVE_SHARING}"
echo "nonlinear_R_curve_seed=${NONLINEAR_R_CURVE_SEED}"
echo "output_save_path=${OUTPUT_SAVE_PATH}"
echo "run_tag=${RUN_TAG}"

source ./launch_scripts/run_toggle_ode_mixed_ft.sh

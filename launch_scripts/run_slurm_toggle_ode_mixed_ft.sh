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
################################################################################################

source activate base
conda activate scanbase

export MODEL_NAME_OVERRIDE="${MODEL_NAME_OVERRIDE:-TIMMPCNetNoBatchNorm_PCConvReLU6_0.0eps_ODEXInitFFFB_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_C100_3K1S96C_0.25Dropout_16Layers4l5l4_2Pool_srrlDistill_a0p3_t2p0_scanGFI_6REP}"
export PULSE_MISMATCH_TRAINING_MODE="${PULSE_MISMATCH_TRAINING_MODE:-post_quant_amplitude}"
export R_VAL="${R_VAL:-67e3}"
export MISMATCH_LEVEL="${MISMATCH_LEVEL:-0.25}"
export RUN_TAG="${RUN_TAG:?RUN_TAG must be set}"
export EXP_OVERRIDE="pulse_qat_mt_${RUN_TAG}"
export OUTPUT_SAVE_PATH="./saved_ckpt_runs/${RUN_TAG}"

export ODE_BLOCK_OVERRIDE="ToggleODEXInitFFFB"
export ODEXINIT_SCALING_MODE="direct"
export ACTIVATION_CORNER_MODE="${ACTIVATION_CORNER_MODE:?ACTIVATION_CORNER_MODE must be fixed or random_per_forward}"
export TRAIN_CONV_EXPANDED="${TRAIN_CONV_EXPANDED:-false}"
export NONLINEAR_R_TRAIN_MODE="${NONLINEAR_R_TRAIN_MODE:-none}" # exact_curve or mean
export NONLINEAR_R_CORNER_RANGE="${NONLINEAR_R_CORNER_RANGE:-all}" # or passing in corners "FF_V0_T0,FF_V1_T0"
export ENABLE_NONLINEAR_R="${ENABLE_NONLINEAR_R:-false}"
export NONLINEAR_R_TABLE="${NONLINEAR_R_TABLE:-./hardware_data/mc_45_corners/coupler_monte}"
export NONLINEAR_R_MC_QUANTITY="${NONLINEAR_R_MC_QUANTITY:-conductance}"
export NONLINEAR_R_CURVE_SHARING="${NONLINEAR_R_CURVE_SHARING:-shared}"
export NONLINEAR_R_CURVE_SEED="${NONLINEAR_R_CURVE_SEED:-none}"


echo "model=${MODEL_NAME_OVERRIDE}"
echo "pulse_mismatch_training_mode=${PULSE_MISMATCH_TRAINING_MODE}"
echo "R=${R_VAL}"
echo "mismatch_level=${MISMATCH_LEVEL}"
echo "ode_block_override=${ODE_BLOCK_OVERRIDE}"
echo "odexinit_scaling_mode=${ODEXINIT_SCALING_MODE}"
echo "activation_corner_mode=${ACTIVATION_CORNER_MODE}"
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

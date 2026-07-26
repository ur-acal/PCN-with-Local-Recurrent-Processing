#!/bin/bash
#SBATCH -p ising
#SBATCH -c 16
#SBATCH --gres=gpu:1
#SBATCH -t 90:10:00
#SBATCH -o /scratch/rzeng7/repos/PCN-with-Local-Recurrent-Processing/logs/slurm_jobs/slurm_%j.out

source activate base
conda activate scanbase

export MODEL_NAME_OVERRIDE="${MODEL_NAME_OVERRIDE:-TIMMPCNetNoBatchNorm_PCConvReLU6_0.0eps_ODEXInitFFFB_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_C100_3K1S96C_0.25Dropout_16Layers4l5l4_2Pool_srrlDistill_a0p3_t2p0_scanGFI_6REP}"
export PULSE_MISMATCH_TRAINING_MODE="post_quant_amplitude"
export RUN_TAG="${RUN_TAG:?RUN_TAG must be set}"
export EXP_OVERRIDE="pulse_qat_mt_${RUN_TAG}"
export OUTPUT_SAVE_PATH="./saved_ckpt_runs/${RUN_TAG}"

export ODE_BLOCK_OVERRIDE="ToggleODEXInitFFFB"
export ODEXINIT_SCALING_MODE="direct"
export ACTIVATION_CORNER_MODE="${ACTIVATION_CORNER_MODE:?ACTIVATION_CORNER_MODE must be fixed or random_per_forward}"


echo "model=${MODEL_NAME_OVERRIDE}"
echo "pulse_mismatch_training_mode=${PULSE_MISMATCH_TRAINING_MODE}"
echo "ode_block_override=${ODE_BLOCK_OVERRIDE}"
echo "odexinit_scaling_mode=${ODEXINIT_SCALING_MODE}"
echo "activation_corner_mode=${ACTIVATION_CORNER_MODE}"
echo "output_save_path=${OUTPUT_SAVE_PATH}"
echo "run_tag=${RUN_TAG}"

source ./launch_scripts/run_toggle_ode_mixed_ft.sh

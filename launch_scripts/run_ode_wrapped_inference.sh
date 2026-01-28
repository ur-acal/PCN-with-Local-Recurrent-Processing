#!/usr/bin/env bash
set -eu
trap '' HUP   # ignore hangup so the children survive

# ─────────────── fixed params ───────────────
export MODEL_DIR="./saved_ckpt"
export tol="1e-6"
export R_VAL="20e3"
export R_MAX="300e3"
#export CKPT="full_param_best"
export CKPT="best"

# ─────────────── noise toggles ───────────────
#N_BITS_VALS=(4 5 6 7 8)
N_BITS_VALS=()
#N_BITS_VALS=(5)
for ((i=0; i<20; i++)); do N_BITS_VALS+=(5); done
#N_BITS_VALS=(15 15 15 15 15 15 15 15 15 15)
METHOD_VALS=("dopri5")
CAP_VALS=("49e-15")
#METHOD_VALS=("euler")
#METHOD_VALS=("rk4")

export BASE_LOGDIR="./logs/test_ode_noisy"
# MASTER_LOG and JOB_LOG will be set per noise combination

# ─────────────── model list ───────────────
MODEL_NAMES=(
#  "PCNetNoBatchNorm_PCConvHardTanh_dopri5Solver_0.75TEnd_0.0001Tol_0.001WD_noBPtied_noBP_withPC_128BS_0.01LR_0.25Dropout_7Layers_1REP" # NODE baseline
#  "PCNetNoBatchNorm_PCConvHardTanh_dopri5Solver_1.0TEnd_0.0001Tol_0.001WD_noBPtied_noBP_withPC_128BS_0.01LR_0.25Dropout_7Layers_1REP" # NODE baseline

#  "PCNetNoBatchNorm_PCConvReLU6_ODEActInpNoMinus_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_noBPtied_noBP_128BS_0.01LR_0.25Dropout_7Layers_1REP" # zero init
#  "PCNetNoBatchNorm_PCConvHardTanh2_ODEActInpNoMinus_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_noBPtied_noBP_128BS_0.01LR_0.25Dropout_7Layers_1REP" # zero init

#  "PCNetNoBatchNorm_PCConvHardTanh_ODEBlockPCMinusY_dopri5Solver_0.75TEnd_0.001Tol_0.001WD_noBPtied_noBP_128BS_0.01LR_0.25Dropout_7Layers_2REP" # no max_g_norm
#  "PCNetNoBatchNorm_PCConvReLU6_ODEBlockPCMinusY_dopri5Solver_0.75TEnd_0.001Tol_0.001WD_noBPtied_noBP_128BS_0.01LR_0.25Dropout_7Layers_2REP" # no max_g_norm

#  "PCNetNoBatchNorm_PCConvHardTanh_0.2eps_ODEFixNoiseOffset_dopri5Solver_0.75TEnd_0.0001Tol_0.001WD_noBPtied_noBP_128BS_0.01LR_0.25Dropout_7Layers_1REP"
#  "ftPCNetNoBatchNorm_PCConvHardTanh_0.4eps_ODEFixNoiseOffset_dopri5Solver_0.75TEnd_0.0001Tol_0.001WD_noBPtied_noBP_128BS_0.0001LR_0.25Dropout_7Layers_1REP"
#  "PCNetNoBatchNorm_PCConvReLU6_0.2eps_ODEFixNoiseOffset_dopri5Solver_0.75TEnd_0.0001Tol_0.001WD_noBPtied_noBP_128BS_0.01LR_0.25Dropout_7Layers_1REP"
#  "ftPCNetNoBatchNorm_PCConvReLU6_0.4eps_ODEFixNoiseOffset_dopri5Solver_0.75TEnd_0.0001Tol_0.001WD_noBPtied_noBP_128BS_0.0001LR_0.25Dropout_7Layers_1REP" # passive baseline

#  "PCNetNoBatchNorm_PCConvReLU6_0.2eps_ODEFixNoiseOffset_dopri5Solver_0.75TEnd_0.0001Tol_0.001WD_noBPtied_noBP_128BS_0.01LR_0.25Dropout_5Layers_2Pool_1REP" # 2 max pooling

#  "ftPCNetNoBatchNorm_PCConvReLU6_0.4eps_ODEFixNoiseOffset_dopri5Solver_0.75TEnd_0.0001Tol_0.001WD_noBPtied_noBP_128BS_0.0001LR_0.25Dropout_6Layers_1REP" # 3 max pooling
#  "PCNetNoBatchNorm_PCConvReLU6_0.002eps_S2NoMinusZChgZNoisyI_dopri5Solver_1.5TEnd_0.0001Tol_0.001WD_128BS_0.01LR_0.25Dropout_7Layers_2Pool_1REP"
#  "PCNetNoBatchNorm_PCConvReLU6_0.002eps_S2NoMinusZChgZNoisyI_dopri5Solver_1.5TEnd_0.0001Tol_0.001WD_128BS_0.01LR_0.25Dropout_7Layers_2Pool_2REP" # S2NoMinusZChgZNoisyI baseline

#  "PCNetNoBatchNorm_PCConvReLU6_0.2eps_S2NoMinusZChgZMinusNoisyI_dopri5Solver_1.5TEnd_0.0001Tol_0.001WD_128BS_0.01LR_0.25Dropout_7Layers_2Pool_rggb_2REP"
#  "PCNetNoBatchNorm_PCConvReLU6_0.2eps_S2NoMinusZChgZMinusNoisyI_dopri5Solver_1.5TEnd_0.0001Tol_0.001WD_128BS_0.01LR_0.25Dropout_7Layers_2Pool_cycleisp_2REP"
#  "PCNetNoBatchNorm_PCConvReLU6_0.2eps_S2NoMinusZChgZNoisyI_dopri5Solver_1.5TEnd_0.0001Tol_0.001WD_128BS_0.01LR_0.25Dropout_7Layers_2Pool_cycleisp_2REP"

#  "QAT4bPCNetNoBatchNorm_PCConvReLU6_0.0eps_S2NoMinusZChgZNoisyI_dopri5Solver_1.5TEnd_0.0001Tol_0.001WD_128BS_0.01LR_0.25Dropout_7Layers_2Pool_3REP"
#  "QAT4bPCNetNoBatchNorm_PCConvReLU6_0.0eps_S2NoMinusZChgZNoisyI_dopri5Solver_1.5TEnd_0.0001Tol_0.001WD_128BS_0.01LR_0.25Dropout_7Layers_2Pool_5REP" # Use relu6 scaled
#  "QAT6bPCNetNoBatchNorm_PCConvReLU6_0.0eps_S2NoMinusZChgZNoisyI_dopri5Solver_1.5TEnd_0.0001Tol_0.001WD_128BS_0.01LR_0.25Dropout_7Layers_2Pool_1REP"

#  "QAT5bPCNetNoBatchNorm_PCConvReLU6_0.0eps_S2NoMinusZChgZNoisyI_dopri5Solver_1.5TEnd_0.0001Tol_0.001WD_128BS_0.01LR_0.25Dropout_7Layers_2Pool_3REP" # QAT that uses W^Q in [-1,1]
#  "QAT5bPCNetNoBatchNorm_PCConvReLU6_0.0eps_S2NoMinusZChgZNoisyI_dopri5Solver_1.5TEnd_0.0001Tol_0.001WD_128BS_0.01LR_0.25Dropout_7Layers_2Pool_4REP" # Trained with R=C=1
#  "QAT5bPCNetNoBatchNorm_PCConvReLU6_0.0eps_S2NoMinusZChgZNoisyI_dopri5Solver_1.5TEnd_0.0001Tol_0.001WD_128BS_0.01LR_0.25Dropout_7Layers_2Pool_5REP" # R=C=1 and tol=1e-6
#  "QAT5bPCNetNoBatchNorm_PCConvReLU6_0.0eps_S2NoMinusZChgZNoisyI_dopri5Solver_1.5TEnd_0.0001Tol_0.001WD_128BS_0.01LR_0.25Dropout_7Layers_2Pool_6REP" # R=C=1 and v_dd=10
#  "QAT5bPCNetNoBatchNorm_PCConvReLU6_0.0eps_S2NoMinusZChgZNoisyI_dopri5Solver_1.5TEnd_0.0001Tol_0.001WD_128BS_0.01LR_0.25Dropout_7Layers_2Pool_7REP" # R=C=1, lr=1e-3
#  "QAT5bPCNetNoBatchNorm_PCConvReLU6_0.0eps_S2NoMinusZChgZNoisyI_dopri5Solver_1.5TEnd_0.0001Tol_0.001WD_128BS_0.01LR_0.25Dropout_7Layers_2Pool_8REP" # R=1e5, C=49e-15 lr=1e-3
#  "QAT5bPCNetNoBatchNorm_PCConvReLU6_0.0eps_S2NoMinusZChgZNoisyI_dopri5Solver_1.5TEnd_0.0001Tol_0.001WD_128BS_0.01LR_0.25Dropout_7Layers_2Pool_10REP" # LR=1e-2 Cosine Annealing, R=1e5, C=49e-15
#  "QAT4bPCNetNoBatchNorm_PCConvReLU6_0.0eps_S2NoMinusZChgZNoisyI_dopri5Solver_1.5TEnd_0.0001Tol_0.001WD_128BS_0.01LR_0.25Dropout_7Layers_2Pool_7REP" # Same as above, but 4-bit
#  "QAT4bLSQWeightPCNetNoBatchNorm_PCConvReLU6_0.0eps_S2NoMinusZChgZNoisyI_dopri5Solver_1.5TEnd_0.0001Tol_0.001WD_128BS_0.01LR_0.25Dropout_7Layers_2Pool_1REP"

#  "QAT5bLSQWeightPCNetNoBatchNorm_PCConvReLU6_0.0eps_S2NoMinusZChgZNoisyI_dopri5Solver_1.5TEnd_0.0001Tol_0.001WD_128BS_0.01LR_0.25Dropout_7Layers_2Pool_3REP" # LSQ
#  "QAT5bLSQWeightPCNetNoBatchNorm_PCConvReLU6_0.0eps_S2NoMinusZChgZNoisyI_dopri5Solver_1.5TEnd_0.0001Tol_0.001WD_128BS_0.01LR_0.25Dropout_7Layers_2Pool_7REP" # R=C=1, lr=1e-3
#  "QAT5bLSQWeightPCNetNoBatchNorm_PCConvReLU6_0.0eps_S2NoMinusZChgZNoisyI_dopri5Solver_1.5TEnd_0.0001Tol_0.001WD_128BS_0.01LR_0.25Dropout_7Layers_2Pool_5REP" # R=1e5, C=49e-15
#  "QAT5bLSQWeightPCNetNoBatchNorm_PCConvReLU6_0.0eps_S2NoMinusZChgZNoisyI_dopri5Solver_1.5TEnd_0.0001Tol_0.001WD_128BS_0.01LR_0.25Dropout_7Layers_2Pool_6REP" # R=1e5, C=49e-15 lr=1e-3

#  "PCNetNoBatchNorm_PCConvReLU6_0.2eps_S2NoMinusZChgZNoisyI_dopri5Solver_1.5TEnd_0.0001Tol_0.001WD_128BS_0.01LR_0.25Dropout_7Layers_2Pool_rggb_2REP" # RGGB
#  "PCNetNoBatchNorm_PCConvReLU6_0.2eps_S2NoMinusZChgZNoisyI_dopri5Solver_1.5TEnd_0.0001Tol_0.001WD_128BS_0.01LR_0.25Dropout_6Layers_2Pool_rggb_1REP"
#  "PCNetNoBatchNorm_PCConvReLU6_0.2eps_S2NoMinusZChgZNoisyI_dopri5Solver_1.5TEnd_0.0001Tol_0.001WD_128BS_0.01LR_0.25Dropout_4Layers_2Pool_rggb_1REP"

#  "PCNetNoBatchNorm_PCConvReLU6_0.0eps_S2NoMinusZChgZNoisyI_dopri5Solver_1.5TEnd_0.0001Tol_0.001WD_128BS_0.01LR_0.25Dropout_5Layers_2Pool_scanGFI_1REP"
#  "PCNetNoBatchNorm_PCConvReLU6_0.0eps_S2NoMinusZChgZNoisyI_dopri5Solver_1.5TEnd_0.0001Tol_0.001WD_128BS_0.01LR_0.25Dropout_5Layers_2Pool_scanGFI_3REP"
#  "ftNT0p4mulPCNetNoBatchNorm_PCConvReLU6_0.0eps_S2NoMinusZChgZNoisyI_dopri5Solver_1.5TEnd_0.0001Tol_0.001WD_128BS_0.01LR_0.25Dropout_5Layers_2Pool_scanGFI_1REP"
#  "QAT5bNT0p4mulPCNetNoBatchNorm_PCConvReLU6_0.0eps_S2NoMinusZChgZNoisyI_dopri5Solver_1.5TEnd_0.0001Tol_0.001WD_128BS_0.01LR_0.25Dropout_5Layers_2Pool_scanGFI_2REP"

#  "QAT5bNT0p4mulPCNetNoBatchNorm_PCConvReLU6_0.0eps_S2NoMinusZChgZNoisyI_dopri5Solver_1.5TEnd_0.0001Tol_0.001WD_128BS_0.01LR_3K1S_0.25Dropout_6Layers_2Pool_scanGFI_1REP" # model on bluehive
#  "QAT5bNT0p4mulPCNetNoBatchNorm_PCConvReLU6_0.0eps_S2NoMinusZChgZNoisyI_dopri5Solver_1.5TEnd_0.0001Tol_0.001WD_128BS_0.01LR_3K1S_0.25Dropout_6Layers_2Pool_scanGFI_2REP"
#  "PCNetNoBatchNorm_PCConvReLU6_0.0eps_S2NoMinusZChgZNoisyI_dopri5Solver_1.5TEnd_0.0001Tol_0.001WD_128BS_0.01LR_3K1S64C_0.25Dropout_5Layers_1Pool_scanGFI_2REP"
#  "KD4p0T0p8w0p2w0PCNetNoBatchNorm_PCConvReLU6_0.0eps_S2NoMinusZChgZNoisyI_dopri5Solver_1.5TEnd_0.0001Tol_0.001WD_128BS_0.01LR_3K1S32C_0.25Dropout_18Layers_1Pool_scanGFI_1REP"
#  "QAT5bNT0p2mulKD4p0T0p8w0p2w0PCNetNoBatchNorm_PCConvReLU6_0.0eps_S2NoMinusZChgZNoisyI_dopri5Solver_1.5TEnd_0.0001Tol_0.001WD_128BS_0.01LR_3K1S32C_0.25Dropout_18Layers_1Pool_scanGFI_1REP"
#  "QAT5bNT0p2mulKD4p0T0p8w0p2w0PCNetNoBatchNorm_PCConvReLU6_0.0eps_S2NoMinusZChgZNoisyI_dopri5Solver_1.5TEnd_0.0001Tol_0.001WD_128BS_0.01LR_3K1S32C_0.25Dropout_18Layers_1Pool_scanGFI_2REP"
#  "QAT5bNT0p25mulKD4p0T0p8w0p2w0PCNetNoBatchNorm_PCConvReLU6_0.0eps_S2NoMinusZChgZNoisyI_dopri5Solver_1.5TEnd_0.0001Tol_0.001WD_128BS_0.01LR_3K1S32C_0.25Dropout_18Layers_1Pool_scanGFI_1REP"

#  "PCNetNoBatchNorm_PCConvReLU6_0.0eps_S2NoisyIYAsXZAsX_dopri5Solver_0.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_3K1S64C_0.25Dropout_7Layers_2Pool_scanGFI_1REP"
#  "PCNetNoBatchNorm_PCConvReLU6_0.0eps_S2NoisyIYAs0ZAsX_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_3K1S64C_0.25Dropout_7Layers_2Pool_scanGFI_1REP"
#  "PCNetNoBatchNorm_PCConvReLU6_0.0eps_S2NoisyIYAsXZAs0_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_3K1S64C_0.25Dropout_7Layers_2Pool_scanGFI_1REP"
#
#  "PCNetNoBatchNorm_PCConvReLU6_0.0eps_S2NoisyIYAsXZAsX_dopri5Solver_0.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_3K1S64C_0.25Dropout_7Layers_1Pool_scanGFI_1REP"
#  "PCNetNoBatchNorm_PCConvReLU6_0.0eps_S2NoisyIYAs0ZAsX_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_3K1S64C_0.25Dropout_7Layers_1Pool_scanGFI_1REP"
#  "PCNetNoBatchNorm_PCConvReLU6_0.0eps_S2NoisyIYAsXZAs0_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_3K1S64C_0.25Dropout_7Layers_1Pool_scanGFI_1REP"
#
#  # YAs0ZAsX Failed to train for the 18 Layer model
#  "PCNetNoBatchNorm_PCConvReLU6_0.0eps_S2NoisyIYAsXZAsX_dopri5Solver_0.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_3K1S64C_0.25Dropout_18Layers_2Pool_scanGFI_1REP"
#  "PCNetNoBatchNorm_PCConvReLU6_0.0eps_S2NoisyIYAsXZAs0_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_3K1S64C_0.25Dropout_18Layers_2Pool_scanGFI_1REP"
#
#  "PCNetNoBatchNorm_PCConvReLU6_0.0eps_S2NoisyIYAsXZAsX_dopri5Solver_0.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_3K1S64C_0.25Dropout_18Layers_1Pool_scanGFI_1REP"
#  "PCNetNoBatchNorm_PCConvReLU6_0.0eps_S2NoisyIYAsXZAs0_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_3K1S64C_0.25Dropout_18Layers_1Pool_scanGFI_1REP"

#  "8P4PS5PCPCNetNoBatchNorm_PCConvReLU6_0.0eps_S2CircYAsXZas0_dopri5Solver_1.25TEnd_0.0001Tol_0.001WD_128BS_0.01LR_3K1S64C_0.25Dropout_7Layers_2Pool_scanGFI_1REP"
#  "PCNetNoBatchNorm_PCConvReLU6_0.0eps_S2NoisyIYAsXZAs0_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_3K1S48C_0.25Dropout_18Layers_2Pool_scanGFI_1REP"
#  "8P8PS1PCPCNetNoBatchNorm_PCConvReLU6_0.0eps_S2CircYAsXZas0_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_3K1S64C_0.25Dropout_11Layers_2Pool_scanGFI_2REP"

#  "PCNetNoBatchNorm_PCConvReLU6_0.0eps_S2NoisyIYAsXZAs0_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_3K1S48C_0.25Dropout_18Layers_2Pool_scanGFI_2REP"
#  ""  # R_max=30e3
#  "QAT5bNT0p1mulPCNetNoBatchNorm_PCConvReLU6_0.0eps_S2NoisyIYAsXZAs0_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_3K1S48C_0.25Dropout_18Layers_2Pool_scanGFI_5REP"  # R=50e3, R_max=180e3

#  "PCNetNoBatchNorm_PCConvReLU6_0.0eps_S2NoisyIYAsXZAs0_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_3K1S4C_0.25Dropout_8Layers_2Pool_scanGFI_1REP"
#  "PCNetNoBatchNorm_PCConvReLU6_0.0eps_S2NoisyIYAsXZAs0_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_C100_3K1S80C_0.25Dropout_10Layers_2Pool_scanGFI_1REP"

#  "PCNetNoBatchNorm_PCConvReLU6_0.0eps_S2NoisyIYAsXZAs0_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_3K1S72C_0.25Dropout_13Layers_2Pool_scanGFI_1REP"
#  "QAT5bNT0p1mulPCNetNoBatchNorm_PCConvReLU6_0.0eps_S2NoisyIYAsXZAs0_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_3K1S72C_0.25Dropout_13Layers_2Pool_scanGFI_4REP"
#  "QAT5bNT0p1mulQAT5bNT0p1mulPCNetNoBatchNorm_PCConvReLU6_0.0eps_S2NoisyIYAsXZAs0_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_3K1S72C_0.25Dropout_13Layers_2Pool_scanGFI_1REP" # grad at _k==1 is w_scalar
#  "QAT5bNT0p1mulQAT5bNT0p1mulPCNetNoBatchNorm_PCConvReLU6_0.0eps_S2NoisyIYAsXZAs0_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_3K1S72C_0.25Dropout_13Layers_2Pool_scanGFI_2REP" # grad is _delta * q_max for _k > 0
#  "QAT5bNT0p1mulQAT5bNT0p1mulPCNetNoBatchNorm_PCConvReLU6_0.0eps_S2NoisyIYAsXZAs0_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_3K1S72C_0.25Dropout_13Layers_2Pool_scanGFI_3REP" # grad = 0 when _k == 0, _delta * q_max o.w.
#  "QAT5bNT0p2mulQAT5bNT0p1mulPCNetNoBatchNorm_PCConvReLU6_0.0eps_S2NoisyIYAsXZAs0_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_3K1S72C_0.25Dropout_13Layers_2Pool_scanGFI_1REP"

#  "PCNetNoBatchNorm_PCConvReLU6_0.0eps_S2NoisyIYAsXZAs0_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_C100_3K1S72C_0.25Dropout_13Layers_2Pool_scanGFI_1REP"
#  "PCNetNoBatchNorm_PCConvReLU6_0.0eps_S2NoisyIYAsXZAs0_dopri5Solver_1.75TEnd_0.0001Tol_0.0001WD_128BS_0.01LR_C100_3K1S128C_0.0Dropout_7Layers_2Pool_kd_crdDistill_a0p3_t2p0_scanGFI_1REP" # need one_over_q to be high (40)
#  "QAT5bNT0p4mulPCNetNoBatchNorm_PCConvReLU6_0.0eps_S2NoMinusZChgZNoisyI_dopri5Solver_1.5TEnd_0.0001Tol_0.001WD_128BS_0.01LR_3K1S128C_0.25Dropout_7Layers_2Pool_scanGFI_2REP"
#  "QAT5bNT0p15mulPCNetNoBatchNorm_PCConvReLU6_0.0eps_S2NoisyIYAsXZAs0_dopri5Solver_1.75TEnd_0.0001Tol_0.0001WD_128BS_0.01LR_C100_3K1S128C_0.0Dropout_7Layers_2Pool_kd_crdDistill_a0p3_t2p0_scanGFI_2REP"

#  "PCNetNoBatchNorm_PCConvReLU6_0.0eps_ODEXInitFFFB_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_3K1S72C_0.25Dropout_13Layers_2Pool_scanGFI_1REP"
#  "QAT5bNT0p15mulPCNetNoBatchNorm_PCConvReLU6_0.0eps_ODEXInitFFFB_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_3K1S72C_0.25Dropout_13Layers_2Pool_scanGFI_1REP"
#  "QAT5bNT0p15mulQAT5bNT0p15mulPCNetNoBatchNorm_PCConvReLU6_0.0eps_ODEXInitFFFB_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_3K1S72C_0.25Dropout_13Layers_2Pool_scanGFI_1REP"
#  "PCNetNoBatchNorm_PCConvReLU6_0.0eps_ODEXInitFFFB_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_3K1S128C_0.25Dropout_7Layers_2Pool_scanGFI_1REP"
#  "PCNetNoBatchNorm_PCConvReLU6_0.0eps_S2NoisyIYAsXZAs0_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_3K1S128C_0.25Dropout_7Layers_2Pool_scanGFI_1REP"

#  "PCNetNoBatchNorm_PCConvReLU6_0.0eps_ODEXInitFFFB_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_C100_3K1S128C_0.25Dropout_10Layers2l2l3_2Pool_scanGFI_2REP"
  "QAT5bNT0p15mulPCNetNoBatchNorm_PCConvReLU6_0.0eps_ODEXInitFFFB_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_C100_3K1S128C_0.25Dropout_10Layers2l2l3_2Pool_scanGFI_1REP"
)

# ─────────────── prepare logs ───────────────
mkdir -p "$BASE_LOGDIR"
for n_bits in "${N_BITS_VALS[@]}"; do
  for cap_val in "${CAP_VALS[@]}"; do
    for method in "${METHOD_VALS[@]}"; do
      for name in "${MODEL_NAMES[@]}"; do
        mkdir -p "$BASE_LOGDIR/${name}_method_${method}_tol_${tol}_nbits_${n_bits}_cap_${cap_val}"
        > "$BASE_LOGDIR/${name}_method_${method}_tol_${tol}_nbits_${n_bits}_cap_${cap_val}/job.log"
      done
    done
  done
done

# ─────────────── helper function ───────────────
run_model(){
  local name="$1"
  local method="$2"
  local n_bits="$3"
  local cap_val="$4"

  if [[ "$name" == *C100* ]]; then
    local _task="cifar100"
  else
    local _task="cifar10"
  fi

  local __rest="${name#*_}"
  local _pc_conv="${__rest%%_*}"

  IFS='_' read -r -a parts <<< "$name"
  local _ode_block="${parts[3]}"
  prev="${parts[${#parts[@]}-2]}"
  if [[ "$prev" == *Layers || "$prev" == *Pool ]]; then
    local _img_type="rgb"
  else
    local _img_type="$prev"
  fi

#  local _ode_wrapper="ODEWrapperRC"
#  local _ode_wrapper="WrapQuantizeW"
  local _ode_wrapper="ODEWrapper1State"
  if [[ "$name" == *S2* || "$name" == *State2* ]]; then
    if [[ "$name" == *QAT* && "$CKPT" != *full_param* ]]; then
      _ode_wrapper="QATTester2State"
    else
      _ode_wrapper="ODEWrapper2State"
    fi
  else
    if [[ "$name" == *QAT* && "$CKPT" != *full_param* ]]; then
      _ode_wrapper="QATTester1State"
    fi
  fi

  local test_bs=128
  if [[ "${_ode_block}" == *Circ* ]]; then test_bs=128; fi
  ########################################
  # only fuse_bn when noise is added to bn
  ########################################
  set -o pipefail
  python -u ode_inference.py \
    --model_name      "$name" \
    --ckpt            "${CKPT}" \
    --task            "${_task}" \
    --model_dir       "$MODEL_DIR" \
    --test_bs         "${test_bs}" \
    --method          "$method" \
    --tol             "$tol" \
    --n_steps         100 \
    --ts_scale        1 \
    --d_start         0 \
    --d_end           1 \
    --n_sweep_left    0 \
    --n_sweep_right   1 \
    --thermal_noise   "true" \
    --sde_noise_type  "mul" \
    --sweep_eps       "false" \
    --R               "$R_VAL" \
    --R_max           "$R_MAX" \
    --C               "$cap_val" \
    --v_dd            "0.2" \
    --w_bits          "$n_bits" \
    --patch_node      "8" \
    --patch_stride    "8" \
    --patch_cycle     "1" \
    --patch_pad       "0" \
    --fold_scalar     "1" \
    --tie_cap         "false" \
    --one_over_q      "6" \
    --pc_conv         "${_pc_conv}Noisy" \
    --ode_block       "${_ode_block}" \
    --ode_wrapper     "${_ode_wrapper}" \
    --img_type        "${_img_type}" \
    --test_expanded   "false" \
    --nonlinear_R     "false" \
    --test_only       "true" \
    2>&1 | tee -a "$BASE_LOGDIR/${name}_method_${method}_tol_${tol}_nbits_${n_bits}_cap_${cap_val}/job.log"
}
export -f run_model

# ─────────────── loop over methods ───────────────
#for n_bits in "${N_BITS_VALS[@]}"; do
for method in "${METHOD_VALS[@]}"; do
  ##########################################################################################
  # Modify log name here before each run
  ##########################################################################################
#  EXP_NAME="0818_3pooling_wrapped_${n_bits}bits_ODESumAsBInitY_${method}Method_${tol}Tol.log"
  EXP_NAME="1228_scanGFI_deep_${method}Method_${tol}Tol.log"
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
    run_model {1} "${method}" {2} {3} \
    ::: "${MODEL_NAMES[@]}" \
    ::: "${N_BITS_VALS[@]}" \
    ::: "${CAP_VALS[@]}"
  echo "All jobs finished — merging logs into $MASTER_LOG"
  # ─────────────── merge logs sequentially ───────────────
  : >"$MASTER_LOG"
  for n_bits in "${N_BITS_VALS[@]}"; do
    for cap_val in "${CAP_VALS[@]}"; do
      for name in "${MODEL_NAMES[@]}"; do
        printf '========== %s | method=%s | n_bits=%s | cap=%s ==========\n' \
               "$name" "$method" "$n_bits" "$cap_val" >>"$MASTER_LOG"
        logfile="$BASE_LOGDIR/${name}_method_${method}_tol_${tol}_nbits_${n_bits}_cap_${cap_val}/job.log"
        if ! grep -A 100 "Model name: ${name} " "$logfile" >>"$MASTER_LOG"; then
          echo "[Final Result not found] $logfile" >>"$MASTER_LOG"
        fi
        printf '\n\n' >>"$MASTER_LOG"
      done
    done
  done
  echo "All summaries written to $MASTER_LOG"
done
#done

#######################################################
# running
# nohup bash run_ode_inference.sh > logs/run_script_output/launcher.out 2>&1 &
# tail -f logs/run_script_output/launcher.out
# After the run is finished, the master_log file will be printed out
# then cat master_log
# then use shell_utils/parse_noise_logs.py to convert the log into csv
#######################################################

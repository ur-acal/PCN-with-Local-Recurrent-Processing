#!/usr/bin/env bash
# Reproduce the August-2026 PCN physical fine-tuning/evaluation recipe, with
# opt-in CiFAIR-10/ResNet support and one selected TT nonlinear-R curve.
#
# Required:
#   MODEL_KIND=pcn|wrn_c27|resnet110
#   TASK=cifar10|cifar100 (default: cifar100, preserving historical runs)
#   COUPLER_FAMILY=coupler_monte|coupler_monte_v2
#   TT_SINGLETON=<CSV path or filename under hardware_data/mc_45_corners>
# Optional:
#   STAGE=ft|eval|all       (default: all)
#   N_TRIALS=<integer>     (default: 1)
#   RUN_TAG=<unique artifact name> (default includes singleton filename)
#   FUSE_MEASURED_ACTIVATION=true|false (default: true)
#   WRN_RECOVERY_CKPT=<latest_ckpt.pth> explicitly continues a WRN FT run.
#   PREFLIGHT_ONLY=true validates the actual Python commands without training.
#   CORNER_IDS="..."       (default: all 45 corners)
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO_ROOT}"

MODEL_KIND="${MODEL_KIND:?MODEL_KIND must be pcn, wrn_c27 or resnet110}"
TASK="${TASK:-cifar100}"
case "${TASK}" in cifar10|cifar100) ;; *) echo "Unsupported TASK=${TASK}" >&2; exit 2;; esac
DATASET_TAG="CiFAIR${TASK#cifar}"
COUPLER_FAMILY="${COUPLER_FAMILY:?COUPLER_FAMILY must be coupler_monte or coupler_monte_v2}"
STAGE="${STAGE:-all}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-false}"
REHEARSAL_ONLY="${REHEARSAL_ONLY:-false}"
if [[ "${REHEARSAL_ONLY}" == true ]]; then
  export REHEARSAL_ONLY
  export REHEARSAL_ARTIFACT_ROOT="${REHEARSAL_ARTIFACT_ROOT:?Set an isolated rehearsal artifact directory}"
  export REHEARSAL_ARTIFACT_ROOT="$(realpath -m "${REHEARSAL_ARTIFACT_ROOT}")"
  case "${REHEARSAL_ARTIFACT_ROOT}" in
    /tmp/*|"${REPO_ROOT}"/results/rehearsals/*) ;;
    *) echo "Rehearsals must use /tmp/... or results/rehearsals/..." >&2; exit 2;;
  esac
  export STUDY_ARTIFACT_ROOT="${REHEARSAL_ARTIFACT_ROOT}"
  export PYTHONPATH="${REPO_ROOT}/diagnostic_scripts/rehearsal_hooks${PYTHONPATH:+:${PYTHONPATH}}"
fi
if [[ "${PREFLIGHT_ONLY}" == true ]]; then
  PREFLIGHT_PYTHON="${PYTHON:-$(command -v python)}"
  export PREFLIGHT_PYTHON REPO_ROOT
  python() {
    "${PREFLIGHT_PYTHON}" "${REPO_ROOT}/diagnostic_scripts/verify_tt_singleton_launch.py" "$@"
  }
  export -f python
fi
TT_SINGLETON="${TT_SINGLETON:?TT_SINGLETON must specify the training singleton CSV}"

case "${COUPLER_FAMILY}" in
  coupler_monte|coupler_monte_v2) ;;
  *)
    echo "Unsupported COUPLER_FAMILY=${COUPLER_FAMILY}" >&2
    exit 2
    ;;
esac

if [[ ! -f "${TT_SINGLETON}" ]]; then
  TT_SINGLETON="hardware_data/mc_45_corners/${TT_SINGLETON}"
fi
[[ -f "${TT_SINGLETON}" ]] || {
  echo "Missing training singleton CSV: ${TT_SINGLETON}" >&2
  exit 1
}
TT_SINGLETON="$(realpath "${TT_SINGLETON}")"
singleton_tag="${TT_SINGLETON##*/}"
singleton_tag="${singleton_tag%.csv}"

case "${STAGE}" in
  ft|eval|all) ;;
  *) echo "STAGE must be ft, eval, or all; got ${STAGE}" >&2; exit 2 ;;
esac

TEACHER_CKPT="${TEACHER_CKPT:-./checkpoint/efficientnet_v2_l_${TASK}_CiFAIR_OldNoTimm_MatchDistill.pth}"
PCN_RAW_NAME="TIMMPCNetNoBatchNorm_PCConvReLU6_0.0eps_ToggleODEXInitFFFB_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_C100_3K1S96C_0.25Dropout_16Layers4l5l4_2Pool_srrlDistill_a0p3_t2p0_CiFAIR_1REP"
PCN_SOURCE_ROOT="saved_ckpt_runs/coupler_v2_CiFAIR100_qf1_noENOB_matchdistill_srrl_noRE_toggle_odexinit"
WRN_NAME="wrn_flexible_cifar_avgpool_shortcut"
WRN_SOURCE_CKPT="saved_ckpt_runs/coupler_full_range_CiFAIR100_wrn28_c27_avgpool_shortcut_fixedTiming15ns_rescaled_pretrain_iq12/cifar100/custom_noresize/${WRN_NAME}/custom_noresize_cifar100_${WRN_NAME}/custom_noresize_cifar100_${WRN_NAME}_best_ckpt.pth"
if [[ "${TASK}" == cifar10 ]]; then
  PCN_RAW_NAME="${PCN_RAW_NAME/_C100_/_}"
  PCN_SOURCE_ROOT="saved_ckpt_runs/coupler_v2_CiFAIR10_qf1_noENOB_matchdistill_srrl_noRE_ftlr0p002_e200_toggle_odexinit"
fi
PCN_QAT_NAME="TIMMQAT5bNoneaNT0p0mul${PCN_RAW_NAME}"
PCN_SOURCE_ROOT="${PCN_PRETRAIN_ROOT:-${PCN_SOURCE_ROOT}}"
if [[ "${MODEL_KIND}" == resnet110 ]]; then
  WRN_NAME=resnet110_cifar
  export INTERMEDIATE_ACTIVATION="${INTERMEDIATE_ACTIVATION:-relu6}"
  export FINAL_DROPOUT_RATE="${FINAL_DROPOUT_RATE:-0.25}"
  export CNN_ARCH_OVERRIDE="${CNN_ARCH_OVERRIDE:-intermediate_activation=${INTERMEDIATE_ACTIVATION},final_dropout_rate=${FINAL_DROPOUT_RATE}}"
fi
WRN_SOURCE_CKPT="${CNN_PRETRAIN_CKPT:-${WRN_SOURCE_CKPT}}"
if [[ "${MODEL_KIND}" == resnet110 || ( "${MODEL_KIND}" == wrn_c27 && "${TASK}" == cifar10 ) ]]; then
  : "${CNN_PRETRAIN_CKPT:?Specify the new CNN pretrained checkpoint}"
fi
WRN_ONE_OVER_Q="${WRN_ONE_OVER_Q:-}"

if [[ "${MODEL_KIND}" == wrn_c27 || "${MODEL_KIND}" == resnet110 ]]; then
  [[ -n "${WRN_ONE_OVER_Q}" ]] || {
    echo "WRN_ONE_OVER_Q must be set explicitly after the exact-recipe WRN scan" >&2
    exit 2
  }
  RUN_TAG="${RUN_TAG:-tt_singleton_50ep_${DATASET_TAG}_${MODEL_KIND}_${COUPLER_FAMILY}_fixedTiming15ns_vdd0p1_qinv${WRN_ONE_OVER_Q}_${singleton_tag}}"
else
  RUN_TAG="${RUN_TAG:-tt_singleton_50ep_${DATASET_TAG}_${MODEL_KIND}_${COUPLER_FAMILY}_${singleton_tag}}"
fi
ARTIFACT_PREFIX="${STUDY_ARTIFACT_ROOT:+${STUDY_ARTIFACT_ROOT}/}"
FT_ROOT="${ARTIFACT_PREFIX}saved_ckpt_runs/${RUN_TAG}_ft"
EVAL_ROOT="${ARTIFACT_PREFIX}results/${RUN_TAG}_mc45"
# Physical feedforward preprocessing is part of the saved-model namespace.
# The trainer appends this suffix after resolving the source checkpoint's
# preprocessing metadata.
WRN_FT_ROOT="${FT_ROOT}_iq12"

for required in \
  "${TEACHER_CKPT}" \
  "${TT_SINGLETON}"; do
  [[ -f "${required}" ]] || { echo "Missing required file: ${required}" >&2; exit 1; }
done

run_pcn_ft() {
  local source_ckpt="${PCN_SOURCE_ROOT}/${PCN_RAW_NAME}/${PCN_RAW_NAME}_best_ckpt.pth"
  local output_ckpt="${FT_ROOT}/${PCN_QAT_NAME}/${PCN_QAT_NAME}_best_ckpt.pth"
  [[ -f "${source_ckpt}" ]] || { echo "Missing PCN source checkpoint: ${source_ckpt}" >&2; exit 1; }

  (
    unset RHS_CURRENT_SUMMARY EXTRA_OVERRIDE PRETRAIN_RESUME_CKPT
    export MODEL_NAME_OVERRIDE="${PCN_RAW_NAME}" \
           MODEL_DIR="${PCN_SOURCE_ROOT}" \
           OUTPUT_SAVE_PATH="${FT_ROOT}" \
           EXP_OVERRIDE="${RUN_TAG}" \
           LOGDIR="${ARTIFACT_PREFIX}logs/${RUN_TAG}" \
           IMG_TYPE=CiFAIR \
           PCN=PCNetNoBatchNorm \
           ODE_BLOCK_OVERRIDE=ToggleODEXInitFFFB \
           ODEXINIT_SCALING_MODE=direct \
           R_VAL=50e3 \
           R_MAX=none \
           C_VAL=500e-15 \
           V_DD=0.1 \
           TOGGLE_ONE_OVER_Q=1 \
           TOGGLE_TIMING_MODE=derived \
           TOGGLE_Y_TIME=5e-9 \
           Z_OVER_Y_TIME=3 \
           TOGGLE_N_CYCLES=5 \
           TOGGLE_TIME_SPLIT=0.5 \
           TOGGLE_FAST_PATH=true \
           SCALE_TRAIN_RECIPE=false \
           MISMATCH_LEVEL=0.0 \
           PULSE_MISMATCH_TRAINING_MODE=post_quant_amplitude \
           WEIGHT_QUANT_FACTOR_BITS=1 \
           ENOB=none \
           INPUT_QUANT_BITS=none \
           CENTER_STUDENT_INPUT=false \
           ENABLE_SPIN_VARIATION=true \
           SIGMA_SPIN=0.10 \
           ENABLE_SUMMING_CURRENT_NOISE=true \
           SUMMING_CURRENT_P=0.6e-12 \
           ENABLE_COUPLER_NOISE=true \
           COUPLER_NOISE_P=0.6e-12 \
           ENABLE_SLOW_SUMMING_CURRENT=false \
           ENABLE_SLOW_COUPLER_NOISE=false \
           ENABLE_NONLINEAR_R=true \
           NONLINEAR_R_TABLE="${TT_SINGLETON}" \
           NONLINEAR_R_TRAIN_MODE=exact_curve \
           NONLINEAR_R_CORNER_RANGE=all \
           NONLINEAR_R_MC_QUANTITY=conductance \
           NONLINEAR_R_CURVE_SHARING=shared \
           TRAIN_CONV_EXPANDED=false \
           ENABLE_MEASURED_ACTIVATION=true \
           MEASURED_ACTIVATION_SCOPE=pc_only \
           FUSE_MEASURED_ACTIVATION="${FUSE_MEASURED_ACTIVATION:-true}" \
           ACTIVATION_CURVE_PATH=./hardware_data/relu_current_0p2uA_finer.csv \
           ACTIVATION_CORNER=TT \
           ACTIVATION_CORNER_MODE=fixed \
           ACTIVATION_INTERPOLATION=piecewise_linear \
           ACTIVATION_SPLINE_PARAMETERS=10 \
           ACTIVATION_FIT_CONSTRAINT=auto \
           SCALE_MEASURED_ACTIVATION=false \
           ENABLE_MEASURED_POOLING=true \
           FINAL_HEAD_TYPE=old_ideal \
           TIMM_AUG_LEVEL=no_aug \
           TIMM_RE_PROB=0.0 \
           FT_LEARNING_RATE=0.005 \
           FT_NUM_EPOCHS=50 \
           WARMUP_FT=0 \
           EVAL_EVERY=2 \
           DISTILL_METHOD=srrl \
           SRRL_WEIGHT=1.0 \
           TEACHER_CKPT="${TEACHER_CKPT}" \
           TEACHER_ARCH=efficientnet_v2_l \
           TEACHER_ARCH_SOURCE=auto \
           TEACHER_INPUT_SIZE=224 \
           TEACHER_CENTER_CROP=true \
           ADAPT_PIL_TEACHER=false

    bash -o pipefail ./launch_scripts/run_toggle_ode_mixed_ft.sh
  )
  [[ "${PREFLIGHT_ONLY}" == true ]] && return 0
  # Require both successful exit (including Python's tee pipeline) and the
  # artifact. A stale checkpoint must not hide a failed trainer.
  [[ -f "${output_ckpt}" ]] || {
    echo "PCN fine-tuning did not produce its expected checkpoint: ${output_ckpt}" >&2
    return 1
  }
}

run_wrn_ft() {
  [[ -f "${WRN_SOURCE_CKPT}" ]] || { echo "Missing WRN source checkpoint: ${WRN_SOURCE_CKPT}" >&2; exit 1; }
  local input_ckpt="${WRN_RECOVERY_CKPT:-${WRN_SOURCE_CKPT}}"
  [[ -f "${input_ckpt}" ]] || { echo "Missing WRN recovery checkpoint: ${input_ckpt}" >&2; exit 1; }

  (
    unset RHS_CURRENT_SUMMARY EXTRA_OVERRIDE PRETRAIN_RESUME_CKPT
    export MODEL_NAME="${WRN_NAME}" \
           MODEL_CKPT="${input_ckpt}" \
           TASK="${TASK}" \
           IMG_TYPE=CiFAIR \
           OUTPUT_DIR="${FT_ROOT}" \
           VALIDATION_MODE=false \
           TRAINING_SEED=4096 \
           NUM_WORKERS=2 \
           PHYSICAL_LEVEL=2 \
           R_VAL=50e3 \
           C_VAL=500e-15 \
           V_DD=0.1 \
           TOGGLE_ONE_OVER_Q="${WRN_ONE_OVER_Q}" \
           TOGGLE_TIMING_MODE=fixed \
           TOGGLE_Y_TIME=15e-9 \
           Z_OVER_Y_TIME=1 \
           SCALE_TRAIN_RECIPE=1 \
           MISMATCH_LEVEL=0.0 \
           PULSE_MISMATCH_TRAINING_MODE=post_quant_amplitude \
           W_BITS=5 \
           WEIGHT_QUANT_FACTOR_BITS=1 \
           ENOB=none \
           INPUT_QUANT_BITS=12 \
           CENTER_STUDENT_INPUT=false \
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
           NONLINEAR_R_TABLE="${TT_SINGLETON}" \
           NONLINEAR_R_TRAIN_MODE=exact_curve \
           NONLINEAR_R_CORNER_RANGE=all \
           MC_COUPLER_NONLINEAR_VARIATION_QUANTITY=conductance \
           NONLINEAR_R_CURVE_SHARING=shared \
           ENABLE_MEASURED_ACTIVATION=true \
           MEASURED_ACTIVATION_SCOPE=pc_only \
           FUSE_MEASURED_ACTIVATION="${FUSE_MEASURED_ACTIVATION:-true}" \
           ACTIVATION_CURVE_PATH=./hardware_data/relu_current_0p2uA_finer.csv \
           ACTIVATION_CORNER=TT \
           ACTIVATION_CORNER_MODE=fixed \
           ACTIVATION_CURVE_SHARING=per_model \
           ACTIVATION_INTERPOLATION=piecewise_linear \
           ACTIVATION_SPLINE_PARAMETERS=10 \
           ACTIVATION_FIT_CONSTRAINT=auto \
           SCALE_MEASURED_ACTIVATION=false \
           ENABLE_MEASURED_POOLING=true \
           FINAL_HEAD_TYPE=old_ideal \
           FT_TIMM_AUG_LEVEL=no_aug \
           TIMM_RE_PROB=0.0 \
           FT_LEARNING_RATE=0.005 \
           FT_NUM_EPOCHS=50 \
           WARMUP_FT=0 \
           WEIGHT_DECAY=0.001 \
           BATCH_SIZE=128 \
           EXTRA_OVERRIDE="lr=0.005,num_epochs=50,warmup_epoch=0,weight_decay=0.001,batch_size=128,skip_eval_epochs=0${CNN_ARCH_OVERRIDE:+,${CNN_ARCH_OVERRIDE}}" \
           FT_EVAL_EVERY=2 \
           FINAL_EVAL_ONLY=false \
           DISTILL_METHOD=srrl \
           SRRL_WEIGHT=1.0 \
           TEACHER_CKPT="${TEACHER_CKPT}" \
           TEACHER_ARCH=efficientnet_v2_l \
           TEACHER_ARCH_SOURCE=auto \
           TEACHER_INPUT_SIZE=224 \
           TEACHER_CENTER_CROP=true \
           ADAPT_PIL_TEACHER=false \
           TC_FEEDFORWARD=false

    unset WRN_DEPTH WRN_FIRST_STAGE_CHANNELS TIMM_AUG_LEVEL
    if [[ "${MODEL_KIND}" == wrn_c27 ]]; then export WRN_DEPTH=28 WRN_FIRST_STAGE_CHANNELS=27; fi
    bash ./launch_scripts/run_feedforward_physical_ft.sh
  )
  [[ "${PREFLIGHT_ONLY}" == true ]] && return 0
  local name="custom_noresize_${TASK}_${WRN_NAME}"
  local output_ckpt="${WRN_FT_ROOT}/${TASK}/custom_noresize/${WRN_NAME}/${name}/${name}_full_param_best_ckpt.pth"
  [[ -f "${output_ckpt}" ]] || {
    echo "WRN fine-tuning did not produce its expected checkpoint: ${output_ckpt}" >&2
    return 1
  }
}

run_pcn_eval() {
  local checkpoint="${FT_ROOT}/${PCN_QAT_NAME}/${PCN_QAT_NAME}_best_ckpt.pth"
  [[ "${PREFLIGHT_ONLY}" == true || "${REHEARSAL_ONLY}" == true || -f "${checkpoint}" ]] || { echo "PCN FT checkpoint not ready: ${checkpoint}" >&2; exit 1; }

  (
    unset RHS_CURRENT_SUMMARY
    export MODEL_NAME="${PCN_QAT_NAME}" \
           MODEL_DIR="${FT_ROOT}" \
           CKPT=best \
           OUTPUT_DIR="${EVAL_ROOT}" \
           N_TRIALS="${N_TRIALS:-1}" \
           MAX_EVAL_BATCHES=0 \
           BASE_SEED=20260723 \
           CORNER_IDS="${CORNER_IDS:-}" \
           W_BITS=5 \
           WEIGHT_QUANT_FACTOR_BITS=1 \
           ENOB=none \
           V_DD=0.1 \
           ONE_OVER_Q=1 \
           TOGGLE_TIMING_MODE=derived \
           TOGGLE_Y_TIME=5e-9 \
           Z_OVER_Y_TIME=3 \
           TOGGLE_N_CYCLES=5 \
           TOGGLE_TIME_SPLIT=0.5 \
           TOGGLE_FAST_PATH=true \
           RELU6_BOUND_MODE=scaled \
           INPUT_QUANT_BITS=none \
           CENTER_STUDENT_INPUT=false \
           FULL_45_CORNER_C=500e-15 \
           MC_COUPLER_NONLINEAR_VARIATION_SOURCE="${COUPLER_FAMILY}" \
           MC_COUPLER_NONLINEAR_VARIATION_QUANTITY=conductance \
           MC_COUPLER_NOMINAL_R=50e3 \
           ENABLE_NONLINEAR_R=true \
           NONLINEAR_R_CURVE_SHARING=per_coupler \
           NONLINEAR_R_CURVE_SAMPLING=empirical_with_replacement \
           ENABLE_SPIN_VARIATION=true \
           ENABLE_SUMMING_CURRENT_NOISE=true \
           SUMMING_CURRENT_P=0.6e-12 \
           ENABLE_COUPLER_NOISE=true \
           COUPLER_NOISE_P=0.6e-12 \
           ENABLE_SLOW_SUMMING_CURRENT=false \
           ENABLE_SLOW_COUPLER_NOISE=false \
           ENABLE_DTC_NONIDEALITY=true \
           MC_RELU_MONTE_CARLO_SOURCE=relu_monteCarlo \
           ENABLE_MEASURED_ACTIVATION=true \
           MEASURED_ACTIVATION_SCOPE=pc_only \
           FUSE_MEASURED_ACTIVATION="${FUSE_MEASURED_ACTIVATION:-true}" \
           ACTIVATION_CURVE_SHARING=per_spin \
           ACTIVATION_INTERPOLATION=piecewise_linear \
           ACTIVATION_FIT_CONSTRAINT=auto \
           ACTIVATION_NORMALIZE_POSITIVE_ENDPOINT=false \
           ENABLE_MEASURED_POOLING=true \
           ENABLE_DIFF_MISMATCH=false \
           FINAL_HEAD_TYPE=old_ideal \
           IS_SLURM=0

    bash ./launch_scripts/run_mc45_toggle_ablation.sh
  )
}

run_wrn_eval() {
  local name="custom_noresize_${TASK}_${WRN_NAME}"
  local checkpoint="${WRN_FT_ROOT}/${TASK}/custom_noresize/${WRN_NAME}/${name}/${name}_full_param_best_ckpt.pth"
  [[ "${PREFLIGHT_ONLY}" == true || "${REHEARSAL_ONLY}" == true || -f "${checkpoint}" ]] || { echo "WRN FT checkpoint not ready: ${checkpoint}" >&2; exit 1; }

  (
    unset RHS_CURRENT_SUMMARY
    export MODEL_NAME="${WRN_NAME}" \
           MODEL_CKPT="${checkpoint}" \
           TASK="${TASK}" \
           IMG_TYPE=CiFAIR \
           OUTPUT_DIR="${EVAL_ROOT}" \
           EXPANDED_WEIGHT_DIR="${ARTIFACT_PREFIX}expanded_weights/${RUN_TAG}" \
           N_TRIALS="${N_TRIALS:-1}" \
           JOBS=1 \
           MAX_EVAL_BATCHES=0 \
           BASE_SEED=20260723 \
           CORNER_IDS="${CORNER_IDS:-}" \
           PHYSICAL_LEVEL=3 \
           W_BITS=5 \
           WEIGHT_QUANT_FACTOR_BITS=1 \
           ENOB=none \
           V_DD=0.1 \
           ONE_OVER_Q="${WRN_ONE_OVER_Q}" \
           TOGGLE_TIMING_MODE=fixed \
           TOGGLE_Y_TIME=15e-9 \
           Z_OVER_Y_TIME=1 \
           INPUT_QUANT_BITS=12 \
           CENTER_STUDENT_INPUT=false \
           FULL_45_CORNER_C=500e-15 \
           MC_COUPLER_NONLINEAR_VARIATION_SOURCE="${COUPLER_FAMILY}" \
           MC_COUPLER_NONLINEAR_VARIATION_QUANTITY=conductance \
           MC_COUPLER_NOMINAL_R=50e3 \
           ENABLE_NONLINEAR_R=true \
           NONLINEAR_R_CURVE_SHARING=per_coupler \
           NONLINEAR_R_CURVE_SAMPLING=empirical_with_replacement \
           ENABLE_SPIN_VARIATION=true \
           ENABLE_SUMMING_CURRENT_NOISE=true \
           SUMMING_CURRENT_P=0.6e-12 \
           ENABLE_COUPLER_NOISE=true \
           COUPLER_NOISE_P=0.6e-12 \
           ENABLE_SLOW_SUMMING_CURRENT=false \
           ENABLE_SLOW_COUPLER_NOISE=false \
           ENABLE_DTC_NONIDEALITY=true \
           MC_RELU_MONTE_CARLO_SOURCE=relu_monteCarlo \
           ENABLE_MEASURED_ACTIVATION=true \
           MEASURED_ACTIVATION_SCOPE=pc_only \
           FUSE_MEASURED_ACTIVATION="${FUSE_MEASURED_ACTIVATION:-true}" \
           ACTIVATION_CURVE_SHARING=per_spin \
           ACTIVATION_INTERPOLATION=piecewise_linear \
           ACTIVATION_FIT_CONSTRAINT=auto \
           ACTIVATION_NORMALIZE_POSITIVE_ENDPOINT=false \
           ENABLE_MEASURED_POOLING=true \
           ENABLE_DIFF_MISMATCH=false \
           ENABLE_BN_RECALIBRATION=false \
           FINAL_HEAD_TYPE=old_ideal

    unset WRN_DEPTH WRN_FIRST_STAGE_CHANNELS
    if [[ "${MODEL_KIND}" == wrn_c27 ]]; then export WRN_DEPTH=28 WRN_FIRST_STAGE_CHANNELS=27; fi
    bash ./launch_scripts/run_feedforward_mc45_ablation.sh
  )
}

if [[ "${STAGE}" == ft || "${STAGE}" == all ]]; then
  case "${MODEL_KIND}" in
    pcn) run_pcn_ft ;;
    wrn_c27|resnet110) run_wrn_ft ;;
    *) echo "Unsupported MODEL_KIND=${MODEL_KIND}" >&2; exit 2 ;;
  esac
fi

if [[ "${STAGE}" == eval || "${STAGE}" == all ]]; then
  case "${MODEL_KIND}" in
    pcn) run_pcn_eval ;;
    wrn_c27|resnet110) run_wrn_eval ;;
    *) echo "Unsupported MODEL_KIND=${MODEL_KIND}" >&2; exit 2 ;;
  esac
fi

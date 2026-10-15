#!/bin/bash
# Common local/Slurm pipeline. Each family starts from the same pretraining.
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO_ROOT}"
: "${MODEL_KIND:?pcn, wrn_c27 or resnet110}"
: "${TASK:?cifar10 or cifar100}"
: "${COUPLER_MONTE_TT_SINGLETON:?Pass the selected original-circuit TT CSV}"
: "${COUPLER_MONTE_V2_TT_SINGLETON:?Pass the selected improved-circuit TT CSV}"
case "${TASK}" in cifar10|cifar100) ;; *) exit 2;; esac
case "${MODEL_KIND}" in pcn|wrn_c27|resnet110) ;; *) exit 2;; esac
export REPO_ROOT TASK MODEL_KIND
unset WRN_RECOVERY_CKPT PRETRAIN_RESUME_CKPT MODEL_CKPT EXTRA_OVERRIDE
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export TC_FEEDFORWARD=false VALIDATION_MODE=false FINAL_EVAL_ONLY=false
export INTERMEDIATE_ACTIVATION=relu6 FINAL_DROPOUT_RATE=0.25
export CNN_ARCH_OVERRIDE=intermediate_activation=relu6,final_dropout_rate=0.25
export FUSE_MEASURED_ACTIVATION=true
export FINAL_HEAD_TYPE=old_ideal DISTILL_ALPHA=0.3 DISTILL_TEMPERATURE=2.0 SRRL_WEIGHT=1.0
export TEACHER_ARCH=efficientnet_v2_l TEACHER_ARCH_SOURCE=auto TEACHER_INPUT_SIZE=224
export TEACHER_CENTER_CROP=true ADAPT_PIL_TEACHER=false ORIG_T_INP=false
STUDY_TAG="${STUDY_TAG:-tt_representative_50ep_CiFAIR${TASK#cifar}_${MODEL_KIND}_pipeline}"
if [[ "${REHEARSAL_ONLY:-false}" == true ]]; then
  : "${REHEARSAL_ARTIFACT_ROOT:?An isolated absolute rehearsal directory is required}"
  export REHEARSAL_ARTIFACT_ROOT="$(realpath -m "${REHEARSAL_ARTIFACT_ROOT}")"
  case "${REHEARSAL_ARTIFACT_ROOT}" in
    /tmp/*|"${REPO_ROOT}"/results/rehearsals/*) ;;
    *) echo "Rehearsals must use /tmp/... or results/rehearsals/..." >&2; exit 2;;
  esac
  export STUDY_ARTIFACT_ROOT="${REHEARSAL_ARTIFACT_ROOT}"
  export PYTHONPATH="${REPO_ROOT}/diagnostic_scripts/rehearsal_hooks${PYTHONPATH:+:${PYTHONPATH}}"
fi
PREFIX="${STUDY_ARTIFACT_ROOT:+${STUDY_ARTIFACT_ROOT}/}"
LOG_ROOT="${PREFIX}logs/${STUDY_TAG}"
if [[ "${REHEARSAL_ONLY:-false}" != true && -d "${LOG_ROOT}" && -n "$(ls -A "${LOG_ROOT}")" ]]; then
  echo "Study logs already exist: ${LOG_ROOT}; use a new STUDY_TAG (no implicit resume)." >&2
  exit 2
fi
mkdir -p "${LOG_ROOT}"
trap 's=$?; echo "STUDY_EXIT status=${s} model=${MODEL_KIND} dataset=${TASK}" | tee -a "${LOG_ROOT}/status.log"' EXIT
for csv in "${COUPLER_MONTE_TT_SINGLETON}" "${COUPLER_MONTE_V2_TT_SINGLETON}"; do
  [[ -f "${csv}" ]] || { echo "Missing singleton: ${csv}" >&2; exit 1; }
done
export TEACHER_CKPT="${TEACHER_CKPT:-./checkpoint/efficientnet_v2_l_${TASK}_CiFAIR_OldNoTimm_MatchDistill.pth}"
[[ -f "${TEACHER_CKPT}" ]] || { echo "Missing teacher ${TEACHER_CKPT}" >&2; exit 1; }

if [[ "${MODEL_KIND}" != pcn ]]; then
  if [[ "${MODEL_KIND}" == wrn_c27 ]]; then
    export MODEL_NAME=wrn_flexible_cifar_avgpool_shortcut WRN_DEPTH=28 WRN_FIRST_STAGE_CHANNELS=27
  else
    export MODEL_NAME=resnet110_cifar
    unset WRN_DEPTH WRN_FIRST_STAGE_CHANNELS
  fi
  PRETRAIN_ROOT="${PREFIX}saved_ckpt_runs/${STUDY_TAG}_pretrain_iq12"
  NAME="custom_noresize_${TASK}_${MODEL_NAME}"
  export CNN_PRETRAIN_CKPT="${PRETRAIN_ROOT}/${TASK}/custom_noresize/${MODEL_NAME}/${NAME}/${NAME}_best_ckpt.pth"
  # No implicit resume or stale-checkpoint shortcut. Pretraining must succeed.
  (
    unset TIMM_AUG_LEVEL PRETRAIN_RESUME_CKPT MODEL_CKPT
    export OUTPUT_DIR="${PRETRAIN_ROOT}" IMG_TYPE=CiFAIR TRAINING_SEED=4096 \
      NUM_WORKERS="${NUM_WORKERS:-2}" TIMM_AUG_LEVEL=none TIMM_RE_PROB=0.0 \
      DISTILL_METHOD=srrl ADAPT_PIL_TEACHER=false \
      R_VAL=50e3 C_VAL=500e-15 V_DD=0.5 TOGGLE_ONE_OVER_Q=5 \
      TOGGLE_TIMING_MODE=fixed TOGGLE_Y_TIME=15e-9 Z_OVER_Y_TIME=1 \
      SCALE_TRAIN_RECIPE=1 INPUT_QUANT_BITS=12 CENTER_STUDENT_INPUT=false \
      UNITLESS_MEASURED_PULLBACK_MODE=none \
      EXTRA_OVERRIDE="lr=0.1,num_epochs=300,warmup_epoch=0,weight_decay=0.001,batch_size=128,${CNN_ARCH_OVERRIDE}"
    bash ./launch_scripts/run_feedforward_cifar_pretrain.sh
  ) 2>&1 | tee "${LOG_ROOT}/pretrain.log"
  [[ -f "${CNN_PRETRAIN_CKPT}" ]] || { echo "Missing pretrain handoff: ${CNN_PRETRAIN_CKPT}" >&2; exit 1; }
fi

for FAMILY in coupler_monte coupler_monte_v2; do
  if [[ "${FAMILY}" == coupler_monte ]]; then CSV="${COUPLER_MONTE_TT_SINGLETON}"; else CSV="${COUPLER_MONTE_V2_TT_SINGLETON}"; fi
  export COUPLER_FAMILY="${FAMILY}" TT_SINGLETON="${CSV}" RUN_TAG="${STUDY_TAG}_${FAMILY}"
  if [[ "${MODEL_KIND}" != pcn ]]; then
    SELECTION="${PREFIX}results/${RUN_TAG}_selection"
    read -r -a CANDIDATES <<< "${ONE_OVER_Q_CANDIDATES:-1 2 4 5 8 16 25 32 50 64 128}"
    REHEARSAL_SELECTION=true python -u diagnostic_scripts/select_feedforward_one_over_q.py \
      --output "${SELECTION}" --candidates "${CANDIDATES[@]}" \
      --model_name "${MODEL_NAME}" --checkpoint "${CNN_PRETRAIN_CKPT}" \
      ${WRN_DEPTH:+--wrn_depth ${WRN_DEPTH}} ${WRN_FIRST_STAGE_CHANNELS:+--wrn_first_stage_channels ${WRN_FIRST_STAGE_CHANNELS}} \
      --dataset "${TASK}" --img_type CiFAIR --checkpoint_kind unitless_physical \
      --intermediate_activation relu6 --final_dropout_rate 0.25 \
      --evaluation_split train --selection_samples "${SELECTION_SAMPLES:-1280}" \
      --batch_size 128 --max_eval_batches "${SELECTION_BATCHES:-10}" \
      --physical true --physical_level 2 --use_expanded_weights false --bn_recalibrate false \
      --R 50e3 --C 500e-15 --v_dd 0.1 --one_over_q 5 \
      --toggle_timing_mode fixed --toggle_y_time 15e-9 --z_over_y_time 1 \
      --w_bits 5 --weight_quant_factor_bits 1 --enob none \
      --input_quant_bits 12 --center_student_input false --final_head_type old_ideal \
      --enable_nonlinear_R true --nonlinear_R_table "${CSV}" \
      --nonlinear_R_train_mode exact_curve --nonlinear_R_curve_sharing shared \
      --enable_measured_activation true --measured_activation_scope pc_only \
      --activation_curve_path hardware_data/relu_current_0p2uA_finer.csv \
      --activation_corner TT --activation_curve_sharing per_model --fuse_measured_activation true \
      --enable_measured_pooling true --enable_spin_variation true --sigma_spin 0.10 \
      --enable_summing_current_noise true --summing_current_p 0.6e-12 \
      --enable_coupler_noise true --coupler_noise_p 0.6e-12 --enable_dtc_nonideality false \
      --enable_slow_summing_current false --enable_slow_coupler_noise false \
      --data_seed 4096 --spin_variation_seed 4096 --summing_noise_seed 4096 \
      --coupler_noise_seed 4096 --nonlinear_R_curve_seed 4096 --activation_curve_seed 4096 \
      --result_path "${SELECTION}/sanity" \
      2>&1 | tee "${LOG_ROOT}/${FAMILY}_selection.log"
    export WRN_ONE_OVER_Q="$(<"${SELECTION}/selected_one_over_q.txt")"
    [[ -n "${WRN_ONE_OVER_Q}" ]] || exit 1
  fi
  STAGE=all bash ./launch_scripts/run_cifair100_tt_singleton_ft_eval.sh \
    2>&1 | tee "${LOG_ROOT}/${FAMILY}_ft_eval.log"
done
echo "STUDY_COMPLETE rehearsal=${REHEARSAL_ONLY:-false}" | tee -a "${LOG_ROOT}/status.log"

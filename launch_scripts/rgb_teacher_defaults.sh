# Match the outputs of run_rgb_teacher.sbatch; preserve explicit selections.
source ./launch_scripts/normalize_validation_mode.sh || return 2
if [[ "${IMG_TYPE,,}" == rgb ]]; then
  export TEACHER_CKPT="${TEACHER_CKPT:-./checkpoint/efficientnet_v2_l_${TASK}_rgb_OldNoTimm_MatchDistill.pth}"
  if [[ "${VALIDATION_MODE}" == true && "${TEACHER_CKPT}" != *"_val5k"* ]]; then
    export TEACHER_CKPT="${TEACHER_CKPT%.pth}_val5k.pth"
  fi
  export TEACHER_ARCH="${TEACHER_ARCH:-efficientnet_v2_l}"
  export TEACHER_ARCH_SOURCE="${TEACHER_ARCH_SOURCE:-torchvision}"
  export ADAPT_PIL_TEACHER="${ADAPT_PIL_TEACHER:-false}"
fi

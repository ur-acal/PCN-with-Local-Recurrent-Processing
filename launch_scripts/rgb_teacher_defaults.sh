# Match the outputs of run_rgb_teacher.sbatch; preserve explicit selections.
if [[ "${IMG_TYPE,,}" == rgb ]]; then
  export TEACHER_CKPT="${TEACHER_CKPT:-./checkpoint/efficientnet_v2_l_${TASK}_rgb_OldNoTimm_MatchDistill.pth}"
  export TEACHER_ARCH="${TEACHER_ARCH:-efficientnet_v2_l}"
  export TEACHER_ARCH_SOURCE="${TEACHER_ARCH_SOURCE:-torchvision}"
  export ADAPT_PIL_TEACHER="${ADAPT_PIL_TEACHER:-false}"
fi

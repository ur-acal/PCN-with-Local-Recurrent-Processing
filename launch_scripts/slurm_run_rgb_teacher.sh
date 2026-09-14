#!/bin/bash
# Run module swap in your interactive shell, THEN source this submitter.
# Uses the same inherited-environment pattern as slurm_run_rgb_ode_train.sh.
submit_rgb_teacher() {
  local root="${REPO_ROOT:-/scratch/rzeng7/repos/PCN-with-Local-Recurrent-Processing}"
  local recipe="${RGB_TEACHER_RECIPE:-legacy}"
  case "$recipe" in legacy|timm_oldaugs) ;; *) echo 'RGB_TEACHER_RECIPE must be legacy or timm_oldaugs' >&2; return 2 ;; esac
  local data="${RGB_DATA_ROOT:-${root}/../data}"
  local checkpoints="${TEACHER_OUTPUT_DIR:-${root}/checkpoint}"
  local logs="${TEACHER_LOG_DIR:-${root}/logs/teacher_rgb}"
  local exports="ALL,IS_SLURM=1,REPO_ROOT=${root},RGB_TEACHER_RECIPE=${recipe},RGB_DATA_ROOT=${data},TEACHER_OUTPUT_DIR=${checkpoints},TEACHER_LOG_DIR=${logs}"
  local -a command=(sbatch --parsable --chdir="$root" --gres=gpu:1
    --time="${SBATCH_TIMELIMIT:-90:10:00}" --export="$exports"
    --output="${logs}/slurm_%j.out" "${root}/launch_scripts/run_rgb_teacher.sbatch")
  if [[ "${TEACHER_DRY_RUN:-false}" == true ]]; then
    printf '%q ' "${command[@]}"; printf '\n'
    return 0
  fi
  mkdir -p "$logs" || return
  local jid
  jid=$("${command[@]}") || return
  echo "Submitted ${jid}: RGB CIFAR-10 then CIFAR-100; recipe=${recipe}; logs=${logs}"
}
submit_rgb_teacher

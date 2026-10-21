#!/bin/bash -l
#SBATCH -N 1
#SBATCH -p ising
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH -t 90:10:00
#SBATCH -o logs/slurm_jobs/slurm_%j.out

_run_pcn_pipeline() {
  _PCN_JOB_PREPARED=0
  _PCN_LOCK_FD=''
  cd "${REPO_ROOT:?REPO_ROOT is required}" || return
  if [[ "${IS_SLURM:-0}" == 1 ]]; then
    source activate base || return
    conda activate scanbase || return
  fi
  # Source the proven workers in this very shell, without activating twice.
  export IS_SLURM=0
  local job="${PIPELINE_JOB_DIR:?PIPELINE_JOB_DIR is required}"
  if [[ "${PCCONV:-PCConv}" != PCConv ]]; then
    echo "PCN pipeline requires PCCONV=PCConv (matched PCConvNoisy evaluation)." >&2
    return 2
  fi
  export PCCONV=PCConv
  export TRAIN_SAVE_PATH="${job}/checkpoints" TRAIN_LOG_DIR="${job}/logs"
  export TRAIN_COMPLETION_JSON="${job}/train_complete.json"
  export OUTPUT_ROOT="${job}/results" EXPANDED_W_DIR="${job}/expanded_weights"
  export EVAL_MODE=mismatch PC_CONV=PCConvNoisy RESULT_TAG="${ARCHITECTURE}"
  export FORCE_RERUN=false DATASETS=all
  if [[ -n "${REHEARSAL_ROOT:-}" ]]; then
    # Redirect interpreter/library scratch files as well as explicit artifacts.
    export PYTHONDONTWRITEBYTECODE=1 TMPDIR="${job}/tmp" MPLCONFIGDIR="${job}/mpl"
    export XDG_CACHE_HOME="${job}/xdg" TORCH_HOME="${job}/home" PYTHONPYCACHEPREFIX="${job}/pycache"
    export CUDA_CACHE_PATH="${job}/xdg/cuda" TRITON_CACHE_DIR="${job}/xdg/triton"
    export TORCHINDUCTOR_CACHE_DIR="${job}/xdg/inductor" NUMBA_CACHE_DIR="${job}/xdg/numba"
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
    export EVAL_DATA_DIR="${EVAL_DATA_DIR:-${REPO_ROOT}/../data}"
  fi
  # The shell retains this descriptor throughout training and evaluation.
  local lock_path
  lock_path=$(python "${REPO_ROOT}/pcn_pipeline.py" lock-path --root "${job}") || return
  exec {_PCN_LOCK_FD}>>"${lock_path}" || return
  python "${REPO_ROOT}/pcn_pipeline.py" prepare --root "${job}" --stage "${STAGE}" --lock-fd "${_PCN_LOCK_FD}" || return
  _PCN_JOB_PREPARED=1
  if [[ "${STAGE}" != test ]]; then
    source "${REPO_ROOT}/launch_scripts/run_rgb_ode_train.sh" || return
  fi
  python "${REPO_ROOT}/pcn_pipeline.py" handoff --root "${job}" || return
  if [[ "${STAGE}" != train ]]; then
    source "${job}/handoff.env" || return
    source "${REPO_ROOT}/launch_scripts/run_rgb_ode_mismatch_eval.sh" || return
    python "${REPO_ROOT}/pcn_pipeline.py" finish --root "${job}" || return
  fi
}

if _run_pcn_pipeline; then
  [[ -z "${_PCN_LOCK_FD}" ]] || exec {_PCN_LOCK_FD}>&-
else
  _pcn_status=$?
  if [[ "${_PCN_JOB_PREPARED}" == 1 ]]; then
    python "${REPO_ROOT}/pcn_pipeline.py" fail --root "${PIPELINE_JOB_DIR}"
  fi
  [[ -z "${_PCN_LOCK_FD}" ]] || exec {_PCN_LOCK_FD}>&-
  # Preserve the worker status whether executed by sbatch or sourced locally.
  return "${_pcn_status}" 2>/dev/null || exit "${_pcn_status}"
fi

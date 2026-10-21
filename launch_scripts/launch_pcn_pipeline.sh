#!/bin/bash
# Source this in the existing login shell for SLURM; local use also supports bash.
# The submission path calls sbatch directly, without Python or environment setup.

_launch_pcn_pipeline() {
  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  export REPO_ROOT="${REPO_ROOT:-$(cd "${script_dir}/.." && pwd)}"
  local backend="${BACKEND:-local}" stage="${STAGE:-train-test}"
  case "${backend}:${stage}" in
    local:train|local:test|local:train-test|slurm:train|slurm:test|slurm:train-test) ;;
    *) echo "Invalid BACKEND/STAGE: ${backend}/${stage}" >&2; return 2 ;;
  esac
  if [[ "${PCCONV:-PCConv}" != PCConv ]]; then
    echo "PCN pipeline requires PCCONV=PCConv (matched PCConvNoisy evaluation)." >&2
    return 2
  fi
  export STAGE="${stage}"
  export RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/logs/pcn_pipeline/$(date +%Y%m%d_%H%M%S)_${BASHPID}}"
  if [[ "${REHEARSAL:-0}" == 1 ]]; then
    [[ "${backend}" == local && "${stage}" == train-test ]] || {
      echo "Run rehearsals locally with STAGE=train-test (no cluster required)." >&2; return 2;
    }
    export REHEARSAL_ROOT="${RUN_ROOT}"
    # The helper rejects roots inside the repository or existing directories.
    PYTHONDONTWRITEBYTECODE=1 python "${REPO_ROOT}/pcn_pipeline.py" init-rehearsal --root "${RUN_ROOT}" || return
  elif [[ -n "${REHEARSAL_ROOT:-}" ]]; then
    echo "REHEARSAL_ROOT requires explicit REHEARSAL=1" >&2; return 2
  elif [[ "${stage}" != test ]]; then
    mkdir -p "$(dirname "${RUN_ROOT}")"
    mkdir "${RUN_ROOT}" || return
  fi
  export ODE_BLOCK="${ODE_BLOCK:-ODEXInitFFFB}" PCCONV="${PCCONV:-PCConv}"
  export T_END="${T_END:-1.75}" METHOD="${METHOD:-dopri5}" AVG_POOLING="${AVG_POOLING:-false}"
  if [[ "${METHOD}" == euler ]]; then
    export N_STEPS="${N_STEPS:-5}"
  else
    export N_STEPS="${N_STEPS:-15}"
  fi
  export WARMUP_EPOCH="${WARMUP_EPOCH:-5}" IS_TIMM="${IS_TIMM:-true}" TIMM_SCHED="${TIMM_SCHED:-cosine}"
  export REUSE_ACCEPTED_STEP_TRAINING="${REUSE_ACCEPTED_STEP_TRAINING:-true}"
  export CONDITIONS="${CONDITIONS:-max_additive,multiplicative,rms_additive}"
  export NOISY_TRIALS="${NOISY_TRIALS:-10}" BASE_SEED="${BASE_SEED:-123}" TEST_BS="${TEST_BS:-128}"
  export MUL_LEVELS="${MUL_LEVELS:-0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4}"
  export MAX_ADD_LEVELS="${MAX_ADD_LEVELS:-0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1}"
  export RMS_ADD_LEVELS="${RMS_ADD_LEVELS:-0.25,0.5,0.75,1.0,1.25}"
  export NOISE_TO_BN=false
  local p d size depth width blocks stage_width i j inputs outputs pools job_id identity index
  local pcns="${PCNS:-PCNetNoBatchNorm PCNetBoundaryBN}"
  local datasets="${DATASETS:-cifar10 cifar100}" sizes="${SIZES:-16_2 16_4 28_2 28_4}"
  if [[ "${stage}" != test ]]; then
    for p in ${pcns//,/ }; do
      for d in ${datasets//,/ }; do
        for size in ${sizes//,/ }; do
          printf '%s\t%s\t%s\n' "${p}" "${d}" "${size}"
        done
      done
    done > "${RUN_ROOT}/planned_jobs.tsv"
  fi
  for p in ${pcns//,/ }; do
    case "${p}" in PCNetNoBatchNorm|PCNetBoundaryBN) ;; *) echo "Unsupported PCN: ${p}"; return 2 ;; esac
    export PCN="${p}"
    if [[ "${p}" == PCNetBoundaryBN ]]; then export BN_EVAL_MODE=paired; else export BN_EVAL_MODE=legacy; fi
    for d in ${datasets//,/ }; do
      case "${d}" in cifar10|cifar100) ;; *) echo "Unsupported dataset: ${d}"; return 2 ;; esac
      export DATASET_NAME="${d}"
      for size in ${sizes//,/ }; do
        [[ "${size}" =~ ^(16|28)_[1-9][0-9]*$ ]] || { echo "Invalid size: ${size}"; return 2; }
        depth="${size%_*}"; width="${size#*_}"; blocks=$(((depth - 4) / 6))
        inputs="3"; outputs="16"; pools="0"
        local previous=16
        for i in 1 2 3; do
          stage_width=$((16 * width * (1 << (i - 1))))
          for ((j=0; j<blocks; j++)); do
            inputs+=" ${previous}"; outputs+=" ${stage_width}"
            if [[ "${i}" -gt 1 && "${j}" == 0 ]]; then pools+=" 1"; else pools+=" 0"; fi
            previous="${stage_width}"
          done
        done
        case "${size}" in 16_2) index=0 ;; 16_4) index=1 ;; 28_2) index=2 ;; 28_4) index=3 ;; *) index=$((depth * 100 + width)) ;; esac
        [[ "${d}" != cifar100 ]] || index=$((index + 4))
        export MODEL_INDEX="${index}" ARCHITECTURE="WRN_${size}"
        export INP_CHANNELS="${inputs}" OUT_CHANNELS="${outputs}" MAX_POOL="${pools}"
        identity="${ODE_BLOCK}_${METHOD}_n${N_STEPS}_t${T_END}_avg${AVG_POOLING}_reuse${REUSE_ACCEPTED_STEP_TRAINING}"
        export PIPELINE_JOB_DIR="${RUN_ROOT}/${p}/${d}/${size}/${identity}"
        if [[ "${backend}" == slurm ]]; then
          mkdir -p "${REPO_ROOT}/logs/slurm_jobs"
          job_id=$(sbatch --parsable \
            --chdir="${REPO_ROOT}" \
            --output="${REPO_ROOT}/logs/slurm_jobs/slurm_%j.out" \
            --gres=gpu:1 \
            --export=ALL,IS_SLURM=1 \
            "${REPO_ROOT}/launch_scripts/run_pcn_pipeline.sh") || return
          printf '%s\t%s\t%s\t%s\t%s\n' "${job_id}" "${p}" "${d}" "${size}" "${PIPELINE_JOB_DIR}" >> "${RUN_ROOT}/submissions.tsv"
          echo "submitted job ${job_id}: ${p} ${d} ${size}, stage=${stage}"
        else
          IS_SLURM=0 source "${REPO_ROOT}/launch_scripts/run_pcn_pipeline.sh" || return
        fi
      done
    done
  done
  if [[ "${REHEARSAL:-0}" == 1 ]]; then
    PYTHONDONTWRITEBYTECODE=1 python "${REPO_ROOT}/pcn_pipeline.py" audit --root "${RUN_ROOT}" || return
  fi
  echo "Pipeline artifacts: ${RUN_ROOT}"
}

_launch_pcn_pipeline

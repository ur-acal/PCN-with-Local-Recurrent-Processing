#!/bin/bash -l

#SBATCH -p ising
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16            # modest CPU request so the node can be shared
#SBATCH --gres=gpu:1                  # exactly ONE GPU; allows packing on 4-GPU nodes
#SBATCH -t 72:10:00
#SBATCH -o /scratch/rzeng7/repos/PCN-with-Local-Recurrent-Processing/logs/slurm_jobs/slurm_%j.out

#set -euo pipefail

# ---- Conda activation (non-interactive safe) ----
source activate base
conda activate scanbase

REPO_ROOT="${REPO_ROOT:-/scratch/rzeng7/repos/PCN-with-Local-Recurrent-Processing}"
cd "${REPO_ROOT}"

DATA_DIR="${DATA_DIR:-/scratch/rzeng7/repos/data}"
MODEL_OUTPUT_DIR="${MODEL_OUTPUT_DIR:-${REPO_ROOT}/checkpoint/baselines}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/logs/baselines}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-baseline/train_baseline_cifar.py}"
EVAL_SCRIPT="${EVAL_SCRIPT:-baseline/run_baseline.py}"

TRAIN_LOG_DIR="${TRAIN_LOG_DIR:-${OUTPUT_DIR}/train_logs}"
EVAL_LOG_DIR="${EVAL_LOG_DIR:-${OUTPUT_DIR}/eval_logs}"
EVAL_RESULTS_DIR="${EVAL_RESULTS_DIR:-${OUTPUT_DIR}/mismatch_eval}"

mkdir -p "${TRAIN_LOG_DIR}" "${EVAL_LOG_DIR}" "${EVAL_RESULTS_DIR}"

MODEL_NAME="${MODEL_NAME:?MODEL_NAME is required}"
DATASET_NAME="${DATASET_NAME:-cifar100}"
CASE_NAME="${CASE_NAME:-auto}"
PRETRAINED="${PRETRAINED:-false}"
PREFER_RESIZE="${PREFER_RESIZE:-false}"
EXTRA_OVERRIDE="${EXTRA_OVERRIDE:-eval_every=5}"
PRINT_ONLY="${PRINT_ONLY:-false}"
IMG_TYPE="${IMG_TYPE:-rgb}"
RGGB_TO_RGB="${RGGB_TO_RGB:-false}"
TIMM_AUG_LEVEL="${TIMM_AUG_LEVEL:-none}"
TIMM_RE_PROB="${TIMM_RE_PROB:-none}"
DISTILL_METHOD="${DISTILL_METHOD:-none}"
TEACHER_CKPT="${TEACHER_CKPT:-}"
TEACHER_ARCH="${TEACHER_ARCH:-}"
TEACHER_ARCH_SOURCE="${TEACHER_ARCH_SOURCE:-auto}"
TEACHER_INPUT_SIZE="${TEACHER_INPUT_SIZE:-224}"
TEACHER_CENTER_CROP="${TEACHER_CENTER_CROP:-true}"
ADAPT_PIL_TEACHER="${ADAPT_PIL_TEACHER:-false}"
ORIG_T_INP="${ORIG_T_INP:-false}"
DISTILL_ALPHA="${DISTILL_ALPHA:-0.3}"
DISTILL_TEMPERATURE="${DISTILL_TEMPERATURE:-2.0}"
SRRL_WEIGHT="${SRRL_WEIGHT:-1.0}"
RUN_EVAL="${RUN_EVAL:-true}"

MULT_NOISE_LEVEL_LIST="${MULT_NOISE_LEVEL_LIST:-0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4}"
ADD_NOISE_LEVEL_LIST="${ADD_NOISE_LEVEL_LIST:-0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1}"
EVAL_NOISY_TRIALS="${EVAL_NOISY_TRIALS:-10}"

make_train_log_path() {
  local model_name="$1"
  local dataset_name="$2"
  local case_name="$3"
  local pretrained="$4"

  local tag="${dataset_name}_${case_name}_${model_name}_pretrained_${pretrained}"
  echo "${TRAIN_LOG_DIR}/train_${tag}.log"
}

extract_ckpt_path() {
  local log_file="$1"

  grep -- "----- Model path:" "${log_file}" \
    | tail -n 1 \
    | sed -E 's/^----- Model path: (.*) -----$/\1/'
}

run_train_one() {
  local model_name="$1"
  local dataset_name="$2"
  local case_name="$3"
  local pretrained="$4"
  local prefer_resize="$5"

  local train_log
  train_log="$(make_train_log_path "${model_name}" "${dataset_name}" "${case_name}" "${pretrained}")"

  echo "======================================================================"
  echo "TRAIN: dataset=${dataset_name}, model=${model_name}, case=${case_name}, pretrained=${pretrained}"
  echo "LOG:   ${train_log}"
  echo "EXTRA_OVERRIDE: ${EXTRA_OVERRIDE}"
  echo "======================================================================"

  CMD=(
    python "${TRAIN_SCRIPT}"
    --model_name "${model_name}"
    --dataset "${dataset_name}"
    --data_dir "${DATA_DIR}"
    --output_dir "${MODEL_OUTPUT_DIR}"
    --case "${case_name}"
    --pretrained "${pretrained}"
    --prefer_resize "${prefer_resize}"
    --print_only "${PRINT_ONLY}"
    --img_type "${IMG_TYPE}"
    --rggb_to_rgb "${RGGB_TO_RGB}"
    --timm_aug_level "${TIMM_AUG_LEVEL}"
    --distill_method "${DISTILL_METHOD}"
    --teacher_arch_source "${TEACHER_ARCH_SOURCE}"
    --teacher_input_size "${TEACHER_INPUT_SIZE}"
    --teacher_center_crop "${TEACHER_CENTER_CROP}"
    --adapt_PIL_teacher "${ADAPT_PIL_TEACHER}"
    --orig_t_inp "${ORIG_T_INP}"
    --distill_alpha "${DISTILL_ALPHA}"
    --distill_temperature "${DISTILL_TEMPERATURE}"
    --srrl_weight "${SRRL_WEIGHT}"
  )

  if [[ "${TIMM_RE_PROB}" != "none" ]]; then
    CMD+=(--timm_re_prob "${TIMM_RE_PROB}")
  fi
  if [[ -n "${TEACHER_CKPT}" ]]; then
    CMD+=(--teacher_ckpt "${TEACHER_CKPT}")
  fi
  if [[ -n "${TEACHER_ARCH}" ]]; then
    CMD+=(--teacher_arch "${TEACHER_ARCH}")
  fi

  if [[ -n "${EXTRA_OVERRIDE}" ]]; then
    CMD+=(--override "${EXTRA_OVERRIDE}")
  fi

  printf 'Running command:'
  printf ' %q' "${CMD[@]}"
  printf '\n'

  : > "${train_log}"
  "${CMD[@]}" 2>&1 | tee "${train_log}"
}

run_eval_one() {
  local model_name="$1"
  local dataset_name="$2"
  local case_name="$3"
  local ckpt_path="$4"
  local prefer_resize="$5"
  local noise_type="$6"
  local noise_level_list="$7"
  local noise_to_norm="$8"

  local tag="${dataset_name}_${case_name}_${model_name}_${noise_type}_noise_to_norm_${noise_to_norm}"
  local eval_log="${EVAL_LOG_DIR}/eval_${tag}.log"
  local results_dir="${EVAL_RESULTS_DIR}/${dataset_name}/${case_name}/${model_name}/${noise_type}_noise_to_norm_${noise_to_norm}"

  mkdir -p "${results_dir}"

  echo "======================================================================"
  echo "EVAL: dataset=${dataset_name}, model=${model_name}, case=${case_name}"
  echo "CKPT: ${ckpt_path}"
  echo "NOISE_TYPE: ${noise_type}"
  echo "NOISE_LEVEL_LIST: ${noise_level_list}"
  echo "NOISY_TRIALS: ${EVAL_NOISY_TRIALS}"
  echo "LOG:  ${eval_log}"
  echo "======================================================================"

  python "${EVAL_SCRIPT}" \
    --model_list "${model_name}" \
    --dataset "${dataset_name}" \
    --data_dir "${DATA_DIR}" \
    --checkpoint_map "${model_name}=${ckpt_path}" \
    --case "${case_name}" \
    --pretrained false \
    --prefer_resize "${prefer_resize}" \
    --noise_level_list "${noise_level_list}" \
    --noisy_trials "${EVAL_NOISY_TRIALS}" \
    --noise_type "${noise_type}" \
    --noise_to_norm "${noise_to_norm}" \
    --results_dir "${results_dir}" \
    2>&1 | tee "${eval_log}"
}

run_one() {
  local model_name="$1"
  local dataset_name="$2"
  local case_name="$3"
  local pretrained="$4"
  local prefer_resize="$5"

  local train_log
  local ckpt_path

  train_log="$(make_train_log_path "${model_name}" "${dataset_name}" "${case_name}" "${pretrained}")"

  run_train_one "${model_name}" "${dataset_name}" "${case_name}" "${pretrained}" "${prefer_resize}"

  ckpt_path="$(extract_ckpt_path "${train_log}")"

  if [[ -z "${ckpt_path}" ]]; then
    echo "ERROR: failed to extract checkpoint path from ${train_log}" >&2
    exit 1
  fi

  if [[ ! -f "${ckpt_path}" ]]; then
    echo "ERROR: extracted checkpoint does not exist: ${ckpt_path}" >&2
    exit 1
  fi

  echo "======================================================================"
  echo "Extracted checkpoint path:"
  echo "${ckpt_path}"
  echo "======================================================================"

  if [[ "${RUN_EVAL}" != "true" ]]; then
    echo "RUN_EVAL=${RUN_EVAL}; skipping legacy RGB mismatch evaluation."
    return 0
  fi

  run_eval_one "${model_name}" "${dataset_name}" "${case_name}" "${ckpt_path}" "${prefer_resize}" \
    "multiplicative" "${MULT_NOISE_LEVEL_LIST}" "false"

  run_eval_one "${model_name}" "${dataset_name}" "${case_name}" "${ckpt_path}" "${prefer_resize}" \
    "multiplicative" "${MULT_NOISE_LEVEL_LIST}" "true"

  run_eval_one "${model_name}" "${dataset_name}" "${case_name}" "${ckpt_path}" "${prefer_resize}" \
    "additive" "${ADD_NOISE_LEVEL_LIST}" "false"

  run_eval_one "${model_name}" "${dataset_name}" "${case_name}" "${ckpt_path}" "${prefer_resize}" \
    "additive" "${ADD_NOISE_LEVEL_LIST}" "true"

  echo "======================================================================"
  echo "Finished train + multiplicative/additive evals with noise_to_norm=false/true"
  echo "Checkpoint path: ${ckpt_path}"
  echo "======================================================================"
}

echo "REPO_ROOT=${REPO_ROOT}"
echo "MODEL_NAME=${MODEL_NAME}"
echo "DATASET_NAME=${DATASET_NAME}"
echo "CASE_NAME=${CASE_NAME}"
echo "PRETRAINED=${PRETRAINED}"
echo "PREFER_RESIZE=${PREFER_RESIZE}"
echo "IMG_TYPE=${IMG_TYPE}"
echo "TIMM_AUG_LEVEL=${TIMM_AUG_LEVEL}"
echo "TIMM_RE_PROB=${TIMM_RE_PROB}"
echo "DISTILL_METHOD=${DISTILL_METHOD}"
echo "TEACHER_CKPT=${TEACHER_CKPT:-auto}"
echo "TEACHER_ARCH=${TEACHER_ARCH:-auto}"
echo "TEACHER_ARCH_SOURCE=${TEACHER_ARCH_SOURCE}"
echo "ADAPT_PIL_TEACHER=${ADAPT_PIL_TEACHER}"
echo "ORIG_T_INP=${ORIG_T_INP}"
echo "RUN_EVAL=${RUN_EVAL}"
echo "DATA_DIR=${DATA_DIR}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "MULT_NOISE_LEVEL_LIST=${MULT_NOISE_LEVEL_LIST}"
echo "ADD_NOISE_LEVEL_LIST=${ADD_NOISE_LEVEL_LIST}"
echo "EVAL_NOISY_TRIALS=${EVAL_NOISY_TRIALS}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
which python
python --version

run_one "${MODEL_NAME}" "${DATASET_NAME}" "${CASE_NAME}" "${PRETRAINED}" "${PREFER_RESIZE}"
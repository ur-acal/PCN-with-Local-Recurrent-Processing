#!/bin/bash -l
#SBATCH -N 1
#SBATCH -p ising
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH -t 90:10:00
#SBATCH -o /scratch/rzeng7/repos/PCN-with-Local-Recurrent-Processing/logs/slurm_jobs/slurm_%j.out

IS_SLURM="${IS_SLURM:-0}"

if [[ "${IS_SLURM}" == 1 ]]; then
  source activate base
  conda activate scanbase
fi

REPO_ROOT="${REPO_ROOT:-/scratch/rzeng7/repos/PCN-with-Local-Recurrent-Processing}"
MODEL_DIR="${MODEL_DIR:-${REPO_ROOT}/saved_ckpt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/logs/pcn_with1stconv_mismatch_slurm}"
MODEL_SET="${MODEL_SET:-existing}"
SHARD_ID="${SHARD_ID:-0}"
NOISY_TRIALS="${NOISY_TRIALS:-10}"
BASE_SEED="${BASE_SEED:-123}"
TEST_BS="${TEST_BS:-128}"
MUL_LEVELS="${MUL_LEVELS:-0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4}"
MAX_ADD_LEVELS="${MAX_ADD_LEVELS:-0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1}"
RMS_ADD_LEVELS="${RMS_ADD_LEVELS:-0.25,0.5,0.75,1.0,1.25}"
FORCE_RERUN="${FORCE_RERUN:-false}"

cd "${REPO_ROOT}"

C10_16_2="TIMMPCNetWith1stConv_PCConv_0.0eps_ODEXInitFFFB_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_3K1S128C_0.25Dropout_7Layers0l1l2_2Pool_1REP"
C10_16_4="TIMMPCNetWith1stConv_PCConv_0.0eps_ODEXInitFFFB_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_3K1S256C_0.25Dropout_7Layers0l1l2_2Pool_1REP"
C10_28_2="TIMMPCNetWith1stConv_PCConv_0.0eps_ODEXInitFFFB_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_3K1S128C_0.25Dropout_13Layers0l3l6_2Pool_1REP"
C10_28_4="TIMMPCNetWith1stConv_PCConv_0.0eps_ODEXInitFFFB_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_3K1S256C_0.25Dropout_13Layers0l3l6_2Pool_1REP"
C100_16_2="TIMMPCNetWith1stConv_PCConv_0.0eps_ODEXInitFFFB_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_C100_3K1S128C_0.25Dropout_7Layers0l1l2_2Pool_1REP"
C100_16_4="TIMMPCNetWith1stConv_PCConv_0.0eps_ODEXInitFFFB_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_C100_3K1S256C_0.25Dropout_7Layers0l1l2_2Pool_1REP"
C100_28_2="TIMMPCNetWith1stConv_PCConv_0.0eps_ODEXInitFFFB_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_C100_3K1S128C_0.25Dropout_13Layers0l3l6_2Pool_1REP"
C100_28_4="TIMMPCNetWith1stConv_PCConv_0.0eps_ODEXInitFFFB_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_C100_3K1S256C_0.25Dropout_13Layers0l3l6_2Pool_1REP"

case "${MODEL_SET}:${SHARD_ID}" in
  existing:0) MODELS=("${C10_28_4}|cifar10|3|WRN_28_4") ;;
  existing:1) MODELS=("${C100_28_4}|cifar100|7|WRN_28_4") ;;
  existing:2) MODELS=("${C10_16_2}|cifar10|0|WRN_16_2" "${C100_16_4}|cifar100|5|WRN_16_4") ;;
  existing:3) MODELS=("${C100_16_2}|cifar100|4|WRN_16_2" "${C100_28_2}|cifar100|6|WRN_28_2") ;;
  pending:0) MODELS=("${C10_16_4}|cifar10|1|WRN_16_4") ;;
  pending:1) MODELS=("${C10_28_2}|cifar10|2|WRN_28_2") ;;
  all:0) MODELS=("${C10_28_4}|cifar10|3|WRN_28_4" "${C100_16_2}|cifar100|4|WRN_16_2") ;;
  all:1) MODELS=("${C100_28_4}|cifar100|7|WRN_28_4" "${C10_16_2}|cifar10|0|WRN_16_2") ;;
  all:2) MODELS=("${C10_16_4}|cifar10|1|WRN_16_4" "${C100_16_4}|cifar100|5|WRN_16_4") ;;
  all:3) MODELS=("${C10_28_2}|cifar10|2|WRN_28_2" "${C100_28_2}|cifar100|6|WRN_28_2") ;;
  *)
    echo "Unsupported MODEL_SET/SHARD_ID: ${MODEL_SET}/${SHARD_ID}"
    exit 2
    ;;
esac

for entry in "${MODELS[@]}"; do
  IFS='|' read -r model_name task model_index architecture <<< "${entry}"
  checkpoint="${MODEL_DIR}/${model_name}/${model_name}_best_ckpt.pth"
  if [[ ! -f "${checkpoint}" ]]; then
    echo "Missing checkpoint: ${checkpoint}"
    exit 3
  fi

  for spec in \
    "multiplicative|mul|max_abs|${MUL_LEVELS}" \
    "max_additive|add|max_abs|${MAX_ADD_LEVELS}" \
    "rms_additive|add|rms|${RMS_ADD_LEVELS}"; do
    IFS='|' read -r condition mismatch_type scale_mode levels <<< "${spec}"
    output_dir="${OUTPUT_ROOT}/${task}/${architecture}/${condition}"
    output_pickle="${output_dir}/result.pkl"
    log_file="${output_dir}/run.log"
    mkdir -p "${output_dir}"

    if [[ "${FORCE_RERUN}" != "true" && -s "${output_pickle}" ]]; then
      echo "Skipping completed result: ${output_pickle}"
      continue
    fi

    echo "Running ${task} ${architecture} ${condition} on shard ${SHARD_ID}"
    python -u ode_inference.py \
      --model_name "${model_name}" \
      --ckpt best \
      --task "${task}" \
      --img_type rgb \
      --model_dir "${MODEL_DIR}" \
      --method dopri5 \
      --tol 0.0001 \
      --n_steps 15 \
      --ts_scale 1 \
      --d_start 0 \
      --d_end 1 \
      --n_sweep_left 0 \
      --n_sweep_right 1 \
      --thermal_noise false \
      --mismatch_type "${mismatch_type}" \
      --additive_scale_mode "${scale_mode}" \
      --noise_to_conv_bias false \
      --noise_level_list "${levels}" \
      --noisy_trials "${NOISY_TRIALS}" \
      --seed "${BASE_SEED}" \
      --model_index "${model_index}" \
      --test_bs "${TEST_BS}" \
      --pc_conv PCConvNoisy \
      --ode_block ODEXInitFFFB \
      --output_pickle "${output_pickle}" \
      >"${log_file}" 2>&1

    status=$?
    if [[ "${status}" != 0 ]]; then
      echo "Evaluation failed with status ${status}: ${log_file}"
      exit "${status}"
    fi
  done
done

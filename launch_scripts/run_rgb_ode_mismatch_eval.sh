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
OUTPUT_ROOT="${OUTPUT_ROOT:-}"
MODEL_SET="${MODEL_SET:-existing}"
EVAL_MODE="${EVAL_MODE:-mismatch}"
CONDITIONS="${CONDITIONS:-multiplicative,max_additive,rms_additive}"
DATASETS="${DATASETS:-all}"
SHARD_ID="${SHARD_ID:-0}"
MODEL_NAME="${MODEL_NAME:-}"
MODEL_INDEX="${MODEL_INDEX:-}"
ARCHITECTURE="${ARCHITECTURE:-}"
RESULT_TAG="${RESULT_TAG:-}"
ODE_BLOCK="${ODE_BLOCK:-}"
PC_CONV="${PC_CONV:-}"
NOISY_TRIALS="${NOISY_TRIALS:-10}"
BASE_SEED="${BASE_SEED:-123}"
TEST_BS="${TEST_BS:-128}"
MUL_LEVELS="${MUL_LEVELS:-0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4}"
MAX_ADD_LEVELS="${MAX_ADD_LEVELS:-0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1}"
RMS_ADD_LEVELS="${RMS_ADD_LEVELS:-0.25,0.5,0.75,1.0,1.25}"
MAX_SQRT_LEVELS="${MAX_SQRT_LEVELS:-0,0.02,0.03,0.05,0.07,0.09,0.1}"
FF_GAIN_LIST="${FF_GAIN_LIST:-0.90,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99,1.00,1.01,1.02,1.03,1.04,1.05,1.06,1.07,1.08,1.09,1.10}"
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

NEW_PC_C100_28_2="TIMMPCNetNoBatchNorm_PCConv_0.0eps_ODEBlockPC_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_C100_3K1S128C_0.25Dropout_13Layers0l3l6_2Pool_1REP"
NEW_XINIT_C100_28_2="TIMMPCNetNoBatchNorm_PCConv_0.0eps_ODEBlockXInit_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_C100_3K1S128C_0.25Dropout_13Layers0l3l6_2Pool_1REP"

OLD_C10_16_2="TIMMPCNetNoBatchNorm_PCConv_0.0eps_ODEXInitFFFB_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_3K1S128C_0.25Dropout_7Layers0l1l2_2Pool_1REP"
OLD_C10_16_4="TIMMPCNetNoBatchNorm_PCConv_0.0eps_ODEXInitFFFB_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_3K1S256C_0.25Dropout_7Layers0l1l2_2Pool_1REP"
OLD_C10_28_2="TIMMPCNetNoBatchNorm_PCConv_0.0eps_ODEXInitFFFB_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_3K1S128C_0.25Dropout_13Layers0l3l6_2Pool_1REP"
OLD_C10_28_4="TIMMPCNetNoBatchNorm_PCConv_0.0eps_ODEXInitFFFB_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_3K1S256C_0.25Dropout_13Layers0l3l6_2Pool_1REP"
OLD_C100_16_2="TIMMPCNetNoBatchNorm_PCConv_0.0eps_ODEXInitFFFB_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_C100_3K1S128C_0.25Dropout_7Layers0l1l2_2Pool_1REP"
OLD_C100_16_4="TIMMPCNetNoBatchNorm_PCConv_0.0eps_ODEXInitFFFB_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_C100_3K1S256C_0.25Dropout_7Layers0l1l2_2Pool_1REP"
OLD_C100_28_2="TIMMPCNetNoBatchNorm_PCConv_0.0eps_ODEXInitFFFB_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_C100_3K1S128C_0.25Dropout_13Layers0l3l6_2Pool_1REP"
OLD_C100_28_4="TIMMPCNetNoBatchNorm_PCConv_0.0eps_ODEXInitFFFB_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_C100_3K1S256C_0.25Dropout_13Layers0l3l6_2Pool_1REP"

if [[ -n "${MODEL_NAME}" ]]; then
  echo "MODEL_NAME is set; using dynamic mode and ignoring MODEL_SET=${MODEL_SET}."
  if [[ -z "${MODEL_INDEX}" || -z "${ARCHITECTURE}" || -z "${ODE_BLOCK}" || -z "${PC_CONV}" ]]; then
    echo "Dynamic model requires MODEL_INDEX, ARCHITECTURE, ODE_BLOCK, and PC_CONV"
    exit 2
  fi
  if [[ "${MODEL_NAME}" == *"_T200_"* || "${MODEL_NAME}" == *"_TINY200_"* ]]; then
    task="tinyimagenet"
  elif [[ "${MODEL_NAME}" == *"_C100_"* ]]; then
    task="cifar100"
  else
    task="cifar10"
  fi
  RESULT_TAG="${RESULT_TAG:-${MODEL_NAME}}"
  OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/logs/pcn_dynamic_mismatch_slurm}"
  MODELS=("${MODEL_NAME}|${task}|${MODEL_INDEX}|${ARCHITECTURE}|${RESULT_TAG}")
else
  if [[ -z "${OUTPUT_ROOT}" ]]; then
    case "${MODEL_SET}" in
      new_x_minus_fb) OUTPUT_ROOT="${REPO_ROOT}/logs/pcn_x_minus_fb_mismatch_slurm" ;;
      legacy_no_x) OUTPUT_ROOT="${REPO_ROOT}/logs/pcn_legacy_no_x_mismatch_slurm" ;;
      *) OUTPUT_ROOT="${REPO_ROOT}/logs/pcn_with1stconv_mismatch_slurm" ;;
    esac
  fi
  ODE_BLOCK="${ODE_BLOCK:-ODEXInitFFFB}"
  PC_CONV="${PC_CONV:-PCConvNoisy}"
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
  new_x_minus_fb:0) MODELS=("${NEW_PC_C100_28_2}|cifar100|6|WRN_28_2|WRN_28_2_ODEBlockPC|ODEBlockPC|PCConvNoisy") ;;
  new_x_minus_fb:1) MODELS=("${NEW_XINIT_C100_28_2}|cifar100|6|WRN_28_2|WRN_28_2_ODEBlockXInit|ODEBlockXInit|PCConvNoisy") ;;
  legacy_no_x:0) MODELS=("${OLD_C10_16_2}|cifar10|0|WRN_16_2|WRN_16_2|ODEXInitFFFB|PCConvNoisy" "${OLD_C100_16_2}|cifar100|4|WRN_16_2|WRN_16_2|ODEXInitFFFB|PCConvNoisy") ;;
  legacy_no_x:1) MODELS=("${OLD_C10_16_4}|cifar10|1|WRN_16_4|WRN_16_4|ODEXInitFFFB|PCConvNoisy" "${OLD_C100_16_4}|cifar100|5|WRN_16_4|WRN_16_4|ODEXInitFFFB|PCConvNoisy") ;;
  legacy_no_x:2) MODELS=("${OLD_C10_28_2}|cifar10|2|WRN_28_2|WRN_28_2|ODEXInitFFFB|PCConvNoisy" "${OLD_C100_28_2}|cifar100|6|WRN_28_2|WRN_28_2|ODEXInitFFFB|PCConvNoisy") ;;
  legacy_no_x:3) MODELS=("${OLD_C10_28_4}|cifar10|3|WRN_28_4|WRN_28_4|ODEXInitFFFB|PCConvNoisy" "${OLD_C100_28_4}|cifar100|7|WRN_28_4|WRN_28_4|ODEXInitFFFB|PCConvNoisy") ;;
  *)
    echo "Unsupported MODEL_SET/SHARD_ID: ${MODEL_SET}/${SHARD_ID}"
    exit 2
    ;;
  esac
fi

for entry in "${MODELS[@]}"; do
  IFS='|' read -r model_name task model_index architecture result_tag entry_ode_block entry_pc_conv <<< "${entry}"
  if [[ "${DATASETS}" != "all" && ",${DATASETS}," != *",${task},"* ]]; then
    echo "Skipping ${model_name}: task=${task} is outside DATASETS=${DATASETS}."
    continue
  fi
  result_tag="${result_tag:-${architecture}}"
  model_ode_block="${entry_ode_block:-${ODE_BLOCK}}"
  model_pc_conv="${entry_pc_conv:-${PC_CONV}}"
  checkpoint="${MODEL_DIR}/${model_name}/${model_name}_best_ckpt.pth"
  if [[ ! -f "${checkpoint}" ]]; then
    echo "Missing checkpoint: ${checkpoint}"
    exit 3
  fi

  run_eval() {
    condition="$1"
    mismatch_type="$2"
    scale_mode="$3"
    levels="$4"
    ff_gain="$5"
    trials="$6"
    output_dir="${OUTPUT_ROOT}/${task}/${result_tag}/${condition}"
    output_pickle="${output_dir}/result.pkl"
    log_file="${output_dir}/run.log"
    mkdir -p "${output_dir}"

    if [[ "${FORCE_RERUN}" != "true" && -s "${output_pickle}" ]]; then
      echo "Skipping completed result: ${output_pickle}"
      return 0
    fi

    echo "Running ${task} ${architecture} ${condition}, ff_gain=${ff_gain}, shard=${SHARD_ID}"
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
      --noisy_trials "${trials}" \
      --seed "${BASE_SEED}" \
      --model_index "${model_index}" \
      --test_bs "${TEST_BS}" \
      --pc_conv "${model_pc_conv}" \
      --ode_block "${model_ode_block}" \
      --ff_gain "${ff_gain}" \
      --output_pickle "${output_pickle}" \
      >"${log_file}" 2>&1

    status=$?
    if [[ "${status}" != 0 ]]; then
      echo "Evaluation failed with status ${status}: ${log_file}"
      exit "${status}"
    fi
  }

  if [[ "${EVAL_MODE}" == "gain" ]]; then
    IFS=',' read -ra gains <<< "${FF_GAIN_LIST}"
    for gain in "${gains[@]}"; do
      gain_tag="${gain/./p}"
      run_eval "ff_gain/gain_${gain_tag}" "add" "max_abs" "0" "${gain}" "1"
    done
  elif [[ "${EVAL_MODE}" == "mismatch" ]]; then
    IFS=',' read -ra requested_conditions <<< "${CONDITIONS}"
    for condition in "${requested_conditions[@]}"; do
      case "${condition}" in
        multiplicative) spec="multiplicative|mul|max_abs|${MUL_LEVELS}" ;;
        max_additive) spec="max_additive|add|max_abs|${MAX_ADD_LEVELS}" ;;
        rms_additive) spec="rms_additive|add|rms|${RMS_ADD_LEVELS}" ;;
        max_sqrt_additive) spec="max_sqrt_additive|add|max_sqrt|${MAX_SQRT_LEVELS}" ;;
        *)
          echo "Unsupported mismatch condition: ${condition}"
          exit 2
          ;;
      esac
      IFS='|' read -r condition_name mismatch_type scale_mode levels <<< "${spec}"
      run_eval "${condition_name}" "${mismatch_type}" "${scale_mode}" "${levels}" "1.0" "${NOISY_TRIALS}"
    done
  else
    echo "Unsupported EVAL_MODE: ${EVAL_MODE}"
    exit 2
  fi
done

#!/usr/bin/env bash
set -euo pipefail

# Train a Level-1 unitless toggle model for the PTQ Level-2/3 comparison.
# Override values with environment variables, e.g.:
#   RESET_MODE=persistent NUM_EPOCHS=50 bash launch_scripts/run_toggle_unitless_train.sh

RESET_MODE="${RESET_MODE:-reset}" # reset or persistent
case "${RESET_MODE}" in
  reset) ODE_BLOCK="ToggleResetZ" ;;
  persistent) ODE_BLOCK="ToggleKeepZ" ;;
  *) echo "RESET_MODE must be reset or persistent, got: ${RESET_MODE}" >&2; exit 2 ;;
esac

DATASET_NAME="${DATASET_NAME:-cifar10}"
IMG_TYPE="${IMG_TYPE:-rgb}"
PCN="${PCN:-PCNetNoBatchNorm}"
PC_CONV="${PC_CONV:-PCConvReLU6}"
NUM_EPOCHS="${NUM_EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LR="${LR:-0.01}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.001}"
METHOD="${METHOD:-euler}"
TOL="${TOL:-0.0001}"
T_END="${T_END:-1.0}"
N_STEPS="${N_STEPS:-10}"
TOGGLE_TIME_SPLIT="${TOGGLE_TIME_SPLIT:-0.5}"
ODEXINIT_SCALING_MODE="${ODEXINIT_SCALING_MODE:-approx}"
TOGGLE_N_CYCLES="${TOGGLE_N_CYCLES:-${N_STEPS}}"
SAVE_PATH="${SAVE_PATH:-saved_ckpt}"
LOGDIR="${LOGDIR:-logs/toggle_ptq_check}"
mkdir -p "${LOGDIR}"

if [[ "${DATASET_NAME}" == "cifar100" ]]; then
  NUM_CLASSES="${NUM_CLASSES:-100}"
else
  NUM_CLASSES="${NUM_CLASSES:-10}"
fi

IFS=' ' read -r -a INP <<< "${INP_CHANNELS:-3 32 32 64}"
IFS=' ' read -r -a OUT <<< "${OUT_CHANNELS:-32 32 64 64}"
IFS=' ' read -r -a POOL <<< "${MAX_POOL:-0 1 0 0}"
IFS=' ' read -r -a KSZ <<< "${KERNEL_SIZE:-3}"
IFS=' ' read -r -a STRIDE <<< "${STRIDE:-1}"

cmd=(python train_ode_cifar.py
  --dataset "${DATASET_NAME}"
  --task "${DATASET_NAME}"
  --img_type "${IMG_TYPE}"
  --save_path "${SAVE_PATH}"
  --num_epochs "${NUM_EPOCHS}"
  --eval_every "${EVAL_EVERY:-1}"
  --batch_size "${BATCH_SIZE}"
  --optim "${OPTIM:-SGD}"
  --learning_rate "${LR}"
  --weight_decay "${WEIGHT_DECAY}"
  --lr_reduce_on "${LR_REDUCE_ON:-80,122,150,225,262}"
  --inp_channels "${INP[@]}"
  --out_channels "${OUT[@]}"
  --max_pool "${POOL[@]}"
  --kernel_size "${KSZ[@]}"
  --stride "${STRIDE[@]}"
  --padding "${PADDING:-1}"
  --dropout "${DROPOUT:-0.25}"
  --num_classes "${NUM_CLASSES}"
  --tie_weights "false"
  --tie_bp "false"
  --bypass "false"
  --method "${METHOD}"
  --tol "${TOL}"
  --n_steps "${N_STEPS}"
  --t_end "${T_END}"
  --pcn "${PCN}"
  --pc_conv "${PC_CONV}"
  --ode_block "${ODE_BLOCK}"
  --toggle_n_cycles "${TOGGLE_N_CYCLES}"
  --toggle_time_split "${TOGGLE_TIME_SPLIT}"
  --toggle_fast_path "${TOGGLE_FAST_PATH:-true}"
  --odexinit_scaling_mode "${ODEXINIT_SCALING_MODE}"
  --distill_method "none"
  --distill_alpha "0.0")

log_path="${LOGDIR}/train_${ODE_BLOCK}_$(date +%Y%m%d_%H%M%S).log"
printf 'Running: %q ' "${cmd[@]}" | tee "${log_path}"
printf '\n' | tee -a "${log_path}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi

"${cmd[@]}" 2>&1 | tee -a "${log_path}"
model_name="$(sed -n 's/.*Model name: //p' "${log_path}" | tail -1)"
if [[ -n "${model_name}" ]]; then
  echo "MODEL_NAME=${model_name}" | tee -a "${log_path}"
  echo "CHECKPOINT=${SAVE_PATH}/${model_name}/${model_name}_best_ckpt.pth" | tee -a "${log_path}"
fi

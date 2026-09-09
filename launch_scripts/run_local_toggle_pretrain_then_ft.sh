#!/usr/bin/env bash
# Local PCN pretraining followed by physical level-2 QAT (no final ablation).
# Activate scanbase first. Defaults reproduce the 7/14/28, 4/5/4 experiment.
# Environment overrides:
#   TASK=cifar10 CHANNELS="24 48 96" STAGE_DEPTHS="4 5 4" RUN_TAG=my_run
#   ACTIVATION_CORNER_MODE=fixed (default: random_per_forward, per_layer)
#   FT_LEARNING_RATE=0.002 FT_NUM_EPOCHS=200
#   DRY_RUN=1 prints configuration without training.
# Stage depths exclude the input convolution and two channel transitions.
# Each invocation gets a timestamped RUN_TAG unless explicitly supplied.
set -eo pipefail
# Propagate pipeline failure handling to the existing Bash launchers.
export SHELLOPTS
cd "$(dirname "${BASH_SOURCE[0]}")/.."

export TOGGLE_MODE="${TOGGLE_MODE:-odexinit}"
export DATASET_NAME="${TASK:-cifar100}"
export IMG_TYPE="${IMG_TYPE:-CiFAIR}"
export TEACHER_CKPT="${TEACHER_CKPT:-./checkpoint/efficientnet_v2_l_${DATASET_NAME}_CiFAIR_OldNoTimm_MatchDistill.pth}"
export NONLINEAR_R_TABLE="${NONLINEAR_R_TABLE:-coupler_full_range}"
export R_VAL="${R_VAL:-50e3}"
export C_VAL="${C_VAL:-500e-15}"
export V_DD="${V_DD:-0.5}"
export TOGGLE_ONE_OVER_Q="${TOGGLE_ONE_OVER_Q:-5}"
export ONE_OVER_Q="$TOGGLE_ONE_OVER_Q"
export TOGGLE_TIMING_MODE="${TOGGLE_TIMING_MODE:-fixed}"
export TOGGLE_Y_TIME="${TOGGLE_Y_TIME:-10e-9}"
export Z_OVER_Y_TIME="${Z_OVER_Y_TIME:-1}"
export SCALE_TRAIN_RECIPE="${SCALE_TRAIN_RECIPE:-1}"
export INPUT_QUANT_BITS="${INPUT_QUANT_BITS:-12}"
export CENTER_STUDENT_INPUT="${CENTER_STUDENT_INPUT:-false}"
export ENABLE_MEASURED_POOLING="${ENABLE_MEASURED_POOLING:-true}"
export WEIGHT_QUANT_FACTOR_BITS="${WEIGHT_QUANT_FACTOR_BITS:-1}"
export MC_RELU_MONTE_CARLO_SOURCE="${MC_RELU_MONTE_CARLO_SOURCE:-0906_RELU_Voltage}"
export ACTIVATION_CORNER_MODE="${ACTIVATION_CORNER_MODE:-random_per_forward}"
export ENOB="${ENOB:-none}"
# Enforced for this local launcher, including inherited nonzero settings.
export NUM_WORKERS=0
export IS_SLURM=0
export PYTHONUNBUFFERED=1

read -r -a channels <<< "${CHANNELS:-7 14 28}"
read -r -a depths <<< "${STAGE_DEPTHS:-4 5 4}"
if (( ${#channels[@]} != 3 || ${#depths[@]} != 3 )); then
  echo 'CHANNELS and STAGE_DEPTHS require three positive integers each.' >&2
  exit 2
fi
for value in "${channels[@]}" "${depths[@]}"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid architecture value: $value" >&2; exit 2; }
done
inp=(); out=(); pool=()
previous=4
for stage in 0 1 2; do
  inp+=("$previous"); out+=("${channels[stage]}"); pool+=(0)
  for ((i=1; i<=depths[stage]; i++)); do
    inp+=("${channels[stage]}"); out+=("${channels[stage]}")
    if (( stage < 2 && i == depths[stage] )); then pool+=(1); else pool+=(0); fi
  done
  previous="${channels[stage]}"
done
export INP_CHANNELS="${inp[*]}" OUT_CHANNELS="${out[*]}" MAX_POOL="${pool[*]}"

architecture="C${channels[0]}_${channels[1]}_${channels[2]}_${depths[0]}l${depths[1]}l${depths[2]}"
export EXP="${RUN_TAG:-${DATASET_NAME}_${IMG_TYPE}_${architecture}_$(date +%Y%m%d_%H%M%S)}"
export EXP_OVERRIDE="$EXP"
export OUTPUT_SAVE_PATH="${OUTPUT_SAVE_PATH:-./saved_ckpt_runs/${EXP}}"
suffix=""
if [[ "${INPUT_QUANT_BITS,,}" != none ]]; then suffix="_iq${INPUT_QUANT_BITS}"; fi
if [[ "${CENTER_STUDENT_INPUT,,}" == true ]]; then suffix+="_ctr"; fi
if [[ -n "$suffix" && "${OUTPUT_SAVE_PATH,,}" != *"${suffix,,}" ]]; then OUTPUT_SAVE_PATH+="$suffix"; fi
export MODEL_DIR="$OUTPUT_SAVE_PATH"

echo "Experiment: $EXP"
echo "Checkpoint root: $OUTPUT_SAVE_PATH"
echo "Input channels: $INP_CHANNELS"
echo "Output channels: $OUT_CHANNELS"
echo "Pooling: $MAX_POOL"
echo "DataLoader workers: $NUM_WORKERS (both stages)"
echo "FT ReLU: $MC_RELU_MONTE_CARLO_SOURCE / $ACTIVATION_CORNER_MODE"
if [[ "${DRY_RUN:-0}" == 1 ]]; then exit 0; fi

mkdir -p logs/local_runs
pretrain_log="logs/local_runs/${EXP}_pretrain.log"
bash ./launch_scripts/run_ode_train.sh 2>&1 | tee "$pretrain_log"

MODEL_NAME_OVERRIDE=$(sed -n 's/^----- Train finished, Model Name: \(.*\) -----$/\1/p' "$pretrain_log" | tail -n 1)
test -n "$MODEL_NAME_OVERRIDE"
export MODEL_NAME_OVERRIDE

bash ./launch_scripts/run_toggle_ode_mixed_ft.sh 2>&1 | tee "logs/local_runs/${EXP}_ft.log"

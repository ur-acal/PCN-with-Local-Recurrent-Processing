#!/bin/bash -l
#SBATCH -N 1
#SBATCH -p ising
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16            # modest CPU request so the node can be shared
#SBATCH --gres=gpu:1                  # exactly ONE GPU; allows packing on 4-GPU nodes
#SBATCH -t 90:10:00
#SBATCH --nodelist=bhgrb4x0081,bhgrb4x0082
#SBATCH -o /scratch/rzeng7/repos/PCN-with-Local-Recurrent-Processing/logs/slurm_jobs/slurm_%j.out

# OpenMP settings:
#export OMP_NUM_THREADS=1
#export OMP_PLACES=threads
#export OMP_PROC_BIND=spread

IS_SLURM="${IS_SLURM:-0}"

if [[ "${IS_SLURM}" == 1 ]]; then
  source activate base
  conda activate scanbase
fi

if [[ "${IS_SLURM}" == 1 || -n "${SLURM_JOB_ID:-}" ]]; then
  NUM_WORKERS="${NUM_WORKERS:-2}"
else
  NUM_WORKERS="${NUM_WORKERS:-0}"
fi

ORIG_T_INP="${ORIG_T_INP:-false}"
DATASET_NAME="${DATASET_NAME:-cifar10}"
IMG_TYPE="${IMG_TYPE:-scanGFI}"
case "${IMG_TYPE,,}" in
  cifair) IMG_TYPE="CiFAIR" ;;
  scangfi|raw|_raw) IMG_TYPE="scanGFI" ;;
esac
OUTPUT_SAVE_PATH="${OUTPUT_SAVE_PATH:-./saved_ckpt}"
INPUT_QUANT_BITS="${INPUT_QUANT_BITS:-none}"
CENTER_STUDENT_INPUT="${CENTER_STUDENT_INPUT:-false}"
INPUT_PREPROCESS_SUFFIX=""
if [[ "${INPUT_QUANT_BITS,,}" != "none" && -n "${INPUT_QUANT_BITS}" ]]; then
  INPUT_PREPROCESS_SUFFIX="_iq${INPUT_QUANT_BITS}"
fi
if [[ "${CENTER_STUDENT_INPUT,,}" == "true" ]]; then
  INPUT_PREPROCESS_SUFFIX+="_ctr"
fi
if [[ -n "${INPUT_PREPROCESS_SUFFIX}" && "${OUTPUT_SAVE_PATH,,}" != *"${INPUT_PREPROCESS_SUFFIX,,}" ]]; then
  OUTPUT_SAVE_PATH+="${INPUT_PREPROCESS_SUFFIX}"
fi
WARMUP_PRETRAIN="${WARMUP_PRETRAIN:-0}"
TIMM_RE_PROB="${TIMM_RE_PROB:-0.0}"
TOGGLE_TIMING_MODE="${TOGGLE_TIMING_MODE:-derived}"
TOGGLE_Y_TIME="${TOGGLE_Y_TIME:-5e-9}"
Z_OVER_Y_TIME="${Z_OVER_Y_TIME:-3}"
SCALE_TRAIN_RECIPE="${SCALE_TRAIN_RECIPE:-false}"
TOGGLE_MODE="${TOGGLE_MODE:-none}"
R_VAL="${R_VAL:-50e3}"
C_VAL="${C_VAL:-500e-15}"
V_DD="${V_DD:-0.1}"
ONE_OVER_Q="${ONE_OVER_Q:-1}"
ODE_BLOCK="${ODE_BLOCK:-ODEXInitFFFB}"
TOGGLE_ARGS=()
case "${TOGGLE_MODE}" in
  none)
    if [[ "${TOGGLE_TIMING_MODE}" == "fixed" ]]; then
      ODE_BLOCK="ToggleODEXInitFFFB"
    fi
    ;;
  reset) ODE_BLOCK="ToggleResetZ" ;;
  persistent) ODE_BLOCK="ToggleKeepZ" ;;
  odexinit|pulse_odexinit) ODE_BLOCK="ToggleODEXInitFFFB" ;;
  *) echo "ERROR: TOGGLE_MODE must be none/reset/persistent/odexinit/pulse_odexinit, got '${TOGGLE_MODE}'" >&2; exit 2 ;;
esac
if [[ "${TOGGLE_MODE}" != "none" || "${TOGGLE_TIMING_MODE}" == "fixed" ]]; then
  TOGGLE_ARGS=(
    --toggle_n_cycles "${TOGGLE_N_CYCLES:-5}"
    --toggle_time_split "${TOGGLE_TIME_SPLIT:-0.5}"
    --toggle_timing_mode "${TOGGLE_TIMING_MODE}"
    --toggle_y_time "${TOGGLE_Y_TIME}"
    --z_over_y_time "${Z_OVER_Y_TIME}"
    --toggle_fast_path "${TOGGLE_FAST_PATH:-true}"
    --odexinit_scaling_mode "${ODEXINIT_SCALING_MODE:-direct}"
  )
fi
EXP="${EXP:-DS_PCN}"
LOGDIR="./logs/${EXP}"
mkdir -p "${LOGDIR}"

if [[ "${ORIG_T_INP}" == "true" ]]; then
  TEACHER_CKPT="${TEACHER_CKPT:-checkpoint/efficientnet_v2_l_${DATASET_NAME}.pth}"
  TEACHER_ARCH="${TEACHER_ARCH:-efficientnet_v2_l}"
elif [[ "${IMG_TYPE}" == "CiFAIR" ]]; then
  TEACHER_CKPT="${TEACHER_CKPT:-checkpoint/efficientnet_v2_l_${DATASET_NAME}_CiFAIR_timm.pth}"
  TEACHER_ARCH="${TEACHER_ARCH:-efficientnet_v2_l}"
elif [[ "${DATASET_NAME}" == "cifar10" ]]; then
  TEACHER_CKPT="${TEACHER_CKPT:-checkpoint/b4.pth}"
  TEACHER_ARCH="${TEACHER_ARCH:-efficientnet-b4}"
else
  TEACHER_CKPT="${TEACHER_CKPT:-checkpoint/b4_100.pth}"
  TEACHER_ARCH="${TEACHER_ARCH:-efficientnet_v2_l}"
fi
TEACHER_ARCH_SOURCE="${TEACHER_ARCH_SOURCE:-auto}"
TEACHER_INPUT_SIZE="${TEACHER_INPUT_SIZE:-224}"
TEACHER_CENTER_CROP="${TEACHER_CENTER_CROP:-true}"
# Enable only for legacy run_teacher PIL checkpoints (match_distill_preprocess=false).
ADAPT_PIL_TEACHER="${ADAPT_PIL_TEACHER:-false}"

NUM_EPOCHS="${NUM_EPOCHS:-300}"
DISTILL_METHOD="${DISTILL_METHOD:-srrl}"
NEG_SAMPLE="${NEG_SAMPLE:-index}"
CONTRAST_METHOD="${CONTRAST_METHOD:-memory}"
DISTILL_ALPHA="${DISTILL_ALPHA:-0.3}"
DISTILL_TEMPERATURE="${DISTILL_TEMPERATURE:-2.0}"
IS_TIMM="${IS_TIMM:-true}"
RGGB_TO_RGB="${RGGB_TO_RGB:-false}"

# Change exp name here
EXP_SUFFIX="16L96C_${DATASET_NAME}_${NEG_SAMPLE}_${CONTRAST_METHOD}_${DISTILL_ALPHA}_${DISTILL_TEMPERATURE}"

read -r -a INP <<< "${INP_CHANNELS:-4 24 24 24 24 24 48 48 48 48 48 48 96 96 96 96}"
read -r -a OUT <<< "${OUT_CHANNELS:-24 24 24 24 24 48 48 48 48 48 48 96 96 96 96 96}"
read -r -a POOL <<< "${MAX_POOL:-0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0}"
STRIDE=(1)
KSZ=(3)
PADDING=1
NOISE_LEVEL=0.
if [[ "${RGGB_TO_RGB}" == "true" ]]; then
  INP[0]=3
fi

python train_ode_cifar.py \
  --save_path     "${OUTPUT_SAVE_PATH}" \
  --output_save_path "${OUTPUT_SAVE_PATH}" \
  --optim         "SGD" \
  --learning_rate 0.01 \
  --weight_decay  "1e-3" \
  --timm_trainer  "${IS_TIMM}" \
  --timm_sched    "cosine" \
  --timm_re_prob  "${TIMM_RE_PROB}" \
  --img_type      "${IMG_TYPE}" \
  --input_quant_bits "${INPUT_QUANT_BITS}" \
  --center_student_input "${CENTER_STUDENT_INPUT}" \
  --rggb_to_rgb   "${RGGB_TO_RGB}" \
  --dataset       "${DATASET_NAME}" \
  --num_epochs    "${NUM_EPOCHS}" \
  --warmup_epoch  "${WARMUP_PRETRAIN}" \
  --eval_every    5 \
  --offset_eps    0.0 \
  --inp_channels  "${INP[@]}" \
  --out_channels  "${OUT[@]}" \
  --max_pool      "${POOL[@]}" \
  --kernel_size   "${KSZ[@]}" \
  --padding       "${PADDING}" \
  --stride        "${STRIDE[@]}" \
  --avg_pooling   "true" \
  --dropout       0.25 \
  --tie_weights   "false" \
  --tie_bp        "false" \
  --bypass        "false" \
  --batch_size    128 \
  --num_workers   "${NUM_WORKERS}" \
  --method        "dopri5" \
  --n_steps       5 \
  --tol           "0.0001" \
  --t_end         "1.75" \
  --R             "${R_VAL}" \
  --C             "${C_VAL}" \
  --v_dd          "${V_DD}" \
  --one_over_q    "${ONE_OVER_Q}" \
  --scale_train_recipe "${SCALE_TRAIN_RECIPE}" \
  --noise_type    "mul" \
  --pcn           "PCNetNoBatchNorm" \
  --pc_conv       "PCConvReLU6" \
  --ode_block     "${ODE_BLOCK}" \
  "${TOGGLE_ARGS[@]}" \
  --teacher_ckpt "${TEACHER_CKPT}" \
  --teacher_arch "${TEACHER_ARCH}" \
  --teacher_arch_source "${TEACHER_ARCH_SOURCE}" \
  --teacher_input_size "${TEACHER_INPUT_SIZE}" \
  --teacher_center_crop "${TEACHER_CENTER_CROP}" \
    --adapt_PIL_teacher "${ADAPT_PIL_TEACHER}" \
  --distill_method  "${DISTILL_METHOD}" \
  --contrast_method "${CONTRAST_METHOD}" \
  --neg_sample      "${NEG_SAMPLE}" \
  --orig_t_inp      "${ORIG_T_INP}" \
  --distill_alpha "${DISTILL_ALPHA}" \
  --distill_temperature "${DISTILL_TEMPERATURE}" \
  2>&1 | tee "${LOGDIR}/train_${EXP}_${EXP_SUFFIX}.log"

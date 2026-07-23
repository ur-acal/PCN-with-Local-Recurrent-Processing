#!/bin/bash -l
#SBATCH -p ising
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16            # modest CPU request so the node can be shared
#SBATCH --gres=gpu:1                  # exactly ONE GPU; allows packing on 4-GPU nodes
#SBATCH -t 90:10:00
#SBATCH --nodelist=bhgrb4x0081,bhgrb4x0082
#SBATCH -o /scratch/rzeng7/repos/PCN-with-Local-Recurrent-Processing/logs/slurm_jobs/slurm_%j.out

#set -euo pipefail

# ---- Conda activation (non-interactive safe) ----
source activate base
conda activate scanbase

##########################################################################
# Inputs from launcher (packed COMB_LIST)
##########################################################################
IMG_TYPE=${IMG_TYPE:-rgb}
PCN=${PCN:-PCNetNoBatchNorm}
SWITCH_INF=${SWITCH_INF:-false}

TASK="${TASK:-cifar10}"
if [[ "$TASK" == "cifar100" ]]; then
  N_CLASSES=100
else
  N_CLASSES=10
fi

CIRC_CONF="${CIRC_CONF:-NoCirc}"
SAN_BLOCKS="${BLOCKS_LIST:-ALL}"
SAN_BLOCKS="${SAN_BLOCKS//,/+}"

EXP="${EXP:-no_bn_${PCN}_NODE_search_${TASK}_${IMG_TYPE}_${CIRC_CONF:-NoCirc}_job${SLURM_JOB_ID}_Exp}"

ODE_BLOCK="${ODE_BLOCK:-ODEXInitFFFB}"
TOGGLE_MODE="${TOGGLE_MODE:-none}"
TOGGLE_N_CYCLES="${TOGGLE_N_CYCLES:-5}"
TOGGLE_TIME_SPLIT="${TOGGLE_TIME_SPLIT:-0.5}"
TOGGLE_FAST_PATH="${TOGGLE_FAST_PATH:-true}"
TOGGLE_ONE_OVER_Q="${TOGGLE_ONE_OVER_Q:-1}"
ENABLE_MEASURED_ACTIVATION="${ENABLE_MEASURED_ACTIVATION:-false}"
SCALE_MEASURED_ACTIVATION="${SCALE_MEASURED_ACTIVATION:-false}"
ACTIVATION_CURVE_PATH="${ACTIVATION_CURVE_PATH:-./hardware_data/relu_0p3mV.csv}"
ACTIVATION_CORNER="${ACTIVATION_CORNER:-TT}"
ACTIVATION_SPLINE_PARAMETERS="${ACTIVATION_SPLINE_PARAMETERS:-10}"
MEASURED_ACTIVATION_ARGS=(
  --enable_measured_activation "${ENABLE_MEASURED_ACTIVATION}"
  --activation_curve_path "${ACTIVATION_CURVE_PATH}"
  --activation_corner "${ACTIVATION_CORNER}"
  --activation_spline_parameters "${ACTIVATION_SPLINE_PARAMETERS}"
  --activation_normalize_positive_endpoint "${SCALE_MEASURED_ACTIVATION}"
)
ENABLE_UNITLESS_MEASURED_PULLBACK="${ENABLE_UNITLESS_MEASURED_PULLBACK:-false}"
UNITLESS_PULLBACK_Q="${UNITLESS_PULLBACK_Q:-none}"
UNITLESS_PULLBACK_K="${UNITLESS_PULLBACK_K:-1e3}"
UNITLESS_PULLBACK_R="${UNITLESS_PULLBACK_R:-10e3}"
UNITLESS_MEASURED_PULLBACK_ARGS=(
  --enable_unitless_measured_pullback "${ENABLE_UNITLESS_MEASURED_PULLBACK}"
  --unitless_pullback_q "${UNITLESS_PULLBACK_Q}"
  --unitless_pullback_k "${UNITLESS_PULLBACK_K}"
  --unitless_pullback_R "${UNITLESS_PULLBACK_R}"
)
ENABLE_SPIN_VARIATION="${ENABLE_SPIN_VARIATION:-false}"
SIGMA_SPIN="${SIGMA_SPIN:-0.10}"
ENABLE_SUMMING_CURRENT_NOISE="${ENABLE_SUMMING_CURRENT_NOISE:-false}"
SUMMING_CURRENT_P="${SUMMING_CURRENT_P:-18.5e-12}"
FT_SPIN_VARIATION_ARGS=(
  --enable_spin_variation "${ENABLE_SPIN_VARIATION}"
  --sigma_spin "${SIGMA_SPIN}"
)
FT_SUMMING_CURRENT_NOISE_ARGS=(
  --enable_summing_current_noise "${ENABLE_SUMMING_CURRENT_NOISE}"
  --summing_current_p "${SUMMING_CURRENT_P}"
)
CHAN_0=${CHAN_0:-}
NUM_LAYERS=${NUM_LAYERS:-}
CHUNK_ID="${CHUNK_ID:-0}"
CHUNK_TAG="${CHUNK_TAG:-chunk}"
COMB_LIST="${COMB_LIST:-}"

if [[ -z "${COMB_LIST}" ]]; then
  echo "ERROR: Missing COMB_LIST. Launcher must export COMB_LIST (newline-delimited records)." >&2
  exit 2
fi

LOGDIR="/scratch/rzeng7/repos/PCN-with-Local-Recurrent-Processing/logs/${EXP}_${SAN_BLOCKS}"
mkdir -p "$LOGDIR"

if [[ "${CIRC_CONF}" == "NoCirc" ]]; then
  patch_node=""; patch_stride=""; patch_cycle=""; patch_pad=""; fold_scalar=""; patch_bs=""
else
  IFS='|' read -r patch_node patch_stride patch_cycle patch_pad fold_scalar patch_bs <<< "${CIRC_CONF}"
fi
TRAIN_BATCH_SIZE="${patch_bs:-128}"

##########################################################################
# Newly added TIMM + distillation args
##########################################################################
DISTILL_METHOD="${DISTILL_METHOD:-srrl}"
IS_TIMM="${IS_TIMM:-true}"

TEACHER_CKPT="${TEACHER_CKPT:-checkpoint/b4_100.pth}"
TEACHER_ARCH="${TEACHER_ARCH:-efficientnet_v2_l}"
TEACHER_ARCH_SOURCE="${TEACHER_ARCH_SOURCE:-auto}"
TEACHER_INPUT_SIZE="${TEACHER_INPUT_SIZE:-224}"
TEACHER_CENTER_CROP="${TEACHER_CENTER_CROP:-true}"
if [[ "${TASK}" == "cifar10" ]]; then
  TEACHER_CKPT="checkpoint/b4.pth"
  TEACHER_ARCH="efficientnet-b4"
fi
##########################################################################

trap 'echo "Signal caught; killing children..."; kill -- -$$ 2>/dev/null || true' TERM INT

KERNEL=(3)
padding=1
STRIDE=(1)

##########################################################################
# Blocks are passed in (kept for compatibility), but we always use ONE block.
##########################################################################
if [[ -n "${BLOCKS_LIST:-}" ]]; then
  IFS=',' read -r -a BLOCKS <<< "${BLOCKS_LIST}"
fi
BLOCKS=( "${ODE_BLOCK}" )
echo "BLOCKS_LIST=${BLOCKS_LIST}"
echo "Using single ODE block: ${BLOCKS[*]}"

TRAIN_ODE_BLOCK="${ODE_BLOCK}"
FT_ODE_BLOCK="${ODE_BLOCK}"
INF_ODE_BLOCK="${ODE_BLOCK}"
TOGGLE_ARGS=()
case "${TOGGLE_MODE}" in
  none) ;;
  reset)
    TRAIN_ODE_BLOCK="ToggleResetZ"
    FT_ODE_BLOCK="ToggleResetZ"
    INF_ODE_BLOCK="TogglePulseResetZ"
    ;;
  persistent)
    TRAIN_ODE_BLOCK="ToggleKeepZ"
    FT_ODE_BLOCK="ToggleKeepZ"
    INF_ODE_BLOCK="TogglePulseKeepZ"
    ;;
  odexinit)
    TRAIN_ODE_BLOCK="ToggleODEXInitFFFB"
    FT_ODE_BLOCK="ToggleODEXInitFFFB"
    INF_ODE_BLOCK="TogglePulseODEXInitFFFB"
    ;;
  *) echo "ERROR: TOGGLE_MODE must be none/reset/persistent/odexinit, got '${TOGGLE_MODE}'" >&2; exit 2 ;;
esac
if [[ "${TOGGLE_MODE}" != "none" ]]; then
  TOGGLE_ARGS=(
    --toggle_n_cycles "${TOGGLE_N_CYCLES}"
    --toggle_time_split "${TOGGLE_TIME_SPLIT}"
    --toggle_fast_path "${TOGGLE_FAST_PATH}"
  )
fi
echo "Toggle mode=${TOGGLE_MODE}: train=${TRAIN_ODE_BLOCK}, ft=${FT_ODE_BLOCK}, inference=${INF_ODE_BLOCK}"
echo "FT spin variation=${ENABLE_SPIN_VARIATION} (sigma=${SIGMA_SPIN}); summing-current noise=${ENABLE_SUMMING_CURRENT_NOISE} (p=${SUMMING_CURRENT_P})"
##########################################################################

##########################################################################
# Parse COMB_LIST: newline-delimited records; TAB-delimited fields
# Each Element's Record format:
#   comb_tag<TAB>params<TAB>inp_str<TAB>out_str<TAB>pool_str
##########################################################################
mapfile -t COMBS <<< "${COMB_LIST}"
NCOMB=${#COMBS[@]}
if (( NCOMB == 0 )); then
  echo "ERROR: COMB_LIST parsed to zero combos." >&2
  exit 2
fi
echo "Parsed ${NCOMB} combos in this job."
##########################################################################

# Split CPU threads evenly across the selected number of combos (since we run combos in parallel)
CPUS=${SLURM_CPUS_PER_TASK:-16}
PROCS=${NCOMB}
THREADS_PER_PROC=$(( CPUS / PROCS ))
if (( THREADS_PER_PROC < 1 )); then THREADS_PER_PROC=1; fi

# Per-process GPU memory fraction (one proc per combo)
MAX_MEM_FRAC=${MAX_MEM_FRAC:-0.9}
MEM_FRAC=$(awk -v m="$MAX_MEM_FRAC" -v n="$PROCS" 'BEGIN{printf "%.6f", m/n}')
echo "Per-proc --mem_frac=${MEM_FRAC}  (MAX_MEM_FRAC=${MAX_MEM_FRAC}, PROCS=${PROCS})"

echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}  (SLURM set)"
echo "CPUs per task: $CPUS  → OMP_NUM_THREADS per proc: $THREADS_PER_PROC"
echo "EXP=${EXP}  PCN=${PCN}  TASK=${TASK}"
echo "Logs in: $LOGDIR"
echo "CHAN_0=${CHAN_0:-unset} NUM_LAYERS=${NUM_LAYERS:-unset} CHUNK_ID=${CHUNK_ID} CHUNK_TAG=${CHUNK_TAG}"

# Helpers
extract_model_name() {
  local logfile="$1"
  sed -n 's/.*Model Name:[[:space:]]*\(.*\)[[:space:]]*-----.*$/\1/p' "$logfile" \
    | tail -n 1 \
    | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//'
}

# ------------------------ TRAIN ONE COMBO ------------------------
run_one_combo () {
  local comb_line="$1"
  local block="$2"

  local comb_tag params inp_str out_str pool_str
  IFS=$'\t' read -r comb_tag params inp_str out_str pool_str <<< "${comb_line}"

  if [[ -z "${comb_tag}" || -z "${inp_str}" || -z "${out_str}" || -z "${pool_str}" ]]; then
    echo "ERROR: Bad COMB_LIST record (missing fields): ${comb_line}" >&2
    return 2
  fi

  # Parse arrays
  read -r -a INP  <<< "${inp_str}"
  read -r -a OUT  <<< "${out_str}"
  read -r -a POOL <<< "${pool_str}"

  if (( ${#INP[@]} != ${#OUT[@]} || ${#INP[@]} != ${#POOL[@]} )); then
    echo "ERROR: INP/OUT/POOL length mismatch for ${comb_tag}: INP=${#INP[@]} OUT=${#OUT[@]} POOL=${#POOL[@]}" >&2
    echo "  INP_STR=${inp_str}" >&2
    echo "  OUT_STR=${out_str}" >&2
    echo "  POOL_STR=${pool_str}" >&2
    return 2
  fi

  if [[ "$IMG_TYPE" != "rgb" ]]; then
    INP[0]=4
  fi

  local comb_logdir="${LOGDIR}/${comb_tag}"
  mkdir -p "${comb_logdir}"

  local logfile="${comb_logdir}/train_${EXP}_${comb_tag}_${block}.log"
  echo "[$(date +%T)] TRAIN start comb=${comb_tag} params=${params} block=${block} -> ${logfile}"
  echo "  INP=(${INP[*]})" | tee -a "${logfile}"
  echo "  OUT=(${OUT[*]})" | tee -a "${logfile}"
  echo "  POOL=(${POOL[*]})" | tee -a "${logfile}"

  local t_end="1.75"

  train_solver="dopri5"
  if [[ "${SWITCH_INF}" == "true" ]]; then
    train_solver="euler"
  fi

  OMP_NUM_THREADS=${THREADS_PER_PROC} \
  python -u train_ode_cifar.py \
    --optim         "SGD" \
    --timm_trainer  "${IS_TIMM}" \
    --timm_sched    "cosine" \
    --img_type      "${IMG_TYPE}" \
    --dataset       "${TASK}" \
    --num_classes   "${N_CLASSES}" \
    --num_epochs    300 \
    --eval_every    5 \
    --learning_rate 0.01 \
    --offset_eps    0.0 \
    --inp_channels  "${INP[@]}" \
    --out_channels  "${OUT[@]}" \
    --max_pool      "${POOL[@]}" \
    --stride        "${STRIDE[@]}" \
    --avg_pooling   "true" \
    --kernel_size   "${KERNEL[@]}" \
    --padding       "${padding}" \
    --patch_node    "${patch_node}" \
    --patch_stride  "${patch_stride}" \
    --patch_cycle   "${patch_cycle}" \
    --patch_pad     "${patch_pad}" \
    --fold_scalar   "${fold_scalar}" \
    --dropout       0.25 \
    --weight_decay  "1e-3" \
    --tie_weights   "false" \
    --tie_bp        "false" \
    --bypass        "false" \
    --batch_size    "${TRAIN_BATCH_SIZE}" \
    --method        "${train_solver}" \
    --n_steps        5 \
    --tol           "1e-4" \
    --t_end         "${t_end}" \
    --pcn           "${PCN}" \
    --pc_conv       "PCConvReLU6" \
    --ode_block     "${block}" \
    "${TOGGLE_ARGS[@]}" \
    "${MEASURED_ACTIVATION_ARGS[@]}" \
    "${UNITLESS_MEASURED_PULLBACK_ARGS[@]}" \
    --v_dd          "${V_DD}" \
    --one_over_q    "${ONE_OVER_Q}" \
    --teacher_ckpt  "${TEACHER_CKPT}" \
    --teacher_arch  "${TEACHER_ARCH}" \
    --teacher_arch_source "${TEACHER_ARCH_SOURCE}" \
    --teacher_input_size "${TEACHER_INPUT_SIZE}" \
    --teacher_center_crop "${TEACHER_CENTER_CROP}" \
    --distill_method "${DISTILL_METHOD}" \
    --distill_alpha 0.3 \
    --distill_temperature 2.0 \
    --test_only     "false" \
    --mem_frac      "${MEM_FRAC}" \
    2>&1 | tee "${logfile}"
}

# ------------------------ EVAL ONE MODEL ------------------------
MODEL_DIR="./saved_ckpt"
METHOD="dopri5"
TOL="1e-6"
n_bits=5
ENOB=8
V_DD="0.1"
ONE_OVER_Q="1"
if [[ "${TOGGLE_MODE}" != "none" ]]; then
  ONE_OVER_Q="${TOGGLE_ONE_OVER_Q}"
fi
R_val="10e3"
R_MAX="150e3"
C_val="49e-15"
EVAL_BASE="${LOGDIR}/evals"
MASTER_LOG="${LOGDIR}/eval_master_merged.log"
mkdir -p "${EVAL_BASE}"
: > "${MASTER_LOG}"

eval_one_combo_model() {
  local comb_tag="$1"
  local name="$2"
  local tag="$3"         # pre_ft or post_ft
  local test_block="$4"  # always ODE_BLOCK

  local __rest="${name#*_}"
  local pc_conv="${__rest%%_*}"

  local dir="${EVAL_BASE}/${comb_tag}/${tag}/${name}_method_${METHOD}_tol_${TOL}"
  mkdir -p "$dir"

  local ckpt="best"

  local test_bs=128

  local wrapper="ODEWrapper1State"
  if [[ "$name" == *S2* || "$name" == *State2* ]]; then
    if [[ "$name" == *QAT* && "$ckpt" != *full_param* ]]; then
      wrapper="QATTester2State"
    else
      wrapper="ODEWrapper2State"
    fi
  else
    if [[ "$name" == *QAT* && "$ckpt" != *full_param* ]]; then
      wrapper="QATTester1State"
    fi
  fi
  if [[ "${TOGGLE_MODE}" != "none" ]]; then
    wrapper="ToggleQATTester1State"
  fi

  test_expanded="true"
  if [[ "${SWITCH_INF}" == "true" && "${TOGGLE_MODE}" == "none" ]]; then
    # Todo: Hardcode for now
    test_block="ODEXInitFFFBPixelSwitchStretchT"
    test_expanded="false"
  fi

  echo "[EVAL ${tag}] comb=${comb_tag} ${name} (block=${test_block}, ckpt=${ckpt}) → $dir/job.log"
  OMP_NUM_THREADS=${THREADS_PER_PROC} \
  python -u ode_inference.py \
    --model_name      "$name" \
    --ckpt            "${ckpt}" \
    --task            "${TASK}" \
    --model_dir       "$MODEL_DIR" \
    --test_bs         "${test_bs}" \
    --method          "$METHOD" \
    --tol             "$TOL" \
    --n_steps         100 \
    --ts_scale        1 \
    --d_start         0 \
    --d_end           1 \
    --n_sweep_left    0 \
    --n_sweep_right   1 \
    --thermal_noise   "true" \
    --sde_noise_type  "mul" \
    --sweep_eps       "false" \
    --R               "${R_val}" \
    --R_max           "${R_MAX}" \
    --C               "${C_val}" \
    --v_dd            "${V_DD}" \
    --enob            "${ENOB}" \
    --w_bits          "${n_bits}" \
    --patch_node      "${patch_node}" \
    --patch_stride    "${patch_stride}" \
    --patch_cycle     "${patch_cycle}" \
    --patch_pad       "${patch_pad}" \
    --fold_scalar     "${fold_scalar}" \
    --tie_cap         "false" \
    --one_over_q      "${ONE_OVER_Q}" \
    "${TOGGLE_ARGS[@]}" \
    "${MEASURED_ACTIVATION_ARGS[@]}" \
    "${FT_SPIN_VARIATION_ARGS[@]}" \
    "${FT_SUMMING_CURRENT_NOISE_ARGS[@]}" \
    --pc_conv         "${pc_conv}Noisy" \
    --ode_block       "${test_block}" \
    --ode_wrapper     "${wrapper}" \
    --img_type        "${IMG_TYPE}" \
    --switch_period   "" \
    --switch_iter     "5" \
    --noisy_trials    "20" \
    --conv_only       "true" \
    --test_expanded   "${test_expanded}" \
    --diff_mismatch   "true" \
    --nonlinear_R     "false" \
    --mem_frac        "${MEM_FRAC}" \
    2>&1 | tee -a "$dir/job.log"
}

# ------------------------ FINETUNE ONE MODEL ------------------------
finetune_one_combo_model() {
  local comb_tag="$1"
  local name="$2"
  local train_block="$3"

  local comb_logdir="${LOGDIR}/${comb_tag}"
  mkdir -p "${comb_logdir}"
  local logfile="${comb_logdir}/finetune_${train_block}.log"

  echo "[FINETUNE] comb=${comb_tag} ${name} (block=${train_block}) → $logfile"

  local ckpt="best"
  local t_end="1.75"
  local qat_wrapper="QATWrapper1State"
  if [[ "$name" == *S2* || "$name" == *State2* ]]; then
    qat_wrapper="QATWrapper2State"
  fi
  if [[ "${TOGGLE_MODE}" != "none" ]]; then
    qat_wrapper="ToggleQATWrapper1State"
  fi

  local nl="0.25"
  local nt="mul"

  train_solver="dopri5"
  if [[ "${SWITCH_INF}" == "true" ]]; then
    train_solver="euler"
  fi

  OMP_NUM_THREADS=${THREADS_PER_PROC} \
  python -u train_ode_cifar.py \
    --dataset       "${TASK}" \
    --timm_trainer  "${IS_TIMM}" \
    --timm_sched    "cosine" \
    --timm_aug_level "no_aug" \
    --num_classes   "${N_CLASSES}" \
    --ckpt          "${ckpt}" \
    --optim         "SGD" \
    --learning_rate 0.005 \
    --eval_every    2 \
    --num_epochs    140 \
    --img_type      "${IMG_TYPE}" \
    --model_name    "${name}" \
    --offset_eps    0.0 \
    --dropout       0.25 \
    --avg_pooling   "true" \
    --tie_weights   "false" \
    --tie_bp        "false" \
    --bypass        "false" \
    --batch_size    128 \
    --method        "${train_solver}" \
    --n_steps       5 \
    --tol           "${TOL}" \
    --t_end         "${t_end}" \
    --R             "${R_val}" \
    --R_max         "${R_MAX}" \
    --C             "${C_val}" \
    --v_dd          "${V_DD}" \
    --enob          "${ENOB}" \
    --w_bits        "${n_bits}" \
    --patch_node    "${patch_node}" \
    --patch_stride  "${patch_stride}" \
    --patch_cycle   "${patch_cycle}" \
    --patch_pad     "${patch_pad}" \
    --fold_scalar   "${fold_scalar}" \
    --tie_cap       "false" \
    --one_over_q    "${ONE_OVER_Q}" \
    "${TOGGLE_ARGS[@]}" \
    "${MEASURED_ACTIVATION_ARGS[@]}" \
    "${FT_SPIN_VARIATION_ARGS[@]}" \
    "${FT_SUMMING_CURRENT_NOISE_ARGS[@]}" \
    --qat_cls       "SymQuantizeWeight" \
    --pcn           "${PCN}" \
    --ode_wrapper   "${qat_wrapper}" \
    --pc_conv       "PCConvReLU6" \
    --ode_block     "${train_block}" \
    --noise_level   "${nl}" \
    --noise_type    "${nt}" \
    --teacher_ckpt  "${TEACHER_CKPT}" \
    --teacher_arch  "${TEACHER_ARCH}" \
    --teacher_arch_source "${TEACHER_ARCH_SOURCE}" \
    --teacher_input_size "${TEACHER_INPUT_SIZE}" \
    --teacher_center_crop "${TEACHER_CENTER_CROP}" \
    --distill_method "${DISTILL_METHOD}" \
    --distill_alpha 0.3 \
    --distill_temperature 2.0 \
    --contrast_method "memory" \
    --test_only     "false" \
    --mem_frac      "${MEM_FRAC}" \
    2>&1 | tee "${logfile}"
}

###############################################################################
# PHASE 1: TRAIN all combos in parallel
###############################################################################
echo "==== PHASE 1: TRAIN (parallel over combos) ===="
pids=()
for comb in "${COMBS[@]}"; do
  run_one_combo "${comb}" "${TRAIN_ODE_BLOCK}" & pids+=($!)
done

ok=0
for p in "${pids[@]}"; do
  wait "$p" || ok=1
done
echo "Completed TRAIN phase for ${EXP} (job $SLURM_JOB_ID)."

###############################################################################
# Collect model names from train logs (one per combo)
###############################################################################
declare -a COMB_TAGS=()
declare -a MODEL_NAMES_PRE=()
for comb in "${COMBS[@]}"; do
  # record: comb_tag<TAB>params<TAB>inp_str<TAB>out_str<TAB>pool_str
  IFS=$'\t' read -r comb_tag params inp_str out_str pool_str <<< "${comb}"
  COMB_TAGS+=( "${comb_tag}" )
  train_log="${LOGDIR}/${comb_tag}/train_${EXP}_${comb_tag}_${TRAIN_ODE_BLOCK}.log"
  mn="$(extract_model_name "${train_log}")"
  if [[ -z "$mn" ]]; then
    echo "[WARN] Could not extract model name from ${train_log}"
  fi
  MODEL_NAMES_PRE+=( "${mn}" )
done

###############################################################################
# PHASE 2: EVAL pre_ft all combos in parallel
# Skipped
###############################################################################


###############################################################################
# PHASE 3: FINETUNE all combos in parallel
###############################################################################
echo "==== PHASE 3: FINETUNE (parallel over combos) ===="
pids=()
for i in "${!COMB_TAGS[@]}"; do
  comb_tag="${COMB_TAGS[$i]}"
  mn="${MODEL_NAMES_PRE[$i]}"
  if [[ -z "$mn" ]]; then
    echo "[WARN] Skip finetune for comb=${comb_tag} (empty model name)"
    continue
  fi
  finetune_one_combo_model "${comb_tag}" "${mn}" "${FT_ODE_BLOCK}" & pids+=($!)
done
for p in "${pids[@]}"; do wait "$p" || ok=1; done

###############################################################################
# Collect finetuned model names (one per combo)
###############################################################################
declare -a MODEL_NAMES_FT=()
for i in "${!COMB_TAGS[@]}"; do
  comb_tag="${COMB_TAGS[$i]}"
  ft_log="${LOGDIR}/${comb_tag}/finetune_${FT_ODE_BLOCK}.log"
  mn_ft="$(extract_model_name "${ft_log}")"
  if [[ -z "$mn_ft" ]]; then
    echo "[WARN] Could not extract finetuned model name from ${ft_log} (comb=${comb_tag})"
  fi
  MODEL_NAMES_FT+=( "${mn_ft}" )
done

###############################################################################
# PHASE 4: EVAL post_ft all combos in parallel
###############################################################################
echo "==== PHASE 4: EVAL post_ft (parallel over combos) ===="
pids=()
for i in "${!COMB_TAGS[@]}"; do
  comb_tag="${COMB_TAGS[$i]}"
  mn_ft="${MODEL_NAMES_FT[$i]}"
  if [[ -z "$mn_ft" ]]; then
    echo "[WARN] Skip post_ft eval for comb=${comb_tag} (empty finetuned model name)"
    continue
  fi
  eval_one_combo_model "${comb_tag}" "${mn_ft}" "post_ft" "${INF_ODE_BLOCK}" & pids+=($!)
done
for p in "${pids[@]}"; do wait "$p" || ok=1; done

###############################################################################
# Merge summaries into MASTER_LOG (simple: append per-combo job logs)
###############################################################################
echo "Merging summaries into ${MASTER_LOG}"
: > "${MASTER_LOG}"
for i in "${!COMB_TAGS[@]}"; do
  comb_tag="${COMB_TAGS[$i]}"
  # pre="${MODEL_NAMES_PRE[$i]}" # pre ft evaluation skipped
  post="${MODEL_NAMES_FT[$i]}"

  echo "==================== COMB ${comb_tag} ====================" >> "${MASTER_LOG}"

  if [[ -n "$post" ]]; then
    post_log="${EVAL_BASE}/${comb_tag}/post_ft/${post}_method_${METHOD}_tol_${TOL}/job.log"
    printf '---------- %s (post-finetune) ----------\n' "$post" >>"${MASTER_LOG}"
    if [[ -f "$post_log" ]]; then
      cat "$post_log" >> "${MASTER_LOG}"
    else
      echo "[Missing log: $post_log]" >> "${MASTER_LOG}"
    fi
    printf '\n\n' >> "${MASTER_LOG}"
  fi
done

echo "All summaries written to ${MASTER_LOG}"

################################################################################
#                               RUN PARSER
################################################################################
PARSE_WRAPPER="shell_utils/pipeline_node_run_parse.py"
if [[ -f "$PARSE_WRAPPER" ]]; then
  echo "[PARSE] python $PARSE_WRAPPER ${MASTER_LOG}"
  python -u "$PARSE_WRAPPER" "${MASTER_LOG}"
else
  echo "[PARSE] Skipping parse: ${PARSE_WRAPPER} not found."
fi
exit $ok

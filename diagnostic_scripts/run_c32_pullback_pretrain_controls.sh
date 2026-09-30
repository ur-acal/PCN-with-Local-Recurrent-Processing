#!/usr/bin/env bash

# Local controls for the CiFAIR-100 C32->64 measured-ReLU
# pretraining convergence issue.  The baseline mirrors the pretraining phase
# of run_kdcrd_then_ft.sbatch for the submitted 11:11 two-stage architecture;
# only NUM_EPOCHS and NUM_WORKERS differ from the production SLURM run.

set -u
set -o pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

NUM_EPOCHS="${NUM_EPOCHS:-3}"
NUM_WORKERS="${NUM_WORKERS:-0}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-0}"
STAGE0_LAYERS="${STAGE0_LAYERS:-11}"
STAGE1_LAYERS="${STAGE1_LAYERS:-11}"
STAGE0_CHANNELS="${STAGE0_CHANNELS:-32}"
STAGE1_CHANNELS="${STAGE1_CHANNELS:-64}"
SMALL_SRRL_WEIGHT="${SMALL_SRRL_WEIGHT:-0.1}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
CASES="${CASES:-baseline boundary_bn no_srrl srrl_0p1 no_timm_aug}"

output_root="saved_ckpt_runs/diagnostics/c32_pullback_controls_${RUN_ID}"
log_root="logs/local_runs/c32_pullback_controls_${RUN_ID}"
summary_path="${log_root}/summary.txt"
curve_root="${output_root}/activation_curves"
mkdir -p "${output_root}" "${log_root}"

python diagnostic_scripts/generate_mc18_activation_controls.py \
  --source ./hardware_data/mc_45_corners/0906_RELU_Voltage/tt_25_1.csv \
  --output-dir "${curve_root}" \
  | tee "${log_root}/activation_curves.txt"

inp=(4)
out=("${STAGE0_CHANNELS}")
pool=(0)
for ((i = 1; i <= STAGE0_LAYERS; i++)); do
  inp+=("${STAGE0_CHANNELS}")
  out+=("${STAGE0_CHANNELS}")
  if ((i == STAGE0_LAYERS)); then pool+=(1); else pool+=(0); fi
done
inp+=("${STAGE0_CHANNELS}")
out+=("${STAGE1_CHANNELS}")
pool+=(0)
for ((i = 1; i <= STAGE1_LAYERS; i++)); do
  inp+=("${STAGE1_CHANNELS}")
  out+=("${STAGE1_CHANNELS}")
  pool+=(0)
done

printf 'RUN_ID=%s\nNUM_EPOCHS=%s\nWARMUP_EPOCHS=%s\nSTAGE_CHANNELS=%s:%s\nSTAGE_LAYERS=%s:%s\nINP=%s\nOUT=%s\nPOOL=%s\nCASES=%s\n' \
  "${RUN_ID}" "${NUM_EPOCHS}" "${WARMUP_EPOCHS}" \
  "${STAGE0_CHANNELS}" "${STAGE1_CHANNELS}" \
  "${STAGE0_LAYERS}" "${STAGE1_LAYERS}" \
  "${inp[*]}" "${out[*]}" "${pool[*]}" "${CASES}" \
  | tee "${summary_path}"

run_case() {
  local case_name="$1"
  local pcn="$2"
  local distill_method="$3"
  local srrl_weight="$4"
  local timm_aug_level="$5"
  local pc_conv="${6:-PCConvReLU6}"
  local activation_curve_path="${7:-./hardware_data/mc_45_corners/0906_RELU_Voltage/tt_25_1.csv}"
  local normalize_endpoint="${8:-false}"
  local pullback_mode="${9:-direct}"
  local enable_measured_activation="${10:-true}"
  local case_output="${output_root}/${case_name}"
  local case_log="${log_root}/${case_name}.log"
  mkdir -p "${case_output}"

  printf '\n[%s] Starting %s: pcn=%s distill=%s srrl_weight=%s timm_aug=%s\n' \
    "$(date --iso-8601=seconds)" "${case_name}" "${pcn}" \
    "${distill_method}" "${srrl_weight}" "${timm_aug_level}" \
    | tee -a "${summary_path}"

  OMP_NUM_THREADS=16 python -u train_ode_cifar.py \
    --save_path "${case_output}" \
    --output_save_path "${case_output}" \
    --optim SGD \
    --timm_trainer true \
    --timm_sched cosine \
    --timm_aug_level "${timm_aug_level}" \
    --timm_re_prob 0.0 \
    --img_type CiFAIR \
    --input_quant_bits 12 \
    --center_student_input false \
    --dataset cifar100 \
    --validation_mode false \
    --validation_split_seed 4096 \
    --num_classes 100 \
    --num_epochs "${NUM_EPOCHS}" \
    --eval_every 5 \
    --final_eval_only false \
    --health_check_epochs 30,75 \
    --health_check_batches 4 \
    --health_check_seed 4096 \
    --warmup_epoch "${WARMUP_EPOCHS}" \
    --learning_rate 0.01 \
    --offset_eps 0.0 \
    --inp_channels "${inp[@]}" \
    --out_channels "${out[@]}" \
    --max_pool "${pool[@]}" \
    --stride 1 \
    --avg_pooling true \
    --kernel_size 3 \
    --padding 1 \
    --dropout 0.25 \
    --weight_decay 1e-3 \
    --tie_weights false \
    --tie_bp false \
    --bypass false \
    --batch_size 128 \
    --num_workers "${NUM_WORKERS}" \
    --method dopri5 \
    --n_steps 5 \
    --tol 1e-4 \
    --t_end 1.75 \
    --R 50e3 \
    --C 500e-15 \
    --scale_train_recipe 1 \
    --pcn "${pcn}" \
    --pc_conv "${pc_conv}" \
    --ode_block ToggleODEXInitFFFB \
    --toggle_n_cycles 5 \
    --toggle_time_split 0.5 \
    --toggle_fast_path true \
    --odexinit_scaling_mode direct \
    --toggle_timing_mode fixed \
    --toggle_y_time 10e-9 \
    --z_over_y_time 1 \
    --enable_measured_activation "${enable_measured_activation}" \
    --activation_curve_path "${activation_curve_path}" \
    --activation_corner MC18 \
    --activation_corner_mode fixed \
    --activation_random_curve_sharing per_layer \
    --activation_interpolation piecewise_linear \
    --fuse_measured_activation true \
    --activation_spline_parameters 10 \
    --activation_fit_constraint auto \
    --activation_normalize_positive_endpoint "${normalize_endpoint}" \
    --unitless_measured_pullback_mode "${pullback_mode}" \
    --unitless_pullback_q none \
    --unitless_pullback_k 1e3 \
    --unitless_pullback_R 10e3 \
    --v_dd 0.5 \
    --one_over_q 5 \
    --teacher_ckpt ./checkpoint/efficientnet_v2_l_cifar100_CiFAIR_OldNoTimm_MatchDistill.pth \
    --teacher_arch efficientnet_v2_l \
    --teacher_arch_source auto \
    --teacher_input_size 224 \
    --teacher_center_crop true \
    --adapt_PIL_teacher false \
    --distill_method "${distill_method}" \
    --srrl_weight "${srrl_weight}" \
    --distill_alpha 0.3 \
    --distill_temperature 2.0 \
    --test_only false \
    --mem_frac 1 \
    2>&1 | tee "${case_log}"
  local status="${PIPESTATUS[0]}"

  printf '[%s] Finished %s with exit code %s\n' \
    "$(date --iso-8601=seconds)" "${case_name}" "${status}" \
    | tee -a "${summary_path}"
  tr '\r' '\n' < "${case_log}" \
    | grep 'Iter=390/390' \
    | sed -E 's/^.*Iter=/Iter=/' \
    | awk '!seen[$0]++' >> "${summary_path}" || true
  return "${status}"
}

for case_name in ${CASES}; do
  case "${case_name}" in
    baseline)
      run_case "${case_name}" PCNetNoBatchNorm srrl 1.0 none || true
      ;;
    boundary_bn)
      run_case "${case_name}" PCNetBoundaryBN srrl 1.0 none || true
      ;;
    no_srrl)
      run_case "${case_name}" PCNetNoBatchNorm none 1.0 none || true
      ;;
    srrl_0p1)
      run_case "${case_name}" PCNetNoBatchNorm srrl "${SMALL_SRRL_WEIGHT}" none || true
      ;;
    no_timm_aug)
      run_case "${case_name}" PCNetNoBatchNorm srrl 1.0 no_aug || true
      ;;
    ideal_relu5)
      run_case "${case_name}" PCNetNoBatchNorm srrl 1.0 none \
        PCConvReLU5 unused false none false || true
      ;;
    endpoint_normalized)
      run_case "${case_name}" PCNetNoBatchNorm srrl 1.0 none \
        PCConvReLU6 \
        ./hardware_data/mc_45_corners/0906_RELU_Voltage/tt_25_1.csv \
        true direct true || true
      ;;
    zero_offset)
      run_case "${case_name}" PCNetNoBatchNorm srrl 1.0 none \
        PCConvReLU6 "${curve_root}/mc18_zero_offset.csv" \
        false direct true || true
      ;;
    zero_offset_relu)
      run_case "${case_name}" PCNetNoBatchNorm srrl 1.0 none \
        PCConvReLU6 "${curve_root}/mc18_zero_offset_relu.csv" \
        false direct true || true
      ;;
    *)
      printf 'Unknown case: %s\n' "${case_name}" | tee -a "${summary_path}"
      ;;
  esac
done

printf '\n[%s] Diagnostic suite finished.\n' "$(date --iso-8601=seconds)" \
  | tee -a "${summary_path}"

#!/usr/bin/env bash
# One matched trial per switched-update ordering for the active model in
# run_ode_switch_inf.sh. Leakage is disabled. Runs sequentially and fails loud.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

MODEL_NAME="${MODEL_NAME:-TIMMQAT5bNT0p25mulTIMMPCNetNoBatchNorm_PCConvReLU6_0.0eps_ODEXInitFFFB_eulerSolver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_C100_3K1S72C_0.25Dropout_12Layers11l0l0_1Pool5_srrlDistill_a0p3_t2p0_scanGFI_3REP}"
SOLVER_METHOD="${SOLVER_METHOD:-euler}"
SOLVER_TOL="${SOLVER_TOL:-1e-6}"
RESULT_ROOT="${RESULT_ROOT:-./results/switch_update_order_ablation_${SOLVER_METHOD}_tol${SOLVER_TOL}}"
mkdir -p "$RESULT_ROOT"

run_case() {
  local case_name="$1"
  local ode_block="$2"
  local log_path="$RESULT_ROOT/${case_name}.log"
  echo "Starting ${case_name}: ${ode_block}" | tee "$log_path"
  python -u ode_inference.py \
    --model_name "$MODEL_NAME" \
    --model_dir ./saved_ckpt \
    --ckpt best \
    --task cifar100 \
    --img_type scanGFI \
    --test_bs 128 \
    --method "$SOLVER_METHOD" \
    --tol "$SOLVER_TOL" \
    --n_steps 5 \
    --ts_scale 1 \
    --d_start 0 \
    --d_end 1 \
    --n_sweep_left 0 \
    --n_sweep_right 1 \
    --thermal_noise true \
    --sde_noise_type mul \
    --mismatch_type mul \
    --sweep_eps false \
    --noise_level_list 0 \
    --R 20e3 \
    --R_max 300e3 \
    --C 49e-15 \
    --v_dd 0.1 \
    --enob 8 \
    --w_bits 5 \
    --patch_node 8 \
    --patch_stride 8 \
    --patch_cycle 1 \
    --patch_pad 0 \
    --fold_scalar 1 \
    --tie_cap false \
    --one_over_q 1 \
    --pc_conv PCConvReLU6Noisy \
    --ode_block "$ode_block" \
    --ode_wrapper QATTester1State \
    --noisy_trials 1 \
    --hardware_seed 4096 \
    --data_seed 4096 \
    --switch_period none \
    --switch_iter "${SWITCH_ITER:-5}" \
    --switch_block_size "${SWITCH_BLOCK_SIZE:-1}" \
    --i_leak none \
    --conv_only true \
    --test_expanded false \
    --diff_mismatch false \
    --nonlinear_R false \
    --test_only false \
    2>&1 | tee -a "$log_path"
  grep -q 'Average test acc over' "$log_path" || {
    echo "ERROR: ${case_name} completed without an accuracy result" >&2
    return 1
  }
  echo "Finished ${case_name}" | tee -a "$log_path"
}

case "${SWITCH_CASE:-all}" in
  frozen) run_case frozen_old_values ODEXInitFFFBPixelSwitchEfficient ;;
  updated) run_case forward_updated_values ODEXInitFFFBPixelSwitchExplicit ;;
  strang) run_case strang_forward_reverse ODEXInitFFFBPixelSwitchStrang ;;
  all)
    run_case frozen_old_values ODEXInitFFFBPixelSwitchEfficient
    run_case forward_updated_values ODEXInitFFFBPixelSwitchExplicit
    run_case strang_forward_reverse ODEXInitFFFBPixelSwitchStrang
    ;;
  *) echo "SWITCH_CASE must be frozen, updated, strang, or all" >&2; exit 2 ;;
esac

#!/usr/bin/env bash
# Restart CIFAR-100 22L6l7l6 FT from pretraining, changing only the solver.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
state="${1:-1}"
case "$state" in
  1) block=ODEXInitFFFB ;;
  2) block=S2NoisyIYAsXZAs0 ;;
  *) echo 'Usage: bash launch_scripts/local_tc_rgb_ft_euler5.sh {1|2} [--dry-run]' >&2; exit 2 ;;
esac
export TC_NONIDEALITIES=true TC_STATE="$state" TOGGLE_MODE=none SWITCH_INF=false
source ./launch_scripts/tc_nonideality_args.sh ft
model="TIMMPCNetNoBatchNorm_PCConvReLU6_0.0eps_${block}_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_C100_3K1S64C_0.25Dropout_22Layers6l7l6_2Pool_srrlDistill_a0p3_t2p0_2REP"
root="./saved_ckpt_runs/tc_rgb_cifar100_state${state}_pcn_resnet_depth_study"
teacher=./checkpoint/efficientnet_v2_l_cifar100_rgb_OldNoTimm_MatchDistill.pth
for checkpoint in "${root}/${model}/${model}_last_ckpt.pth" "$teacher"; do
  [[ -s "$checkpoint" ]] || { echo "Missing checkpoint: $checkpoint" >&2; exit 2; }
done
cmd=(python -u train_ode_cifar.py
  --model_name "$model" --save_path "$root" --output_save_path "${root}_ft_euler5"
  --dataset cifar100 --num_classes 100 --img_type rgb --ckpt last
  --timm_trainer true --timm_sched cosine --timm_aug_level no_aug --timm_re_prob 0.0
  --optim SGD --learning_rate 0.005 --num_epochs 140 --warmup_epoch 0
  --eval_every 2 --final_eval_only true
  --health_check_epochs 20,50 --health_check_batches 4 --health_check_seed 4096
  --input_quant_bits none --center_student_input false
  --offset_eps 0.0 --dropout 0.25 --avg_pooling true
  --tie_weights false --tie_bp false --bypass false
  --batch_size 128 --tol 1e-6 --t_end 1.75 --scale_train_recipe false
  --patch_node "" --patch_stride "" --patch_cycle "" --patch_pad "" --fold_scalar ""
  --tie_cap false --qat_cls SymQuantizeWeight --pcn PCNetNoBatchNorm --pc_conv PCConvReLU6
  --noise_type mul --pulse_mismatch_training_mode post_quant_amplitude
  --activation_corner_mode fixed --activation_random_curve_sharing per_layer
  --activation_interpolation piecewise_linear --activation_spline_parameters 10
  --activation_fit_constraint auto --activation_normalize_positive_endpoint false
  --slow_summing_current 2.47e-9 --slow_coupler_noise 2.47e-9
  --nonlinear_R_mc_quantity conductance --train_conv_expanded false --nonlinear_R_corner_range all
  --teacher_ckpt "$teacher" --teacher_arch efficientnet_v2_l --teacher_arch_source torchvision
  --teacher_input_size 224 --teacher_center_crop true --adapt_PIL_teacher false
  --distill_method srrl --distill_alpha 0.3 --distill_temperature 2.0
  --reviewkd_weight 1.0 --reviewkd_warmup_epochs 20 --reviewkd_num_stages 4
  --contrast_method memory --test_only false --mem_frac 0.9 --num_workers 2
  "${TC_ARGS[@]}"
  # Must follow TC_ARGS, which otherwise selects dopri5.
  --method euler --n_steps 5)
if [[ "${2:-}" == --dry-run ]]; then
  printf '%q ' "${cmd[@]}"; printf '\n'
else
  exec "${cmd[@]}"
fi

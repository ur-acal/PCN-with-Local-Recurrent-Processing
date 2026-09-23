#!/usr/bin/env bash
# Local equivalent of docs/training_recovery.md's 1-state FT redo.
set -o pipefail
cd "$(dirname "$0")/.." || exit 2
mkdir -p logs/local_runs || exit 2
source /home/rongzeng/anaconda3/etc/profile.d/conda.sh
conda activate scanbase || exit 2
export OMP_NUM_THREADS=16
export TC_NONIDEALITIES=true TC_STATE=1 TOGGLE_MODE=none SWITCH_INF=false
source ./launch_scripts/tc_nonideality_args.sh ft || exit 2

MODEL_NAME="TIMMPCNetNoBatchNorm_PCConvReLU6_0.0eps_ODEXInitFFFB_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_3K1S64C_0.25Dropout_22Layers6l7l6_2Pool_srrlDistill_a0p3_t2p0_2REP"
checkpoint="./saved_ckpt_runs/tc_rgb_cifar10_state1_pcn_resnet_depth_study/${MODEL_NAME}/${MODEL_NAME}_last_ckpt.pth"
if [[ ! -s "$checkpoint" ]]; then
  echo "Missing pretrained checkpoint: $checkpoint" >&2
  exit 2
fi

python -u train_ode_cifar.py \
  --model_name "$MODEL_NAME" \
  --save_path ./saved_ckpt_runs/tc_rgb_cifar10_state1_pcn_resnet_depth_study \
  --output_save_path ./saved_ckpt_runs/tc_rgb_cifar10_state1_pcn_resnet_depth_study \
  --dataset cifar10 --num_classes 10 --img_type rgb --ckpt last \
  --timm_trainer true --timm_sched cosine --timm_aug_level no_aug --timm_re_prob 0.0 \
  --optim SGD --learning_rate 0.005 --num_epochs 200 --warmup_epoch 5 \
  --eval_every 2 --final_eval_only true \
  --health_check_epochs 20,50 --health_check_batches 4 --health_check_seed 4096 \
  --input_quant_bits none --center_student_input false \
  --offset_eps 0.0 --dropout 0.25 --avg_pooling true \
  --tie_weights false --tie_bp false --bypass false \
  --batch_size 128 --method dopri5 --n_steps 5 --tol 1e-6 --t_end 1.75 \
  --scale_train_recipe false \
  --patch_node "" --patch_stride "" --patch_cycle "" --patch_pad "" --fold_scalar "" \
  --tie_cap false --qat_cls SymQuantizeWeight --pcn PCNetNoBatchNorm --pc_conv PCConvReLU6 \
  --noise_type mul --pulse_mismatch_training_mode post_quant_amplitude \
  --activation_corner_mode fixed --activation_random_curve_sharing per_layer \
  --activation_interpolation piecewise_linear --activation_spline_parameters 10 \
  --activation_fit_constraint auto --activation_normalize_positive_endpoint false \
  --slow_summing_current 2.47e-9 --slow_coupler_noise 2.47e-9 \
  --nonlinear_R_mc_quantity conductance --train_conv_expanded false --nonlinear_R_corner_range all \
  --teacher_ckpt ./checkpoint/efficientnet_v2_l_cifar10_rgb_OldNoTimm_MatchDistill.pth \
  --teacher_arch efficientnet_v2_l --teacher_arch_source torchvision \
  --teacher_input_size 224 --teacher_center_crop true --adapt_PIL_teacher false \
  --distill_method srrl --distill_alpha 0.3 --distill_temperature 2.0 \
  --reviewkd_weight 1.0 --reviewkd_warmup_epochs 20 --reviewkd_num_stages 4 \
  --contrast_method memory --test_only false --mem_frac 0.9 \
  --num_workers 0 \
  "${TC_ARGS[@]}"
status=$?
printf '\nFT_EXIT_CODE=%s\n' "$status"
printf '%s\n' "$status" > logs/local_runs/tc_state1_22L6l7l6_2REP_ft_warmup5_e200.exitcode
exit "$status"

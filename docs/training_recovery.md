# Epoch recovery

Both trainers atomically replace `<model>_latest_ckpt.pth` after each completed
epoch, after validation (when scheduled) and scheduler advancement. This file
contains trainable/QAT weights and buffers, optimizer and scheduler state,
distillation state, completed epoch, metric histories, launch arguments, global
RNG states, and hardware generators/assignments. It is not a flattened inference
checkpoint.

Use the same architecture, training stage, recipe, dataset and device layout.
Explicitly supply the latest file through the existing CNN
`PRETRAIN_RESUME_CKPT` (pretraining) or `MODEL_CKPT` (fine-tuning) environment
variable. These forward to `--resume_checkpoint`. PCN uses its existing
`--model_name`, `--save_path`, and `--ckpt latest` selection. No latest file is
discovered automatically. Ordinary checkpoints retain weights-only loading.

Recovery resumes at the next epoch and restores RNG state after trainer and
teacher setup. Exact reproducibility requires non-persistent data-loader workers,
the same data/worker configuration and deterministic operators; persistent-worker
recovery is rejected rather than pretending to restore inaccessible worker RNGs.
An interrupted epoch is rerun from its beginning.

Existing best/last checkpoint writes and validation scheduling are unchanged.
Latest is deleted only after the existing last-checkpoint writer succeeds.
Completion queues continue to watch last, never latest. Checkpoints made before
this feature cannot recover optimizer/RNG state that was never saved.


# Recover only the finetune stage of the pipelined launch of rgb:
for tc_state in 1 2; do
  for task_id in 10 100; do
    task="cifar${task_id}"
    run_name="tc_rgb_${task}_state${tc_state}_pcn_resnet_depth_study"

    (
      export TC_NONIDEALITIES=true \
             TC_STATE="${tc_state}" \
             TOGGLE_MODE=none \
             SWITCH_INF=false \
             TASK="${task}" \
             IMG_TYPE=rgb \
             EXP_PREFIX="${run_name}" \
             TEACHER_CKPT="./checkpoint/efficientnet_v2_l_${task}_rgb_OldNoTimm_MatchDistill.pth" \
             TEACHER_ARCH=efficientnet_v2_l \
             TEACHER_ARCH_SOURCE=torchvision \
             DISTILL_METHOD=srrl \
             TIMM_RE_PROB=0.0 \
             PCN_CHAN_0_LIST=16 \
             PCN_NUM_LAYERS_LIST="22 28" \
             NUM_COMB_PER_NUM_LAYER=3 \
             COMB_SEL_SET="2 3" \
             MAX_TASKS_PER_GPU=1 \
             N_TRIALS=10

      source ./launch_scripts/slurm_search_config.sh
    ) > "./logs/scheduler_slurm/${run_name}.log" 2>&1 < /dev/null &

    echo "${run_name}: scheduler PID $!"
  done
done



## 1-state
module swap slurm slurm/24.05.0.b1

cd /scratch/rzeng7/repos/PCN-with-Local-Recurrent-Processing
mkdir -p logs/slurm_jobs

for layout in 6l7l6 7l6l6; do
  model="TIMMPCNetNoBatchNorm_PCConvReLU6_0.0eps_ODEXInitFFFB_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_3K1S64C_0.25Dropout_22Layers${layout}_2Pool_srrlDistill_a0p3_t2p0_2REP"

  sbatch --export=ALL,MODEL_NAME="${model}" <<'SBATCH'
#!/bin/bash -l
#SBATCH -p ising
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH -t 90:10:00
#SBATCH --nodelist=bhgrb4x0081,bhgrb4x0082
#SBATCH -o /scratch/rzeng7/repos/PCN-with-Local-Recurrent-Processing/logs/slurm_jobs/slurm_%j.out

set -o pipefail
source activate base
conda activate scanbase

export OMP_NUM_THREADS=16
export TC_NONIDEALITIES=true TC_STATE=1 TOGGLE_MODE=none SWITCH_INF=false
source ./launch_scripts/tc_nonideality_args.sh ft || exit 2

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
  "${TC_ARGS[@]}"
SBATCH
done


## 2-state
module swap slurm slurm/24.05.0.b1

cd /scratch/rzeng7/repos/PCN-with-Local-Recurrent-Processing
mkdir -p logs/slurm_jobs

sbatch <<'SBATCH'
#!/bin/bash -l
#SBATCH -p ising
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH -t 90:10:00
#SBATCH --nodelist=bhgrb4x0081,bhgrb4x0082
#SBATCH -o /scratch/rzeng7/repos/PCN-with-Local-Recurrent-Processing/logs/slurm_jobs/slurm_%j.out

set -eo pipefail

cd /scratch/rzeng7/repos/PCN-with-Local-Recurrent-Processing
source activate base
conda activate scanbase

export TC_NONIDEALITIES=true
export TC_STATE=2
export TOGGLE_MODE=none
export SWITCH_INF=false
export TASK=cifar10
export IMG_TYPE=rgb
export OMP_NUM_THREADS=16

MODEL_NAME="TIMMPCNetNoBatchNorm_PCConvReLU6_0.0eps_S2NoisyIYAsXZAs0_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_3K1S64C_0.25Dropout_22Layers6l7l6_2Pool_srrlDistill_a0p3_t2p0_2REP"
MODEL_DIR="./saved_ckpt_runs/tc_rgb_cifar10_state2_pcn_resnet_depth_study"

test -s "${MODEL_DIR}/${MODEL_NAME}/${MODEL_NAME}_last_ckpt.pth"
test -s ./checkpoint/efficientnet_v2_l_cifar10_rgb_OldNoTimm_MatchDistill.pth

source ./launch_scripts/tc_nonideality_args.sh ft

python -u train_ode_cifar.py \
  --save_path "${MODEL_DIR}" \
  --output_save_path "${MODEL_DIR}" \
  --model_name "${MODEL_NAME}" \
  --ckpt last \
  --dataset cifar10 \
  --num_classes 10 \
  --img_type rgb \
  --input_quant_bits none \
  --center_student_input false \
  --timm_trainer true \
  --timm_sched cosine \
  --timm_aug_level no_aug \
  --timm_re_prob 0.0 \
  --optim SGD \
  --learning_rate 0.005 \
  --num_epochs 140 \
  --warmup_epoch 0 \
  --batch_size 128 \
  --eval_every 2 \
  --final_eval_only true \
  --health_check_epochs 20,50 \
  --health_check_batches 4 \
  --health_check_seed 4096 \
  --offset_eps 0.0 \
  --dropout 0.25 \
  --avg_pooling true \
  --tie_weights false \
  --tie_bp false \
  --bypass false \
  --n_steps 5 \
  --tol 1e-6 \
  --t_end 1.75 \
  --scale_train_recipe false \
  --tie_cap false \
  --qat_cls SymQuantizeWeight \
  --pcn PCNetNoBatchNorm \
  --pc_conv PCConvReLU6 \
  --noise_type mul \
  --pulse_mismatch_training_mode post_quant_amplitude \
  --teacher_ckpt ./checkpoint/efficientnet_v2_l_cifar10_rgb_OldNoTimm_MatchDistill.pth \
  --teacher_arch efficientnet_v2_l \
  --teacher_arch_source torchvision \
  --teacher_input_size 224 \
  --teacher_center_crop true \
  --adapt_PIL_teacher false \
  --distill_method srrl \
  --distill_alpha 0.3 \
  --distill_temperature 2.0 \
  --contrast_method memory \
  --test_only false \
  --mem_frac 0.9 \
  "${TC_ARGS[@]}"
SBATCH
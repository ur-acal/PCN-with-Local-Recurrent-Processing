#!/bin/bash
#set -euo pipefail

REPO_ROOT="/scratch/rzeng7/repos/PCN-with-Local-Recurrent-Processing"
SBATCH_SCRIPT="${REPO_ROOT}/launch_scripts/run_baseline_cifar_train.sh"

###############################################################################################
# running with
# module swap slurm slurm/24.05.0.b1
# ( source ./launch_scripts/slurm_run_baseline.sh ) \
#  > ./logs/scheduler_slurm/slurm_baseline_cifar.log 2>&1 < /dev/null &
###############################################################################################
#EXTRA_OVERRIDE="eval_every=5,noise_level=0.1,noise_type=mul"
EXTRA_OVERRIDE="eval_every=5"

MAX_TASKS_PER_GPU="${MAX_TASKS_PER_GPU:-1}"
GPUS_PER_JOB="${GPUS_PER_JOB:-1}"

CUSTOM_CIFAR_MODELS=(
  resnet20_cifar
  resnet32_cifar
  resnet44_cifar
  resnet56_cifar
  preact_resnet164_cifar
  wrn_28_10_cifar
)

ADAPT_NORESIZE_TIMM_MODELS=(
  resnet18
  resnet34
  resnet50
  resnext26ts
  resnext50_32x4d
  seresnet18
  seresnet34
  seresnet50
  mobilenetv2_100
  vgg19
)

RESIZE_FINETUNE_TIMM_MODELS=(
  efficientnet_b0
  mobilenetv3_small_100
  vit_tiny_patch16_224
  deit_tiny_patch16_224
  mixer_b16_224
  convnext_tiny
)

DATASETS=(cifar10 cifar100)

for dataset_name in "${DATASETS[@]}"; do

  # 1a. Custom CIFAR models: no resize, scratch only.
  for model_name in "${CUSTOM_CIFAR_MODELS[@]}"; do
    jid=$(
      sbatch --parsable \
        --gres=gpu:${GPUS_PER_JOB} \
        --export=ALL,IS_SLURM=1,MODEL_NAME="${model_name}",DATASET_NAME="${dataset_name}",CASE_NAME="custom_noresize",PRETRAINED="false",PREFER_RESIZE="false" \
        "${SBATCH_SCRIPT}"
    )
    echo "submitted job ${jid}: dataset=${dataset_name}, model=${model_name}, case=custom_noresize, pretrained=false"
    sleep "1"
  done

  # 1b. Built-in timm models adapted to native CIFAR resolution: scratch only.
  for model_name in "${ADAPT_NORESIZE_TIMM_MODELS[@]}"; do
    jid=$(
      sbatch --parsable \
        --gres=gpu:${GPUS_PER_JOB} \
        --export=ALL,IS_SLURM=1,MODEL_NAME="${model_name}",DATASET_NAME="${dataset_name}",CASE_NAME="adapt_noresize_scratch",PRETRAINED="false",PREFER_RESIZE="false" \
        "${SBATCH_SCRIPT}"
    )
    echo "submitted job ${jid}: dataset=${dataset_name}, model=${model_name}, case=adapt_noresize_scratch, pretrained=false"
    sleep "1"
  done

  # 2. Resize CIFAR and fine-tune ImageNet-pretrained models.
  for model_name in "${RESIZE_FINETUNE_TIMM_MODELS[@]}"; do
    jid=$(
      sbatch --parsable \
        --gres=gpu:${GPUS_PER_JOB} \
        --export=ALL,IS_SLURM=1,MODEL_NAME="${model_name}",DATASET_NAME="${dataset_name}",CASE_NAME="resize_finetune",PRETRAINED="true",PREFER_RESIZE="true" \
        "${SBATCH_SCRIPT}"
    )
    echo "submitted job ${jid}: dataset=${dataset_name}, model=${model_name}, case=resize_finetune, pretrained=true"
    sleep "1"
  done

done
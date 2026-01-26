#!/usr/bin/env bash

EXP="no_bn_pcn_NODE_0111_ODEXInitFFFB"
LOGDIR="./logs/${EXP}"
mkdir -p "${LOGDIR}"

##STRIDE=(1  1  2  1  2   1   1)
STRIDE=(1)
KSZ=(3)
PADDING=1 # For ksz=5, padding=2; o.w. padding=1
TASK="cifar10"
if [[ "$TASK" == "cifar100" ]]; then
  N_CLASSES=100
else
  N_CLASSES=10
fi
ODE_BLK="ODEXInitFFFB"

python train_ode_cifar.py \
  --optim         "SGD" \
  --img_type      "scanGFI" \
  --task          "$TASK" \
  --num_classes   "$N_CLASSES" \
  --num_epochs    150 \
  --eval_every    10 \
  --learning_rate 0.01 \
  --offset_eps    0.0 \
  --inp_channels  4  32 32 64 64  128 128 \
  --out_channels  32 32 64 64 128 128 128 \
  --max_pool      0  0  1  0  1   0   0  \
  --kernel_size   "${KSZ[@]}" \
  --padding       "${PADDING}" \
  --stride        "${STRIDE[@]}" \
  --avg_pooling   "false" \
  --dropout       0.25 \
  --weight_decay  "1e-3" \
  --lr_reduce_on  "80,122" \
  --tie_weights   "false" \
  --tie_bp        "false" \
  --bypass        "false" \
  --batch_size    128 \
  --method        "dopri5" \
  --tol           "1e-4" \
  --t_end         "1.75" \
  --pcn           "PCNetNoBatchNorm" \
  --pc_conv       "PCConvReLU6" \
  --ode_block     "ODEXInitFFFB" \
  2>&1 | tee "${LOGDIR}/train_${EXP}_L_ODEXInitFFFB.log"

python train_ode_cifar.py \
  --optim         "SGD" \
  --img_type      "scanGFI" \
  --task          "$TASK" \
  --num_classes   "$N_CLASSES" \
  --num_epochs    150 \
  --eval_every    10 \
  --learning_rate 0.01 \
  --offset_eps    0.0 \
  --inp_channels  4  32 32 64 64  128 128 \
  --out_channels  32 32 64 64 128 128 128 \
  --max_pool      0  0  1  0  1   0   0  \
  --kernel_size   "${KSZ[@]}" \
  --padding       "${PADDING}" \
  --stride        "${STRIDE[@]}" \
  --avg_pooling   "false" \
  --dropout       0.25 \
  --weight_decay  "1e-3" \
  --lr_reduce_on  "80,122" \
  --tie_weights   "false" \
  --tie_bp        "false" \
  --bypass        "false" \
  --batch_size    128 \
  --method        "dopri5" \
  --tol           "1e-4" \
  --t_end         "1.75" \
  --pcn           "PCNetNoBatchNorm" \
  --pc_conv       "PCConvReLU6" \
  --ode_block     "S2NoisyIYAsXZAs0" \
  2>&1 | tee "${LOGDIR}/train_${EXP}_L_S2NoisyIYAsXZAs0.log"

#python train_ode_cifar.py \
#  --optim         "SGD" \
#  --img_type      "scanGFI" \
#  --task          "$TASK" \
#  --num_classes   "$N_CLASSES" \
#  --num_epochs    150 \
#  --eval_every    10 \
#  --learning_rate 0.01 \
#  --offset_eps    0.0 \
#  --inp_channels  4  18 18 18 18 36 36 36 36 72 72 72 72 \
#  --out_channels  18 18 18 18 36 36 36 36 72 72 72 72 72 \
#  --max_pool      0  0  0  1  0  0  0  1  0  0  0  0  0  \
#  --kernel_size   "${KSZ[@]}" \
#  --padding       "${PADDING}" \
#  --stride        "${STRIDE[@]}" \
#  --avg_pooling   "false" \
#  --patch_node    "8" \
#  --patch_stride  "8" \
#  --patch_cycle   "2" \
#  --patch_pad     "0" \
#  --fold_scalar   "1" \
#  --dropout       0.25 \
#  --weight_decay  "1e-3" \
#  --lr_reduce_on  "80,122" \
#  --tie_weights   "false" \
#  --tie_bp        "false" \
#  --bypass        "false" \
#  --batch_size    128 \
#  --method        "dopri5" \
#  --tol           "1e-4" \
#  --t_end         "1.75" \
#  --pcn           "PCNetNoBatchNorm" \
#  --pc_conv       "PCConvReLU6" \
#  --ode_block     "$ODE_BLK" \
#  2>&1 | tee "${LOGDIR}/train_${EXP}_rggb_72C13L.log"

#python train_ode_cifar.py \
#  --optim         "SGD" \
#  --img_type      "scanGFI" \
#  --task          "$TASK" \
#  --num_classes   "$N_CLASSES" \
#  --num_epochs    150 \
#  --eval_every    10 \
#  --learning_rate 0.01 \
#  --offset_eps    0.0 \
#  --inp_channels  4  20 20 20 40 40 40 80 80 80 \
#  --out_channels  20 20 20 40 40 40 80 80 80 80 \
#  --max_pool      0  0  1  0  0  1  0  0  0  0  \
#  --kernel_size   "${KSZ[@]}" \
#  --padding       "${PADDING}" \
#  --stride        "${STRIDE[@]}" \
#  --avg_pooling   "false" \
#  --patch_node    "8" \
#  --patch_stride  "8" \
#  --patch_cycle   "2" \
#  --patch_pad     "0" \
#  --fold_scalar   "1" \
#  --dropout       0.25 \
#  --weight_decay  "1e-3" \
#  --lr_reduce_on  "80,122" \
#  --tie_weights   "false" \
#  --tie_bp        "false" \
#  --bypass        "false" \
#  --batch_size    128 \
#  --method        "dopri5" \
#  --tol           "1e-4" \
#  --t_end         "1.75" \
#  --pcn           "PCNetNoBatchNorm" \
#  --pc_conv       "PCConvReLU6" \
#  --ode_block     "$ODE_BLK" \
#  2>&1 | tee "${LOGDIR}/train_${EXP}_rggb_80C10L.log"

#python train_ode_cifar.py \
#  --optim         "SGD" \
#  --img_type      "scanGFI" \
#  --num_epochs    150 \
#  --eval_every    10 \
#  --learning_rate 0.01 \
#  --offset_eps    0.0 \
#  --inp_channels  4  16 16 32 32 64 64 64 64 64 64 \
#  --out_channels  16 16 32 32 64 64 64 64 64 64 64 \
#  --max_pool      0  1  0  1  0  0  0  0  0  0  0  \
#  --kernel_size   "${KSZ[@]}" \
#  --padding       "${PADDING}" \
#  --stride        "${STRIDE[@]}" \
#  --avg_pooling   "true" \
#  --patch_node    "8" \
#  --patch_stride  "8" \
#  --patch_cycle   "2" \
#  --patch_pad     "0" \
#  --fold_scalar   "1" \
#  --dropout       0.25 \
#  --weight_decay  "1e-3" \
#  --lr_reduce_on  "80,122" \
#  --tie_weights   "false" \
#  --tie_bp        "false" \
#  --bypass        "false" \
#  --batch_size    128 \
#  --method        "dopri5" \
#  --tol           "1e-4" \
#  --t_end         "1.75" \
#  --pcn           "PCNetNoBatchNorm" \
#  --pc_conv       "PCConvReLU6" \
#  --ode_block     "S2CircYAsXZas0" \
#  2>&1 | tee "${LOGDIR}/train_${EXP}_rggb_S2Circ.log"


#python train_ode_cifar.py \
#  --optim         "SGD" \
#  --img_type      "scanGFI" \
#  --num_epochs    150 \
#  --eval_every    10 \
#  --offset_eps    0.0 \
#  --inp_channels  4  64 64 64 64 64 64 \
#  --out_channels  64 64 64 64 64 64 64 \
#  --max_pool      0  0  0  1  0  0  0  \
#  --kernel_size   "${KSZ[@]}" \
#  --padding       "${PADDING}" \
#  --stride        "${STRIDE[@]}" \
#  --dropout       0.25 \
#  --weight_decay  "1e-3" \
#  --lr_reduce_on  "80,122" \
#  --tie_weights   "false" \
#  --tie_bp        "false" \
#  --bypass        "false" \
#  --batch_size    128 \
#  --method        "dopri5" \
#  --tol           "1e-4" \
#  --t_end         "1.25" \
#  --pcn           "PCNetNoBatchNorm" \
#  --pc_conv       "PCConvReLU6" \
#  --ode_block     "S2Circ" \
#  2>&1 | tee "${LOGDIR}/train_${EXP}_rggb_S2Circ.log"

# 6L64 Chan
#python train_ode_cifar.py \
#  --optim         "SGD" \
#  --img_type      "scanGFI" \
#  --num_epochs    150 \
#  --eval_every    10 \
#  --offset_eps    0.0 \
#  --inp_channels  4  64 64 64 64 64 64 \
#  --out_channels  64 64 64 64 64 64 64 \
#  --max_pool      0  1  0  1  0  0  0   \
#  --kernel_size   "${KSZ[@]}" \
#  --padding       "${PADDING}" \
#  --stride        "${STRIDE[@]}" \
#  --dropout       0.25 \
#  --tie_weights   "false" \
#  --tie_bp        "false" \
#  --bypass        "false" \
#  --batch_size    128 \
#  --method        "dopri5" \
#  --tol           "0.0001" \
#  --t_end         "1.5" \
#  --pcn           "PCNetNoBatchNorm" \
#  --pc_conv       "PCConvReLU6" \
#  --ode_block     "S2NoMinusZChgZNoisyI" \
#  2>&1 | tee "${LOGDIR}/train_${EXP}_rggb_S2NoMinusZChgZNoisyI_hw_friendly.log"

## No 2
#python train_ode_cifar.py \
#  --optim         "SGD" \
#  --num_epochs    150 \
#  --eval_every    10 \
#  --offset_eps    0.2 \
#  --inp_channels  3  16 16 16 16 16 16 32 32 32 32 32 64 64 64 64 64 64 \
#  --out_channels  16 16 16 16 16 16 32 32 32 32 32 64 64 64 64 64 64 64 \
#  --max_pool      0  0  0  0  0  0  1  0  0  0  0  1  0  0  0  0  0  0   \
#  --dropout       0.25 \
#  --tie_weights   "false" \
#  --tie_bp        "false" \
#  --bypass        "false" \
#  --batch_size    128 \
#  --method        "dopri5" \
#  --tol           "0.0001" \
#  --t_end         "0.75" \
#  --pcn           "PCNetNoBatchNorm" \
#  --pc_conv       "PCConvReLU6" \
#  --ode_block     "ODEFixNoiseXInitFFFB" \
#  2>&1 | tee "${LOGDIR}/train_${EXP}_No_2_ReLU6_Deep_ODEFixNoiseXInitFFFB.log"
#
#echo "Completed."

## Purely FF ODE dynamics
## No 2
#python train_ode_cifar.py \
#  --optim         "SGD" \
#  --num_epochs    150 \
#  --eval_every    1 \
#  --inp_channels  3  32 32 64 64  128 \
#  --out_channels  32 32 64 64 128 128 \
#  --max_pool      0  1  1  0  1   0   \
#  --dropout       0.25 \
#  --tie_weights   "false" \
#  --tie_bp        "false" \
#  --bypass        "false" \
#  --batch_size    128 \
#  --method        "dopri5" \
#  --tol           "0.0001" \
#  --t_end         "0.75" \
#  --pcn           "PCNetNoBatchNorm" \
#  --pc_conv       "PCConvFFReLU6" \
#  --ode_block     "ODEFFConv" \
#  2>&1 | tee "${LOGDIR}/train_${EXP}_No_2_ReLU6_Wide_ODEFFConv.log"
#
#echo "Completed."

##STRIDE=(1  1  2  1  2   1   1)
#STRIDE=(1)
## No 2
#python train_ode_cifar.py \
#  --optim         "SGD" \
#  --num_epochs    150 \
#  --eval_every    10 \
#  --offset_eps    0.002 \
#  --inp_channels  3  32 32 64 64  128 128 \
#  --out_channels  32 32 64 64 128 128 128 \
#  --max_pool      0  0  1  0  1   0   0  \
#  --stride        "${STRIDE[@]}" \
#  --dropout       0.25 \
#  --tie_weights   "false" \
#  --tie_bp        "false" \
#  --bypass        "false" \
#  --batch_size    128 \
#  --method        "dopri5" \
#  --tol           "0.0001" \
#  --t_end         "1.5" \
#  --pcn           "PCNetNoBatchNorm" \
#  --pc_conv       "PCConvReLU6" \
#  --ode_block     "S2NoMinusZChgZMinusNoisyI" \
#  2>&1 | tee "${LOGDIR}/train_${EXP}_No_2_ReLU6_Wide_SelfCUAbsSumFFFB.log"
#
#echo "Completed."

## No 2
#python train_ode_cifar.py \
#  --optim         "SGD" \
#  --num_epochs    150 \
#  --offset_eps    0.1 \
#  --inp_channels  3  32 32 64 64  128 128 \
#  --out_channels  32 32 64 64 128 128 128 \
#  --max_pool      0  0  1  0  1   0   0   \
#  --dropout       0.25 \
#  --tie_weights   "false" \
#  --tie_bp        "false" \
#  --bypass        "false" \
#  --batch_size    128 \
#  --method        "dopri5" \
#  --tol           "0.0001" \
#  --t_end         "0.75" \
#  --pcn           "PCNetNoBatchNorm" \
#  --pc_conv       "PCConvReLU6" \
#  --ode_block     "ODEFixNoiseXInitFFFB" \
#  2>&1 | tee "${LOGDIR}/train_${EXP}_No_2_ReLU6_7Layers.log"
#
#echo "Completed."

## No 2
#python train_ode_cifar.py \
#  --optim         "SGD" \
#  --num_epochs    150 \
#  --offset_eps    0.2 \
#  --inp_channels  3  32 32 64 64  128 128 \
#  --out_channels  32 32 64 64 128 128 128 \
#  --max_pool      0  1  1  0  1   0   0   \
#  --dropout       0.25 \
#  --tie_weights   "false" \
#  --tie_bp        "false" \
#  --bypass        "false" \
#  --batch_size    128 \
#  --method        "dopri5" \
#  --tol           "0.0001" \
#  --t_end         "0.75" \
#  --pcn           "PCNetNoBatchNorm" \
#  --pc_conv       "PCConvReLU6" \
#  --ode_block     "ODEFixNoiseOffset" \
#  2>&1 | tee "${LOGDIR}/train_${EXP}_No_2_ReLU6_7Layers_minus_y_0p75_1e-4.log"
#
#echo "Completed."
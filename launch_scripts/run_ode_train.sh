#!/usr/bin/env bash

EXP="no_bn_pcn_NODE_1031_scan_gfi_SML"
LOGDIR="./logs/${EXP}"
mkdir -p "${LOGDIR}"

##STRIDE=(1  1  2  1  2   1   1)
STRIDE=(1)
KSZ=(3)
PADDING=1
# S model (0.26 M)
python train_ode_cifar.py \
  --optim         "SGD" \
  --img_type      "scanGFI" \
  --num_epochs    150 \
  --eval_every    1 \
  --offset_eps    0.0 \
  --inp_channels  4  32 64 64 64 \
  --out_channels  32 64 64 64 64 \
  --max_pool      0  1  0  1  0  \
  --kernel_size   "${KSZ[@]}" \
  --padding       "${PADDING}" \
  --stride        "${STRIDE[@]}" \
  --dropout       0.0 \
  --weight_decay  "1e-4" \
  --lr_reduce_on  "80,122" \
  --tie_weights   "false" \
  --tie_bp        "false" \
  --bypass        "false" \
  --batch_size    128 \
  --method        "dopri5" \
  --tol           "0.0001" \
  --t_end         "1.5" \
  --pcn           "PCNetNoBatchNorm" \
  --pc_conv       "PCConvReLU6" \
  --ode_block     "ODEBlockXInit" \
  2>&1 | tee "${LOGDIR}/train_${EXP}_ODEBlockXInit.log"

# M model (0.57 M)
python train_ode_cifar.py \
  --optim         "SGD" \
  --img_type      "scanGFI" \
  --num_epochs    150 \
  --eval_every    1 \
  --offset_eps    0.0 \
  --inp_channels  4  32 32 64 64  128 \
  --out_channels  32 32 64 64 128 128 \
  --max_pool      0  0  1  0  1   0  \
  --kernel_size   "${KSZ[@]}" \
  --padding       "${PADDING}" \
  --stride        "${STRIDE[@]}" \
  --dropout       0.0 \
  --weight_decay  "1e-4" \
  --lr_reduce_on  "80,122" \
  --tie_weights   "false" \
  --tie_bp        "false" \
  --bypass        "false" \
  --batch_size    128 \
  --method        "dopri5" \
  --tol           "0.0001" \
  --t_end         "1.5" \
  --pcn           "PCNetNoBatchNorm" \
  --pc_conv       "PCConvReLU6" \
  --ode_block     "ODEBlockXInit" \
  2>&1 | tee "${LOGDIR}/train_${EXP}_ODEBlockXInit.log"

# L model (0.86 M)
python train_ode_cifar.py \
  --optim         "SGD" \
  --img_type      "scanGFI" \
  --num_epochs    150 \
  --eval_every    1 \
  --offset_eps    0.0 \
  --inp_channels  4  32 32 64 64  128 128 \
  --out_channels  32 32 64 64 128 128 128 \
  --max_pool      0  0  1  0  1   0   0  \
  --kernel_size   "${KSZ[@]}" \
  --padding       "${PADDING}" \
  --stride        "${STRIDE[@]}" \
  --dropout       0.0 \
  --weight_decay  "1e-4" \
  --lr_reduce_on  "80,122" \
  --tie_weights   "false" \
  --tie_bp        "false" \
  --bypass        "false" \
  --batch_size    128 \
  --method        "dopri5" \
  --tol           "0.0001" \
  --t_end         "1.5" \
  --pcn           "PCNetNoBatchNorm" \
  --pc_conv       "PCConvReLU6" \
  --ode_block     "ODEBlockXInit" \
  2>&1 | tee "${LOGDIR}/train_${EXP}_ODEBlockXInit.log"

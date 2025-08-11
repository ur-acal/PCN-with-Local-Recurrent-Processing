#!/usr/bin/env bash

EXP="no_bn_pcn_NODE_0809_ODEFixNoiseOffset_0.2eps_relu6_3maxpooling_cosineAnnealLR"
LOGDIR="./logs/${EXP}"
mkdir -p "${LOGDIR}"

# No 2
python train_ode_cifar.py \
  --optim         "SGD" \
  --num_epochs    150 \
  --offset_eps    0.2 \
  --inp_channels  3  32 32 64 64  128 \
  --out_channels  32 32 64 64 128 128 \
  --max_pool      0  1  1  0  1   0 \
  --dropout       0.25 \
  --cosine_t0     10 \
  --tie_weights   "false" \
  --tie_bp        "false" \
  --bypass        "false" \
  --batch_size    128 \
  --method        "dopri5" \
  --tol           "0.0001" \
  --t_end         "0.75" \
  --pcn           "PCNetNoBatchNorm" \
  --pc_conv       "PCConvReLU6" \
  --ode_block     "ODEFixNoiseOffset" \
  2>&1 | tee "${LOGDIR}/train_${EXP}_No_2_ReLU6_5Layers_minus_y_0p75_1e-4_lr0.01T010.log"

echo "Completed."


# No 2
python train_ode_cifar.py \
  --optim         "SGD" \
  --num_epochs    150 \
  --offset_eps    0.2 \
  --inp_channels  3  32 32 64 64  128 \
  --out_channels  32 32 64 64 128 128 \
  --max_pool      0  1  1  0  1   0 \
  --dropout       0.25 \
  --cosine_t0     20 \
  --tie_weights   "false" \
  --tie_bp        "false" \
  --bypass        "false" \
  --batch_size    128 \
  --method        "dopri5" \
  --tol           "0.0001" \
  --t_end         "0.75" \
  --pcn           "PCNetNoBatchNorm" \
  --pc_conv       "PCConvReLU6" \
  --ode_block     "ODEFixNoiseOffset" \
  2>&1 | tee "${LOGDIR}/train_${EXP}_No_2_ReLU6_5Layers_minus_y_0p75_1e-4_lr0.01T020.log"

echo "Completed."

# No 2
python train_ode_cifar.py \
  --optim         "SGD" \
  --num_epochs    150 \
  --offset_eps    0.2 \
  --inp_channels  3  32 32 64 64  128 \
  --out_channels  32 32 64 64 128 128 \
  --max_pool      0  1  1  0  1   0 \
  --dropout       0.25 \
  --learning_rate 0.05 \
  --cosine_t0     10 \
  --tie_weights   "false" \
  --tie_bp        "false" \
  --bypass        "false" \
  --batch_size    128 \
  --method        "dopri5" \
  --tol           "0.0001" \
  --t_end         "0.75" \
  --pcn           "PCNetNoBatchNorm" \
  --pc_conv       "PCConvReLU6" \
  --ode_block     "ODEFixNoiseOffset" \
  2>&1 | tee "${LOGDIR}/train_${EXP}_No_2_ReLU6_5Layers_minus_y_0p75_1e-4_lr0.05T010.log"

echo "Completed."

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
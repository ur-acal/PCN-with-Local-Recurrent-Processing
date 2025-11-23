#!/usr/bin/env bash

EXP="no_bn_pcn_NODE_0729_ODEFixNoiseOffset_0.0eps_withBN"
LOGDIR="./logs/${EXP}"
mkdir -p "${LOGDIR}"

# No 2
#python train_ode_cifar.py \
#  --optim         "SGD" \
#  --num_epochs    150 \
#  --learning_rate 0.01 \
#  --aug           "true" \
#  --inp_channels  3   256 256 256 256 256 256 \
#  --out_channels  256 256 256 256 256 256 256 \
#  --max_pool      0   0   0   0   0   0   0   0   0   \
#  --separable         "d" "p" "d" "p" "d" "p" "d" "p" \
#  --kernel_size   5 \
#  --dropout       0.25 \
#  --tie_weights   "false" \
#  --tie_bp        "false" \
#  --bypass        "false" \
#  --batch_size    128 \
#  --method        "dopri5" \
#  --tol           "0.0001" \
#  --t_end         "0.75" \
#  --patch_dim     4 \
#  --pcn           "PCNetSepBN" \
#  --pc_conv       "PCConvReLU6" \
#  --ode_block     "ODEFixNoiseOffset" \
#  2>&1 | tee "${LOGDIR}/train_${EXP}_No_2_ReLU6_minus_y_0p75_1e-4.log"
#
#echo "Completed."

# No 2
python train_ode_cifar.py \
  --optim         "SGD" \
  --num_epochs    100 \
  --model_name    "PCNetSepBN_PCConvReLU6_ODEFixNoiseOffset_dopri5Solver_0.75TEnd_0.0001Tol_0.001WD_noBPtied_noBP_128BS_0.01LR_0.25Dropout_7Layers_3REP" \
  --learning_rate 0.0001 \
  --lr_reduce_on  "40,80" \
  --aug           "true" \
  --inp_channels  3   256 256 256 256 256 256 \
  --out_channels  256 256 256 256 256 256 256 \
  --max_pool      0   0   0   0   0   0   0   0   0   \
  --separable         "d" "p" "d" "p" "d" "p" "d" "p" \
  --kernel_size   5 \
  --dropout       0.25 \
  --tie_weights   "false" \
  --tie_bp        "false" \
  --bypass        "false" \
  --batch_size    128 \
  --method        "dopri5" \
  --tol           "0.0001" \
  --t_end         "0.75" \
  --patch_dim     4 \
  --pcn           "PCNetSepBN" \
  --pc_conv       "PCConvReLU6" \
  --ode_block     "ODEFixNoiseOffset" \
  2>&1 | tee "${LOGDIR}/train_${EXP}_No_2_ReLU6_minus_y_0p75_1e-4.log"

echo "Completed."

############################################################
# launch in this way:
# nohup bash launch_scripts/run_ode_train.sh > ./logs/master_single.log 2>&1 &
# tail -f ./logs/master_single.log
# after train finished
# cat ./logs/master_single.log | grep "Train finished" -A 3
############################################################
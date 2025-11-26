#!/usr/bin/env bash

EXP="no_bn_pcn_1122_FFFBReLU6_scanGFI"
LOGDIR="./logs/${EXP}"
mkdir -p "${LOGDIR}"

PCCONV="FFFBReLU6NoLastConvYasX"

# No 2 - Small
python train_cifar.py \
  --optim         "SGD" \
  --img_type      "scanGFI" \
  --num_epochs    150 \
  --inp_channels  4  32 64 64 64 \
  --out_channels  32 64 64 64 64 \
  --max_pool      0  1  0  1  0   \
  --dropout       0.25 \
  --lr_pc         0.15 \
  --cls           5 \
  --tie_weights   "false" \
  --tie_bp        "false" \
  --relu_between  "true" \
  --bypass        "false" \
  --batch_size    128 \
  --pcn           "PCNetNoBatchNorm" \
  --pc_conv       "${PCCONV}" \
  2>&1 | tee "${LOGDIR}/train_${EXP}_No_2_5L2pSmall.log"

# No 2 - 7L
python train_cifar.py \
  --optim         "SGD" \
  --img_type      "scanGFI" \
  --num_epochs    150 \
  --inp_channels  4  32 32 64 64  128 128 \
  --out_channels  32 32 64 64 128 128 128 \
  --max_pool      0  0  1  0  1   0   0   \
  --dropout       0.25 \
  --lr_pc         0.15 \
  --cls           5 \
  --tie_weights   "false" \
  --tie_bp        "false" \
  --relu_between  "true" \
  --bypass        "false" \
  --batch_size    128 \
  --pcn           "PCNetNoBatchNorm" \
  --pc_conv       "${PCCONV}" \
  2>&1 | tee "${LOGDIR}/train_${EXP}_No_2_7L2p.log"

# No 2 - 5L
python train_cifar.py \
  --optim         "SGD" \
  --img_type      "scanGFI" \
  --num_epochs    150 \
  --inp_channels  4  32 64 64  128 \
  --out_channels  32 64 64 128 128 \
  --max_pool      0  1  0  1   0   \
  --dropout       0.25 \
  --lr_pc         0.15 \
  --cls           5 \
  --tie_weights   "false" \
  --tie_bp        "false" \
  --relu_between  "true" \
  --bypass        "false" \
  --batch_size    128 \
  --pcn           "PCNetNoBatchNorm" \
  --pc_conv       "${PCCONV}" \
  2>&1 | tee "${LOGDIR}/train_${EXP}_No_2_5L2p.log"

## No 2 - Tiny
#python train_cifar.py \
#  --optim         "SGD" \
#  --img_type      "scanGFI" \
#  --num_epochs    150 \
#  --inp_channels  4  8 8 8 8 8 8  16 16 16 16 16 32 32 32 32 32 32 \
#  --out_channels  8  8 8 8 8 8 16 16 16 16 16 32 32 32 32 32 32 32 \
#  --max_pool      0 0 0 0 0 0  1  0  0  0  0  0  0  0  0  0  0  0   \
#  --dropout       0.25 \
#  --lr_pc         0.15 \
#  --cls           5 \
#  --tie_weights   "false" \
#  --tie_bp        "false" \
#  --relu_between  "true" \
#  --bypass        "false" \
#  --batch_size    128 \
#  --pcn           "PCNetNoBatchNorm" \
#  --pc_conv       "FFFBReLU6" \
#  2>&1 | tee "${LOGDIR}/train_${EXP}_No_2_18L1p.log"
#
## No 2 - Tiny, Pool on the second expansion
#python train_cifar.py \
#  --optim         "SGD" \
#  --img_type      "scanGFI" \
#  --num_epochs    150 \
#  --inp_channels  4  8 8 8 8 8 8  16 16 16 16 16 32 32 32 32 32 32 \
#  --out_channels  8  8 8 8 8 8 16 16 16 16 16 32 32 32 32 32 32 32 \
#  --max_pool      0 0 0 0 0 0  0  0  0  0  0  1  0  0  0  0  0  0   \
#  --dropout       0.25 \
#  --lr_pc         0.15 \
#  --cls           5 \
#  --tie_weights   "false" \
#  --tie_bp        "false" \
#  --relu_between  "true" \
#  --bypass        "false" \
#  --batch_size    128 \
#  --pcn           "PCNetNoBatchNorm" \
#  --pc_conv       "FFFBReLU6" \
#  2>&1 | tee "${LOGDIR}/train_${EXP}_No_2_18L1p_pool_later.log"
#
#echo "Completed."

############################################################
# launch in this way:
# nohup bash ./launch_scripts/run_single_scan_ppcn.sh > ./logs/master_single.log 2>&1 &
# tail -f ./logs/master_single.log
# after train finished
# cat ./logs/master_single.log | grep "Train finished" -A 3
############################################################
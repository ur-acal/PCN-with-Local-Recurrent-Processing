#!/usr/bin/env bash

EXP="no_bn_pcn_1130_PPCN_scanGFI_QAT"
LOGDIR="./logs/${EXP}"
mkdir -p "${LOGDIR}"

MODEL_NAMES=(
#  "PCNetNoBatchNorm_FFFBReLU6NoLastConvYasX_5CLS_0.15LRPC_0.001WD_noBPtied_noBP_128BS_0.25Dropout_5Layers_64Chan_2Pool_scanGFI_1REP"
#  "PCNetNoBatchNorm_FFFBReLU6NoLastConvYasX_5CLS_0.15LRPC_0.001WD_noBPtied_noBP_128BS_0.25Dropout_5Layers_128Chan_2Pool_scanGFI_1REP"
#  "PCNetNoBatchNorm_FFFBReLU6NoLastConvYasX_5CLS_0.15LRPC_0.001WD_noBPtied_noBP_128BS_0.25Dropout_7Layers_128Chan_2Pool_scanGFI_1REP"

  "PCNetNoBatchNorm_FFFBReLU6NoLastConvYasX_5CLS_0.15LRPC_0.001WD_noBPtied_noBP_128BS_0.25Dropout_5Layers_C100_64Chan_2Pool_scanGFI_1REP"
)
W_BITS="4"
ACT_BITS="4"
AGG_BITS="8"
ACT_PERC="0.9999"
# No 2 - Small
for _name in "${MODEL_NAMES[@]}"; do
  __rest="${_name#*_}"
  pc_conv="${__rest%%_*}"

  if [[ "$_name" == *C100* ]]; then
    _task="cifar100"
  else
    _task="cifar10"
  fi

  python train_cifar.py \
    --optim         "SGD" \
    --img_type      "scanGFI" \
    --task          "${_task}" \
    --model_name    "${_name}" \
    --learning_rate 0.005 \
    --cosine_t0     20 \
    --num_epochs    80 \
    --dropout       0.25 \
    --lr_pc         0.15 \
    --cls           5 \
    --tie_weights   "false" \
    --tie_bp        "false" \
    --relu_between  "true" \
    --bypass        "false" \
    --batch_size    128 \
    --pcn           "PCNetNoBatchNorm" \
    --qat_cls       "QATHelper" \
    --act_qat_cls   "PercQATHelper" \
    --act_perc      "${ACT_PERC}" \
    --agg_bits      "${AGG_BITS}" \
    --w_quant_type  "per_channel" \
    --w_bits        "${W_BITS}" \
    --act_bits      "${ACT_BITS}" \
    --pc_conv       "${pc_conv}" \
    2>&1 | tee "${LOGDIR}/train_${EXP}_${_name}_${W_BITS}w${ACT_BITS}a_No_2_QAT.log"
done

echo "Train finished"
ACT_PERC="0.999"
echo "Starting act percentile ${ACT_PERC}"

for _name in "${MODEL_NAMES[@]}"; do
  __rest="${_name#*_}"
  pc_conv="${__rest%%_*}"

  if [[ "$_name" == *C100* ]]; then
    _task="cifar100"
  else
    _task="cifar10"
  fi

  python train_cifar.py \
    --optim         "SGD" \
    --img_type      "scanGFI" \
    --task          "${_task}" \
    --model_name    "${_name}" \
    --learning_rate 0.005 \
    --cosine_t0     20 \
    --num_epochs    80 \
    --dropout       0.25 \
    --lr_pc         0.15 \
    --cls           5 \
    --tie_weights   "false" \
    --tie_bp        "false" \
    --relu_between  "true" \
    --bypass        "false" \
    --batch_size    128 \
    --pcn           "PCNetNoBatchNorm" \
    --qat_cls       "QATHelper" \
    --act_qat_cls   "PercQATHelper" \
    --act_perc      "${ACT_PERC}" \
    --agg_bits      "${AGG_BITS}" \
    --w_quant_type  "per_channel" \
    --w_bits        "${W_BITS}" \
    --act_bits      "${ACT_BITS}" \
    --pc_conv       "${pc_conv}" \
    2>&1 | tee "${LOGDIR}/train_${EXP}_${_name}_${W_BITS}w${ACT_BITS}a_No_2_QAT_0p999.log"
done

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
# nohup bash ./launch_scripts/run_ppcn_qat.sh > ./logs/master_ppcn_qat.log 2>&1 &
# tail -f ./logs/master_ppcn_qat.log
# after train finished
# cat ./logs/master_ppcn_qat.log | grep "Train finished" -A 3
############################################################
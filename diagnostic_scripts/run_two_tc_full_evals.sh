#!/usr/bin/env bash
# Sequential, full-dataset, one-trial Gaussian evaluations of the audited pair.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON="${PYTHON:-/home/rongzeng/anaconda3/envs/scanbase/bin/python}"
export OMP_NUM_THREADS=1
trap 'status=$?; if (( status != 0 )); then echo "FAILED: TC evaluation sequence exited with status ${status}"; fi' EXIT
mkdir -p results/tc_full_optimized
for schedule in 6l7l6 7l6l6; do
  checkpoints=(saved_ckpt_runs/tc_rgb_cifar100_state1_pcn_resnet_depth_study/TIMMQAT*22Layers${schedule}*/*1REP_last_ckpt.pth)
  if [[ ${#checkpoints[@]} != 1 || ! -f "${checkpoints[0]}" ]]; then
    echo "Expected exactly one finished flattened checkpoint for ${schedule}" >&2
    exit 2
  fi
  echo "START ${schedule}: ${checkpoints[0]}"
  "$PYTHON" -u diagnostic_scripts/run_tc_audited_eval.py \
    --checkpoint "${checkpoints[0]}" \
    --output-dir "results/tc_full_optimized/${schedule}" \
    --batch-size 16 --max-batches 0 \
    > "results/tc_full_optimized/${schedule}.log" 2>&1
  echo "FINISHED ${schedule}"
done
echo 'Both full TC evaluations finished successfully.'

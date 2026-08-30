#!/bin/bash

BASELINE_DIR="${BASELINE_DIR:-results/nonlinearR_exact_curve_noMT}"
PATCHED_DIR="${PATCHED_DIR:-results/nonlinearR_exact_curve_noMT_measuredPool_patchedGaussian}"
OUTPUT_DIR="${OUTPUT_DIR:-results/mc45_accuracy_distribution_plots}"

python scripts/plot_mc45_accuracy_distributions.py \
  --baseline_dir "${BASELINE_DIR}" \
  --patched_dir "${PATCHED_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  "$@"


#!/bin/bash

TASK="${TASK:-cifar100}"

python diagnostic_scripts/plot_corner_curves.py \
  --task "${TASK}" \
  --corner_means \
  "$@"

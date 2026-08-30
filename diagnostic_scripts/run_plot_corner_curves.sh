#!/bin/bash

# Copy-ready corner selections from the completed reference evaluations.
#
# CIFAR-100: results/coupler_monte_v2_cifar100_qf1
# Lowest five:
# ./diagnostic_scripts/run_plot_corner_curves.sh --corners "FS_V0_T0,FF_V2_T0,FF_V0_T0,FF_V0_T1,TT_V0_T0"
# Highest five:
# ./diagnostic_scripts/run_plot_corner_curves.sh --corners "TT_V1_T2,SF_V2_T2,SF_V1_T2,FS_V1_T2,SF_V0_T2"
#
# CIFAR-10: results/coupler_monte_v2_cifar10
# Lowest five:
# TASK=cifar10 ./diagnostic_scripts/run_plot_corner_curves.sh --corners "FS_V0_T0,TT_V0_T0,FF_V0_T0,FF_V0_T1,SF_V0_T0"
# Highest five:
# TASK=cifar10 ./diagnostic_scripts/run_plot_corner_curves.sh --corners "SF_V2_T2,SF_V1_T2,FS_V1_T2,TT_V1_T2,SS_V1_T2"

TASK="${TASK:-cifar100}"

python diagnostic_scripts/plot_corner_curves.py --task "${TASK}" "$@"


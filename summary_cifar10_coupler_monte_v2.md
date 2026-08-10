# CIFAR-10 Coupler-v2: 45-Corner Accuracy Summary

This summarizes full-CIFAR-10 test-dataset accuracy across 45 aligned process, voltage, and temperature corners for:

`TIMMQAT5b8aNT0p0mulTIMMPCNetNoBatchNorm_PCConvReLU6_0.0eps_ToggleODEXInitFFFB_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_3K1S96C_0.25Dropout_16Layers4l5l4_2Pool_srrlDistill_a0p3_t2p0_scanGFI_1REP`

For every corner, the table reports the accuracy mean and population standard deviation across independently sampled hardware realizations. This version uses the corresponding corner data for spin statistics, empirical coupler conductance curves, measured ReLU curves, measured pooling, and DTC pulse-width statistics. Current-noise densities are scaled using the corner temperature.

## Corner-level mapping

`V0`, `V1`, and `V2` are ordered voltage levels matched level-to-level across the characterization sources; they are not one common physical voltage.

| Source | V0 | V1 | V2 |
|---|---:|---:|---:|
| Spin | 1.08 V | 1.20 V | 1.32 V |
| DTC pulse width | 0.72 V | 0.80 V | 0.88 V |
| Measured ReLU | 0.90 V | 1.00 V | 1.10 V |
| Coupler (`coupler_monte_v2`) | 0.98 V | 1.00 V | 1.02 V |

| Temperature level | Celsius | Kelvin |
|---|---:|---:|
| T0 | -20 °C | 253.15 K |
| T1 | 25 °C | 298.15 K |
| T2 | 85 °C | 358.15 K |

Noise spectral densities were scaled by `sqrt(T_K / 298.15 K)`.

## TT process corners

| Statistic over 9 TT corners | Result |
|---|---:|
| Lowest corner mean | **TT_V0_T0: 84.9480%** |
| Highest corner mean | **TT_V1_T2: 87.8660%** |
| Mean of corner means | **87.0451%** |
| Std of corner means | **0.8289%** |

| Corner | Mean (%) | Std (%) |
|---|---:|---:|
| TT_V0_T0 | 84.9480 | 1.9613 |
| TT_V0_T1 | 86.5300 | 0.5110 |
| TT_V0_T2 | 87.0240 | 0.2672 |
| TT_V1_T0 | 87.2580 | 0.1997 |
| TT_V1_T1 | 87.5580 | 0.2833 |
| TT_V1_T2 | 87.8660 | 0.2048 |
| TT_V2_T0 | 87.1220 | 0.3888 |
| TT_V2_T1 | 87.4460 | 0.2188 |
| TT_V2_T2 | 87.6540 | 0.1132 |

## All 45 corners

| Statistic over all 45 corners | Result |
|---|---:|
| Lowest corner mean | **FS_V0_T0: 83.9540% ± 1.7413%** |
| Highest corner mean | **SF_V2_T2: 87.9760%** |
| Highest single-trial accuracy | **SF_V1_T2: 88.4100%** |
| Mean of corner means | **87.0240%** |
| Std of corner means | **0.8846%** |

| Corner | Mean (%) | Std (%) |
|---|---:|---:|
| TT_V0_T0 | 84.9480 | 1.9613 |
| TT_V0_T1 | 86.5300 | 0.5110 |
| TT_V0_T2 | 87.0240 | 0.2672 |
| TT_V1_T0 | 87.2580 | 0.1997 |
| TT_V1_T1 | 87.5580 | 0.2833 |
| TT_V1_T2 | 87.8660 | 0.2048 |
| TT_V2_T0 | 87.1220 | 0.3888 |
| TT_V2_T1 | 87.4460 | 0.2188 |
| TT_V2_T2 | 87.6540 | 0.1132 |
| FF_V0_T0 | 85.1180 | 0.5026 |
| FF_V0_T1 | 85.1800 | 1.2765 |
| FF_V0_T2 | 85.6820 | 0.6449 |
| FF_V1_T0 | 86.8940 | 0.4151 |
| FF_V1_T1 | 86.9840 | 0.0845 |
| FF_V1_T2 | 87.4300 | 0.1833 |
| FF_V2_T0 | 86.5040 | 0.6139 |
| FF_V2_T1 | 86.8540 | 0.2392 |
| FF_V2_T2 | 87.1740 | 0.2017 |
| SS_V0_T0 | 86.1520 | 0.5612 |
| SS_V0_T1 | 87.4120 | 0.3095 |
| SS_V0_T2 | 87.7320 | 0.3121 |
| SS_V1_T0 | 87.0140 | 0.2781 |
| SS_V1_T1 | 87.6720 | 0.2534 |
| SS_V1_T2 | 87.8480 | 0.2955 |
| SS_V2_T0 | 87.4880 | 0.2041 |
| SS_V2_T1 | 87.1180 | 0.2709 |
| SS_V2_T2 | 87.5300 | 0.2184 |
| FS_V0_T0 | 83.9540 | 1.7413 |
| FS_V0_T1 | 86.3400 | 1.0188 |
| FS_V0_T2 | 87.3360 | 0.3943 |
| FS_V1_T0 | 87.0440 | 0.3275 |
| FS_V1_T1 | 87.5160 | 0.1710 |
| FS_V1_T2 | 87.8740 | 0.1744 |
| FS_V2_T0 | 87.2760 | 0.2767 |
| FS_V2_T1 | 87.4700 | 0.2871 |
| FS_V2_T2 | 87.8000 | 0.2315 |
| SF_V0_T0 | 85.6640 | 1.3491 |
| SF_V0_T1 | 87.2340 | 0.3873 |
| SF_V0_T2 | 87.7480 | 0.2802 |
| SF_V1_T0 | 87.2120 | 0.4102 |
| SF_V1_T1 | 87.6000 | 0.3057 |
| SF_V1_T2 | 87.9120 | 0.2838 |
| SF_V2_T0 | 87.4320 | 0.2431 |
| SF_V2_T1 | 87.5020 | 0.2125 |
| SF_V2_T2 | 87.9760 | 0.1881 |

# CIFAR-100 Coupler-v2 QF1: 45-Corner Accuracy Summary

This summarizes full-CIFAR-100 test-dataset accuracy across 45 aligned process, voltage, and temperature corners for:

`TIMMQAT5b8aNT0p0mulTIMMPCNetNoBatchNorm_PCConvReLU6_0.0eps_ToggleODEXInitFFFB_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_C100_3K1S96C_0.25Dropout_16Layers4l5l4_2Pool_srrlDistill_a0p3_t2p0_scanGFI_1REP`

For every corner, the table reports the accuracy mean and population standard deviation across independently sampled hardware realizations. This version uses the corresponding corner data for spin statistics, empirical coupler conductance curves, measured ReLU curves, measured pooling, and DTC pulse-width statistics. Current-noise densities are scaled using the corner temperature.

The evaluated model uses `weight_quant_factor_bits=1`, 5-bit pulse weights, and 8-bit output quantization.

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
| Lowest corner mean | **TT_V0_T0: 60.9320%** |
| Highest corner mean | **TT_V1_T2: 64.3680%** |
| Mean of corner means | **62.9689%** |
| Std of corner means | **0.9927%** |

| Corner | Mean (%) | Std (%) |
|---|---:|---:|
| TT_V0_T0 | 60.9320 | 1.6548 |
| TT_V0_T1 | 62.8240 | 0.5372 |
| TT_V0_T2 | 63.4400 | 0.4685 |
| TT_V1_T0 | 62.7940 | 0.3327 |
| TT_V1_T1 | 63.6360 | 0.3455 |
| TT_V1_T2 | 64.3680 | 0.1528 |
| TT_V2_T0 | 61.8100 | 0.5591 |
| TT_V2_T1 | 63.1240 | 0.2174 |
| TT_V2_T2 | 63.7920 | 0.3020 |

## All 45 corners

| Statistic over all 45 corners | Result |
|---|---:|
| Lowest corner mean | **FS_V0_T0: 59.7660% ± 1.2851%** |
| Highest corner mean | **TT_V1_T2: 64.3680%** |
| Highest single-trial accuracy | **SS_V0_T2: 64.8400%** |
| Mean of corner means | **62.7817%** |
| Std of corner means | **1.1572%** |

| Corner | Mean (%) | Std (%) |
|---|---:|---:|
| TT_V0_T0 | 60.9320 | 1.6548 |
| TT_V0_T1 | 62.8240 | 0.5372 |
| TT_V0_T2 | 63.4400 | 0.4685 |
| TT_V1_T0 | 62.7940 | 0.3327 |
| TT_V1_T1 | 63.6360 | 0.3455 |
| TT_V1_T2 | 64.3680 | 0.1528 |
| TT_V2_T0 | 61.8100 | 0.5591 |
| TT_V2_T1 | 63.1240 | 0.2174 |
| TT_V2_T2 | 63.7920 | 0.3020 |
| FF_V0_T0 | 60.5000 | 0.4669 |
| FF_V0_T1 | 60.9020 | 1.2023 |
| FF_V0_T2 | 61.7500 | 0.7588 |
| FF_V1_T0 | 62.1120 | 0.7826 |
| FF_V1_T1 | 62.5120 | 0.5016 |
| FF_V1_T2 | 63.6360 | 0.2755 |
| FF_V2_T0 | 60.2880 | 1.7226 |
| FF_V2_T1 | 61.3620 | 0.7692 |
| FF_V2_T2 | 62.7220 | 0.5824 |
| SS_V0_T0 | 61.9560 | 0.5555 |
| SS_V0_T1 | 63.5560 | 0.2919 |
| SS_V0_T2 | 64.1480 | 0.4880 |
| SS_V1_T0 | 62.7600 | 0.1740 |
| SS_V1_T1 | 63.4180 | 0.2225 |
| SS_V1_T2 | 63.8680 | 0.2409 |
| SS_V2_T0 | 62.5540 | 0.1632 |
| SS_V2_T1 | 61.8740 | 0.6898 |
| SS_V2_T2 | 63.2420 | 0.3277 |
| FS_V0_T0 | 59.7660 | 1.2851 |
| FS_V0_T1 | 62.3020 | 0.9514 |
| FS_V0_T2 | 63.5380 | 0.4454 |
| FS_V1_T0 | 62.5340 | 0.2889 |
| FS_V1_T1 | 63.5400 | 0.1313 |
| FS_V1_T2 | 64.3320 | 0.0694 |
| FS_V2_T0 | 61.9120 | 0.1594 |
| FS_V2_T1 | 63.3240 | 0.4258 |
| FS_V2_T2 | 64.1980 | 0.1434 |
| SF_V0_T0 | 61.4220 | 1.1695 |
| SF_V0_T1 | 63.3420 | 0.2060 |
| SF_V0_T2 | 64.2660 | 0.2492 |
| SF_V1_T0 | 62.8280 | 0.3373 |
| SF_V1_T1 | 63.4320 | 0.2233 |
| SF_V1_T2 | 64.3520 | 0.1805 |
| SF_V2_T0 | 62.6160 | 0.3211 |
| SF_V2_T1 | 63.2380 | 0.1955 |
| SF_V2_T2 | 64.3540 | 0.1651 |

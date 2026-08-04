# CIFAR-10 45-Corner Accuracy Summary

This summarizes full-CIFAR-10 test-dataset accuracy across 45 aligned process, voltage, and temperature corners for:

`TIMMQAT5b8aNT0p0mulTIMMPCNetNoBatchNorm_PCConvReLU6_0.0eps_ToggleODEXInitFFFB_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_3K1S96C_0.25Dropout_16Layers4l5l4_2Pool_srrlDistill_a0p3_t2p0_scanGFI_1REP`

For every corner, the table reports accuracy mean and population standard deviation across independently sampled hardware realizations. This version uses the corresponding corner's spin statistics, empirical coupler conductance curves, measured ReLU Monte Carlo curves, and DTC pulse-width statistics. Current-noise densities are scaled using the corner temperature.

## Corner-level mapping

`V0`, `V1`, and `V2` are ordered voltage levels matched level-to-level across the four characterization sources; they are not one common physical voltage.

| Source | V0 | V1 | V2 |
|---|---:|---:|
| Spin | 1.08 V | 1.20 V | 1.32 V |
| DTC pulse width | 0.72 V | 0.80 V | 0.88 V |
| Measured ReLU | 0.90 V | 1.00 V | 1.10 V |
| Coupler (`coupler_monte`) | 0.98 V | 1.00 V | 1.02 V |

| Temperature level | Celsius | Kelvin |
|---|---:|---:|
| T0 | -20 °C | 253.15 K |
| T1 | 25 °C | 298.15 K |
| T2 | 85 °C | 358.15 K |

Noise spectral densities were scaled by `sqrt(T_K / 298.15 K)`.

## TT process corners

| Statistic over 9 TT corners | Result |
|---|---:|
| Lowest corner mean | **TT_V0_T0: 83.6620%** |
| Highest corner mean | **TT_V0_T2: 87.3700%** |
| Mean of corner means | **86.2304%** |
| Std of corner means | **1.1763%** |

| Corner | Mean (%) | Std (%) |
|---|---:|---:|
| TT_V0_T0 | 83.6620 | 2.4889 |
| TT_V0_T1 | 86.9640 | 0.1116 |
| TT_V0_T2 | 87.3700 | 0.1188 |
| TT_V1_T0 | 85.1860 | 0.8484 |
| TT_V1_T1 | 86.8060 | 0.1240 |
| TT_V1_T2 | 87.3280 | 0.1258 |
| TT_V2_T0 | 85.3140 | 0.1891 |
| TT_V2_T1 | 86.5360 | 0.1904 |
| TT_V2_T2 | 86.9080 | 0.2083 |

## All 45 corners

| Statistic over all 45 corners | Result |
|---|---:|
| Lowest corner mean | **FF_V0_T0: 72.5600%** |
| Highest corner mean | **FS_V1_T2: 87.3940%** |
| Mean of corner means | **85.3942%** |
| Std of corner means | **3.3897%** |

| Corner | Mean (%) | Std (%) |
|---|---:|---:|
| TT_V0_T0 | 83.6620 | 2.4889 |
| TT_V0_T1 | 86.9640 | 0.1116 |
| TT_V0_T2 | 87.3700 | 0.1188 |
| TT_V1_T0 | 85.1860 | 0.8484 |
| TT_V1_T1 | 86.8060 | 0.1240 |
| TT_V1_T2 | 87.3280 | 0.1258 |
| TT_V2_T0 | 85.3140 | 0.1891 |
| TT_V2_T1 | 86.5360 | 0.1904 |
| TT_V2_T2 | 86.9080 | 0.2083 |
| FF_V0_T0 | 72.5600 | 4.0051 |
| FF_V0_T1 | 86.0340 | 0.6722 |
| FF_V0_T2 | 87.3520 | 0.1856 |
| FF_V1_T0 | 77.8160 | 5.8243 |
| FF_V1_T1 | 86.3840 | 0.1616 |
| FF_V1_T2 | 87.2940 | 0.1191 |
| FF_V2_T0 | 81.3840 | 2.1882 |
| FF_V2_T1 | 86.1860 | 0.2075 |
| FF_V2_T2 | 86.9180 | 0.2120 |
| SS_V0_T0 | 86.5920 | 0.1028 |
| SS_V0_T1 | 87.0780 | 0.1342 |
| SS_V0_T2 | 87.2360 | 0.0926 |
| SS_V1_T0 | 86.7480 | 0.1433 |
| SS_V1_T1 | 87.0320 | 0.1926 |
| SS_V1_T2 | 87.1020 | 0.2003 |
| SS_V2_T0 | 86.6100 | 0.1367 |
| SS_V2_T1 | 86.7000 | 0.1594 |
| SS_V2_T2 | 86.9620 | 0.2178 |
| FS_V0_T0 | 73.4200 | 5.0747 |
| FS_V0_T1 | 86.0820 | 0.7475 |
| FS_V0_T2 | 87.3020 | 0.0968 |
| FS_V1_T0 | 78.2120 | 3.5072 |
| FS_V1_T1 | 86.8340 | 0.1040 |
| FS_V1_T2 | 87.3940 | 0.1477 |
| FS_V2_T0 | 83.7740 | 0.8276 |
| FS_V2_T1 | 86.7860 | 0.1679 |
| FS_V2_T2 | 87.2220 | 0.1565 |
| SF_V0_T0 | 84.7420 | 0.9239 |
| SF_V0_T1 | 86.7800 | 0.1207 |
| SF_V0_T2 | 87.2860 | 0.1362 |
| SF_V1_T0 | 84.7500 | 0.5903 |
| SF_V1_T1 | 86.5260 | 0.1695 |
| SF_V1_T2 | 87.1280 | 0.1552 |
| SF_V2_T0 | 85.0300 | 0.2455 |
| SF_V2_T1 | 86.3040 | 0.1392 |
| SF_V2_T2 | 87.1040 | 0.2461 |

## Common inference configuration

- Level-3 expanded `TogglePulseODEXInitFFFB`, direct scaling, five toggle cycles.
- `R = 67 kΩ`, `C = 282 fF`, 5-bit weights, and 8-bit output quantization.
- Coupler source: `coupler_monte`, interpreted as conductance; one empirical curve sampled with replacement per physical coupler and fixed for the complete trial.
- `diff_mismatch=false`, avoiding duplication of the coupler variation already represented by sampled curves.
- Spin variation, measured piecewise-linear ReLU, nonlinear-R, summing-current noise, coupler noise, and DTC nonideality enabled.
- Base summing-current and coupler-noise densities: `0.6 pA/sqrt(Hz)`, with corner-temperature scaling.
- ReLU curves sampled from `relu_monteCarlo`; spin statistics from `PVT_Monte_Carlo_Results_SPIN.csv`; DTC statistics from `PVT_45corner_DTC_pulse_width.csv`.

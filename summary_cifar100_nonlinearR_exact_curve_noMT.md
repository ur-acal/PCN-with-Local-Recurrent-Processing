# Nonlinear-R Exact-Curve No-MT: 45-Corner Accuracy Summary

This summarizes full-CIFAR-100 test-dataset accuracy across 45 aligned process, voltage, and temperature corners for:

`TIMMQAT5b8aNT0p0mulTIMMPCNetNoBatchNorm_PCConvReLU6_0.0eps_ToggleODEXInitFFFB_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_C100_3K1S96C_0.25Dropout_16Layers4l5l4_2Pool_srrlDistill_a0p3_t2p0_scanGFI_1REP`

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
| Lowest corner mean | **TT_V0_T0: 59.8100%** |
| Highest corner mean | **TT_V0_T2: 64.3560%** |
| Mean of corner means | **62.5516%** |
| Std of corner means | **1.3953%** |

| Corner | Mean (%) | Std (%) |
|---|---:|---:|
| TT_V0_T0 | 59.8100 | 2.8511 |
| TT_V0_T1 | 63.5860 | 0.2385 |
| TT_V0_T2 | 64.3560 | 0.1877 |
| TT_V1_T0 | 61.2980 | 0.6741 |
| TT_V1_T1 | 63.0800 | 0.3065 |
| TT_V1_T2 | 63.8800 | 0.3688 |
| TT_V2_T0 | 61.2320 | 0.5364 |
| TT_V2_T1 | 62.6040 | 0.4059 |
| TT_V2_T2 | 63.1180 | 0.7816 |

## All 45 corners

| Statistic over all 45 corners | Result |
|---|---:|
| Lowest corner mean | **FF_V0_T0: 46.6820%** |
| Highest corner mean | **TT_V0_T2: 64.3560%** |
| Mean of corner means | **61.6241%** |
| Std of corner means | **3.8370%** |

| Corner | Mean (%) | Std (%) |
|---|---:|---:|
| TT_V0_T0 | 59.8100 | 2.8511 |
| TT_V0_T1 | 63.5860 | 0.2385 |
| TT_V0_T2 | 64.3560 | 0.1877 |
| TT_V1_T0 | 61.2980 | 0.6741 |
| TT_V1_T1 | 63.0800 | 0.3065 |
| TT_V1_T2 | 63.8800 | 0.3688 |
| TT_V2_T0 | 61.2320 | 0.5364 |
| TT_V2_T1 | 62.6040 | 0.4059 |
| TT_V2_T2 | 63.1180 | 0.7816 |
| FF_V0_T0 | 46.6820 | 4.2237 |
| FF_V0_T1 | 62.8480 | 0.7165 |
| FF_V0_T2 | 64.2960 | 0.1201 |
| FF_V1_T0 | 52.9120 | 6.6758 |
| FF_V1_T1 | 62.8060 | 0.3372 |
| FF_V1_T2 | 63.9400 | 0.3074 |
| FF_V2_T0 | 56.7180 | 2.3045 |
| FF_V2_T1 | 62.3020 | 0.4939 |
| FF_V2_T2 | 63.2240 | 0.5539 |
| SS_V0_T0 | 62.9920 | 0.2065 |
| SS_V0_T1 | 63.8200 | 0.3611 |
| SS_V0_T2 | 64.0300 | 0.1507 |
| SS_V1_T0 | 63.2840 | 0.3342 |
| SS_V1_T1 | 63.4640 | 0.5625 |
| SS_V1_T2 | 63.3360 | 0.4859 |
| SS_V2_T0 | 62.7680 | 0.4673 |
| SS_V2_T1 | 62.6720 | 0.6054 |
| SS_V2_T2 | 63.1600 | 0.4025 |
| FS_V0_T0 | 48.7700 | 5.3831 |
| FS_V0_T1 | 62.8760 | 0.8664 |
| FS_V0_T2 | 64.0960 | 0.2348 |
| FS_V1_T0 | 54.0040 | 3.9741 |
| FS_V1_T1 | 63.5240 | 0.2975 |
| FS_V1_T2 | 63.8720 | 0.3574 |
| FS_V2_T0 | 60.3760 | 0.6974 |
| FS_V2_T1 | 62.8640 | 0.5121 |
| FS_V2_T2 | 63.4580 | 0.3347 |
| SF_V0_T0 | 60.3900 | 1.1716 |
| SF_V0_T1 | 63.2660 | 0.3130 |
| SF_V0_T2 | 63.9880 | 0.3004 |
| SF_V1_T0 | 60.5560 | 0.7249 |
| SF_V1_T1 | 62.7340 | 0.4167 |
| SF_V1_T2 | 63.7880 | 0.2822 |
| SF_V2_T0 | 60.4080 | 0.4119 |
| SF_V2_T1 | 62.1360 | 0.4784 |
| SF_V2_T2 | 63.7620 | 0.4567 |

## Common inference configuration

- Level-3 expanded `TogglePulseODEXInitFFFB`, direct scaling, five toggle cycles.
- `R = 67 kΩ`, `C = 282 fF`, 5-bit weights, and 8-bit output quantization.
- Coupler source: `coupler_monte`, interpreted as conductance; one empirical curve sampled with replacement per physical coupler and fixed for the complete trial.
- `diff_mismatch=false`, avoiding duplication of the coupler variation already represented by sampled curves.
- Spin variation, measured piecewise-linear ReLU, nonlinear-R, summing-current noise, coupler noise, and DTC nonideality enabled.
- Base summing-current and coupler-noise densities: `0.6 pA/sqrt(Hz)`, with corner-temperature scaling.
- ReLU curves sampled from `relu_monteCarlo`; spin statistics from `PVT_Monte_Carlo_Results_SPIN.csv`; DTC statistics from `PVT_45corner_DTC_pulse_width.csv`.

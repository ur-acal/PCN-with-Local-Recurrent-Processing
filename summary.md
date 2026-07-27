# All-on measured-activation corner sweep

Recorded CIFAR-100 accuracy by fixed measured-activation corner, using:

- model: `TIMMQAT5b8aNT0p25mulTIMMPCNetNoBatchNorm_PCConvReLU6_0.0eps_ToggleODEXInitFFFB_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_C100_3K1S96C_0.25Dropout_16Layers4l5l4_2Pool_srrlDistill_a0p3_t2p0_scanGFI_6REP`;
- piecewise-linear activation from `relu_current_0p2uA_all.csv`;
- spin variation using a fitted `sigma_spin=0.10` obtained from data of different
  corners;
- multiplicative coupler variation using a fitted `sigma=0.181` (18.1%)
  obtained from data of different
  corners;
- input-dependent nonlinear-R characterization curve;
- coupler noise using a noise spectral density value of `0.6 pA/sqrt(Hz)`;
- DTC variation/jitter using fitted timing standard deviations obtained from data of different
  corners: zero leading-edge variation, `1.8%` width
  variation, and `0.5%` leading- and falling-edge jitter, all relative to the
  minimum pulse width.

This run uses one
representative fitted value per effect from data of different corners. The measured ReLU remains corner-specific: all 45 process, voltage, and
temperature curves in `relu_current_0p2uA_all.csv` are evaluated.

## Accuracy by fixed activation corner

Each cell is the recorded top-1 accuracy in percent.

| Process | VDD | -20 C | 25 C | 85 C |
|---|---:|---:|---:|---:|
| FF | 0.9 V | 62.2100 | 62.3000 | 62.3300 |
| FF | 1.0 V | 62.2100 | 62.3000 | 62.4200 |
| FF | 1.1 V | 61.6300 | 61.7600 | 61.9500 |
| FS | 0.9 V | 62.2000 | 62.2200 | 62.1700 |
| FS | 1.0 V | 62.2700 | 62.3700 | 62.3600 |
| FS | 1.1 V | 61.8700 | 62.1100 | 62.2000 |
| SF | 0.9 V | 61.3700 | 61.2600 | 61.2600 |
| SF | 1.0 V | 61.7700 | 61.7700 | 61.7300 |
| SF | 1.1 V | 61.9100 | 61.9100 | 61.8100 |
| SS | 0.9 V | 61.0900 | 61.1100 | 61.1600 |
| SS | 1.0 V | 61.5300 | 61.6300 | 61.6000 |
| SS | 1.1 V | 61.8300 | 61.8300 | 61.8900 |
| TT | 0.9 V | 62.0000 | 62.0000 | 61.8600 |
| TT | 1.0 V | 62.2000 | 62.0200 | 62.0400 |
| TT | 1.1 V | 62.4200 | 62.2800 | 62.2600 |

## Summary

- TT reference (`TT_VDD1_T25`): **62.0200%**.
- Mean over all 45 fixed corners: **61.9204%**.
- Lowest accuracy: **61.0900%** (`SS_VDD0P9_TM20`).
- Highest accuracy: **62.4200%** (`FF_VDD1_T85`).

# Updated 45-corner inference results

This reproduction package contains two empirical Monte Carlo 45-corner studies.
Each detailed table reports every process, ordered voltage, and temperature corner.
This version uses aligned data from all 45 corners for spin variation, coupler conductance variation and nonlinearity, measured ReLU behavior, and DTC pulse-width variation. Current-noise scaling follows each corner's temperature.

| Study | Dataset | Mean of 45 corner means | Std of 45 corner means | Detailed table |
|---|---|---:|---:|---|
| `nonlinearR_exact_curve_noMT` | CIFAR-100 | 61.6241% | 3.8370% | [CIFAR-100 summary](summary_cifar100_nonlinearR_exact_curve_noMT.md) |
| `0803_kd_crd_ft_ToggleODEXInitFFFB_toggle_odexinit` | CIFAR-10 | 85.3942% | 3.3897% | [CIFAR-10 summary](summary_cifar10_0803_kd_crd_ft_ToggleODEXInitFFFB_toggle_odexinit.md) |

Both studies use the same characterized-corner evaluation method and differ in the trained model and dataset.

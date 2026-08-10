# Updated 45-corner inference results

This reproduction package contains two empirical Monte Carlo 45-corner studies. Each detailed table reports every process, ordered voltage, and temperature corner. This version uses aligned corner data for spin variation, coupler conductance variation and nonlinearity, measured ReLU behavior, DTC pulse-width variation, and temperature-scaled current noise. The CIFAR-100 study also uses measured physical-domain pooling.

| Study | Dataset | Mean of 45 corner means | Std of 45 corner means | Lowest corner | Lowest-corner mean | Lowest-corner std | Detailed table |
|---|---|---:|---:|---|---:|---:|---|
| `coupler_monte_v2_cifar100_qf1` | CIFAR-100 | 62.7817% | 1.1572% | `FS_V0_T0` | 59.7660% | 1.2851% | [CIFAR-100 summary](summary_cifar100_coupler_monte_v2_qf1.md) |
| `0803_kd_crd_ft_ToggleODEXInitFFFB_toggle_odexinit` | CIFAR-10 | 85.3942% | 3.3897% | `FF_V0_T0` | 72.5600% | 4.0051% | [CIFAR-10 summary](summary_cifar10_0803_kd_crd_ft_ToggleODEXInitFFFB_toggle_odexinit.md) |

Both studies use the same characterized-corner evaluation method and differ in the trained model, dataset, and study-specific hardware configuration.

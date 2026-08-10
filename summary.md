# Updated 45-corner inference results

This reproduction package contains two empirical Monte Carlo 45-corner studies. Each detailed table reports every process, ordered voltage, and temperature corner. This version uses aligned corner data for spin variation, coupler conductance variation and nonlinearity, measured ReLU behavior, DTC pulse-width variation, and temperature-scaled current noise. Both studies use measured physical-domain pooling.

| Study | Dataset | Mean of 45 corner means | Std of 45 corner means | Lowest corner | Lowest-corner mean | Lowest-corner std | Detailed table |
|---|---|---:|---:|---|---:|---:|---|
| `coupler_monte_v2_cifar100_qf1` | CIFAR-100 | 62.7817% | 1.1572% | `FS_V0_T0` | 59.7660% | 1.2851% | [CIFAR-100 summary](summary_cifar100_coupler_monte_v2_qf1.md) |
| `coupler_monte_v2_cifar10` | CIFAR-10 | 87.0240% | 0.8846% | `FS_V0_T0` | 83.9540% | 1.7413% | [CIFAR-10 summary](summary_cifar10_coupler_monte_v2.md) |

| Study | Highest corner | Highest-corner mean | Highest single-trial accuracy |
|---|---|---:|---:|
| `coupler_monte_v2_cifar100_qf1` | `TT_V1_T2` | 64.3680% | `SS_V0_T2`: 64.8400% |
| `coupler_monte_v2_cifar10` | `SF_V2_T2` | 87.9760% | `SF_V1_T2`: 88.4100% |

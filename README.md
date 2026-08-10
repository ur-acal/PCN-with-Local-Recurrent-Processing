# ScAN-PCN empirical 45-corner inference reproduction

This package reproduces two all-on hardware-nonideality accuracy studies over 45 aligned process, voltage, and temperature corners.

This version uses aligned corner-specific characterization data for spin variation, coupler conductance variation and nonlinearity, measured ReLU behavior, measured pooling, and DTC pulse-width variation. Current-noise densities are scaled using the temperature of the selected corner.

## Result summaries

The headline results are in [`summary.md`](summary.md). Complete 45-corner tables are provided in:

- [`summary_cifar100_coupler_monte_v2_qf1.md`](summary_cifar100_coupler_monte_v2_qf1.md)
- [`summary_cifar10_coupler_monte_v2.md`](summary_cifar10_coupler_monte_v2.md)

## Evaluated nonidealities

For each full-dataset hardware trial, static hardware quantities are sampled once and retained across every input and batch. Dynamic noise is resampled during pulse execution.

The evaluation includes:

- corner-specific spin-gain mean and local-mismatch standard deviation from `PVT_Monte_Carlo_Results_SPIN.csv`;
- one empirical conductance curve sampled with replacement per expanded physical coupler from the corresponding empirical corner bank (`coupler_monte_v2` for both studies);
- a measured piecewise-linear ReLU curve sampled from the 100 Monte Carlo curves for the selected corner;
- measured physical-domain average pooling using the selected corner conductance bank;
- input-dependent conductance evaluated from the sampled empirical coupler curve;
- summing-current and per-coupler current-noise densities of `0.6 pA/sqrt(Hz)`, scaled by `sqrt(T_K / 298.15 K)`;
- corner-specific DTC pulse-width mean and local-mismatch standard deviation, plus leading- and falling-edge jitter.

The provided launcher uses `R = 50 kOhm`, `C = 500 fF`, five toggle cycles, 5-bit pulse weights, `weight_quant_factor_bits=1`, 8-bit output quantization, direct RC timing, and expanded convolution weights for both datasets.

## Package contents

The reproduction ZIP contains:

- the inference source and local MC45 launcher;
- the complete characterized data under `hardware_data/mc_45_corners/`;
- both recorded result summaries;
- both required model checkpoints under `saved_ckpt_runs/`;
- `environment.yml`;
- the vendored scangen 0.3.0 source under `third_party/scangen/`.

The ZIP intentionally excludes the test datasets.

## External test data

Two HDF5 files shipped with scangen 0.3.0 are used:

- `cifar100_raw.h5` for the `coupler_v2_cifar100_qf1` model;
- `cifar10_raw.h5` for the `coupler_v2_cifar10` model.

Pass their absolute paths with:

```bash
export SCAN_TEST_CIFAR100_DATA=/absolute/path/to/cifar100_raw.h5
export SCAN_TEST_CIFAR10_DATA=/absolute/path/to/cifar10_raw.h5
```

`SCAN_TEST_DATA` remains a generic fallback when running only one dataset. If no environment variable is set, the loader checks `data/cifar100_raw.h5` or `data/cifar10_raw.h5` according to the selected model.

## Environment

```bash
conda env create -f environment.yml
conda activate scan_test
```

Check the installation and both datasets:

```bash
python - <<PY
import os
import h5py
import torch
from importlib.metadata import version

print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("scangen:", version("scangen"))
for variable in ("SCAN_TEST_CIFAR100_DATA", "SCAN_TEST_CIFAR10_DATA"):
    path = os.environ[variable]
    with h5py.File(path, "r") as handle:
        print(variable, path, handle["images"].shape,
              int((~handle["train"][:]).sum()), "test samples")
PY
```

## Run the CIFAR-100 study

The launcher defaults to this study. `N_TRIALS` may be set to choose the number of independent hardware realizations evaluated per corner:

```bash
SCAN_TEST_CIFAR100_DATA=/absolute/path/to/cifar100_raw.h5 \
./launch_scripts/run_mc45_toggle_ablation.sh
```

Equivalent explicit configuration:

```bash
OUTPUT_DIR=results/coupler_monte_v2_cifar100_qf1 \
MODEL_NAME=TIMMQAT5b8aNT0p0mulTIMMPCNetNoBatchNorm_PCConvReLU6_0.0eps_ToggleODEXInitFFFB_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_C100_3K1S96C_0.25Dropout_16Layers4l5l4_2Pool_srrlDistill_a0p3_t2p0_scanGFI_1REP \
MODEL_DIR=saved_ckpt_runs/coupler_v2_cifar100_qf1 \
SCAN_TEST_CIFAR100_DATA=/absolute/path/to/cifar100_raw.h5 \
./launch_scripts/run_mc45_toggle_ablation.sh
```

## Run the CIFAR-10 study

```bash
OUTPUT_DIR=results/coupler_monte_v2_cifar10 \
MODEL_NAME=TIMMQAT5b8aNT0p0mulTIMMPCNetNoBatchNorm_PCConvReLU6_0.0eps_ToggleODEXInitFFFB_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_3K1S96C_0.25Dropout_16Layers4l5l4_2Pool_srrlDistill_a0p3_t2p0_scanGFI_1REP \
MODEL_DIR=saved_ckpt_runs/coupler_v2_cifar10 \
SCAN_TEST_CIFAR10_DATA=/absolute/path/to/cifar10_raw.h5 \
./launch_scripts/run_mc45_toggle_ablation.sh
```

## Quick selected-corner check

Set `CORNER_IDS` to evaluate a selected subset without changing the hardware configuration:

```bash
CORNER_IDS="FF_V0_T0 SF_V1_T0" \
OUTPUT_DIR=results/quick_cifar100_check \
SCAN_TEST_CIFAR100_DATA=/absolute/path/to/cifar100_raw.h5 \
./launch_scripts/run_mc45_toggle_ablation.sh
```

The launcher writes `corner_trials.csv`, `summary.md`, per-corner logs, and `run_config.txt` under `OUTPUT_DIR`. Expanded convolution weights are generated on first use and cached under `expanded_weights/`.

# ScAN-PCN 45-corner inference reproduction

This package reproduces the CIFAR-100 accuracy sweep over 45 characterized
measured-activation corners.

## Result summary

The reported 45-corner accuracy table is stored in [`summary.md`](summary.md)
in Markdown format. A new sweep also writes a Markdown summary in its selected
output directory and prints that table to the console.

All corners use the same model and the same non-activation non-idealities:

- spin variation with `sigma_spin=0.10`;
- multiplicative coupler variation with `sigma=0.181` (18.1%);
- input-dependent conductance from `hardware_data/res_vs_vin_10k_150k.csv`;
- per-coupler current noise of `0.6 pA/sqrt(Hz)`;
- DTC width variation of `1.8%` of the minimum pulse width;
- DTC leading- and falling-edge jitter of `0.5%` of the minimum pulse width.

The fixed piecewise-linear measured ReLU curve is the corner-dependent setting
among the 45 process, voltage, and temperature corners in
`hardware_data/relu_current_0p2uA_all.csv`.

## 1. Package contents

The ZIP is expected to contain:

- this source tree;
- `summary.md`, the previously recorded 45-corner table;
- `data/scangen_cifar100_noise.json`, documenting the fixed input-noise path;
- the required best checkpoint under `saved_ckpt/`;
- `environment.yml`, which creates the tested `scan_test` environment;
- `third_party/scangen`, the complete clean scangen 0.3.0 source tree.

The ZIP intentionally does not contain the test dataset. The required file is
`cifar100_raw.h5`, which was shipped with scangen 0.3.0. Keep that file
separately and pass its location through `SCAN_TEST_DATA`.

## 2. Create the environment

An NVIDIA driver compatible with CUDA 12.8 and a Conda installation are
required.

Extract the package and enter its root directory:

```bash
unzip scan_test_45_corner_reproduction.zip -d scan_test_45_corner_reproduction
cd scan_test_45_corner_reproduction
```

Then create and activate the tested environment:

```bash
conda env create -f environment.yml
conda activate scan_test
```

Set the path to the separately supplied test dataset:

```bash
export SCAN_TEST_DATA=/absolute/path/to/cifar100_raw.h5
test -f "${SCAN_TEST_DATA}"
```

The environment pins the versions used for validation, including Python 3.13.7,
PyTorch 2.9.0 with CUDA 12.8, torchvision 0.24.0, NumPy 2.2.6, h5py 3.14.0,
and torchdiffeq 0.2.5.

Check the installation:

```bash
python - <<'PY'
import h5py
import os
import torch
import torchvision
import torchdiffeq
from importlib.metadata import version

print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("CUDA available:", torch.cuda.is_available())
print("scangen:", version("scangen"))
with h5py.File(os.environ["SCAN_TEST_DATA"], "r") as handle:
    print("images:", handle["images"].shape)
    print("test samples:", int((~handle["train"][:]).sum()))
PY
```

The expected dataset shape is `(60000, 4, 16, 16)`, with 10,000 test samples.

## 3. scangen modification

The package includes the complete clean source tree for `scangen==0.3.0`.
Compared with the original version, this copy contains the dataset-indexing edit
that prevents training samples from being mixed into test evaluation.

The required HDF5 file is `cifar100_raw.h5`, shipped with scangen 0.3.0 but not
included in this ZIP. Set `SCAN_TEST_DATA` to its absolute path before running
the sweep. If the variable is not set, the runner falls back to
`data/cifar100_raw.h5`.

## 4. Quick run

Run two corners with one trial each:

```bash
N_TRIALS=1 \
CORNERS="TT_VDD1_T25 SS_VDD0P9_TM20" \
OUTPUT_DIR=results/two_corner_validation \
SCAN_TEST_DATA=/absolute/path/to/cifar100_raw.h5 \
./launch_scripts/run_activation_corner_sweep.sh
```

The launcher evaluates the complete CIFAR-100 test set for each selected corner,
then prints a two-corner Markdown table. Expanded convolution weights are
created automatically on first use and cached under `expanded_weights/`.

## 5. Run the complete 45-corner sweep

The launcher defaults to 10 trials per corner:

```bash
./launch_scripts/run_activation_corner_sweep.sh
```

Equivalent explicit form:

```bash
N_TRIALS=10 \
OUTPUT_DIR=results/activation_corner_reproduction \
SCAN_TEST_DATA=/absolute/path/to/cifar100_raw.h5 \
./launch_scripts/run_activation_corner_sweep.sh
```

The launch path is:

1. `scripts/run_activation_corner_sweep.py` discovers all 45 activation curves;
2. `scripts/run_toggle_nonideality_ablation.py` runs the full test dataset for
   every requested trial and corner;
3. `scripts/summarize_activation_corner_sweep.py` writes
   `results/activation_corner_reproduction/summary.md`;
4. the final Markdown table is printed to the console.

Generated logs, raw per-trial CSV files, expanded weights, and new results are
ignored by Git.

## 6. Useful overrides

```bash
# Evaluate selected corners.
CORNERS="FF_VDD0P9_T25 TT_VDD1_T25" \
./launch_scripts/run_activation_corner_sweep.sh

# Change test batch size if GPU memory is limited.
TEST_BS=64 ./launch_scripts/run_activation_corner_sweep.sh

# Use alternate locations without changing the package.
SCAN_TEST_DATA=/absolute/path/to/cifar100_raw.h5 \
MODEL_DIR=/absolute/path/to/saved_ckpt \
EXPANDED_W_DIR=/absolute/path/to/expanded_weights \
OUTPUT_DIR=/absolute/path/to/results \
./launch_scripts/run_activation_corner_sweep.sh
```

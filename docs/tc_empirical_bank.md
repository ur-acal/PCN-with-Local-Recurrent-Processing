# TC empirical-bank inference

TC PCN unrolled evaluation optionally uses a saved finite bank instead of
independent Gaussian curve storage. Existing Gaussian behavior remains default.
This evaluation change does not alter measured pooling, noise or solver settings,
and does not rerun training.

From the repository root:

```bash
python diagnostic_scripts/sample_tc_empirical_bank.py --output results/tc_bank100.npz
```

Add **both** settings to the existing TC evaluation launch environment:

```bash
NONLINEAR_R_CURVE_SAMPLING=empirical_with_replacement \
TC_EMPIRICAL_CURVE_BANK=./results/tc_bank100.npz \
bash ./launch_scripts/run_tc_nonidealities.sh eval
```

Retain the model, checkpoint and other settings from the existing command.
The equivalent Python options are `--nonlinear_R_curve_sampling empirical_with_replacement`
and `--tc_empirical_curve_bank PATH`. A bank path alone does **not** select the
sampler: Gaussian remains the default and ignores the empirical path.
`TC_CURVE_SAMPLING=histogram` is the separate dense-training code-selection
option, not this inference setting.

Legacy NPZ banks replace **coupler sampling only**. Measured pooling retains its
Gaussian mean/covariance configuration, including `MEASURED_POOLING_CURVE_PATH`,
`MEASURED_POOLING_NOMINAL_R` and `TC_COVARIANCE_TABLE`; keep those settings from the
original all-on command. This differs from the multi-level measured CSV-directory
bank, which supplies empirical curves for both couplers and pooling. NPZ banks
do not contain corner labels and therefore do not support corner-coupled FT.

NPZ arrays: `v_grid` [V], `programmed_resistances` [codes], `curves`
[codes, draws, V], with absolute resistances in ohms. The default generator draws
100 curves for each of 15 resistance codes using the existing means/covariance
and positivity guard. Codes must match the evaluated model's mapping; use
`--weight-scale` if that mapping differs from the default.

Every nonzero physical coupler independently selects an index within its code's
bank, fixed for the trial. Zero weights remain open circuits. The existing trial
seed offsets control assignment; the bank's generation seed is separate. Lookup
and accumulation reuse the existing empirical helpers. A finite bank approximates
the Gaussian distribution; it is not an identical hardware realization.

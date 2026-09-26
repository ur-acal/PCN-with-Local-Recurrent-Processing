# TC empirical-bank inference

TC PCN unrolled evaluation optionally uses a saved finite bank instead of
independent Gaussian curve storage. Existing Gaussian behavior remains default.
This option does not alter measured pooling, noise, solver settings, or training.

From the repository root:

```bash
python diagnostic_scripts/sample_tc_empirical_bank.py --output results/tc_bank100.npz
```

Add `TC_EMPIRICAL_CURVE_BANK=./results/tc_bank100.npz` to the existing TC evaluation
launch environment (Python: `--tc_empirical_curve_bank PATH`). Supplying this
path selects empirical-with-replacement sampling automatically. Do not pass it
to FT. `TC_CURVE_SAMPLING=histogram` is the separate dense-training code-selection
option, not this inference setting.

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

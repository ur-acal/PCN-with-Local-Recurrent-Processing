# Diagnostic scripts

The model-dependent scripts default to the latest CiFAIR-100 checkpoint. Set
`MODEL_NAME` and `MODEL_DIR` to inspect another checkpoint; the CIFAR task and
input representation are inferred from the model name. The latest CiFAIR-10
checkpoint is the other built-in default and is selected by supplying its model
name (or by passing `--model_name` and `--model_dir` directly).

```bash
./diagnostic_scripts/run_print_quant_timing.sh
./diagnostic_scripts/run_print_weight_distribution.sh

# Paired one-batch forward: propagated output SNR after every ODE layer.
./diagnostic_scripts/run_print_layer_snr.sh

# Paired clean/all-on forward: propagated SINAD and equivalent ENOB.
./diagnostic_scripts/run_print_layer_sinad.sh

# Exact per-layer, per-toggle-step deterministic/total current statistics.
./diagnostic_scripts/run_print_toggle_current_stats.sh

# Exact pulse-slice z/y trajectories for every layer in one forward pass,
# with paired equivalence validation.
./diagnostic_scripts/run_plot_spin_trajectories.sh

# Optionally run the full test accuracy with first-batch instrumentation.
./diagnostic_scripts/run_plot_spin_trajectories.sh --full_accuracy

# With no curve flags or corners, plot both complete 4500-curve banks.
./diagnostic_scripts/run_plot_corner_curves.sh

# Plot one Monte Carlo mean curve for each of the 45 hardware corners.
./diagnostic_scripts/run_plot_corner_mean_curves.sh

# Plot selected corner banks only.
./diagnostic_scripts/run_plot_corner_curves.sh \
  --nonlinear_R --relu --corners "tt_v1_t0,fs_v2_t0"

# Reproduce the existing accuracy histogram and violin PDFs unchanged.
./diagnostic_scripts/run_plot_accuracy_distributions.sh
```

The accuracy launcher accepts `BASELINE_DIR`, `PATCHED_DIR`, and `OUTPUT_DIR`
environment overrides. The quantization/timing script accepts `--csv PATH`.

The runtime scripts use the up-to-date 45-corner configuration. Override their
defaults with `MODEL_NAME`, `MODEL_DIR`, `CORNER`, `TRIAL`, and `BATCH_SIZE`; both accept
`--seed`, `--device`, and `--csv PATH` where applicable. The trajectory
launcher additionally accepts optional `LAYER`, plus `TOGGLE_STEP` and
`N_SPINS`. If `LAYER` is omitted, every ODE layer is recorded. Their
instrumentation is installed only inside the diagnostic process and does not change `ode_pc.py` or production
launch scripts.

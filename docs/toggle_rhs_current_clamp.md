# Toggle Level-3 total-current clamp

Add these exports to the existing toggle MC45 launch command:

```bash
export RHS_CURRENT_SUMMARY=/absolute/path/to/total/separate/summary.md
export RHS_CURRENT_BOUND_PERCENTILE=99
```

Use `total/combined/summary.md` for pooled FF/FB bounds. These are central
fitted-Gaussian intervals from the table, not measured empirical quantiles.
The selected columns must exist. Missing, duplicate, extra, or invalid layer
rows fail explicitly. An empty `RHS_CURRENT_SUMMARY` disables the feature.

The scheduler exports the settings; the MC45 worker, pipeline evaluation,
measured-ReLU-only worker, and direct toggle wrapper forward them. Python CLI
names are `--rhs_current_summary` and `--rhs_current_bound_percentile`.
`RHS_CURRENT_AUDIT_PATH` / `--rhs_current_audit_path` optionally appends JSONL
reports. Each record identifies the corner/case, model and checkpoint; file
appends are locked so one path can be shared safely by scheduler shards. Without
it, mappings and counts are still printed to the evaluation log.

The mapping uses the same one-based `model.PcConvs` order used to create the
current summaries. PCN z updates use FB rows and y updates use FF rows.
The analog classifier is a one-shot FF block: its internal stage is **z**,
but its summary row is **final_linear / FF**. Pooled tables assign the same
layer row to both PCN stages; they never add the FF and FB currents together.

For each pulse slice, after generating the normal summing/coupler noise samples:

```
I = C_stage * rhs + C_stage / dt * (summing_delta_v + coupler_delta_v)
I_limited = clamp(I, summary_lower_A, summary_upper_A)
v_next = project_state(v + dt / C_stage * I_limited)
```

The RHS already includes nonlinear coupler behavior, spin variation, and any
enabled slow current terms. Dynamic noise here is its effective slice-average
current, matching the saved total-current distributions. The operation is a
current limit before integration; the existing state-voltage rail projection
still follows it. Only expanded Level-3 toggle fast-path evaluation is supported.

## Verification and experiment

`diagnostic_scripts/verify_rhs_current_clamp.py` calls the production evaluator.
Its observer independently parses the table and takes the actual block labels
from the original `CurrentRecorder`. It checks every source row, branch, stage,
bound, production current, clipped current, and unprojected capacitor update.
It does not supply or replace the clamp implementation. A two-image batch
executes 41 mappings and 3,015 pulse-slice updates for this model.

`diagnostic_scripts/run_rhs_current_clamp_study.py` first captures arguments
from the actual MC45 shell worker, checks their propagation through the corner
driver, and runs the production-path proof. After all eight proofs pass, it runs
the actual worker for eight complete 10,000-image evaluations at batch size 128:
two corners × separate/pooled × 95/99. The source model and seed are read from
the current-distribution run's metadata. The second corner is selected with a
fixed random seed. Ordinary full evaluations use no diagnostic observer.

```bash
conda activate scanbase
python diagnostic_scripts/run_rhs_current_clamp_study.py \
  --source results/toggle_summing_current_distribution/C100_C36_72_8_10_allReLU_analog_biasfalse_adc05_iq12_toggle_odexinit_FS_V2_T1_1trials
```

Results and per-row proof tables: [study README](../results/rhs_current_clamp_study/README.md).
The same limits from FS_V2_T1 training data are used at both evaluation corners.

## Custom limits and pooling-defined stages

`--rhs_current_summary` (shell: `RHS_CURRENT_SUMMARY`) also accepts an explicit
table headed `# Custom separate current limits` or `# Custom combined current limits`:
columns `Layer | Branch | Lower (A) | Upper (A)`. Supported units include µA/uA,
nA and mA. Explicit tables use their bounds directly; the percentile argument is
irrelevant and the audit reports `bounds_source=explicit`, `percentile=null`.
The update arithmetic and all existing launcher wiring are unchanged.

Generate a lossless copy of an existing percentile table:

```bash
conda activate scanbase
python diagnostic_scripts/make_current_limits.py \
  --summary results/toggle_summing_current_distribution/C100_C36_72_8_10_allReLU_analog_biasfalse_adc05_iq12_toggle_odexinit_FS_V2_T1_1trials/total/separate/summary.md \
  --percentile 95 --granularity layer --branches separate --num_layers 20 --pool_positions 9
```

For the pooled table use `total/combined/summary.md` and `--branches pooled`.
Default output is `hardware_data/summing_current_limit/` with filename
`current_limits_<layer|stage>_<separate|pooled>_<p95|p99|manual>.md`.
`--name short.md` overrides the filename; `--overwrite` is required to replace it.
The filename is checked in bytes against the filesystem's `NAME_MAX`.
The generated table uses amperes to preserve the parsed floating-point limits
exactly. Derived tables also record the source-summary hash and available run
metadata; manual tables identify themselves as manually specified. The evaluator
still validates layer/branch compatibility rather than enforcing provenance.
Manual CLI values below are in **µA**.

Example manual stage bounds (illustrative values, not recommended limits):

```bash
python diagnostic_scripts/make_current_limits.py \
  --granularity stage --branches pooled --num_layers 20 --pool_positions 9 \
  --bound stage_01 COMBINED -10 12 \
  --bound stage_02 COMBINED -20 22 \
  --bound final_linear COMBINED -30 15
```

For separate limits provide both FF and FB for each layer/stage, but only FF for
`final_linear`. For per-layer input use `--granularity layer` and `layer_01`, etc.
Every assignment is required; duplicates, extra rows, nonfinite/reversed bounds
are errors. `--no_final_linear` omits the classifier for models without an analog
head. The helper expands stage rows into explicit per-layer rows, so ordinary
evaluation never needs to interpret stages.

Stage statistics: add `--granularity stage` to the existing
`diagnostic_scripts/plot_toggle_current_distributions.py` command. All other
dataset/corner/trial controls stay unchanged. Pool positions are read from the
actual model; pooling after layer N places layer N in the preceding stage.
For this model: stage_01 = layers 1–9, stage_02 = layers 10–20; final_linear is
separate. Both passes aggregate actual samples across all constituent layers,
steps and slices. This is not an average of layer statistics. The usual separate
and combined plots/tables are generated, with stage rows; `run_config.json`
records the membership. Default stage output directories carry a `_stage` suffix.
Pass a stage summary to the helper with `--granularity stage` and matching
`--num_layers`/`--pool_positions` to expand its bounds.
If the source run metadata is present, the helper checks that these stage
boundaries agree with the recorded model layout.

Existing layer histograms are not silently rebinned into stage histograms:
their different bin edges do not permit exact reconstruction. Full stage
histograms require the two-pass collection; replotting existing stage results
does not require inference.

Custom-table regression: `diagnostic_scripts/verify_custom_current_limits.py`
converts both previous 95% summaries, checks every parsed bound for exact
equality, executes two-image independent production proofs for both modes, then
runs **one** matching full separate95 FS_V2_T1 trial through the MC45 worker.
It compares accuracy and all per-layer current/clipping counts against the
original trial. Output: `results/custom_current_limits_verification/README.md`.

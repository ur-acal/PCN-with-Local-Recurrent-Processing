# TC optimization audit and toggle lookup experiment

2026-09-24. Baseline: the **corrected FF-expanded** pinned CiFAIR-100
16-layer/96-channel toggle model, FS_V2_T1, empirical coupler_full_range,
0906 per-spin ReLU, QF1, ENOB=None, fixed timing, all configured nonidealities.
Exact CLI is retained in `results/toggle_lookup_optimization/reference.json`.

## TC audit

Read the solver, TC noise lifecycle, Gaussian kernel, MVMConv dispatch, tests,
profiler and supplied result reports. No new correctness blocker found in the
intended TC paths; this is not a claim of bitwise GPU equivalence.

- `AdaptiveGridSolver.integrate_search_grids`: reuse is gated by gradients
  disabled, TC context, Dopri5 (including ProjDopri5), no state reload, and no
  energy meter. It reuses the final accepted candidate, not a rejected one.
  The TC noise tape advances at `accepted()`, not on RHS evaluation, so skipping
  replay does not skip a fresh physical noise sample. Noise addition, projection
  and interval acceptance remain after the selected candidate. Training replay
  is unchanged. Toggle does not use this adaptive solve path.
- `tc_edge_inference.gaussian_edge_forward`: same per-edge normalized Gaussian
  resistance, interpolation, signed source-current formula and existing curve
  storage. Zero pulse values are masked. This path is restricted to no-grad
  evaluation, CUDA float32, TC per-coupler Gaussian curves, supported projection,
  and no energy observer. Its atomic sum changes accumulation order. The
  supplied 12.09s/0.80s result therefore does **not** demonstrate exact logits;
  the reported 1.51e-4 difference is explicitly documented. The two full-run
  accuracies are not paired reference-versus-optimized accuracy tests.
- Reran the supplied optimization tests, plus TC inference tests imported by
  that suite, the new lookup tests, expansion/cache regressions, and a separate
  state-reload guard check. All passed. No TC production code was changed here.

## Initial lookup-only implementation (historical experiment)

The pinned toggle uses empirical curves, not TC Gaussian curves. Reusing the
whole TC atomic-reduction kernel would neither cover those curves nor satisfy
the requested exact-equality constraint.

`toggle_edge_inference.py` instead fuses **only empirical resistance lookup**:
projection, per-curve bounds, left-sided interval search and
`R_left + R_slope * (query - left_v)`. Floating-point multiply/add fusion is
disabled. Different curve lengths and grids are supported. Sampling is not
performed here; the existing assignment tensors are read unchanged.

The original `pulse_values * source * nominal_R / R_eff`, active-edge selection,
chunk order and `index_add_` remain in `validation.py`. No changes to weights,
FF/FB expansion, DTC, white/slow noise, mismatch, ReLU, pooling, quantization,
capacitance, timing or rail projection. Training, CPU, unsupported dtypes,
custom projections and energy-observed calls fall back to PyTorch. Gaussian
toggle sampling remains on the old path. There are no new launcher arguments.

**Enabled by default for supported toggle inference.** The user accepted the
measured 4.7684e-6 maximum logit difference as a numerical match. Ordinary
45-corner commands need no extra arguments. The diagnostic script can set
`_toggle_fused_lookup=False` to restore the reference implementation. This is
numerical equivalence within the measured roundoff, not bitwise equality or a
guarantee of identical predictions for every possible sample/corner.

## Lookup-only measurements

RTX 4090; three consecutive batches of 128, identical seeds/configuration;
first batch excluded from timing to avoid compilation/warm-up cost.

| Measurement | Reference | Lookup optimized |
|---|---:|---:|
| Mean timed forward (seconds/batch) | 6.814 | 2.202 |
| Speedup | 1x | 3.09x |

- All **384/384 predictions match**; maximum absolute logit difference
  **4.7684e-6**, RMS **4.6761e-7**. Not bitwise equal.
- Repeating the reference alone: maximum difference **6.1989e-6** across 256
  samples. The baseline itself is not bitwise reproducible.
- Diagnostic-only `torch.use_deterministic_algorithms(True)` and CUBLAS
  workspace configuration: reference/optimized maximum **2.3842e-6**;
  reference/reference maximum **3.3379e-6**, 256 predictions matching in both
  comparisons. Those settings were not enabled in production. They did not
  eliminate all numerical variation in this execution path.
- Every fused lookup on a complete batch-4 forward was compared at runtime
  against PyTorch on the **same voltage and curve-index tensors**: **4,458
  exact matches**, zero tolerance. Repeating this audit at the actual batch
  size 128 also gave **4,458 exact matches**. Unit tests also check batch 1/4/128,
  noncontiguous inputs, curve knots, adjacent floats, clipping, unequal curve
  lengths, fallback conditions, and no RNG consumption.
- These are bounded forward checks, **not new full-test accuracies**.

## Reproduce

From the repository root with `scanbase` activated:

```bash
DATALOADER_NUM_WORKERS=0 python diagnostic_scripts/profile_toggle_lookup.py \
  --log results/ff_unroll_fix_FS_V2_T1_2trials/FS_V2_T1.log \
  --mode reference --output results/toggle_lookup_optimization/reference.json

DATALOADER_NUM_WORKERS=0 python diagnostic_scripts/profile_toggle_lookup.py \
  --log results/ff_unroll_fix_FS_V2_T1_2trials/FS_V2_T1.log \
  --mode optimized --output results/toggle_lookup_optimization/optimized.json
```

`--verify-lookups --batches 1` compares every actual resistance lookup exactly;
its timing is not a benchmark. `--deterministic` enables the diagnostic-only
control described above. JSON files retain arguments, module types, timings
and raw logits. See `results/toggle_lookup_optimization/` for all measurements.

## Full empirical-current fusion (current default)

The subsequent investigation is restricted to lookup + signed current +
accumulation fusion. **No pulse-slice reuse or sparse-index caching was added.**

`MVMConv._forward_pulse_per_edge` now dispatches supported empirical toggle
inference to `toggle_edge_inference.empirical_edge_forward`. It keeps the
original `pulse_values != 0` active-edge selection, then runs one Triton kernel
for those edges. Each edge reads its original physical curve assignment,
interpolates resistance, computes `pulse_value * source * nominal_R / R_eff`,
and atomically accumulates into its output spin. The input source is not
replaced by the clipped interpolation query. Fractional pulse values retain
DTC overlaps and mismatch; off edges contribute zero.

Noise, DTC sampling, ReLU, pooling, rails, quantization, expansion and physical
time integration are unchanged. The kernel draws no random values and modifies
no assignments. Atomic addition changes floating-point summation order; the
same roundoff-equivalence criterion accepted for the earlier optimization is
used, not a bitwise claim. Training, unsupported devices/dtypes, other sharing
modes, Gaussian curves, and energy observation retain their previous paths.

The initial all-edge fusion scanned off couplers too and was slower than
lookup-only (2.96 versus 2.21 seconds/batch). It was replaced, not retained as
a separate production mode. The active-edge version is the current code.

`profile_toggle_lookup.py --mode reference` explicitly disables **both** toggle
optimizations. `--mode optimized` selects the previous lookup-only version;
`--mode fused` selects full empirical-current fusion. Ordinary 45-corner
launch commands require no changes. Final raw logits and measurements are
under `results/toggle_full_fusion/`; see its `summary.md` for the comparison.
The earlier `fused.json` is the rejected all-edge prototype, not final code.

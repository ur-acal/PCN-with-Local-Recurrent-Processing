# TC inference fast paths

No launcher changes are required. Existing Gaussian per-coupler TC evaluation
automatically enables:

1. Reuse of accepted Dopri5/ProjDopri5 search results when gradients are disabled.
   Noise, rail projection and step lifecycle still execute once in their original
   locations. Training retains its graph-building replay.
2. Fused Gaussian per-edge interpolation, current calculation and accumulation
   on CUDA float32 with Triton, for modules in evaluation mode.

Energy-metered runs use the original paths. The fused kernel also falls back
for CPU, unsupported dtypes/projections, empirical banks and enabled gradients.
Neither optimization changes the curve distribution, quantization or solver
tolerance. Accumulation order may introduce small floating-point differences;
do not expect bitwise-identical GPU logits.

For diagnostics only, `profile_tc_eval_cost.py --optimization` selects
`reference`, `reuse`, `fused`, or `both`. Add `--timing-only` to disable nested
profiling events. The script saves logits for numerical comparisons.

Measurements and numerical differences are recorded in
`results/tc_inference_optimizations/summary.md`.

## Compact Gaussian curve storage

TC unrolled inference now stores curves only for nonzero programmed weights.
An int64 physical-edge-to-curve index preserves CSR connectivity, zero-weight
physical-site counting, sampling order/seeds, and fixed-per-trial lifetimes.
Both fused and fallback lookup paths use this index; energy accounting is
unchanged. Empirical banks and toggle storage are unchanged. No new launch
argument or expanded-weight cache rebuild is required; restart evaluation to
use the new code.

For E physical edge slots, A active edges and K float32 curve points, persistent
curve storage changes from 4EK bytes to 4AK + 8E bytes (including the index).
This is not the total GPU memory footprint or a guarantee against OOM.

`tests/test_tc_compact_curves.py` runs batch-two CNN/one-state PCN/two-state PCN
through their production expansion validators on CUDA, using the measured
mean/covariance package. It asserts compact storage and actual fused calls,
and compares outputs and energy fallback against the old full-table layout.

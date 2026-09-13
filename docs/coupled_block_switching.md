# Coupled spatial-block switching

Implemented in `switch.py`; no solver, trainer, or full NODE changes in this task.
Previous dirty-worktree changes are retained.

## Interface and semantics

Construct the existing switching class with `block_size=m` (default 1).
The common `local_block_flow(x, y, (row0, col0, row1, col1), duration)`
returns the evolved full state; rectangle upper bounds are exclusive.
`x` and `y` are already in the wrapped block's input/state domain, not raw
dataset coordinates. `duration` is signed effective local time. Negative
duration negates the installed RHS over a positive solver interval. The
legacy `scale_RHS` clock conversion is accounted for internally.

All active pixels evolve in the SAME ODE state and read each other's current
values at every RHS evaluation. Only the required intermediate FB sites are
computed; overlapping output receptive fields share their intermediate sites.
Outside pixels are frozen, including during projection and final interpolation.
The installed instance RHS factory is used, preserving wrapper input/current
and derivative scaling. No raw base-class factory bypass is introduced.

The primitive contains no sweep ordering. Lie commits each block immediately;
Jacobi starts every block from the same macro-step snapshot and merges their
active results. Strang orders block half-flows forward then backward. Yoshida
continues composing signed Strang steps through that same primitive. Singleton
Jacobi retains its existing vectorized independent-pixel optimization; this
optimization is NEVER used as a substitute for coupling an m>1 block.

`iter_spatial_blocks(H, W)` partitions rectangular images and returns smaller
edge blocks. `block_geometry(block, (H, W))` obtains the intermediate coordinates
from FF kernel/padding/dilation and the actual FB output shape. Supported local
operators are stride-one, ungrouped, zero-padded convolutions preserving the
full composition's spatial shape, including the existing restricted transposed
FB convolution. Per-pixel leakage is not silently extended to coupled blocks:
the decay variant rejects m>1 with nonzero leakage.

For two 3x3 convolutions, interior spatial operator counts are:

| m | FF sites | FB sites |
|---:|---:|---:|
| 1 | 1 | 9 |
| 2 | 4 | 16 |
| 3 | 9 | 25 |
| 4 | 16 | 36 |

These are spatial operator copies, not scalar resistor counts. Virtual padding
sites require no physical FB operator and are excluded from `fb_count`.

## Validation results

The first real trained ODE block and fixed scanGFI CIFAR-100 input used in the
earlier switching diagnostics were used. The complete model/checkpoint
identifier and physical wrapper recipe are in
`docs/switched_dynamics_diagnostic_snapshot/debug_strang_real_block.py`.
Input and pre-edit outputs are saved in `results/coupled_switch_legacy.pt`.
State shape is 16x16 spatially; batch size 1, two macro-iterations, float32.
Both the full unsplit ODEXInitFFFB reference and mini-solvers use Dopri5,
rtol=atol=1e-8. No new hardware nonidealities were added.

Relative physical-state L2 error versus the full reference, BEFORE output
ENOB quantization:

| m | Jacobi | Lie | Strang |
|---:|---:|---:|---:|
| 1 | 0.482195 | 0.184540 | 0.088957 |
| 2 | 0.258480 | 0.168444 | 0.092350 |
| 3 | 0.195982 | 0.137078 | 0.075316 |
| 4 | 0.147507 | 0.107957 | 0.060008 |

These are block-output errors, not classification accuracy. This small sweep
does not establish an asymptotic convergence order or monotonicity in m.

Singleton regression caveat: the old solver could numerically rewrite even
inactive state entries through interpolation. The new primitive explicitly
preserves them. Thus float32 Lie/Strang prequantization relative differences
from the old singleton implementation are 8.06e-5 / 1.73e-4; output rounding
can amplify these into quantization-bin differences. In float64 the same
comparisons are 1.21e-9 / 1.24e-9, confirming numerical agreement rather than
bitwise float32 identity. Optimized singleton Jacobi is unchanged (zero
output difference). The new outside-state invariant is bitwise exact.

Thirteen focused tests pass, covering geometry, non-divisible boundaries,
bidirectional active-pixel influence, signed wrapper transformations, frozen
versus updated reads, symmetric/Yoshida durations, transposed FB geometry,
and whole-image active flow versus unsplit integration. Real-block checks also
verify unchanged outside entries and bidirectional active-neighbor sensitivity
at m=2,3,4, including the m=3 one-pixel edge block.

Artifacts:

- `results/coupled_switch_validation.json`: full errors, runtimes, singleton
  comparisons before/after output quantization, float64 controls.
- `results/coupled_switch_local_checks.json`: real block boundary/influence checks.
- `results/coupled_switch_capture.log`, `results/coupled_switch_validation.log`,
  `results/coupled_switch_local_checks.log`: logs.
- `scripts/validate_coupled_switch_blocks.py`: reproducible diagnostic;
  `--local-only` runs only local invariants. `--capture` deliberately replaces
  the baseline and must NOT be rerun to regenerate a pre-change baseline.
- `tests/test_switch_coupled_blocks.py`, `tests/test_switch_strang.py`: tests.

No full CIFAR accuracy evaluation or retraining was performed.

## Extended sweep reference convention

The subsequent 108-case sweep (m=2,3,4; iterations=5,8,10,20;
Jacobi/Lie/Strang; mini tolerance=1e-6,1e-7,1e-8) deliberately fixes the
unsplit Dopri5 reference at **rtol=atol=1e-6**, for historical comparability.
It does not reuse this document's earlier 1e-8 reference. The live full table
and status are in `results/coupled_switch_sweep_ref1e6/summary.md` and
`results/coupled_switch_sweep_ref1e6/results.json`.

## Completed coupled-block sweep: matched comparisons

All **108/108 cases completed**. The tables below contain every measured
relative error from this sweep. Reference: unsplit ODEXInitFFFB, Dopri5
rtol=atol=1e-6, fixed across all cases; mini-solver tolerance varies by row.
Errors are physical-state relative L2 before output quantization, not accuracy.

### m=2

| Iterations | Mini tolerance | Jacobi | Lie | Strang |
|---:|---:|---:|---:|---:|
| 5 | 1e-06 | 0.09618834 | 0.07454270 | 0.02764343 |
| 5 | 1e-07 | 0.09930205 | 0.07535299 | 0.02755069 |
| 5 | 1e-08 | 0.10067447 | 0.07581977 | 0.02792917 |
| 8 | 1e-06 | 0.04110354 | 0.04771424 | 0.01477261 |
| 8 | 1e-07 | 0.04174197 | 0.04737134 | 0.01337314 |
| 8 | 1e-08 | 0.04384102 | 0.04773693 | 0.01268324 |
| 10 | 1e-06 | 0.02191751 | 0.03800763 | 0.01262745 |
| 10 | 1e-07 | 0.02295802 | 0.03808736 | 0.00954353 |
| 10 | 1e-08 | 0.02407983 | 0.03838706 | 0.00871727 |
| 20 | 1e-06 | 0.01350144 | 0.02082519 | 0.01081876 |
| 20 | 1e-07 | 0.00880198 | 0.01939816 | 0.00670104 |
| 20 | 1e-08 | 0.00716823 | 0.01921834 | 0.00406555 |

### m=3

| Iterations | Mini tolerance | Jacobi | Lie | Strang |
|---:|---:|---:|---:|---:|
| 5 | 1e-06 | 0.07142223 | 0.06750436 | 0.02717363 |
| 5 | 1e-07 | 0.07233620 | 0.06598635 | 0.02443263 |
| 5 | 1e-08 | 0.07289003 | 0.06582319 | 0.02401319 |
| 8 | 1e-06 | 0.03025069 | 0.04463084 | 0.01470934 |
| 8 | 1e-07 | 0.03083901 | 0.04285535 | 0.01272290 |
| 8 | 1e-08 | 0.03121560 | 0.04248460 | 0.01184691 |
| 10 | 1e-06 | 0.01956487 | 0.03702549 | 0.01211782 |
| 10 | 1e-07 | 0.01894487 | 0.03510265 | 0.00987610 |
| 10 | 1e-08 | 0.01894764 | 0.03437855 | 0.00836402 |
| 20 | 1e-06 | 0.01164898 | 0.02069957 | 0.00899554 |
| 20 | 1e-07 | 0.00813951 | 0.01922557 | 0.00601298 |
| 20 | 1e-08 | 0.00688663 | 0.01814564 | 0.00413075 |

### m=4

| Iterations | Mini tolerance | Jacobi | Lie | Strang |
|---:|---:|---:|---:|---:|
| 5 | 1e-06 | 0.05524441 | 0.05176839 | 0.01953878 |
| 5 | 1e-07 | 0.05588876 | 0.05162986 | 0.01923476 |
| 5 | 1e-08 | 0.05616946 | 0.05173745 | 0.01934847 |
| 8 | 1e-06 | 0.02374460 | 0.03376418 | 0.01024276 |
| 8 | 1e-07 | 0.02373591 | 0.03343609 | 0.00903272 |
| 8 | 1e-08 | 0.02385289 | 0.03340978 | 0.00906009 |
| 10 | 1e-06 | 0.01451425 | 0.02706114 | 0.00757608 |
| 10 | 1e-07 | 0.01419895 | 0.02681834 | 0.00630704 |
| 10 | 1e-08 | 0.01424197 | 0.02689826 | 0.00625855 |
| 20 | 1e-06 | 0.00786231 | 0.01421016 | 0.00590711 |
| 20 | 1e-07 | 0.00574646 | 0.01378475 | 0.00333685 |
| 20 | 1e-08 | 0.00525437 | 0.01366611 | 0.00224674 |

### Does increasing block size help?

Within this new sweep, m=4 beats both m=2 and m=3 in **all 12 matched
iteration/tolerance pairs for each method** (36/36 comparisons against each
smaller block size). Increasing block size is not universally monotonic:
for example Strang at 20 iterations, 1e-8 gives m=2: 0.00406555 versus
m=3: 0.00413075.

Against historical m=1 measurements at the same reference and mini tolerance,
only iterations 5,10,20 have a complete matching tolerance table; no historical
8-iteration m=1 table is available. Counts below mean strictly lower measured
error, not statistical significance:

| Method | m=2 beats m=1 | m=3 beats m=1 | m=4 beats m=1 |
|---|---:|---:|---:|
| Jacobi | 7/9 | 8/9 | 9/9 |
| Lie | 0/9 | 2/9 | 9/9 |
| Strang | 4/9 | 5/9 | 9/9 |

Thus **m=4 consistently helps in these measurements; m>1 by itself is not a
guarantee of improvement**. In particular m=2 Lie is worse than singleton
Lie at all nine historical matching points.

Illustration at 20 iterations and mini tolerance 1e-8:

| Method | Historical m=1 | m=2 | m=3 | m=4 |
|---|---:|---:|---:|---:|
| Jacobi | 0.008419 | 0.00716823 | 0.00688663 | 0.00525437 |
| Lie (corrected) | 0.017129 | 0.01921834 | 0.01814564 | 0.01366611 |
| Strang | 0.006306 | 0.00406555 | 0.00413075 | 0.00224674 |

The best measured point is m=4 Strang, 20 iterations, mini tolerance 1e-8:
E=0.00224674. This is close to the previously measured approximately 0.00244
reference-tolerance sensitivity; it is not evidence of that accuracy relative
to an exact flow. Error cancellation can produce an error below this sensitivity.

Comparability caveats: singleton values are historical, not rerun here; the
new primitive enforces exactly frozen outside states, whereas old singleton
Lie/Strang allowed float32 interpolation drift. The implementation validation
quantified that difference. Jacobi m=1 also uses its independent-pixel batching
optimization. These comparisons describe the measured implementations, not a
pure isolation of mathematical splitting error. No classification improvement
or asymptotic order is inferred from this single trained block/input.

At the same iteration count the hardware budget is not equal: interior m=4
uses 16 FF and 36 FB spatial sites versus 1 FF and 9 FB for m=1. Larger blocks
also need fewer local solves. GPU runtimes are diagnostic only.

Raw provenance: [results JSON](../../papers/hardware-native-neural-ode/shared/evidence/pixel-switched/data/historical/coupled_switch_sweep_ref1e6/results.json), [full runtime/NFE/solve-count table](../../papers/hardware-native-neural-ode/shared/evidence/pixel-switched/data/historical/coupled_switch_sweep_ref1e6/summary.md), [manifest](../../papers/hardware-native-neural-ode/shared/evidence/pixel-switched/data/historical/coupled_switch_sweep_ref1e6/manifest.json).

### Matched tables at 5, 8, and 10 iterations

Mini-solver tolerance is 1e-8; the full unsplit reference remains fixed at 1e-6.
All values are relative physical-state L2 errors. Historical m=1 caveats above apply.

#### 5 iterations

| Method | Historical m=1 | m=2 | m=3 | m=4 |
|---|---:|---:|---:|---:|
| Jacobi | 0.187350 | 0.100674 | 0.072890 | 0.056169 |
| Lie | 0.068910 | 0.075820 | 0.065823 | 0.051737 |
| Strang | 0.024168 | 0.027929 | 0.024013 | 0.019348 |

#### 8 iterations

| Method | Historical m=1 | m=2 | m=3 | m=4 |
|---|---:|---:|---:|---:|
| Jacobi | — | 0.043841 | 0.031216 | 0.023853 |
| Lie | — | 0.047737 | 0.042485 | 0.033410 |
| Strang | — | 0.012683 | 0.011847 | 0.009060 |

#### 10 iterations

| Method | Historical m=1 | m=2 | m=3 | m=4 |
|---|---:|---:|---:|---:|
| Jacobi | 0.048573 | 0.024080 | 0.018948 | 0.014242 |
| Lie | 0.032246 | 0.038387 | 0.034379 | 0.026898 |
| Strang | 0.009341 | 0.008717 | 0.008364 | 0.006259 |

There are no matching historical m=1 measurements at 8 iterations.

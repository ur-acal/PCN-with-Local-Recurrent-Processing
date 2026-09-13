# Locality-aware color grouping

This is an opt-in PIXEL_SWITCHED software extension. Existing `switch.py`, physical wrappers, training, and numerical solver code are unchanged.

## Dependency and grouping

`spatial_coloring.py` contains only convolution-support analysis, graph construction, deterministic graph coloring, and validation. For each output block, it traces FF output coordinates backward through valid intermediate sites and then through FB to spatial state coordinates. It handles stride, dilation, padding, and transposed-convolution preimages explicitly. Out-of-map zero padding is excluded. Pointwise self-dependence is included conservatively for wrapper terms; learned zeros do not remove structural dependencies.

This assumes the activation and physical transformations between the convolutions are spatially pointwise. Nonlocal transforms require a new dependency description. Zero-padding convolution types are checked; the existing local-flow primitive imposes its additional supported-geometry constraints.

An edge exists if either block's write region intersects the other's read region. Raster-ID greedy coloring uses no solver-order assumption and validates every same-color pair afterward. Returned groups are ordered block IDs; `plan_for` supplies corresponding rectangles and dependency sets. The common two-3×3 case produces nine singleton colors on 16×16; this is a result, not a hard-coded formula. On 4×4, m=2 or 3 yields four interacting blocks and four colors; m=4 yields one.

## Execution

`switch_coloring.py` adds `ColorLie` and `ColorStrang` without replacing existing classes. For a color, each block calls the existing `local_block_flow` with the same current global state. Block outputs are collected before a simultaneous disjoint commit. Physical scaling stays inside the installed wrapper-aware factory.

ColorLie visits colors once per macro-step with duration h. ColorStrang visits them forward and backward with duration h/2 per color visit. Central halves are deliberately unmerged: there are 2K sequential color stages per Strang macro-step (K for Lie), preserving local subsolve durations for comparison. Same-color blocks are evaluated sequentially on the GPU from a shared snapshot, not batched into one adaptive solve; this implementation makes no GPU speedup claim. Mathematically their independent block fields commute, while finite numerical solves retain the existing solver behavior.

## Validation and reruns

`tests/test_switch_coloring.py` checks all same-color structural pairs, rejects invalid groups, tests irregular/edge shapes, checks convolution dependencies against autograd support, numerically perturbs same-color blocks in both directions, verifies within-color permutation invariance and shared snapshots, and checks symmetric scheduling and installed physical scaling. The existing seven coupled-block tests remain applicable.

`python scripts/study_color_grouped.py` first compares raster and color Strang at m=1,2,3,4 on the first real trained layer, N=5/tol=1e-8, using one saved input image and an unsplit Dopri5 1e-6 solve on that same image. It structurally validates every pair and numerically probes a representative pair per color in both directions through the installed real RHS. Full numerical pair checks on small/irregular synthetic tensors run in the unit tests; singleton-only colors have no pairs to probe.

The runner then evaluates ColorStrang N=5,m=2/3/4 and ColorLie N=10,m=1/2, both isolated and propagated, on the original two batches of 128. References and noisy inputs are reused with hashes, physical weights are checked, and new results are isolated from the original studies. No full-network reference rerun is needed. One additional GPU worker runs with memory guards.

Results: `results/switch_color_grouped_r10k_2b/`. `validation.json` records M, K, ordered memberships, stages, reference-relative errors, and the numerical checks; `iso/` and `prop/` contain per-case artifacts and manifests. The paper-facing live summary is `papers/hardware-native-neural-ode/shared/evidence/pixel-switched/color_grouped_results.md` relative to the collaboration workspace.

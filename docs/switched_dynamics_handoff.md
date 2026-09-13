# Switched-dynamics study: handoff for a new chat

Later update: the completed endpoint/projection/composition diagnosis is recorded
under **Yoshida cause diagnosis (2026-09-10)** in the linked study document.
Read that section before relying on the earlier unresolved diagnosis below.
Evidence identifies severe endpoint-overshoot effects and a five-iteration
negative-stage instability in the trained block's frozen exact-flow control;
the composition and wrapped negative-flow checks pass exactly.

Snapshot: 2026-09-10, approximately 18:30 EDT. Read the current logs before
reporting progress; completion and process status below are not timeless facts.

## Start here

Workspace root: `/home/rongzeng/_workspce_old/repos/pcn/collaboration`.
Code repository: `ScAN-PCN`, working branch `colab_all`.
Do not alter `ScAN-PCN-mismatch_analysis` or unrelated training/TC/toggle work.
The code worktree contains many pre-existing uncommitted changes. This document
is not a statement that the worktree is ready to commit.

Primary study document, containing the complete Jacobi/Lie/Strang tolerance
tables and classification progress:
[pixel_switch_method_study.md](../../papers/hardware-native-neural-ode/shared/pixel_switch_method_study.md).
The paths below are relative to `ScAN-PCN` unless explicitly absolute.

Purpose: compare pixel-local approximations of the continuous PCN with its
full unsplit dynamics, and eventually relate block-output error to classifier
accuracy. Do not infer numerical convergence order from classification accuracy.
Do not retrain, redesign switching, or launch full-dataset runs without a
specific request. The latest authorized experiment is the Yoshida tolerance
diagnostic at 5/10 iterations and tolerances 1e-9, 1e-10, 1e-11.

## Mathematics and current implementation

Full dynamics, omitting physical conversion factors for notation:

$$
\dot y=F(y)=W_{\rm FF}f(W_{\rm FB}y).
$$

For spatial pixel p, F_p changes only that pixel (all its channels), evaluating
the surrounding convolution from the current state. Let Phi_p(h) denote its
local continuous flow. Let T be the block's actual wrapped integration horizon,
N the switching iteration count, P the number of spatial pixels, and h=T/N.
The model's nominal unitless t_end is 1.75; physical wrapping converts the horizon.

| Method | Class in `switch.py` | Update and time convention |
|---|---|---|
| Full reference | `ODEXInitFFFB` in `ode_pc.py` | All pixels evolve together |
| Jacobi/old values | `ODEXInitFFFBPixelSwitchEfficient` | Other pixels are frozen at the iteration's old state; batches local solves |
| Lie/updated values | `ODEXInitFFFBPixelSwitchExplicit` | Raster order, immediately commit each result; local duration h |
| Strang | `ODEXInitFFFBPixelSwitchStrang` | Forward raster then reverse raster; duration h/2 per visit |
| Yoshida-4 | `ODEXInitFFFBPixelSwitchYoshida4` | Compose three existing Strang steps, S2(a h), S2(b h), S2(a h) |

Lie, Strang, and Yoshida default to stretchT/direct local time (`scale_RHS=False`).
Do not divide h by P with an unscaled RHS. The older equivalent convention
multiplies F_p by P and divides local time by P. Recorded direct/scaled comparisons
agree exactly in the tested float32 outputs. Recorded trajectory times are
bookkeeping times and must not be confused with each local solve's interval.

Strang visits each pixel twice, including the final raster pixel consecutively;
the total local time per pixel per macro-step is h.

$$
a=\frac{1}{2-2^{1/3}},\qquad b=-\frac{2^{1/3}}{2-2^{1/3}},
\qquad 2a+b=1,\quad 2a^3+b^3=0.
$$

Yoshida reuses `_run_strang_macro_step`. A negative coefficient means a positive
local interval |b|h/2 and a negative final wrapped derivative. It does not negate
W_FB. Nominal fourth order requires smoothness assumptions; ReLU and projection
mean it must be assessed empirically here.

## Bugs already fixed, and regression requirements

1. Earlier Strang constructed the base class RHS directly, bypassing the
   wrapper-installed physical transformations (including k/R and 1/(RC)).
   Always construct through `self._make_ode_fn(x)` and pin the active pixel.
   Near-random results from the superseded Strang must not be used as evidence
   against the corrected algorithm.
2. Earlier Lie inferred its active pixel from solver time; evaluations at the
   interval endpoint could use the next pixel. The corrected implementation
   pins `_explicit_active_pixel` throughout the solve. Old Lie errors
   0.3314/0.08055/0.04336 at N=1/5/20 are superseded.
3. Strang pins `_strang_active_pixel` and removes it in `finally`. Yoshida uses
   the same path for positive and negative stages. Jacobi passes pixel indices
   explicitly and is not affected by that Lie endpoint-dispatch bug.

Relevant tests: [tests/test_switch_strang.py](../tests/test_switch_strang.py).
Previous session reported six focused tests passing. Recheck after any edits.
The recorded negative-stage control checks equality of signed RHS values; it
is not by itself a complete negative-flow integration/convergence test.

## Exact diagnostic model and protocol

Model identifier:

```text
TIMMQAT5b8aNT0p25mulTIMMPCNetNoBatchNorm_PCConvReLU6_0.0eps_ODEXInitFFFB_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_C100_3K1S96C_0.25Dropout_16Layers4l5l4_2Pool_srrlDistill_a0p3_t2p0_scanGFI_1REP
```

Checkpoint: `saved_ckpt/<identifier>/<identifier>_best_ckpt.pth`.
Builder source: archived `switched_dynamics_diagnostic_snapshot/debug_strang_real_block.py`.
It loads the checkpoint separately for each method, uses `net.PcConvs[0]`, and
feeds the first batch of `get_test_data(test_bs=1, img_type="scanGFI",
task="cifar100")` after `torch.manual_seed(4096)`. State spatial size is 16x16,
so P=256. Data samples are generated by that loader, not a saved input tensor.

Actual builder settings include `QATTester1State`, R=20k ohm, R_max=300k ohm,
C=49e-15 F, k=1000, v_dd=0.1, w_bits=5, **enob=8**, one_over_q=1,
noise_level=0, nonlinear_R=false, measured activation=false, i_leak=None.
It passes thermal_noise=true; inspect the wrapper to establish its actual effect
before describing the experiment as noise-free. Keep these settings for a
matched repeat. Do not replace them with later TC nonideality defaults.

The builder uses dense convolution modules and does not request expansion.
“Full reference” means spatially unsplit dynamics, not verified unrolled hardware
convolution. The fixed reference is evaluated at Dopri5 rtol=atol=1e-6.
The checkpoint name contains `0.0001Tol`; do not claim its actual training
tolerance was 1e-6 without checking the saved training configuration/log.

$$
E=\frac{\|y_{\rm method}-y_{\rm ref}\|_2}{\|y_{\rm ref}\|_2}.
$$

## Established results and where to find all records

| Result | File/directory |
|---|---|
| Complete prior tolerance sweep, reference controls | `results/switch_solver_tolerance_floor_diagnostic.json` and `.log` |
| Corrected direct-time Lie/Strang equivalence | `results/switch_direct_time_sanity.json` and `.log` |
| Initial real-block debugging/convergence | `results/switch_strang_debug_real_block.json`, `results/switch_strang_convergence_real_block.json` |
| Jacobi diagnostics | `results/switch_jacobi_efficient_convergence_real_block.log` (use Efficient, not a substitute class) |
| Corrected Lie sweep plus early Yoshida work | `results/switch_yoshida4_diagnostic.log` |
| Dedicated Yoshida sweep | `results/switch_yoshida4_only_diagnostic.log`; expected final `results/switch_yoshida4_only_diagnostic.json` |
| Completed tight Yoshida sweep | `results/switch_yoshida4_tight_tolerance.json` |
| Full reference and 20-iteration Jacobi accuracy | `results/switch_strang_niter20_dopri5_tol1e-6/` |
| Corrected Strang partial accuracy | `results/switch_strang_niter20_dopri5_tol1e-6_corrected/`, `results/switch_strang_niter8_dopri5_tol1e-7_direct/`, `results/switch_strang_niter5_dopri5_tol1e-8_direct/` |

Corrected direct-time errors at mini tolerance 1e-6:

| N | Jacobi | Lie | Strang |
|---:|---:|---:|---:|
| 1 | 0.77461 | 0.31926 | 0.17773 |
| 5 | 0.18082 | 0.06385 | 0.02274 |
| 20 | 0.00802 | 0.01864 | 0.00863 |

Full reference differences versus 1e-6: 0.002456 at 1e-7 and 0.002439 at 1e-8.
The study MD has every entry of the N=5/10/20/40, tolerance 1e-5 through 1e-8
Jacobi/Lie/Strang sweep. Its older sweep JSON contains pre-correction Lie data;
use the corrected Lie table and `switch_yoshida4_diagnostic.log` for that method.

Completed full-dataset accuracy: reference 65.15%, Jacobi N=20 64.64%.
Stopped Strang partial results: N=20/tol1e-6 64.40% at 7/79 batches;
N=8/tol1e-7 64.45% at 4/79; N=5/tol1e-8 62.05% at 7/79.
Early batches tend to have higher accuracy. Unequal-coverage partial values
are not final accuracy comparisons. No full-dataset Yoshida run was requested.

## Yoshida results at this snapshot

Entries below were read from the dedicated log, and are relative block errors.

| N | 1e-5 | 1e-6 | 1e-7 | 1e-8 |
|---:|---:|---:|---:|---:|
| 1 | — | 0.912426 | — | — |
| 2 | — | 0.764397 | — | — |
| 5 | 0.905634 | 0.895538 | 0.728825 | 0.525205 |
| 10 | 0.783710 | 0.661937 | 0.297993 | 0.074358 |
| 20 | 0.590722 | 0.473516 | 0.154174 | 0.035315 |
| 40 | 0.168793 | pending | pending | pending |

Yoshida solve count is 6PN. Raw log includes NFE, runtime, negative-stage maximum.
GPU contention affects runtime; do not interpret mixed runs as fair speed tests.

Completed tight sweep, verified from `results/switch_yoshida4_tight_tolerance.json`:

| N | Requested tolerance | E | NFE | Runtime seconds |
|---:|---:|---:|---:|---:|
| 5 | 1e-9 | 0.25528324 | 157257 | 94.70 |
| 5 | 1e-10 | 0.20120645 | 201364 | 108.45 |
| 5 | 1e-11 | 0.19247451 | 279358 | 132.16 |
| 10 | 1e-9 | 0.04049016 | 259508 | 171.76 |
| 10 | 1e-10 | 0.03295045 | 290000 | 181.07 |
| 10 | 1e-11 | 0.03134842 | 361155 | 212.92 |

All six outputs were finite. The initial detached tight-sweep launch
failed before producing output, so `switch_yoshida4_tight_tolerance.log` is empty.
The replacement ran in terminal session 90635 and successfully wrote its final JSON.
The original sweep was PID 174248. Neither a remembered PID nor an absent process
in a sandbox namespace proves current status: check GPU processes and log growth.

## Solver caveats and unresolved diagnosis

Actual solver is the repository's `TorchDiffEqPack`, not torchdiffeq Dopri5:
`odesolver/ode_solver.py`, `adaptive_grid_solver.py`, `base.py`.
It accepts the requested small positive tolerances without a lower clamp;
adaptive `EPS=0`. Error normalization uses atol + rtol*max(abs(y),abs(y_new)).
The tested model/states are float32. Near 0.1, float32 spacing is about 7.45e-9;
tolerances below that can change step selection but cannot guarantee matching
absolute solution accuracy. The controller also has a `step_dif_ratio` fallback
that can accept a step despite the requested error threshold.

Time handling contains explicit `.float()` conversions, including `integrate`
and `check_t`. Merely applying `.double()` to a model is not enough to certify a
fully float64 solver experiment. A valid float64 control has not been established.

**Do not carry forward the previous confident claim that projection is the
dominant cause of Yoshida failure as a proven conclusion.** Projection is present
and negative compositions with irreversible clipping are a legitimate concern,
but the recorded `_yoshida_negative_max_abs` includes internal RHS stages,
including rejected trial evaluations, as well as returned subflow states.
An internal maximum above v_dd does not prove the committed state clipped.
The strong tolerance sensitivity also demonstrates a numerical contribution.
Inspect accepted states, projection frequency, and endpoint/interpolation behavior
before deciding how much error comes from clipping, splitting, or the mini-solver.
The study MD's description of the original Strang failure solely as active-pixel
handling is incomplete: the wrapper bypass was also a critical fault.

## Reproduction, archived scripts, and next actions

Exact diagnostic-script snapshots are saved under
`docs/switched_dynamics_diagnostic_snapshot/` so a new chat need not depend on
ephemeral `/tmp` files. These are provenance copies, not a newly validated CLI.
They retain hardcoded ROOT and `/tmp` imports; inspect and adjust those imports
before using the copies as independent entry points. Files:

- `debug_strang_real_block.py`: shared model builder and older trace diagnostic.
- `diagnose_switch_yoshida4.py`: dedicated Yoshida sweep and counting helpers.
- `diagnose_yoshida_tight_tolerance.py`: requested 5/10, 1e-9/1e-10/1e-11 sweep.
- `check_switch_direct_time.py`: direct/scaled control.
- `diagnose_switch_solver_floor.py`: previous tolerance-sweep driver.

Model-wide entry point is `ode_inference.py`; historical launchers are
`launch_scripts/run_ode_switch_inf.sh` and
`launch_scripts/run_switch_update_order_ablation.sh`.
The latter hardcodes the **older 12-layer/72-channel Euler-trained model**;
do not run it unchanged to reproduce the current 16-layer/96-channel study.
Use actual logged commands for model-wide reruns.

Next: inspect completion of the original Yoshida sweep; the tight sweep is complete.
Recover remaining rows without duplicating completed experiments; update the study MD with complete
tables and clearly qualified conclusions. Review solver precision and accepted
state behavior if diagnosing the remaining error. Preserve full-reference
tolerance 1e-6 as the historical comparison, with tighter references as controls.
Do not silently change ENOB, physical wrapping, model/input, or switching primitive.

Suggested first message in a new chat:

> Read `ScAN-PCN/docs/switched_dynamics_handoff.md` and the linked study document.
> Continue the Yoshida block-error diagnostics from the existing outputs. Check
> the two sweeps' status first. Verify code/log evidence before interpreting the
> high error or launching anything; preserve unrelated dirty-worktree changes.

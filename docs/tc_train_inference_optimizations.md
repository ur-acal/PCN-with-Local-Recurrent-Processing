# TC training and inference optimizations

## TC solver inference-history controls

Both controls default to `true`, preserving existing behavior. They are wired
through the TC PCN and TC feedforward local/Slurm launchers.

| Environment / Python argument | Set `false` to | Safety behavior |
|---|---|---|
| `TC_SOLVER_DENSE_OUTPUT` / `tc_solver_dense_output` | Stop retaining every solver step's voltage states | Requested in-range evaluation points still work; retrospective dense evaluation/extrapolation requires history. |
| `TC_NOISE_REPLAY_CACHE` / `tc_noise_replay_cache` | Release noise tensors after each accepted interval during no-grad inference | Current-interval reuse remains intact. Grad-enabled training retains history for backward replay. Explicit solve replay/graph regeneration without history raises an error. |

These controls do not disable noise, projection, or change random-number draws.
For PCN Euler with exactly 100 steps, pass `--n_steps 100 --ts_scale 1`;
the inference Python CLI defaults `ts_scale` to 1. CNN uses
`--tc_method euler --tc_num_steps 100`.

## Historical solver, weight-cache and ReLU benchmarks

These measurements predate the dense nonlinear-R backend changes below.
In those runs, batch-128 TC training OOMed without activation checkpointing.
Checkpointing was therefore the fixed memory-saving baseline for the
batch-128 speed comparison below; it is not counted as a speed optimization.
The three speed optimizations are accepted-step reuse, the live-QAT weight
cache, and fused measured ReLU.

### Batch 64: all OFF versus speed optimizations plus checkpointing

Then-current CIFAR-10 state-1/state-2 models, all TC nonidealities, production
shared/histogram convolution, and matched randomness. The four optimizations
are activation checkpointing plus the three speed optimizations. Values are
five-repeat post-warm-up medians from the current `colab_all` branch.

| Model and ON checkpoint policy | Total time OFF → ON | Peak allocated OFF → ON | PyTorch CUDA launches OFF → ON |
|---|---:|---:|---:|
| State 1; full (original) | 1.371 s → 1.049 s (-23.5%) | 14.06 GiB → 2.88 GiB (-79.5%) | 199,631 → 78,715 (-60.6%) |
| State 2; full (original) | 1.649 s → 1.162 s (-29.5%) | 16.07 GiB → 5.43 GiB (-66.2%) | 248,465 → 106,050 (-57.3%) |
| State 1; 0.5 (11/22 layers) | 1.369 s → 0.770 s (-43.7%) | 14.06 GiB → 4.62 GiB (-67.1%) | 199,631 → 74,695 (-62.6%) |
| State 2; 0.5 (11/22 layers) | 1.644 s → 0.960 s (-41.6%) | 16.07 GiB → 6.77 GiB (-57.9%) | 248,465 → 103,194 (-58.5%) |

The launch column records PyTorch `cudaLaunchKernel` calls. The profiler used
for this matched rerun did not count Triton `cuLaunchKernelEx` separately.

### Batch 128: full checkpointing versus speed optimizations plus checkpointing

| Model and ON checkpoint policy | Total time OFF → ON | Time reduction | Peak allocated OFF → ON | CUDA launches OFF → ON |
|---|---:|---:|---:|---:|
| State 1; full (original) | 2.289 s → 1.162 s | 49.2% | 5.79 GiB → 5.78 GiB | Not recorded → 86,178 |
| State 2; full (original) | 2.723 s → 1.424 s | 47.7% | 10.90 GiB → 10.85 GiB | Not recorded → 112,106 |
| State 1; auto = 2/22 layers | 2.348 s → 0.871 s | 62.9% | 5.79 GiB → 16.15 GiB | 245,273 → 72,927 |
| State 2; auto = 5/22 layers | 2.729 s → 1.145 s | 58.0% | 10.87 GiB → 16.33 GiB | 291,287 → 102,000 |

The new batch-128 rows compare the slow, full-checkpoint baseline with the
three speed optimizations and the smallest checkpointed prefix selected by
the automatic memory guard. Partial checkpointing deliberately spends more
memory than full checkpointing while remaining below the guard's 80% limit.

**OOM boundary:** with activation checkpointing disabled, both batch-128
models OOM during forward, regardless of accepted-step reuse: state 1 exceeds
22.37 GiB and state 2 exceeds 19.50 GiB before failure. This is why
checkpointing remains enabled on both sides of the batch-128 speed table.

### Local versus Slurm slowdown

A matched profile measured 2.930 s/batch on the local RTX 4090 and 5.257
s/batch on the Slurm L40S. Actual GPU kernel execution was nearly identical
(1.056 versus 1.018 s), and both runs submitted 186,992 kernels. The gap was
host-side: PyTorch/autograd work and CUDA submission were slower on Slurm,
including about 3.5 us per kernel launch versus 2.0 us locally. CPU/NUMA
binding tests did not remove the gap, and the exact driver/platform component
was not isolated. Reducing PyTorch operations and CUDA submissions benefits
both systems and should help Slurm more because its per-operation host
overhead is higher.

## Optimization controls and applicability

| Optimization | Purpose | Control | Training | Inference | Main exclusions/no-op |
|---|---|---|---|---|---|
| Accepted-step reuse (training and inference) | Speed | `REUSE_ACCEPTED_STEP_TRAINING` / `reuse_accepted_step_training` | Opt-in for RK12, RK23, Dopri5, and ProjDopri5 | Automatic for adaptive Dopri5 and ProjDopri5; the training flag does not control it | Energy metering and `reload_state`; predefined grids have no replay. Training additionally excludes ODE23s, Sym12Async, `regenerate_graph`, and cached-RHS variants |
| RHS activation checkpoint | Memory | `CHECKPOINT_ODE_RHS_TRAINING` plus `CHECKPOINT_ODE_RHS_PORTION`; Python: `checkpoint_ode_rhs_training`, `checkpoint_ode_rhs_portion` | Yes; numeric portion or `auto` | No-op under `no_grad` | Adjoint, energy metering, training BatchNorm, cached-RHS variants, ODE23s, Sym12Async |
| Live-QAT weight cache | Speed | Default ON; auto fallback for unsupported cases | Yes | Yes when live QAT exists; baked QAT inference is a no-op | Non-QAT and stochastic parametrizations |
| Fused measured ReLU | Speed | `FUSE_MEASURED_ACTIVATION` / `fuse_measured_activation` | Yes | Yes | PyTorch fallback for CPU, non-FP32 CUDA, missing Triton, trainable/mismatched curves, cubic spline, or higher-order gradients |
| Fused dense nonlinear-R input correction | Speed and memory | Default ON for supported TC calls; no public launcher flag. Diagnostic OFF: `_tc_fused_correction=False` | Yes; first-order backward supported | Dense only | Combines resistance interpolation and input correction for Gaussian and empirical sampling. Single-curve and batched multi-curve layouts are specializations of this same fusion; ordinary/grouped convolution runs separately. PyTorch fallback for CPU, unsupported dtypes, missing Triton, or differentiable curve constants |
| Fused per-edge nonlinear-R convolution | Speed | Default ON; auto fallback for unsupported cases | No | Expanded only | Combines resistance interpolation, edge-current calculation, and output accumulation for CUDA float32 TC Gaussian or empirical per-coupler curves. Empirical TC reuses the toggle native-grid kernel. Exclusions include energy metering, unsupported projections/devices/dtypes/layouts, and enabled gradients |
| Nonzero-edge TC resistance-curve storage | Memory | Default ON; auto fallback for unsupported cases | No | Yes | Stores sampled curves only for nonzero expanded TC edges; empirical banks and toggle storage are unchanged |
| Expanded Gaussian curve CPU offload | Memory | `EXPANDED_GAUSSIAN_CURVE_CPU_OFFLOAD` / `expanded_gaussian_curve_cpu_offload`; default OFF | No | Yes | Static expanded Gaussian per-coupler curves only. Supports TC PCN state 1/2, TC CNN, Toggle pulse PCN, pulse CNN, and their expanded analog head; falls back to synchronous one-owner staging when asynchronous prefetch is unavailable or unsafe |

Production branch: `colab_all`

## Purpose

An adaptive Runge--Kutta solver first evaluates a candidate step to determine
whether its error is acceptable. The legacy training path evaluates that
candidate without autograd, then evaluates the accepted step a second time
with autograd to advance the solution.

Accepted-step reuse instead evaluates each candidate with autograd, detaches
only the values used by the error controller, discards a rejected candidate's
graph, and retains the accepted candidate and its graph as the actual state
advancement. It removes the accepted-step replay without changing the
step-size acceptance rule.

This is a solver optimization. It is not specific to TC nonidealities or to a
particular model wrapper.

## Interface

The optimization is enabled for training with:

```bash
REUSE_ACCEPTED_STEP_TRAINING=true ...
```

The corresponding Python option is
`reuse_accepted_step_training=True`. It defaults to false. The training entry
points put it into each ODE block's solver options before wrappers are
constructed, so dynamic wrapper snapshots retain it. It is not shipped through
the TC wrapper. No-gradient adaptive Dopri5 and ProjDopri5 inference
automatically reuse accepted candidates; the training option does not control
that inference behavior.

The launcher audit covered every shell/Slurm file under `launch_scripts`.
CIFAR and ImageNet PCN launchers converge on their respective `train_ode_*`
parser and pass the option through `make_ode_block` into
`ODEBlockPC.option_aca`. CIFAR feedforward launchers pass it through
`tc_feedforward_args.sh`, `tc_feedforward_cli.conversion_options`, and
`TCPhysicalBasicBlock` into its direct solver call. MNIST forwards it to both
its PCN and CNN construction paths. Local checkpoint recovery preserves the
saved value, accepts the same environment override, and gives an explicit CLI
value highest priority. Slurm submission paths preserve the parent environment
with `--export=ALL`. No wrapper receives a second public copy of the option.

`accepted_step_reuse_safe` is an internal per-block guard, not a second user
option. It controls only gradient-enabled training reuse; it does not disable
automatic no-gradient Dopri5/ProjDopri5 inference reuse.

## Applicability

- Adaptive solvers can use the optimization when their search candidate is
  also a valid state advancement. Training enables it for RK12, RK23, and
  Dopri5 (including ProjDopri5). No-gradient inference enables it automatically
  for Dopri5 and ProjDopri5, independently of TC nonidealities.
- ODE23s is not enabled. Its step implementation requests a differentiable
  Jacobian only when the state requires gradients. A grad-enabled search
  candidate therefore produces a different error estimate and step sequence
  from the legacy detached search candidate.
- Sym12Async is deliberately not enabled in this change.
- Fixed-grid and predefined-grid solves have no candidate search followed by
  accepted-step replay. The option therefore has no work to perform and may be
  ignored naturally; they do not need an explicit exclusion.
- Deterministic RHS implementations and stochastic RHS implementations are
  both valid. Fresh randomness inside an RHS is not a reason to disable the
  optimization.
- Stateful solver reload and accepted-step energy accounting currently retain
  the replay path because their surrounding lifecycle has not yet been adapted
  to candidate reuse.
- `regenerate_graph=True` retains replay during step search because that mode
  deliberately rebuilds the complete graph later on the accepted grid; keeping
  candidate graphs during search would add work and memory without benefiting
  the returned graph.
- `ODEXInitFFFBPixelSwitchParallel` with cached patches and all
  `ODEXInitFFFBPixelSwitchEfficient` variants retain graph-connected tensors
  inside an RHS closure. They explicitly retain replay because a rejected
  candidate graph cannot safely be reused or released through those caches.
  `PerturbODEXInitFFFB` replaces that cached RHS with a stateless perturbation
  RHS and therefore explicitly re-enables reuse.

## Stochastic RHS semantics

The accepted candidate used to determine the step size becomes the state
advancement. A rejected candidate is discarded and the retry reevaluates the
RHS. The behavior depends on the noise lifetime already defined by the RHS.

### Noise cached for an accepted interval

`TCNoiseLifecycle`, including one-state FB current noise, indexes noise by the
accepted interval. All RK stages and rejected retries before `accepted()` see
the same cached sample. Accepted-step reuse preserves that behavior: the
step-size decision and the accepted advancement use the same interval noise,
then `accepted()` advances the noise tape.

### Fresh noise on every RHS call

Some legacy blocks, including `ODENoisyOffset` and
`SelfCUAbsSumFFFBNoisy`, call `torch.randn_like` inside the RHS. Each RK stage
and each rejected retry therefore receives fresh noise. Under accepted-step
reuse, the stages of the accepted candidate are retained rather than replayed.
Thus the exact random values on which acceptance was decided are also the ones
that advance the state.

This remains consistent with those classes' original fresh-per-RHS-call noise
design. They are not excluded from the optimization. It is also more coherent
than the legacy replay behavior, where a candidate could be accepted using one
set of random draws and then advanced using an independent set that was never
checked by the error controller.

Removing the replay changes the number and ordering of RNG draws. Consequently,
fresh-per-RHS-noise runs are not expected to be seed-for-seed identical to the
legacy path. This is an intentional consequence of eliminating redundant RHS
evaluations, not a change from fresh noise to fixed noise.

## End-to-end batch-64 verification

RTX 4090. All runs use the production `shared`/histogram TC training path, all
TC nonidealities, batch 64, `tol=1e-6`, and no activation checkpointing unless
stated otherwise.
Each successful table row reports one forward/backward from separate
identical-seed replay and reuse processes. This controls the sampled hardware,
adaptive grid, outputs, gradients, and memory comparison. Alternating
post-warmup runs independently confirmed the speedup but are not used below.

The legacy scanGFI checkpoint named in `run_ode_wrapped_inference.sh` was the
common model for RK12, RK23, and Dopri5:

| Solver | Replay total | Reuse total | Change | Replay/reuse peak allocated |
|---|---:|---:|---:|---:|
| RK12 | OOM | OOM | n/a | 21.88 / 21.91 GiB before failure |
| RK23 | 1.8985 s | 1.4572 s | -23.2% | 6.327 / 6.324 GiB |
| Dopri5 | 1.2322 s | 1.0307 s | -16.4% | 3.543 / 3.540 GiB |

RK12 OOMed in the forward pass under both modes even when the benchmark's
diagnostic convolution checkpointing was enabled. RK23 and Dopri5 completed
without it and produced bitwise-identical logits and every parameter gradient.
ProjDopri5 uses the same Dopri5 step and was checked separately with an explicit
projection in the solver tests.

The pinned 22-layer RGB CIFAR-100 one-state and two-state all-on TC models both
completed without activation checkpointing:

| Model | Replay total | Reuse total | Change | Replay/reuse peak allocated |
|---|---:|---:|---:|---:|
| One-state | 1.8105 s | 1.5202 s | -16.0% | 12.888 / 12.879 GiB |
| Two-state | 1.9038 s | 1.5701 s | -17.5% | 12.306 / 12.289 GiB |

For both models, separate identical-seed uncheckpointed processes produced
bitwise-identical logits and every parameter gradient. Peak allocated memory
in those matched processes decreased by 9 MiB and 18 MiB respectively; there
is no measured memory increase. Diagnostic convolution checkpointing was also
compatible with the optimization, but it is not integrated into the production
training path.

The benchmark previously defaulted `mode=all` to the grouped implementation.
Those older timing and OOM records did not exercise the selected production
training approximation and are invalid evidence for this optimization. The
benchmark now defaults `mode=all` to `shared`, records the effective method,
rejects confounded multi-comparison runs, and writes unique tensor artifacts.

For a generic solve whose automatically selected first step is a tensor, the
legacy controller mutates that tensor before replay. The optimized path keeps
that first replay to preserve existing results exactly, then reuses accepted
candidates for the remaining scalar-valued step proposals. TC solves already
normalize their bounded step to a scalar, so this exception does not apply to
the TC training benchmarks above.

## ODE RHS activation checkpointing

Checkpointing is controlled with two arguments:

```bash
CHECKPOINT_ODE_RHS_TRAINING=true CHECKPOINT_ODE_RHS_PORTION=0.5 ...
```

The Python options are `checkpoint_ode_rhs_training=True` and
`checkpoint_ode_rhs_portion`. The portion defaults to `1.0`, accepts a number
in `[0,1]`, or the literal `auto`. A numeric value selects the first
`floor(portion * N + 0.5)` ODE layers. The decision remains layerwise: each
selected layer checkpoints every RHS evaluation, and each unselected layer
checkpoints none. The Boolean remains the master switch.

With `auto`, a pre-training guard profiles three real training batches per
candidate, including forward, loss, backward, and optimizer update. It first
verifies that all layers fit, binary-searches the smallest safe prefix, and
confirms the selected prefix with a second three-batch run. Safe means peak
reserved CUDA memory is at most 80% of usable capacity. Model, optimizer,
scheduler, RNG, data-loader RNG, hardware-noise state, and cached QAT state are
restored between candidates. The original first-epoch sample stream is
preserved: multi-worker loaders replay the three profiled batches and continue
their retained iterator, while zero-worker loaders restore their RNG and
recreate the iterator. The guard runs inside
the existing trainer process; no separate launcher or third public threshold
argument is used.

The per-layer Boolean is installed in `ODEBlockPC.option_aca` when the block
is constructed, inherited by `option_init`, `option_patch`, interval copies,
and wrapper snapshots, and consumed at the common solver RHS boundary.
Runtime portion selection updates those live dictionaries and wrapper
snapshots together. It is not a TC wrapper option and is not shipped through
the nonlinear-resistance package.

Gradient-enabled RHS calls use non-reentrant PyTorch activation checkpointing.
No-gradient evaluation calls the original RHS directly. Tensor and tuple
states, fixed and predefined grids, Euler, RK2, RK4, RK12, RK23, Dopri5, and
ProjDopri5 use the same implementation. ODE23s and Sym12Async are deferred by
scope rather than claimed incompatible.

Ordinary CPU/CUDA RNG state is preserved by checkpointing. For
`TCNoiseLifecycle`, each checkpoint captures its accepted-interval tape index;
backward recomputation temporarily restores that index and then restores the
live index. Thus one-state FB noise and the other interval-cached TC noise use
the identical stored sample during recomputation. Two-state solver-update
noise remains outside the checkpointed RHS.

The option fails explicitly rather than silently falling back for:

- adjoint solvers, which already implement their own recomputation;
- energy-metered RHS execution, whose observations would otherwise be counted
  again during backward;
- the cached parallel/efficient pixel-switch RHS variants marked unsafe for
  graph reuse;
- training-mode BatchNorm inside an RHS, whose running statistics would
  otherwise be updated again during recomputation;
- ODE23s and Sym12Async while their support remains deferred.

The direct feedforward TC solver receives the same option. Its expensive
convolution/current construction currently occurs before its constant-drift
RHS, so generic RHS checkpointing is correct there but is not expected to save
meaningful memory.

Known minor limitation: checkpoint activation is decided when `odesolve`
constructs a solver. Do not construct a solver under `torch.no_grad()` with
`return_solver=True` and later reuse that solver for gradient-enabled training;
that unusual sequence retains the no-gradient decision. Ordinary training and
evaluation construct and execute the solver in the same gradient context and
are unaffected.

### Partial-checkpoint correctness and automatic selection

For the current CIFAR-10 state-1 and state-2 all-nonideality models at batch
64, checkpoint portions `0.25`, `0.5`, and `0.75` produced bit-identical
logits and every parameter gradient relative to full checkpointing. These
were full production shared/histogram forward/backward runs with accepted-step
reuse, live-QAT caching, and fused measured ReLU enabled.

At batch 128, the automatic guard selected and confirmed:

| Model | Selected prefix | Effective portion | Smallest lower trial | Selected peak reserved | 80% limit |
|---|---:|---:|---:|---:|---:|
| State 1 | 2/22 layers | 0.090909 | 1/22 unsafe | 18.21 GiB | 18.36 GiB |
| State 2 | 5/22 layers | 0.227273 | 4/22 unsafe | 17.90 GiB | 18.36 GiB |

Each candidate and the final confirmation used three batches. The reported
search memory includes the real teacher, CE/SRRL loss, backward, and SGD path;
each 22-layer search evaluated six candidates including confirmation, for 18
profile batch executions. The profiled data are then reused by the real first
epoch rather than discarded. The compact timing table at the top uses the
established student-only matched
benchmark so it remains comparable with the original rows.

### CIFAR-10 batch-128 OOM result

RTX 4090; production shared/histogram training path; RGB CIFAR-10; pinned
22-layer, 64-channel 2REP checkpoints; all TC nonidealities; Dopri5,
`tol=1e-6`; batch 128. Times and peak allocated memory are medians of three
post-warm-up successful runs. OOM rows failed during forward on every attempt.
The benchmark resets the global and per-module TC random generators before
each configuration, so all four configurations use the same stochastic draws.

| Model | Accepted-step reuse | RHS checkpoint | Result | Total time | Peak allocated |
|---|---:|---:|---|---:|---:|
| One-state | off | off | OOM | n/a | >22.37 GiB before failure |
| One-state | on | off | OOM | n/a | >22.37 GiB before failure |
| One-state | off | on | pass | 2.289 s | 5.79 GiB |
| One-state | on | on | pass | 1.932 s | 5.78 GiB |
| Two-state | off | off | OOM | n/a | >19.54 GiB before failure |
| Two-state | on | off | OOM | n/a | >19.50 GiB before failure |
| Two-state | off | on | pass | 2.723 s | 10.90 GiB |
| Two-state | on | on | pass | 2.242 s | 10.83 GiB |

Separate identical-seed batch-1 processes compared the unoptimized and
combined configurations. Both one-state and two-state runs produced
bitwise-identical logits and all 46 parameter gradients. Both combined
batch-128 configurations also completed two consecutive batches through the
production `train_one_epoch` path: real data loading, EfficientNet teacher,
CE/SRRL losses, backward, and SGD. Parameters changed and remained finite.
The two-batch medians were 2.126 s and 6.52 GiB allocated for one-state and
2.378 s and 11.67 GiB for two-state. Standalone teacher validation and
checkpoint serialization were intentionally outside this bounded check.

### Batch-64 comparison with gradient accumulation

The matched batch-64 benchmark gives the following post-warm-up medians:

| Model | Configuration | Total time | Peak allocated |
|---|---|---:|---:|
| One-state | neither | 1.353 s | 14.06 GiB |
| One-state | accepted-step reuse only | 1.002 s | 14.05 GiB |
| One-state | RHS checkpoint only | 1.972 s | 2.89 GiB |
| One-state | both | 1.730 s | 2.88 GiB |
| Two-state | neither | 1.634 s | 16.07 GiB |
| Two-state | accepted-step reuse only | 1.288 s | 16.05 GiB |
| Two-state | RHS checkpoint only | 2.233 s | 5.45 GiB |
| Two-state | both | 1.861 s | 5.43 GiB |

At equal effective batch size, the batch-128 combined run is approximately 4%
faster than two batch-64 accepted-step-reuse steps for one-state and 13% faster
for two-state. This comparison excludes optimizer, teacher, data-loading, and
SRRL overhead equally from both sides.

### Legacy model cross-solver result

The supplied 16-layer CIFAR-100 scanGFI checkpoint was run at batch 64 with
all TC nonidealities through the production shared path. RHS checkpointing
completed forward and backward for Euler, RK2, RK4, RK12, RK23, and Dopri5.
The QAT wrapper's projection made the Dopri5 run an actual projected solve.
RK12 changed from an uncheckpointed forward OOM at 22.41 GiB allocated to a
successful checkpointed backward at 10.33 GiB. Separate solver tests cover
forced rejection, full trajectories, predefined grids, tuple states, and
projection output/gradient equivalence.

## Common QAT training host-overhead optimizations

These optimizations are independent of TC state count, dataset, and solver.
They are enabled on the normal production paths and require no TC-specific
training scheme.

### Per-solve live-QAT weight cache

Every RHS evaluation used to read `FFconv.weight` and `FBconv.weight` through
their live QAT parametrizations, repeating the same deterministic
quantization many times although an optimizer cannot update a weight in the
middle of one ODE solve. Each `SymQuantizeWeight`-family parametrization now
caches its quantized tensor for the wrapped block call. With RHS activation
checkpointing, the cache remains available through backward recomputation and
is released after its accumulated gradient. The next training iteration then
recomputes the scale and quantized weight from the updated parameter.

This is implemented at the common QAT wrapper boundary. It covers symmetric,
pulse, and LSQ QAT; one-state, two-state, With-X, and toggle blocks; TC and
non-TC training; and direct physical feedforward blocks. With RHS
checkpointing, multiple forwards before one combined backward share the cache
when the parameters are unchanged. An in-place parameter update while a
checkpointed solve is still pending raises an explicit error instead of
silently using a stale weight.

There is no model-name, dataset, adaptive-solver, or inference exclusion.
Baked `QATTester` inference has no live QAT parametrization, so it takes the
natural no-op path. Full-parameter evaluation that deliberately retains live
QAT parametrizations can use the same deterministic cache safely. Arbitrary
non-QAT or stochastic parametrizations are not cached.

### Fused measured piecewise-linear activation

The single control is:

```bash
FUSE_MEASURED_ACTIVATION=true ...
```

Its Python form is `fuse_measured_activation=True`, and it is forwarded by
the CIFAR PCN, toggle, feedforward, MNIST, local, and Slurm launch paths. When
piecewise-linear measured activation is active, CUDA float32, Triton is
available, and the fixed curve buffers match the input device and dtype, one
Triton kernel performs interpolation and scaling in forward and one performs
the input gradient in backward. Uniform and nonuniform grids, explicit and
sampled corners, per-model/per-layer/per-spin sharing, endpoint normalization,
and a fixed coordinate pullback are supported.

CPU, non-float32 CUDA, unavailable Triton, empty input, trainable or mismatched
curve buffers, and a trainable or nonscalar pullback use the original PyTorch
implementation. Cubic-spline activation is unchanged. The fused custom
backward is first-order only, so a workload requiring higher-order activation
gradients must disable this option; no repository training path currently
requests those gradients.

The first fused prototype differed because reassociation, approximate
division, and fused multiply-add changed float32 rounding. The final kernel
preserves the unfused operation order, uses correctly rounded division, and
disables floating-point contraction. Unit tests cover all supported sharing,
grid, and pullback combinations with bitwise-identical forward and backward
results.

### Batch-64 full-model results

RTX 4090; deterministic cuDNN and TF32 disabled. The three TC rows use all
nonidealities, the production shared/histogram path, Dopri5, accepted-step
reuse, and RHS checkpointing. Times are medians of two post-warm-up
forward/backward runs.

| Model | Neither | Weight cache | Fused ReLU | Both |
|---|---:|---:|---:|---:|
| Legacy CIFAR-100 TC | 1.006 s | 0.748 s (-25.6%) | 0.782 s (-22.3%) | 0.560 s (-44.4%) |
| Current CIFAR-100 TC state 1 | 1.587 s | 1.165 s (-26.6%) | 1.367 s (-13.8%) | 0.943 s (-40.6%) |
| Current CIFAR-100 TC state 2 | 1.626 s | 1.265 s (-22.2%) | 1.402 s (-13.8%) | 1.037 s (-36.2%) |
| Legacy toggle | 0.113 s | 0.094 s (-16.8%) | 0.107 s (-5.2%) | 0.087 s (-22.8%) |

All four models produced bitwise-identical logits and zero class changes.
Fused-activation-only gradients were bitwise identical for every parameter.
Weight caching changes the order in which repeated quantizer-gradient
contributions are accumulated: maximum absolute differences were
`4.47e-8`, `2.24e-8`, `1.12e-8`, and `2.33e-10` for legacy TC, current state
1, current state 2, and toggle respectively; relative L2 differences were
between `4.71e-8` and `1.43e-7`. Three matched toggle optimizer steps retained
bitwise-identical logits, with no growth in this gradient-scale roundoff.
Four consecutive matched optimizer updates on each current TC state-1/state-2
model likewise retained bitwise-identical logits; their gradient relative L2
differences remained in the same `1.27e-7` to `1.45e-7` range.

On the current state-1 model, profiler counts explain the host-side saving:

| Configuration | CUDA kernel launches | CUDA memcpy calls | Profiler self CPU |
|---|---:|---:|---:|
| Neither | 166,495 | 10,886 | 1.759 s |
| Weight cache | 128,215 | 3,326 | 1.304 s |
| Fused ReLU | 111,415 | 10,886 | 1.426 s |
| Both | 73,135 | 3,326 | 0.961 s |

Together they remove 93,360 kernel submissions (56.1%) and 7,560 memory-copy
submissions per measured batch. The weight cache gives the larger wall-time
gain even though ReLU fusion removes more launches, because it also removes
quantization dispatch, copies, and custom-autograd work.

## TC inference optimizations

Production inference automatically uses the first two optimizations when their
guards permit:

1. Adaptive Dopri5/ProjDopri5 reuses the accepted search result instead of
   replaying the accepted step. This solver optimization also applies to
   non-TC inference; it is not controlled by the training flag.
2. Expanded Gaussian or empirical per-coupler TC evaluation fuses curve interpolation,
   signed edge-current calculation, and convolution accumulation in one
   Triton kernel on CUDA float32.

An additional opt-in memory mode,
`EXPANDED_GAUSSIAN_CURVE_CPU_OFFLOAD=true`, stores expanded Gaussian curves on
CPU between complete layer/owner calls. It keeps at most two owners' curves on
GPU and overlaps the next transfer with current-layer computation when pinned
memory and GPU headroom permit. Curve generation, random seeds, ODE stages,
and the fixed-per-trial curve lifetime are unchanged.

Energy-metered solves use the original paths. Edge fusion falls
back for training or any enabled gradient, unsupported
devices/dtypes/projections, or modules outside evaluation mode. Neither
optimization changes curve sampling, quantization, solver tolerances, rail
projection, or the accepted-step/noise lifecycle. Atomic accumulation can
change floating-point summation order, so GPU logits are not expected to be
bitwise identical.

For diagnostics only, `profile_tc_eval_cost.py --optimization` selects
`reference`, `reuse`, `fused`, or `both`; `--timing-only` disables nested
profiling events. Raw results and logits are stored under
`results/tc_inference_optimizations/`.

### What the fused per-edge nonlinear-R convolution does

The fused operation is the expanded physical-edge calculation:

1. read the edge's pre-sampled nonlinear-resistance curve;
2. interpolate its effective resistance at the source voltage;
3. calculate the signed current contribution; and
4. atomically accumulate that contribution into the destination output.

Curve **sampling is not performed by this kernel**. Curves are sampled before
the solve and retain the existing fixed-per-realization lifetime.

TC and toggle use separate lookup backends because they model different
hardware encodings:

| Property | TC Gaussian | Toggle empirical |
|---|---|---|
| Weight magnitude | One of 15 resistance/conductance codes | Pulse duration/count at one nominal resistance |
| Curve distribution | Code-specific mean plus shared covariance | Finite empirical common-resistance curve bank |
| Per-coupler draw | Independent Gaussian draw conditioned on code | Empirical curve sampled with replacement |
| Fused runtime layout | Common voltage grid plus a pre-sampled curve row per active edge | Per-curve grids/slopes/lengths plus an assignment index and instantaneous pulse value |

The final current operation is structurally the same. TC empirical-bank
inference now reuses the existing empirical kernel, while its caller retains
TC's resistance-code-conditioned assignment. Gaussian storage still needs its
different lookup backend. Toggle's common-resistance data bank cannot represent
all TC codes; sharing the kernel does not mean sharing the data bank.
Toggle's existing empirical-current fusion is recorded separately in
`docs/toggle_inference_optimization_audit.md`.

### Nonzero-edge TC resistance-curve storage

TC expanded inference stores curves only for nonzero programmed weights. An
int64 physical-edge-to-curve index preserves CSR connectivity, zero-weight
physical-site counting, sampling order/seeds, and fixed-per-trial lifetimes.
Both fused and fallback lookup paths use this index; energy accounting is
unchanged. Empirical banks and toggle storage are unchanged. No launcher
argument or expanded-weight-cache rebuild is required; evaluation must be
restarted to use newly loaded code.

For `E` physical edge slots, `A` active edges, and `K` float32 curve points,
persistent curve storage changes from `4EK` bytes to `4AK + 8E` bytes. This is
not the total GPU footprint or a guarantee against OOM.

### Expanded Gaussian resistance-curve CPU offload

This mode addresses models whose complete set of compact active-edge curves
still exceeds GPU memory. During shape discovery, each complete owner is
expanded and sampled on GPU and then copied to pinned CPU memory. At most two
owners' curve sets are live on GPU during this initialization pipeline.

The first evaluation forward uses safe synchronous one-owner staging. Its peak
counter is reset at the start of that forward, so expansion or earlier process
activity cannot falsely disable prefetch. The runtime enables two-owner
prefetch only if the next largest owner still fits below 90% of total GPU memory. In
steady state, the current owner computes while the following owner is copied
to GPU; the completed owner is released, and the first owners are prefetched
cyclically near the end of a forward for the next batch. Pageable memory, CPU
execution, insufficient GPU headroom, a one-owner model, or a prefetch OOM
falls back to synchronous one-owner staging.

The option is model- and dataset-independent. It is applicable to TC PCN
state 1/2, TC feedforward CNN, Toggle pulse PCN, pulse feedforward CNN, and an
expanded analog classifier head. It intentionally excludes training,
non-expanded/dense evaluation, shared curves, empirical curve banks, and any
curve sampling other than static multivariate-Gaussian per-coupler sampling.
The OFF path does not install the stager and is unchanged.

RTX 4090 matched benchmark; CIFAR-100 state-1 22Layers7l6l6/64C, batch 4,
three batches, all TC nonidealities, identical seeds:

| Mode | Expansion peak | Forward peak | Batch time | Expansion time | Curve residence |
|---|---:|---:|---:|---:|---|
| OFF: all curves resident | 19.13 GiB | 20.52 GiB | 0.714 s median | 7.82 s | GPU, 14.60 GiB |
| Earlier synchronous one-owner offload | 5.13 GiB | 6.67 GiB | 1.704 s | 12.31 s | CPU between owners |
| Current asynchronous two-owner offload | 5.13 GiB | 7.44 GiB | 0.916 s steady-state median | 13.40 s | CPU; at most two GPU owners |

The first offloaded batch is the synchronous safety pass and took 1.587 s;
the next two batches took 0.923 s and 0.908 s. Compared with synchronous
one-owner staging, prefetch reduced steady-state time by 46.3% while using
0.77 GiB more peak GPU memory. All 12 predictions matched OFF. The maximum
absolute logit difference was `8.58e-6`; the CPU/GPU curve round trip is exact,
while repeated GPU sparse/fused reductions are not bitwise deterministic
across separate processes.

The previously failing CIFAR-100 state-2 28Layers9l8l8/64C model also completed
three batches of 12. Its 18.663 GiB of curves remained on CPU; the largest
owner occupied 0.783 GiB on GPU. Expansion took 15.53 s with a 6.45-GiB peak.
The safe first batch took 2.595 s; the two prefetched batches took 1.534 s and
1.548 s (1.541-s median), with a 9.23-GiB forward peak. The earlier synchronous
one-owner run took 3.782 s per batch with an 8.45-GiB peak. OFF cannot
initialize this model on the 23.61-GiB GPU, so no matched OFF timing exists.

### End-to-end inference benchmark

RTX 4090; finished CIFAR-100 state-1 22Layers6l7l6/64C `last_ckpt`,
`QATTester1State`, all TC nonidealities, independent Gaussian curves per
physical coupler, ENOB disabled, tolerance `1e-6`, and batch size 4. The first
of three batches is warm-up; initialization is excluded.

| Measurement | Reference | Both inference optimizations |
|---|---:|---:|
| Measured batch times | 11.830 s, 12.359 s | 0.784 s, 0.820 s |
| Mean measured time | 12.095 s | 0.802 s |
| Peak forward allocated memory | 22.012 GiB | 21.971 GiB |

This bounded check measured a **15.08x speedup**. All 12 predictions matched.
Maximum absolute logit difference was `1.5116e-4` and RMS difference was
`1.3525e-5`; the arrays satisfied `atol=1e-4, rtol=1e-4`. Two reference runs
also differed by up to `8.49e-5`, so this is numerical rather than bitwise
equivalence and is not a full-test accuracy claim.

Nested-event isolated profiles, mean of batches two and three:

| Path | Seconds/batch | RK step calls across three batches |
|---|---:|---:|
| Reference | 12.622 | 264, 272, 283 |
| Accepted-step reuse only | 7.397 | 151, 155, 161 |
| Fused Gaussian edge operation only | 1.340 | 264, 272, 283 |
| Both | 0.887 | 151, 155, 161 |

The original validation covered 46 targeted tests, including CUDA numerical
checks at batch 1/4/128, signed and zero weights, padding, projection,
one-/two-state solver noise, rejection, endpoint behavior, training fallback,
energy fallback, and legacy regressions. `tests/test_tc_compact_curves.py` also
runs batch-two CNN/one-state/two-state production expansion validators, checks
compact storage and actual fused calls, and compares outputs and the energy
fallback with the former full-table layout.

## Verification coverage

The automated and end-to-end checks cover:

- deterministic outputs and gradients against the replay path;
- a forced rejection, confirming that rejected graphs are released and the
  accepted candidate is not replayed;
- cached interval noise, confirming unchanged accepted-interval advancement;
- fresh-per-RHS noise, confirming fresh draws for stages and retries but no
  second evaluation of the accepted step;
- automatic non-TC Dopri5 and ProjDopri5 inference reuse, confirming bitwise
  deterministic outputs and the documented RNG change for fresh-noise RHSs;
- fixed-grid solves, confirming that the option is ignored;
- parser-to-block and parser-to-direct-CNN-solver wiring for CIFAR, ImageNet,
  MNIST, feedforward launchers, and local recovery;
- wrapper snapshot restoration and the internal cached-RHS safety guard;
- peak training memory and elapsed time on the production configuration;
- parser/environment-to-block RHS-checkpoint wiring for CIFAR, ImageNet,
  MNIST, feedforward CNN, local recovery, derived options, and wrapper
  snapshots;
- RHS-checkpoint output and gradient equivalence for every enabled solver,
  tensor and tuple states, projection, full trajectories, and no-gradient
  evaluation;
- exact TC noise-tape replay during backward recomputation;
- explicit failures for adjoint, energy-metered, cached-RHS, and deferred
  solver combinations;
- batch-128 one-state and two-state OOM recovery with optimization 2 alone and
  combined with accepted-step reuse;
- automatic live-QAT cache coverage for one-state, two-state, With-X, toggle,
  non-TC, and direct feedforward wrappers, including multiple forwards before
  one backward and recomputation on the next iteration;
- exact fused measured-activation forward and first-order backward results for
  every supported grid, curve-sharing, and pullback mode;
- TC Gaussian edge-fusion and compact-storage CUDA checks, including
  one-/two-state models, fallback conditions, and energy accounting;
- batch-64 old TC, current state-1/state-2 TC, and toggle output, gradient,
  timing, and kernel-launch comparisons.

## Multi-level empirical TC curves

`NONLINEAR_R_CURVE_SAMPLING=empirical_with_replacement` selects
`TC_EMPIRICAL_CURVE_BANK` exclusively. Gaussian mode remains the default and
uses only `TC_MEAN_TABLE` and `TC_COVARIANCE_TABLE`. The inactive bank arguments
are ignored, not combined. Local and Slurm PCN/CNN launchers forward the selector.

The new directory contains `pseudo_res_5uS.csv` through `pseudo_res_75uS.csv`
in 5-uS increments. Use `R_VAL=13.3e3`, `R_MAX=200e3`; CNN retains its existing
derived maximum of `15*R` (199.5 kOhm). Pooling defaults to the 75-uS bank at
13.3 kOhm. Each curve retains its own voltage grid. FT samples independently
per forward, layer and convolution: per selected group in shared/top-N mode,
or per nonzero code in full-level mode. The solve and backward replay reuse
that draw. Expanded inference fixes independent code-conditioned
assignments per edge for each trial. Corner-coupled FT uses the corresponding
corner's empirical curves.

The temporary `pseudo_res_15uS.csv` duplicates the supplied 10-uS file, as
requested for implementation testing. `TEST_ONLY_DUPLICATE.json` and a runtime
warning mark this bank as **not suitable for scientific accuracy results**.
The supplied 10-uS file itself measures approximately 15 uS at the nominal
corner; a genuine physical 10-uS level still needs clarification/replacement.

| Optimization | Empirical TC status |
|---|---|
| Accepted-step reuse, live-QAT weight cache, fused measured ReLU | Applicable; existing controls unchanged |
| RHS checkpointing, including automatic/partial selection | Applicable; batch-128 empirical grouped FT fits comfortably with all 22 layers checkpointed (see sweep below). Do not infer that checkpointing can be disabled |
| Gaussian expanded curve CPU offload | Not applicable; leave disabled |
| Fused per-edge nonlinear-R convolution | Default ON for supported empirical TC inference; reuses the native-grid empirical kernel, with `_tc_fused_edges=False` available for diagnostic comparisons |
| Fused dense nonlinear-R input correction | Same optimization as in the main controls table; supports both Gaussian and empirical sampling with first-order backward. CPU/unsupported types use PyTorch. Diagnostic OFF: `_tc_fused_correction=False` |
| Batched multi-curve layout of the same correction fusion | Default ON for supported grouped CUDA FP32 calls. Generates the selected groups' corrections directly in grouped-input layout (15 groups for full-level processing); combines their input-gradient contributions in one backward kernel. Diagnostic OFF: `_tc_batched_correction=False` retains separate fused calls and concatenation; one group always uses the scalar layout |
| Grouped per-code dense convolution | `TC_CONV_METHOD=grouped` combines all 15 empirical code-specific convolutions. With `shared`, `TC_CURVE_TOP_N=0` uses one shared curve/ordinary convolution; N>0 combines the selected groups with the same grouped implementation. Wired for PCN, TC CNN and analog heads; Gaussian N=0 is unchanged |
| Empirical weight-level routing cache | Default ON. Reuses detached weight-level codes and boolean masks across RHS evaluations/replay; does not cache differentiable packed weights. Diagnostic OFF: `_tc_cache_level_routing=False`. Pinned Gaussian shared TC and toggle paths do not use it |
| Compact Gaussian per-active-edge curve storage | Replaced by shared empirical bank plus per-edge indices; no per-edge sampled curve tensor needed |

The measured bank tensors occupy **40.68 MiB** in FP32, including all corner
indices, shared between convolution modules on the same device/dtype. Expanded
assignment storage adds 8 bytes per stored edge, versus 208 bytes per active
edge for a 52-point FP32 Gaussian curve (excluding common connectivity and other
bookkeeping). Pooling has separate small banks. These are storage calculations,
not measured full-model peak GPU memory. Empirical FT may use more activation
memory and time than shared-Gaussian FT when using the full 15-level path;
shared N=0 uses one correction and one convolution. Do not disable checkpointing
on the basis of curve-bank storage alone.

`diagnostic_scripts/rehearse_tc_multilevel_empirical.py` reuses the isolated
rehearsal hooks: batch four, four real FT batches, checkpoint save, then the real
evaluation entry point stopped after its first CSR expansion. It can wait for
an existing pipeline and an idle GPU; it never stops or modifies that pipeline.

### Empirical fusion validation and benchmarks — 2026-10-23

Both real 22-layer CIFAR-100 rehearsals completed: four FT batches, batch four,
checkpoint save, and evaluation stopped after its first genuine expansion.
Subsequent fusion validation completed full expansion and compared final model
outputs on the same four test images, with all TC nonidealities enabled and
matched hardware/noise seeds. Neither assignments nor random-number consumption
are changed by fusion. The complete PCN state-1/state-2 checks below use the
four-batch rehearsal checkpoints; these are numerical/performance tests, not
scientific accuracy results. CUDA tests also cover TC CNN, expanded analog heads,
native-grid boundaries, and unchanged toggle fusion.

Median of three post-warm-up GPU forwards, without a profiler; peak allocated
GPU memory includes the resident expanded model. Initialization is excluded.
The diagnostic uses the identical existing CSR builder on CPU to avoid repeated
GPU scalar synchronization during unmeasured initialization.

| 22-layer model, batch 4 | Empirical inference fusion OFF → ON | Peak allocated OFF → ON | Maximum final-output difference | Prediction changes |
|---|---:|---:|---:|---:|
| State 1 | 5.986 → 1.102 s (5.43× faster) | 4.516 → 4.481 GiB | 7.82e-8 | 0/4 |
| State 2 | 8.102 → 1.591 s (5.09× faster) | 4.519 → 4.484 GiB | 1.99e-7 | 0/4 |

The following historical FT comparison predates empirical correction fusion
and grouped-convolution wiring. It uses the same original pretrained weights per state, real SRRL
teacher, RGB `rgb_mid` recipe, unnormalized student/teacher inputs, all TC
nonidealities, accepted-step reuse, live-QAT cache and fused measured ReLU.
Both sides use fixed RHS checkpointing on **2/22 layers**, not automatic
selection. Gaussian uses its pinned shared/histogram approximation and 10-kOhm
mapping; empirical uses the new 13.3/200-kOhm bank and 15 separate code-specific
convolutions. Thus this measures the two actual curve/computation configurations,
not an isolated comparison of distribution samplers. Data-loader work is excluded:
a real augmented batch is cached, with two warm-up optimizer steps and three
timed optimizer steps. Inference fusion is default-on but **does not run in FT**.

| Model | Batch | Shared Gaussian FT: time / peak allocated | Empirical per-level FT: time / peak allocated |
|---|---:|---:|---:|
| State 1 | 4 | 0.472 s / 0.976 GiB | 5.668 s / 11.339 GiB |
| State 2 | 4 | 0.592 s / 1.049 GiB | 6.540 s / 12.197 GiB |
| State 1 | 128 | 0.643 s / 10.657 GiB | OOM before a measured step |
| State 2 | 128 | 0.822 s / 12.620 GiB | OOM before a measured step |

Do not extrapolate a batch-128 empirical runtime from the OOM cases. More
checkpointing may make those runs fit; that policy search was not benchmarked.
The empirical FT optimizations below supersede the per-level reference timings;
they do not change the historical OOM measurements.

Reproduction: `diagnostic_scripts/benchmark_tc_empirical.py`. Numerical/timing
records are under `results/tc_empirical_fusion_benchmark_20261023/`; batch-128
records are under `results/tc_empirical_fusion_benchmark_20261023_bs128/`.

### Empirical FT correction fusion and grouped convolution

Correction fusion reuses the existing CUDA FP32 kernel and first-order backward
on each of the 15 selected native-grid curves. It is default-on when supported;
`_tc_fused_correction=False` selects the PyTorch reference for diagnostics.
`TC_CONV_METHOD=grouped` combines the 15 convolutions through the existing grouped
implementation. This is opt-in; Gaussian shared/histogram and toggle behavior are
unchanged. Curves are prepared once per solve and retained for backward replay.

Full 22-layer CIFAR-100 checks use batch 4, identical pretrained weights, augmented
inputs, all TC nonidealities, random-per-forward/per-spin ReLU, real SRRL, and
checkpointing on 2/22 layers. Optimizer updates are suppressed. All four variants
draw identical curves. Strict FP32 and deterministic convolution algorithms are
used **for validation only**; production precision settings are not changed.
Repeating the unoptimized reference gives identical outputs and gradients.

| State | Enabled optimizations | Maximum final-output difference | Maximum model weight-gradient difference | Model gradient relative L2 error | Median forward/backward time | Peak allocated memory |
|---|---|---:|---:|---:|---:|---:|
| 1 | Neither | 0 | 0 | 0 | 4.027 s | 10.448 GiB |
| 1 | Correction fusion | 0 | 2.38e-7 | 0.0000253% | 2.497 s | 5.545 GiB |
| 1 | Grouping | 1.14e-7 | 3.91e-4 | 0.0182% | 3.460 s | 10.472 GiB |
| 1 | Both | 1.14e-7 | 3.91e-4 | 0.0182% | 1.870 s | 5.569 GiB |
| 2 | Neither | 0 | 0 | 0 | 4.967 s | 12.170 GiB |
| 2 | Correction fusion | 0 | 1.86e-7 | 0.0000316% | 3.116 s | 6.274 GiB |
| 2 | Grouping | 1.21e-7 | 2.84e-4 | 0.0377% | 4.159 s | 12.197 GiB |
| 2 | Both | 1.21e-7 | 2.84e-4 | 0.0377% | 2.139 s | 6.300 GiB |

No predictions change. Times are medians of three post-warm-up passes without a
profiler, using the same no-update comparison loop. These are not timings of a
complete epoch or an optimizer update. Grouping alone does not reduce peak memory;
the memory reduction comes from fused correction saving fewer backward tensors.

Numerical bounds: outputs use `atol=1e-5, rtol=1e-4`. Full-model input, model and
SRRL auxiliary gradients each satisfy **maximum absolute error <=1e-3 AND relative
L2 error <=0.1%**. These bounds are explicitly looser than the original elementwise
gradient check (`atol=1e-5, rtol=1e-4`), which grouping does **not** pass for every
full-model gradient entry. Raw reports retain those strict-check failures; the
summary records the full-model bounds separately. Smaller complete PCN/CNN,
individual operators, native-grid boundary, and checkpoint replay tests pass the
stricter elementwise check. The Gaussian grouped path also matches the literal
pre-change implementation exactly in the regression test.

Reproduce the full check and benchmark with:

```bash
VERIFY_STRICT_FP32=1 python diagnostic_scripts/benchmark_tc_empirical.py \
  --training-optimizations --ft-only --output ./results/tc_empirical_ft_check
```

Raw results and the explicit-tolerance summary are in
`results/tc_empirical_training_optimizations_20261023_deterministic/`.
Normal TF32/non-deterministic settings were also checked separately. They do
**not** satisfy the same full-model bounds, even for a repeated OFF/OFF forward.
No gradient clipping was used in either precision comparison.

| Precision check | State 1 model-gradient relative L2 difference | State 2 model-gradient relative L2 difference |
|---|---:|---:|
| Repeated OFF/OFF, normal settings | 1.073% | 0.155% |
| Both optimizations versus OFF, normal settings | 5.983% | 1.199% |
| Both optimizations versus OFF, deterministic FP32 | 0.0182% | 0.0377% |

Therefore, the tight parity result is conditional on the stated deterministic
FP32 settings, not a guarantee for ordinary TF32 training. Grouping stays opt-in;
no production precision defaults are silently changed. The normal-settings raw
failure reports are retained in
`results/tc_empirical_training_optimizations_20261023_default_precision/`.

The supplied bank is still test-only (duplicated 15-uS file); these checks establish
numerical/performance behavior, not scientific accuracy.

### Batch-128 empirical FT: checkpoint portion sweep

Both correction fusion and grouped convolution are enabled. These runs use the
same 22-layer CIFAR-100 FT recipe, real SRRL teacher and optimizer updates, and
all TC nonidealities. Normal production precision is retained (TF32 allowed;
no deterministic override). Each configuration starts in a fresh process on an
otherwise idle RTX 4090 with 24 GiB VRAM. Only the checkpointed layer count changes.
Checkpointing selects the first N layers, as in the existing production policy.

Times are medians of **three complete training steps after two warm-ups**. Each
step includes student/teacher computation, backward and optimizer update; the
same real augmented batch of 128 is cached, so data-loader work is excluded.
Memory columns are maxima over **all five steps, including warm-up**, not just
the timed three. OOM cases failed on the first step and have no reported timing.

| State | Checkpointed layers | Portion | Median step time | Peak allocated | Peak reserved | Status |
|---|---:|---:|---:|---:|---:|---|
| 1 | 18/22 | 81.82% | — | — | — | OOM |
| 1 | 20/22 | 90.91% | — | — | — | OOM |
| 1 | 21/22 | 95.45% | 4.052 s | 22.093 GiB | 22.857 GiB | Fits, little headroom |
| 1 | 22/22 | 100% | 4.247 s | 4.520 GiB | 4.697 GiB | Fits |
| 2 | 18/22 | 81.82% | — | — | — | OOM |
| 2 | 20/22 | 90.91% | 4.883 s | 20.096 GiB | 20.793 GiB | Fits |
| 2 | 21/22 | 95.45% | 4.934 s | 15.160 GiB | 15.705 GiB | Fits |
| 2 | 22/22 | 100% | 4.978 s | 7.817 GiB | 8.145 GiB | Fits |

**Recommended starting point: `CHECKPOINT_ODE_RHS_PORTION=1.0` for both states.**
It leaves substantial memory headroom at batch 128, with little extra measured
time compared with the fitting partial portions. These are short memory/time
benchmarks, not evidence of full-training accuracy or a guarantee about every
future batch. The empirical bank is still the test-only bank described above.

Reproduction (diagnostic only; no production checkpoint is overwritten):

```bash
python diagnostic_scripts/benchmark_tc_empirical.py \
  --output ./results/tc_empirical_bs128_checkpoint_sweep \
  --batch-size 128 --checkpoint-layers 22 21 20 18
```

Per-configuration logs and JSON, including per-step loss, allocated/reserved
memory and the selected layer count, are saved in
`results/tc_empirical_bs128_checkpoint_sweep_20261023/`.
`checkpoint_summary.json` combines the eight configurations.

### Empirical weight-level routing cache

`EmpiricalLevels.weight_routing` caches codes and masks for one FF/FB curve draw,
instead of rebuilding them at each RHS evaluation. It holds only a weak reference
to the source weight, and checks tensor identity/version and scalar quantization
settings before reuse. New training draws get independent caches. Weight updates
or replacement invalidate routing; inference-mode and tensor-valued settings use
the uncached fallback. Checkpoint replay reuses the captured draw. Differentiable
weight masking/packing still runs normally, so no weight gradient is detached.
This is separate from the existing live-QAT weight cache.

Storage is **23 bytes per routed weight**: one int64 code and 15 boolean masks.
The 22-layer models route 632,160 FF/FB weights across 44 convolutions, totaling
**14,539,680 bytes = 13.87 MiB per model**, independent of batch size. The cache
does not hold input activations or differentiable packed weights.

Full-model parity checks at batch 128 with all 22 layers checkpointed used
identical weights, inputs, sampled hardware and RNG, with deterministic FP32.
Final outputs and auxiliary gradients are bit-identical. Maximum input-gradient
differences are 1.51e-9 / 6.52e-9 for state 1 / state 2; maximum model-parameter
gradient difference is **4.47e-8** for both, passing the stricter elementwise
`atol=1e-5, rtol=1e-4` check. No predictions change. Tests also verify replay,
cache reuse, invalidation, fallback, ordinary/grouped/transpose convolution and
the unchanged Gaussian path.

The following benchmark changes **only routing cache OFF/ON**. Batch 128,
22/22 checkpointing, correction fusion and grouped convolution are unchanged.
Normal production precision (TF32 allowed), real SRRL and optimizer updates are
used. Each variant runs in a fresh process; times are medians of five measured
steps after two warm-ups. Data loading is excluded, and memory peaks include all
seven steps. The benchmark GPU was otherwise idle.

| State | Median step time OFF → ON | Time reduction | Peak allocated OFF → ON | Peak reserved OFF → ON | Added routing storage |
|---|---:|---:|---:|---:|---:|
| 1 | 4.226 → 4.139 s | 2.1% | 4.627 → 4.675 GiB | 4.799 → 4.832 GiB | 13.87 MiB |
| 2 | 5.026 → 4.920 s | 2.1% | 7.817 → 7.829 GiB | 8.148 → 8.162 GiB | 13.87 MiB |

The observed speed improvement is modest. Total peak differences include
workspace/trajectory variation, not just the persistent routing tensors.
These remain short benchmarks with the test-only empirical bank, not full
training-accuracy evidence. No production checkpoint was overwritten.

```bash
VERIFY_STRICT_FP32=1 VERIFY_CAPTURE_ONLY=1 \
python diagnostic_scripts/benchmark_tc_empirical.py \
  --output ./results/tc_empirical_routing_validation \
  --batch-size 128 --routing-cache-ablation --training-optimizations

python diagnostic_scripts/benchmark_tc_empirical.py \
  --output ./results/tc_empirical_routing_benchmark \
  --batch-size 128 --routing-cache-ablation
```

Raw parity results: `results/tc_empirical_routing_cache_20261023_validation/`.
Raw timings, memory/cache sizes and per-step losses:
`results/tc_empirical_routing_cache_20261023_benchmark/`.

### Batched empirical correction: full-model validation and benchmark

The new path reuses the existing `_correct` arithmetic with a second kernel-grid
dimension for resistance level. Selected native-grid rows and lengths are
prepared once per draw without regridding or a CPU length readback. It writes
directly into `[batch, 15*channels, height, width]`, eliminating the 15 temporary
corrected-input tensors and their concatenation. One first-order backward kernel
accumulates the 15 contributions to the input gradient in fixed reverse level
order. Ordinary PyTorch grouped convolution still computes weight gradients.
When input gradients are unnecessary, no derivative tensor is allocated.

This is default-on only for eligible empirical grouped CUDA FP32 calls with
correction fusion enabled. OFF retains the previous separate fused corrections.
Unsupported device/dtype/input shape or trainable curve constants retain that
reference path. Gaussian shared/level-specific paths, toggle correction, loop
convolution and expanded inference keep their existing computations. The routing
cache remains ON on both sides of this ablation.

**End-to-end check:** complete 22-layer CIFAR-100 state-1/state-2 models, batch
128, checkpointing 22/22 layers, real SRRL loss and all configured TC nonidealities.
Weights, inputs, hardware draws and global/module-local RNG states match.
TF32 is disabled and deterministic cuDNN algorithms are selected for validation
only. Optimizer updates are suppressed for comparison. OFF/OFF is checked first.
The actual new path is instrumented in this diagnostic, not just requested by a
flag: all **44/44 FF/FB convolutions** execute it in both models, including
checkpoint replay (2,664 / 3,238 batched calls in the respective checked passes).

| Model | Maximum final-logit difference | Maximum input-gradient difference | Maximum model-parameter-gradient difference | Maximum SRRL auxiliary-gradient difference | Prediction changes |
|---|---:|---:|---:|---:|---:|
| State 1 | 0 | 2.56e-9 | 5.96e-8 | 0 | 0/128 |
| State 2 | 0 | 8.15e-9 | 7.45e-8 | 0 | 0/128 |

All four categories pass the stricter elementwise `atol=1e-5, rtol=1e-4`
comparison; none of the full-model relaxed bounds needs to be relied on here.
Maximum OFF/OFF model-gradient difference is 4.47e-8. Native-grid knots,
different per-curve lengths, padded grids, rail boundaries, noncontiguous inputs,
nonfinite forward semantics, ordinary/transpose grouped convolutions, analog
heads, and both PCN states with/without checkpoint replay are also tested.

**Production-precision benchmark:** batch 128, checkpointing 22/22, grouped
convolution, scalar correction fusion and routing cache ON in both variants;
only batched correction changes. Fresh sequential processes on an otherwise idle
RTX 4090, TF32 allowed, real SRRL forward/backward and optimizer updates. Times
are medians of five measured steps after two warm-ups; data loading is excluded.
Both memory peaks include all seven steps, including warm-up. No profiler runs
during timing.

| Model | Median step time OFF → ON | Time reduction | Peak allocated OFF → ON | Peak reserved OFF → ON |
|---|---:|---:|---:|---:|
| State 1 | 4.170 → 3.396 s | 18.6% | 4.605 → 4.694 GiB | 4.832 → 5.133 GiB |
| State 2 | 4.937 → 4.070 s | 17.6% | 7.825 → 7.825 GiB | 8.295 → 8.574 GiB |

This gives a clear measured speed improvement, but **does not reduce total peak
training memory under full RHS checkpointing**. The batched path still needs a
packed derivative tensor when input gradients are required; checkpointing already
discards internal activations, and allocation/workspace patterns also differ.
The prior routing-cache and checkpoint-portion tables predate this optimization.
The curve bank remains test-only, so these are numerical/performance results,
not full-training accuracy evidence. No production checkpoint is overwritten.

```bash
VERIFY_STRICT_FP32=1 VERIFY_CAPTURE_ONLY=1 \
python diagnostic_scripts/benchmark_tc_empirical.py \
  --output ./results/tc_batched_validation \
  --batch-size 128 --batched-correction-ablation --training-optimizations

python diagnostic_scripts/benchmark_tc_empirical.py \
  --output ./results/tc_batched_benchmark \
  --batch-size 128 --batched-correction-ablation
```

Raw parity results: `results/tc_batched_correction_20261023_validation/`.
Raw per-step timings, losses and memory: `results/tc_batched_correction_20261023_benchmark/`.

## Histogram top-level groups (2026-10-24)

`TC_CURVE_TOP_N` / `--tc_curve_top_n` defaults to **0** (range 0–L,
where L is the number of nonzero quantization levels; currently 15).
With `TC_CONV_METHOD=shared`, each FF/FB convolution independently counts its
own current nonzero QAT weight codes. The top N occupied levels get separate
curves; the remaining occupied levels share one curve. Its representative level
is drawn from the remaining levels' conditional histogram. Ties select the
lower code first. Empty groups are omitted. Curves and routing masks are held
through the solve and checkpoint recomputation, not resampled at every RHS.
For N>0, the first evaluation solve after training refreshes the groups using
post-update weights; subsequent evaluation batches retain that realization.
As before, explicitly reset realizations after replacing weights during evaluation.

Both Gaussian and empirical-with-replacement sampling are supported, including
the currently selected training corner. Corrections use the selected level's
own nominal resistance. Gaussian N=0 directly retains the original shared
sampler and ordinary convolution: no group masks, packed input, or additional
correction kernels. Pinned Gaussian commands need no changes. Empirical shared
N=0 now means one shared empirical curve, rather than the former implicit
15-level treatment. `TC_CONV_METHOD=loop` or `grouped` retains that full-level
treatment. Nonzero N requires shared/histogram for dense training.

Empirical N=0 skips ranking and weight-sized masks; its PyTorch fallback also
avoids a one-element concatenation. Eligible CUDA calls retain the one-group
native-grid fused correction followed by one ordinary convolution. Native-grid
row gathering still differs from Gaussian curve generation. The current
consolidation uses the same scalar correction kernel for both at N=0; fixed
row lengths are recorded when loading the bank, not copied to CPU per solve.
These fixes have numerical/regression checks, not new speed measurements;
the historical timings below predate them. Sampling follows the package/RNG
device, with routing masks on the weight device, including CPU-built wrappers
subsequently moved to CUDA. Diagnostics require a fresh output directory and
never silently reuse old JSON reports.

Zero codes do not enter the histogram. They remain assigned to the shared
remainder for QAT STE gradients, preserving the original Gaussian shared
behavior (or the last selected group if no remainder exists). Consequently
N=L explicitly represents every occupied level (N=L−1 also covers all levels
when the last remaining level forms the shared group),
but the old full-level implementation's zero-code gradient suppression is not
introduced. Exact recovery checks N (old recipes imply 0). Expanded per-edge
evaluation, ordinary pretraining and toggle paths do not change. Give new
experiments separate output roots, as with other recipe changes.

Legacy empirical recovery needs special handling: before this option, its
method named `shared` actually ran the full-level loop. The PCN recovery entry
point preserves that by resolving it to `loop` when the saved N is absent;
recovery guards reject changing between those semantics and the new shared
approximation. Old Gaussian recovery remains shared N=0.

Validation uses real 22-layer CIFAR-100 state-1/state-2 models, the SRRL teacher,
all TC nonidealities, identical curve/noise draws between comparison passes,
and full RHS checkpointing. Operator tests additionally compare checkpoint
ON/OFF, convolution/transposed convolution and the full-level endpoint. Timing
uses batch 128, two warm-up plus five measured optimizer steps on a cached real
augmented batch, with no profiler or concurrent GPU job. Weight coverage is
measured before optimizer updates and recorded per FF/FB convolution and in
total; percentages exclude zero weights. The empirical bank still contains the
documented test-only duplicated 15-uS file: these are performance/numerical
checks, not scientific accuracy evidence.

### Top-N numerical checks

Full-model final logits and gradients, batch 4, all 22 RHS layers checkpointed,
FP32 with TF32 disabled and deterministic cuDNN. Maxima below are across
N=0,1,2,3. N=0 Gaussian is compared with the literal saved pre-change methods;
other cases compare the optimized correction with native PyTorch correction
and the same grouped-convolution reduction order. All final logits matched
exactly, all predictions matched, and every tensor passed elementwise
`atol=1e-5, rtol=1e-4` (not just an aggregate gradient metric).

| Model | Sampling | Max final-logit difference | Max input-gradient difference | Max parameter-gradient difference | Max SRRL-gradient difference |
|---|---|---:|---:|---:|---:|
| State 1 | Gaussian | 0 | 5.22e-8 | 1.79e-7 | 0 |
| State 1 | Empirical | 0 | 5.96e-8 | 2.98e-7 | 0 |
| State 2 | Gaussian | 0 | 1.18e-7 | 1.49e-7 | 0 |
| State 2 | Empirical | 0 | 1.14e-7 | 4.77e-7 | 0 |

An earlier serial-convolution reference is retained, not discarded. State-2
Gaussian N=3 produced max final-logit difference 6.84e-7, max model-gradient
difference 5.05e-4, and SRRL-gradient relative L2 difference 0.188%, exceeding
the 0.1% aggregate-gradient threshold. Changing convolution reduction order
can amplify small FP32 differences through the solver/SRRL calculation. The
same-grouped reference above isolates correction fusion; small operator tests
also verify the serial sum and grouped sum. This is not a claim of strict
full-model gradient equality across different convolution reduction orders.

### Weight coverage before optimizer updates

Each model has 632,160 FF/FB weights. State 1 has 516,689 nonzero and 115,471
zero weights; state 2 has 538,442 nonzero and 93,718 zero weights. The coverage
is identical between sampling modes for these matched initial weights.

| N | State-1 individually represented | State-1 shared remainder | State-2 individually represented | State-2 shared remainder |
|---:|---:|---:|---:|---:|
| 0 | 0 (0%) | 516,689 | 0 (0%) | 538,442 |
| 1 | 188,856 (36.55%) | 327,833 | 171,775 (31.90%) | 366,667 |
| 2 | 317,694 (61.49%) | 198,995 | 307,969 (57.20%) | 230,473 |
| 3 | 403,538 (78.10%) | 113,151 | 403,271 (74.90%) | 135,171 |

Every raw report contains the histogram, selected codes, weight counts and
percentages for each of the 44 FF/FB convolutions, in addition to these totals.
Validation reports: `results/tc_top_n_20261024_validation_grouped/summary.json`.
Earlier serial-reference results: `results/tc_top_n_20261024_validation/`.

### Top-N training benchmark

RTX 4090, batch 128, full 22/22 RHS checkpointing, reuse/live-QAT/fused measured
activation enabled, production TF32 settings. Median seconds per optimizer
step includes forward, backward, SRRL and the optimizer, but excludes loading.
Memory is the maximum across both warm-up and all five measured steps, in GiB.
Each configuration runs in a fresh process sequentially, without a profiler.

Gaussian uses the unchanged pinned 10-kOhm/150-kOhm mapping and covariance;
empirical uses 13.3-kOhm/200-kOhm and the test bank. Comparisons within each
sampling mode use identical initial weights/settings. Cross-mode timings also
include those physical-data differences. These are actual stochastic optimizer
steps; TF32/atomic numerical variability can make trajectories diverge slightly
even between repeated baseline runs.

| Gaussian setting | State-1 seconds | State-1 allocated / reserved GiB | State-2 seconds | State-2 allocated / reserved GiB |
|---|---:|---:|---:|---:|
| Literal pre-change shared | 0.982 | 4.304 / 4.406 | 1.198 | 7.508 / 7.625 |
| N=0 | 0.996 | 4.304 / 4.406 | 1.181 | 7.508 / 7.625 |
| N=1 | 1.252 | 4.364 / 4.467 | 1.548 | 7.472 / 7.584 |
| N=2 | 1.344 | 4.292 / 4.396 | 1.649 | 7.475 / 7.717 |
| N=3 | 1.470 | 4.329 / 4.445 | 1.798 | 7.520 / 7.830 |

N=0 timing differs by about 1.4% in either direction; peak memory is unchanged
to the displayed precision. The original Gaussian sampler and convolution are
retained directly, with no group allocations or extra correction launches.

| Empirical setting | State-1 seconds | State-1 allocated / reserved GiB | State-2 seconds | State-2 allocated / reserved GiB |
|---|---:|---:|---:|---:|
| N=0 | 1.113 | 4.853 / 4.984 | 1.319 | 8.180 / 8.316 |
| N=1 | 1.374 | 5.257 / 5.420 | 1.643 | 8.923 / 9.119 |
| N=2 | 1.359 | 4.501 / 4.672 | 1.697 | 7.726 / 7.859 |
| N=3 | 1.503 | 4.500 / 4.625 | 1.849 | 7.718 / 7.967 |
| Existing full 15-level grouped | 3.378 | 4.552 / 5.133 | 4.108 | 7.825 / 8.574 |

N=3 reduces step time versus full-level empirical processing by 55.5% for
state 1 and 55.0% for state 2. With full checkpointing, total peak memory is not
monotonic in group count: adaptive trajectories, autograd state and convolution
workspaces also contribute. These measurements do not establish an accuracy
benefit or quantify approximation error against full-level hardware evaluation.

All 20 final benchmark configurations completed with five measured steps each.
Final records, including per-convolution coverage and allocated/reserved memory:
`results/tc_top_n_20261024_benchmark_final/summary.json`.

Reproduce from the repository in the scanbase environment:

```bash
python diagnostic_scripts/benchmark_tc_top_n.py \
  --output results/tc_top_n_validation_repeat --verify --batch-size 4 \
  --original results/tc_top_n_20261024_validation/prechange_ode_pc.py.snapshot

python diagnostic_scripts/benchmark_tc_top_n.py \
  --output results/tc_top_n_benchmark_repeat --batch-size 128 \
  --original results/tc_top_n_20261024_validation/prechange_ode_pc.py.snapshot
```

### Common dense backend consolidation (2026-10-24; performance matrix completed)

Gaussian and empirical samplers remain separate. Already sampled curves feed
one dense backend (`tc_curve_groups.group_convolution`), shared by PCN, CNN
and analog heads. Gaussian full-level rows are adapted to the same prepared
native-grid representation; no voltage-grid resampling is introduced.

- One group: one scalar correction and one ordinary convolution, for either sampler.
- Multiple groups: one batched correction and grouped convolution when eligible.
- OFF reference: the common PyTorch correction with serial convolution/summation.
- CPU/other-dtype fallbacks and explicit optimization-OFF controls remain.
  Serial fused convolution also remains: it is faster for some small late-layer
  groups, and passes the stricter gradient check where grouping sometimes does not.

The sampler-specific duplicate convolution branches were removed. The scalar
and packed autograd wrappers serve different group counts around the same
correction kernel; neither samples curves. N=0 no longer takes a sampler-specific
optimized path. Pinned commands and the default N=0 are unchanged. Expanded
per-edge inference is not routed through this dense backend.

Scalar and packed correction also share row preparation: selected curves are
gathered once with `index_select`, then viewed in scalar or packed layout.
This removes individual CUDA-index readbacks in the scalar empirical path.

Current checks use strict FP32 with matched inputs, weights, sampled-curve hashes
and RNG state. Final logits must satisfy `atol=1e-5, rtol=1e-4`; gradient maximum
absolute error and relative L2 error must both be at most `1e-3`. Prediction
changes are recorded separately. These limits were not relaxed after failures.

The complete 22-layer, batch-4, 22/22-checkpointed comparisons are finished.
The common OFF backend passes against the independent original OFF expression,
and fused serial ON passes against OFF in all 20 configurations. Below, “gradient
limit” means logits passed but at least one gradient measure exceeded the limit.

| State | Sampler; grouped ON versus OFF | N=0 | N=1 | N=2 | N=3 | N=15 | Maximum logit error | Maximum gradient relative L2 error |
|---|---|---|---|---|---|---|---:|---:|
| 1 | Gaussian | Pass | Pass | Pass | Pass | Gradient limit | 1.04e-7 | 0.204% |
| 1 | Empirical | Pass | Pass | Pass | Pass | Gradient limit | 1.08e-7 | 0.305% |
| 2 | Gaussian | Pass | Pass | Pass | Gradient limit | Pass | 6.84e-7 | 0.188% |
| 2 | Empirical | Pass | Pass | Pass | Pass | Pass | 1.79e-7 | 0.0951% |

All sampled-curve hashes matched and no prediction changed. The allowed relative
gradient error is 0.1%, not 1%; therefore the three marked cases are not certified.
The independent OFF expression uses the same curve data, not a separately drawn
hardware realization. Checkpoint-disabled/replay, CNN and analog-head cases also
have focused operator/small-model regression coverage; the real 22-layer numerical
matrix above uses full checkpointing.

For state-1 N=15, grouped logit errors were 1.04e-7 (Gaussian) and 1.08e-7
(empirical), with zero prediction changes. Maximum model-gradient errors were
1.11e-3 and 1.76e-3 respectively. A diagnostic serial group-output reduction did
not remove the Gaussian gradient discrepancy. Do not interpret passing operator
tests or timing results as full SRRL-gradient equivalence.

Batch-128 isolated convolution/transposed-convolution candidate checks all
passed against OFF (228 measurements, each including output and input/weight
gradients). Scalar one-group dispatch was selected instead of the redundant
one-group packed dispatch. Grouping is not universally faster: with 64 channels,
8×8 maps and 2–4 groups, separate fused convolutions beat grouped convolution;
grouping wins other measured shapes. These are operator results, not whole-model
optimizer-step speedups.

The final minimum/full-checkpoint benchmark matrix is under
`results/tc_unification_20261024_state2_counterfix/` (`performance.md` and
`combined_summary.json`). It combines the finished state-1 measurements with
the corrected state-2 rerun. The earlier state-2 failure was diagnostic-only:
the RHS call counter stripped the module's solver/state-dict interface; it now
uses a module pre-hook. No production solver change was needed.
The table reports N, individually covered weight count, minimum fitting layer
count, 22/22 timing, and peak allocated memory including warm-up. Each case uses
batch 128, two warm-ups and five measured optimizer steps, with real SRRL and
the real augmented batch. Minimum counts are established by ascending search;
unfinished searches are marked as such. Numerical failures remain visible in
the parity JSON files even when their timing is measured. The empirical bank
still contains the explicitly requested test-only 15-uS duplicate: no scientific
accuracy claim is made from these checks. No production checkpoint is written.

### Grouped-convolution numerical audit (2026-10-25)

The three full-model gradient-limit failures above reproduce; they are not a
correction-fusion or checkpoint-replay defect. Disabling correction fusion while
retaining grouped convolution reproduces the discrepancy. Repeating serial OFF
is exact, and fused correction with serial convolution passes the unchanged limit.
The grouped operator matches the independent serial expression in FP64 to
`2.78e-17` in outputs, `4.44e-16` in input gradients and `1.39e-17` in weight
gradients; input/weight finite-difference checks pass for convolution and
transposed convolution. Checkpoint replay is exact in these operator tests.

The full-model audit uses the real 22-layer CIFAR-100 checkpoints, the same augmented
batch of four, matched sampled curves, restored hardware/RNG states, SRRL and
22/22 RHS checkpointing. Values below
are the largest relative L2 discrepancy across input, model and SRRL gradients,
against independent serial OFF at the same precision. The limit remains **0.1%**.

| State | Sampler | N | Ordinary FP32 grouped | Diagnostic FP64 convolution + group sum |
|---|---|---:|---:|---:|
| 1 | Gaussian | 15 | 0.204368% | 0.0000713% |
| 1 | Empirical | 15 | 0.305414% | 0.0000708% |
| 2 | Gaussian | 3 | 0.187602% | 0.0000623% |

FP32 grouped convolution/reduction changes rounding, not the mathematical
operator. In the state-2 FP32 comparison, those small forward differences flip
three training-only SRRL ReLU gates, amplifying the gradient discrepancy. Keeping
convolution and group-sum accumulation in FP64 makes all three final-logit
differences exactly zero, changes no SRRL gates and passes the original bounds.
FP32 states, solver and correction arithmetic are retained in that diagnostic;
the final convolution result is cast back to FP32 once after group reduction.

**No production precision, grouping default or tolerance was changed.** These
controls support numerical sensitivity rather than an incorrect grouped backward;
they do not certify the ordinary FP32 grouped cases against serial OFF. Serial
convolution remains the verified alternative when that strict parity is required.
The diagnostic's missing module-name lookup and precision-option forwarding were
repaired; neither repair changes training/evaluation computation.
Raw comparisons: `results/tc_grouped_gradient_audit_20261025/`;
the complete high-precision control is under `fp64_convolution_and_sum/`.

### Removed experiment: dense correction + convolution fusion (2026-10-25)

A separate Triton kernel fused resistance interpolation, input correction and
convolution. Patch reuse and tile tuning improved the prototype, but it remained
slower than the existing correction-only fusion followed by cuDNN. At batch 128
with matched 22/22 RHS checkpointing, measured training steps were 9–85% slower.
With zero RHS checkpoints, N=0 used 35–37% less peak allocated memory but was
22–29% slower. Full-model equivalence also failed: with TF32 permitted, maximum
logit error was `4.26e-4`, worst gradient relative L2 was `5.98%`, and up to two
predictions changed per 128 images. No tolerance was relaxed.

The intermediate cuDNN-backed selective-recomputation alternative was a different
memory-saving strategy, not convolution fusion. Both experimental implementations,
their flag and all active wiring/tests/diagnostics have been removed. Existing
correction-only fusion, grouped convolution and expanded per-edge inference fusion
are retained. Historical measurements and recoverable source snapshots remain under
`results/tc_genuine_conv_20261025/` and `results/tc_dense_conv_diagnosis_20261025/`.

# Adaptive accepted-step reuse

## Optimization summary

| Optimization | Purpose | Control | Training | Inference | Main exclusions/no-op |
|---|---|---|---|---|---|
| Accepted-step reuse | Speed | `REUSE_ACCEPTED_STEP_TRAINING` / `reuse_accepted_step_training` | Opt-in for RK12, RK23, Dopri5, and ProjDopri5 | Automatic for adaptive Dopri5 and ProjDopri5; the training flag does not control it | Energy metering and `reload_state`; predefined grids have no replay. Training additionally excludes ODE23s, Sym12Async, `regenerate_graph`, and cached-RHS variants |
| RHS activation checkpoint | Memory | `CHECKPOINT_ODE_RHS_TRAINING` / `checkpoint_ode_rhs_training` | Yes | No-op under `no_grad` | Adjoint, energy metering, training BatchNorm, cached-RHS variants, ODE23s, Sym12Async |
| Live-QAT weight cache | Speed | Automatic; no public flag | Yes | Yes when live QAT exists; baked QAT inference is a no-op | Non-QAT and stochastic parametrizations |
| Fused measured ReLU | Speed | `FUSE_MEASURED_ACTIVATION` / `fuse_measured_activation` | Yes | Yes | PyTorch fallback for CPU, non-FP32 CUDA, missing Triton, trainable/mismatched curves, cubic spline, or higher-order gradients |

Branch: `perf/tc-training-step-reuse`

Worktree: `ScAN-PCN-tc-training-step-reuse`

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
`reuse_accepted_step_training=True`. It defaults to false while this branch is
experimental. The training entry points put it into each ODE block's solver
options before wrappers are constructed, so dynamic wrapper snapshots retain
it. It is not shipped through the TC wrapper. No-gradient adaptive Dopri5 and
ProjDopri5 inference automatically reuse accepted candidates; the training
option does not control that inference behavior.

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

The second training optimization is enabled with:

```bash
CHECKPOINT_ODE_RHS_TRAINING=true ...
```

The Python option is `checkpoint_ode_rhs_training=True`. It is installed in
`ODEBlockPC.option_aca` when the block is constructed, inherited by
`option_init`, `option_patch`, interval copies, and wrapper snapshots, and
consumed at the common solver RHS boundary. It is not a TC wrapper option and
is not shipped through the nonlinear-resistance package.

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
- batch-64 old TC, current state-1/state-2 TC, and toggle output, gradient,
  timing, and kernel-launch comparisons.

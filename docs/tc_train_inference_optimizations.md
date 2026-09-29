# TC training and inference optimizations

## Training results at a glance

Batch-128 TC training OOMs without activation checkpointing. Activation
checkpointing is therefore the fixed memory-saving baseline for the
batch-128 speed comparison below; it is not counted as a speed optimization.
The three speed optimizations are accepted-step reuse, the live-QAT weight
cache, and fused measured ReLU.

### Batch 64: all OFF versus speed optimizations plus checkpointing

Current CIFAR-10 state-1/state-2 models, all TC nonidealities, production
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
| Fused per-edge nonlinear-R convolution | Speed | Default ON; auto fallback for unsupported cases | No | Yes | Combines resistance interpolation, edge-current calculation, and output accumulation for CUDA float32 TC Gaussian per-coupler curves. Unsupported cases include empirical banks, energy metering, unsupported projections/devices/dtypes, and enabled gradients |
| Nonzero-edge TC resistance-curve storage | Memory | Default ON; auto fallback for unsupported cases | No | Yes | Stores sampled curves only for nonzero expanded TC edges; empirical banks and toggle storage are unchanged |

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

No launcher changes are required. Production inference automatically uses two
optimizations when their guards permit:

1. Adaptive Dopri5/ProjDopri5 reuses the accepted search result instead of
   replaying the accepted step. This solver optimization also applies to
   non-TC inference; it is not controlled by the training flag.
2. Expanded Gaussian per-coupler TC evaluation fuses curve interpolation,
   signed edge-current calculation, and convolution accumulation in one
   Triton kernel on CUDA float32.

Energy-metered solves use the original paths. Gaussian edge fusion also falls
back for training or any enabled gradient, empirical curve banks, unsupported
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

| Property | TC | Toggle |
|---|---|---|
| Weight magnitude | One of 15 resistance/conductance codes | Pulse duration/count at one nominal resistance |
| Curve distribution | Code-specific mean plus shared covariance | Finite empirical common-resistance curve bank |
| Per-coupler draw | Independent Gaussian draw conditioned on code | Empirical curve sampled with replacement |
| Fused runtime layout | Common voltage grid plus a pre-sampled curve row per active edge | Per-curve grids/slopes/lengths plus an assignment index and instantaneous pulse value |

The final current operation is structurally the same, so a future common
expanded-edge interface could share dispatch and accumulation. The lookup
backends must remain distinct. TC empirical-bank sampling could use a
generalized empirical kernel only if assignment remains conditioned on the TC
resistance code; toggle's common-resistance bank cannot represent all TC
codes. Toggle's existing empirical-current fusion is recorded separately in
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

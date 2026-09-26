# Adaptive accepted-step reuse

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
it. It is not shipped through the TC wrapper. Inference behavior is unchanged.

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
the existing no-gradient TC inference reuse.

## Applicability

- Adaptive solvers can use the optimization when their search candidate is
  also a valid state advancement. It is enabled for RK12, RK23, and Dopri5
  (including ProjDopri5).
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

## Current benchmark scope

The available production-path TC training checks show that accepted-step reuse
is faster with effectively unchanged peak allocated memory. It does not make
the RK12 batch-64 case fit. Activation checkpointing is not integrated into
the production training path by this branch. The known CIFAR-10 fine-tuning OOM
configuration has not yet been benchmarked here.

## Verification coverage

The automated and end-to-end checks cover:

- deterministic outputs and gradients against the replay path;
- a forced rejection, confirming that rejected graphs are released and the
  accepted candidate is not replayed;
- cached interval noise, confirming unchanged accepted-interval advancement;
- fresh-per-RHS noise, confirming fresh draws for stages and retries but no
  second evaluation of the accepted step;
- fixed-grid solves, confirming that the option is ignored;
- parser-to-block and parser-to-direct-CNN-solver wiring for CIFAR, ImageNet,
  MNIST, feedforward launchers, and local recovery;
- wrapper snapshot restoration and the internal cached-RHS safety guard;
- peak training memory and elapsed time on the production configuration.

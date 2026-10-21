# Claim 1 and Claim 2: Main Verifications

Implementation: `baseline/run_activation_diagnostics.py` (entry point and
checkpoint inventory), `baseline/activation_diagnostics.py` (shared evaluator).
Only the two main verifications are implemented; the minor Claim 2 diagnostic
and deferred BN-free comparisons are not launched or implemented here.

## Fixed Protocol

- Existing no-X/no-BN PCNetNoBatchNorm (`ODEXInitFFFB`, Dopri5, t_end=1.75,
  tolerance=1e-4) plus original, WD1e-3, and WD1e-3/final-drop0.25 WRNs.
  All four sizes, both CIFAR datasets. No training changes or BN folding.
- Full 10,000-image test dataset, deterministic preprocessing and ordering,
  batch size 128, no AMP, dropout off. Original-BN clean references are cached once.
- Multiplicative levels 0.30/0.35/0.40; tensor-max additive 0.08/0.09/0.10;
  10 trials per level, base seed 123. Restore parameters and buffers per trial.
- Claim 1: perturb only one selected unit; feed its cached clean input.
  For WRN recalibration, calibrate the full model with only that unit perturbed,
  then feed the isolated unit its cached clean test inputs.
- Claim 2: perturb all selected convolution weights and classifier weight/bias,
  then evaluate the entire network. Record unit, final-feature and logit errors,
  accuracy, clean-wrong/noisy-correct and clean-correct/noisy-wrong counts.
- WRN stem versus first PCN layer is included, including in equal-unit means.
  Subsequent pairs are full residual blocks versus PCN recurrent units.
  PCN outputs are pre-external-pooling; WRN outputs are after shortcut addition.
  WRN final features are BEFORE final BN; PCN final features are before head
  ReLU/GAP. Classification still executes the unmodified head.
- Frozen and recalibrated WRN evaluations share the SAME noisy parameters.
  Recalibration reuses the existing helper: 5,120 training images, subset seed
  20240618, batch size 128, dropout disabled, no BN affine mismatch.
- Both main metrics are accumulated over images, not batch means:
  `sqrt(sum ||delta||^2 / sum ||reference||^2)` and
  `mean_i(||delta_i|| / ||reference_i||)`. Exactly zero reference norms are flagged,
  not epsilon-regularized. Undefined-image counts and absolute errors are saved;
  per-image means explicitly give the number of valid images.

The shared FixedMismatchHelper applies mismatch once to an explicit allowlist:
Conv2d/ConvTranspose2d weights and, for Claim 2, Linear weight/bias. This includes
WRN learned shortcuts and PCN FF/FB kernels, but not BN, convolution biases or
unused zero b0. The PCN loader starts at zero mismatch; no later forward call
resamples weights. Noise scales reuse the existing max_abs implementation.
Seeds use the existing model-index/level/trial formula. Tensor traversal differs
between families and isolated/full scopes: this is not an assertion of identical
draws to historical CSV trials or cross-model elementwise pairing.

## Inspect and Run

From the mismatch_analysis worktree, in scanbase:

```bash
# Inventory only; no inference. Inspect exact checkpoint/config paths first.
python -m baseline.run_activation_diagnostics

# Example: one pair/family, both claims, full protocol.
python -m baseline.run_activation_diagnostics --execute \
  --datasets cifar100 --sizes 28_2 --families finaldrop \
  --output ../papers/hardware-native-neural-ode/shared/evidence/generic-mismatch/activation_claims_c100_28_2_finaldrop
```

Families: `pcn,wrn,wd1e3,finaldrop`; sizes: `16_2,16_4,28_2,28_4`.
Filters are comma-separated. `--claims 1` or `--claims 2` selects a main claim.
By default, `--execute` processes all 32 checkpoints sequentially. This is not
a new persistent scheduler or SLURM launcher. No production run was launched
during implementation.

`--pcn-root` defaults to this worktree's `saved_ckpt`; `--wrn-root` defaults to
its `checkpoint`. On this local machine the older PCN checkpoints are under
`../ScAN-PCN/saved_ckpt`, so pass `--pcn-root ../ScAN-PCN/saved_ckpt` for those.
Only checkpoint data is read from that directory, never its Python source.
No silent fallback to another worktree is used. `--data-root` defaults to sibling
`../data`. Missing checkpoint/config files fail preflight before any evaluation.

Small integration check (NOT scientific results):

```bash
python -m baseline.run_activation_diagnostics --execute --smoke --device cpu \
  --datasets cifar10 --sizes 16_2 --families wrn \
  --output ../papers/hardware-native-neural-ode/shared/evidence/generic-mismatch/activation_claims_smoke_wrn
```

Smoke mode uses two test images, two calibration images, one trial and the first
level of each mismatch family. Production does not use these limits.

## Outputs and Validation

The default output is the paper evidence directory, not repository logs:
`../papers/hardware-native-neural-ode/shared/evidence/generic-mismatch/activation_claims/`,
resolved relative to the code repository's parent, independent of the working
directory. All result files, clean caches, BN diagnostics, state files and
manifests are written beneath it. `--output` can select a separate run directory;
the examples above keep these under the same paper evidence folder.

Each invocation requires a NEW output directory to prevent mixing clean caches,
models, settings, smoke results or partial runs. There is no implicit resume.
The selected-top-three queue's explicit `--resume` requires the previous controller
and its recorded workers to have stopped. A queue lock prevents concurrent recovery;
completed runs and completed clean caches are preserved.

- Root `manifest.json`: model inventory, exact arguments, source hashes/revision.
- `<dataset>_<size>_<family>/model.json`: resolved config, checkpoint hash,
  unit names and exact eligible parameter names.
- `state.json`: running, complete or failed, with the failure if any.
- `clean/`: per-batch sample IDs, images, labels, logits and per-unit clean
  inputs/outputs; `clean_summary.json` records clean accuracy. This cache can
  occupy many GB per model. Provision disk space before a full sweep; do not
  accumulate all activations in GPU memory. Cache files are retained for audit.
- `claim*/<unit-or-network>/<condition>/level_*/trial_*/`: perturbation seed/tensors,
  then BN-policy subdirectories with metrics, zero-reference flags, summaries
  and Claim 2 prediction CSVs. No noisy activation tensors are retained.
- `per_trial.csv`, `aggregate.csv`, `layer_means.csv`, `summary.md`: model-level
  summaries; trial standard deviations use population convention. Undefined
  trial metrics are not silently dropped from aggregation.

Tests cover metric batch-partition invariance, zero handling, restoration,
parameter selection, pre-ReLU capture, clean-input replay, paired BN branches,
actual no-X ODE units, output generation and overwrite refusal. CPU integration
smokes additionally exercise the real trained WRN and PCN loading paths. These
are code checks, not full-test scientific results or verification of every
checkpoint's historical clean accuracy.

## Persistent local queue

`baseline/run_activation_queue.py` runs each WRN family (`wrn`, `wd1e3`,
`finaldrop`) at eight concurrent dataset/size pairs, then PCNs at four concurrent
sizes, CIFAR-10 followed by CIFAR-100. Within each group it finishes all workers
at one level before advancing: multiplicative 0.30/0.35/0.40, then max-additive
0.08/0.09/0.10. Both major claims use 10 trials and all 10,000 test images.
WRNs have paired frozen/recalibrated BN branches. Failed waves block later work.

The September 12 queue runs as user service
`pcn-wrn-activation-claims-20260912.service`; it survives the Codex session.
Results are in the paper evidence directory's `activation_claims_20260912/`:
`queue_state.json`, `controller.log`, `logs/`, and `runs/<model-level>/<model>/`.
`clean_cache/<model>/` is shared across levels, guarded by checkpoint, config,
dataset and source signatures. No training or checkpoint modification occurs.
The disposable progress Markdown is `/tmp/activation_claims_monitor/progress.md`.

```bash
systemctl --user status pcn-wrn-activation-claims-20260912.service
cat /tmp/activation_claims_monitor/progress.md
```

The controller supports `--dry-run --output <new-directory> --monitor <tmp-md>`
to print all 192 commands without executing. Tests simulate all 30 waves and
failure barriers; diagnostic tests also check clean-cache reuse and identity.

## Selected top-three WRN extension

The September 15 extension evaluates the fixed dataset/mismatch-specific WRN
selections recorded at the top of its live monitor. It queues only the 40 new
checkpoint/mismatch tracks (120 model-level runs across three levels) at
eight-way parallelism; the already completed `finaldrop`/W2 runs are reused.
Pool-after-add blocks are measured at `block.pre_pool`, before downsampling.

```bash
systemctl --user status pcn-wrn-activation-selected-top3-20260915.service
cat /tmp/activation_claims_selected_top3_monitor/progress.md
```

Results are written to
`papers/hardware-native-neural-ode/shared/evidence/generic-mismatch/activation_claims_selected_top3_20260915/`.
The queue does not invoke the summary generator and therefore does not modify
`claim1_claim2_comparison.md`. After validation, that report can be regenerated
explicitly with `generate_activation_claims_summary.py --selected-source <run-root>`.

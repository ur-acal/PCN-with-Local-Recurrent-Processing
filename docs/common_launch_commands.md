# Common Launch Commands and SLURM Environment Contract

PCN train-to-mismatch chaining now uses [launch_pcn_pipeline.sh](../launch_scripts/launch_pcn_pipeline.sh)
with `BACKEND=local|slurm`, `STAGE=train|test|train-test`, and accepted-step training reuse enabled by default.
See the [PCN pipeline guide](launching_guide.md#pcn-train-to-mismatch-pipeline-local-and-slurm)
for submission, optional AvgPool/five-step Euler, isolated rehearsal, output paths, and packing commands.

Generate all four robustness report/plot modes sequentially with `bash launch_scripts/generate_robustness_reports.sh` (set `PYTHON` if needed); the wrapper documents argument scope, with `--pcn-models P0,P3`, `--color-mode` and `--highlight-top3` applying only to PCN-versus-WRN plots and optional `--uncertainty` applying only to plot modes (off by default).
Only WRNs with all eight complete requested evaluation grids are reported, PCN plots default to P0/P3 while Markdown retains all PCNs, and outputs go to the paper's `shared/evidence/generic-mismatch/` directory with descriptive labels/PDF links in `wrn_variants_and_plots.md`; use `--output-dir` for alternative configurations.

Run remote commands from the mismatch_analysis worktree root. These commands
submit jobs; do not repeat them merely to inspect already submitted jobs.
Full recipe/output details: [launching guide](launching_guide.md).
Task records: [WRN controls](../../papers/hardware-native-neural-ode/shared/evidence/generic-mismatch/wrn_controls_pending.md)
and [PCN boundary BN](../../papers/hardware-native-neural-ode/shared/evidence/generic-mismatch/pcn_boundary_bn_pending.md).

## Required Environment Pattern

Follow the successful `slurm_run_rgb_ode_train.sh` / `run_rgb_ode_train.sh`
and `slurm_search_config.sh` / `run_search_config.sbatch` examples:

```text
Existing login shell: module swap slurm slurm/24.05.0.b1
  -> source submission script (inside a background subshell if desired)
     -> sbatch worker, exporting the requested settings
        -> worker's existing login Bash
           -> cd repository root
           -> source activate base
           -> conda activate scanbase
           -> python trainer/controller directly
              -> controller uses sys.executable for Python children, no shell=True
```

- Do not replace the sourced submission with `bash script.sh`: on this host,
  a fresh Bash previously selected SLURM 16.05 rather than the selected 24.05.
  `--chdir` then failed because the older executable accepted `--workdir` instead.
- Activate `scanbase` inside the compute worker, not as a required login-shell step.
- Do not insert another Bash between worker activation and Python. A new Bash
  may run site initialization via BASH_ENV; do not assume module state survives.
- Do not add mandatory external bookkeeping tools. Job 1891478 failed before
  all eight training launches because Python tried to execute unavailable `git`.
  Git provenance is now optional, with explicit missing metadata and source hashes.
  The precise environment transition that hid Git is not established.
- The WRN submission script still calls `python -m baseline.wrn_control_artifacts`
  on the login node, before submission to write the inventory and after submission
  to record IDs. This is Python-standard-library bookkeeping, not training or Git.
  It uses the login shell's Python (or PYTHON_BIN), not the worker's Conda Python.
- Keep scientific settings and resource headers separate from launcher repairs.
  WRN's existing limit remains 72:10:00; RGB PCN's is 90:10:00. Both request
  one GPU and 16 CPUs. A working header does not establish runtime capacity.
- Validate changed argument forwarding and fake train/test/pack behavior locally.
  Such tests are not claims of remote environment verification.

## WRN SLURM Train and Test

```bash
module swap slurm slurm/24.05.0.b1
mkdir -p logs/scheduler_slurm logs/slurm_jobs
(
  export STAGE=train-test PARALLELISM=4 EVAL_PARALLELISM=8
  export CONDITIONS=max_additive,multiplicative,rms_additive
  source ./launch_scripts/slurm_run_wrn_controls.sh
) > logs/scheduler_slurm/wrn_controls_train_test.log 2>&1 < /dev/null &
```

Default rows: 2,4,5,6,7,8,9,10. Eight dataset/size pairs per row.
Inside each worker: train four concurrently, then evaluate eight concurrently
for each sequential condition. BN rows produce frozen/recalibrated results.
The worker now calls `python -m baseline.run_wrn_controls` directly, without
`bash run_wrn_controls.sh`. The local wrapper remains available independently.
For a confirmed failed row only, add `export ROWS=2` inside the subshell.
Do not resubmit active rows. Partial train logs are not overwritten automatically.

Evaluation only, for verified completed training in the same output root:

```bash
(
  export STAGE=test EVAL_PARALLELISM=8
  export CONDITIONS=max_additive,multiplicative,rms_additive
  source ./launch_scripts/slurm_run_wrn_controls.sh
) > logs/scheduler_slurm/wrn_controls_test.log 2>&1 < /dev/null &
```

Pack completed results (choose a different archive name if it already exists):

```bash
python -m baseline.wrn_control_artifacts pack \
  --output-root "$PWD/logs/wrn_controls_slurm" \
  --archive "$PWD/logs/wrn_controls_slurm_results.tar.gz"
```

Add `--allow-incomplete` when collecting failures for diagnosis.

## New WRN Pooling Campaign

See the new WRN pooling campaign in [launching guide](launching_guide.md#new-pooling-controls-rows-11-18)
for the explicit `ROWS=11,12,13,14,15,16,17,18` command. Old default rows remain unchanged.

## PCN Boundary-BN Training: Eight Models

Historical launch settings; re-running submits another eight jobs.

```bash
module swap slurm slurm/24.05.0.b1
mkdir -p logs/scheduler_slurm logs/slurm_jobs
(
  export IS_SLURM=1 REPO_ROOT="$PWD"
  export PCN=PCNetBoundaryBN ODE_BLOCK=ODEXInitFFFB T_END=1.75
  export WARMUP_EPOCH=5 IS_TIMM=true TIMM_SCHED=cosine PCCONV=PCConv
  for DATASET_NAME in cifar10 cifar100; do
    export DATASET_NAME
    while IFS='|' read -r ARCHITECTURE INP_CHANNELS OUT_CHANNELS MAX_POOL; do
      export INP_CHANNELS OUT_CHANNELS MAX_POOL
      jid=$(sbatch --parsable --chdir="$REPO_ROOT" \
        --output="$REPO_ROOT/logs/slurm_jobs/slurm_%j.out" \
        --gres=gpu:1 --export=ALL \
        "$REPO_ROOT/launch_scripts/run_rgb_ode_train.sh") || exit 1
      echo "submitted job $jid: dataset=$DATASET_NAME, architecture=$ARCHITECTURE, pcn=$PCN, ode_block=$ODE_BLOCK, t_end=$T_END, warmup_epoch=$WARMUP_EPOCH"
    done <<'EOF'
WRN_16_2|3 16 32 32 64 64 128|16 32 32 64 64 128 128|0 0 0 1 0 1 0
WRN_16_4|3 16 64 64 128 128 256|16 64 64 128 128 256 256|0 0 0 1 0 1 0
WRN_28_2|3 16 32 32 32 32 64 64 64 64 128 128 128|16 32 32 32 32 64 64 64 64 128 128 128 128|0 0 0 0 0 1 0 0 0 1 0 0 0
WRN_28_4|3 16 64 64 64 64 128 128 128 128 256 256 256|16 64 64 64 64 128 128 128 128 256 256 256 256|0 0 0 0 0 1 0 0 0 1 0 0 0
EOF
  done
) > logs/scheduler_slurm/slurm_pcn_no_x_boundary_bn_all.log 2>&1 < /dev/null &
```

This trains only. For PCN evaluation, select exact completed checkpoint names
using the dynamic mismatch launcher documented in the launching guide; do not
assume its convenience model sets select Boundary-BN checkpoints.

## Local WRN Row 3

```bash
conda activate scanbase
mkdir -p logs/scheduler_local
nohup env ROWS=3 STAGE=train-test PHASED=1 PARALLELISM=4 EVAL_PARALLELISM=8 \
  CONDITIONS=max_additive,multiplicative,rms_additive \
  bash launch_scripts/run_wrn_controls.sh \
  > logs/scheduler_local/wrn_row3_train_test.log 2>&1 < /dev/null &
```

This is a separate local workload, not permission to overlap it with current GPU jobs.
Local outputs default to `logs/wrn_controls_local`.

## Current Local Activation Diagnostics: Monitor Only

```bash
systemctl --user status pcn-wrn-activation-claims-20260912.service
cat /tmp/activation_claims_monitor/progress.md
```

Do not restart or duplicate this queue to check progress.

To regenerate the single Claim 1/2 comparison Markdown, run
`python baseline/generate_activation_claims_summary.py` (optional `--source` and `--output`).
It validates all 192 full-test runs before writing the paper evidence file
`claim1_claim2_comparison.md`: main tables use recalibrated WRNs only; abnormal-case
tables include frozen and recalibrated WRNs, with no new inference.

## Completed Baseline Trial Recovery

`wrn-baseline-missing-trials-20260914.service` waited for the Claim 1/2 queue's
192 tasks to complete successfully, then evaluated the eight original WRNs
(no retraining). It uses unfused frozen/recalibrated BN, seed 123, 10 trials,
5120 calibration images (subset seed 20240618), batch size 128 and eight-way
parallelism, sequentially across max-additive, RMS-additive and multiplicative.
Levels are 0:.01:.10, .25:.25:1.25 and 0:.05:.40, respectively.

```bash
systemctl --user status wrn-baseline-missing-trials-20260914.service
cat logs/wrn_baseline_unfused_missing_trials/progress.md
```

Controller: `baseline/run_baseline_missing_mismatch_queue.py`. Per-model raw
results are `logs/wrn_baseline_unfused_missing_trials/<condition>/<dataset>/WRN_<size>/`
(`full_per_trial.csv`, `full_aggregate.csv`, `full_clean_sanity.csv`, `evaluation.log`).
Exact commands/checkpoints are in the root `state.json`. The completed 24/24
outputs are loaded by the robustness report generator as the original WRN's
paired unfused frozen/recalibrated source. Override the default source with
`--baseline-paired-root /path/to/results`.

## Arbitrary WRN Widths

Controls now accept any positive integer width, with a valid depth `6n+4`,
for example `SIZES=28_6,28_8`. The four original sizes remain the defaults;
stem width, recipes, row definitions and SLURM environment setup are unchanged.
Registration happens from the model name in each training/evaluation process,
including standalone evaluation. No environment-based registration is needed.

```bash
# SLURM: source from the working submission shell, as above; choose ROWS explicitly.
(
  export ROWS=17,18 SIZES=28_6,28_8 STAGE=train-test
  export PARALLELISM=4 EVAL_PARALLELISM=8
  source ./launch_scripts/slurm_run_wrn_controls.sh
) > logs/scheduler_slurm/wrn_wider_controls.log 2>&1 < /dev/null &

# Local controls: same SIZES selection, with the existing local wrapper.
SIZES=28_6,28_8 ROWS=3 STAGE=train-test PHASED=1 \
  bash launch_scripts/run_wrn_controls.sh

# All four report/plot modes; use a separate output for a new comparison cohort.
bash launch_scripts/generate_robustness_reports.sh \
  --sizes 28_6,28_8 --datasets cifar10,cifar100 \
  --controls-source /path/to/new/control/results \
  --output-dir /path/to/wider-reports
```

Completeness applies to all **selected** dataset/size pairs. By default,
`--wrn-ranking dataset-severe-mean` selects the top three separately for each
dataset and mismatch type by averaging all four architectures over the three
severe levels, then holds those variants fixed across architectures and levels.
Use `--wrn-ranking global-endpoint` only to reproduce the legacy global endpoint
ranking, or `--wrn-ranking per-point` for a separate top three at every exact
dataset/architecture/type/level comparison.
Default reports still require the original eight pairs. New PCN results can be
supplied through the existing `--wide-csv`/`--extra-data` inputs (and Boundary-BN
through `--pcn-bn-root`); missing results are not invented. These wider-model
commands are examples only: no wider training was launched as part of this change.

## Claim 2 Logit Diagnostics and Report

Replay only Claim 2 using the original clean caches, checkpoints, seeds and frozen/recalibrated BN policies; original evidence is not overwritten. The queue uses WRN 8-way and PCN 4-way concurrency, saves raw metrics under `logs/activation_logit_diagnostics`, and regenerates the paper comparison only after all 192 tasks validate against original per-image predictions and logit energies.

```bash
# From the mismatch_analysis repository, with scanbase active:
python -u -m baseline.run_logit_diagnostics

# Report-only regeneration after the queue completes (no inference):
python -m baseline.generate_activation_claims_summary \
  --logit-source logs/activation_logit_diagnostics
```

For logout-safe local execution, run the first command in a user systemd service with the absolute scanbase Python path and this repository as its working directory. Progress is in `logs/activation_logit_diagnostics/progress.md`; failures stop later waves and leave the existing report intact; repeating the command resumes saved validated trials.

The persistent generator adds centered logit relative L2, mean per-image margin degradation, and correct-to-wrong/wrong-to-correct percentages (all test images as denominator), each with population STD across ten trials. It then selects the top six positive original PCN-minus-WRN logit-error gaps with >1-point PCN accuracy advantage over the highest-accuracy recalibrated WRN, and selects three of those by accuracy advantage; new metrics do not influence selection. Output remains the paper evidence `claim1_claim2_comparison.md`; centered zero-reference cases are flagged without epsilon in per-trial JSON.

### Deterministic PCN C2 Replacement and Saved Logits

The original PCN GPU forwards were not deterministic; exact replay validation failed even with the same mismatch seeds. Keep original C1 and the cached clean reference; recompute **all PCN C2 quantities together** with deterministic algorithms, deterministic cuDNN, benchmark off, and `CUBLAS_WORKSPACE_CONFIG=:4096:8`. This is a replacement C2 measurement, not a claim of bitwise equivalence to the original run.

```bash
# PCN only: 48 model-level cases, 10 trials each, 4-way; no clean or C1 inference.
python -u -m baseline.run_logit_diagnostics --pcn-full

# Regenerate after completion, retaining already validated WRN metrics:
python -m baseline.generate_activation_claims_summary \
  --logit-source logs/activation_logit_diagnostics \
  --pcn-c2-source logs/activation_pcn_c2_deterministic
```

The logout-safe local service is `pcn-claim2-deterministic-20260914.service`. Progress: `logs/activation_pcn_c2_deterministic/progress.md`. New outputs do not overwrite the original cache, C1/C2 results, or earlier WRN replay outputs. Completed model-level tasks can be skipped on restart; a failed partial full-C2 task requires a fresh output directory rather than silently mixing trials.

Each model's existing clean logits are consolidated without inference into `clean_logits/<dataset>_<size>_pcn.pt`. Each C2 trial writes `runs/<task>/<model>/claim2/network/<type>/level_<level>/trial_<trial>/no_bn/noisy_logits.pt`, containing full-dataset `logits`, `labels`, and `sample_ids` tensors in matching order. `old_vs_new.json` records per-trial accuracy differences; the provenance JSON records deterministic settings. All layer metrics, summaries, and prediction CSVs are also saved.

Both persistent evaluators now save full noisy logits for every new C2 evaluation, including future WRN frozen/recalibrated runs; this does **not** retroactively add missing noisy logits to already completed WRN replay files. The shared Claim 1/2 evaluator also saves `clean_logits.pt` and `clean_reference.json`; Claim 1 remains an isolated-unit measurement without classifier logits.

## PCN Plus Top-Three WRNs and Baseline Plots

```bash
python -m baseline.generate_robustness_reports --mode pcn-plots --top3-plus-baseline

# Explicit PCN selection; Boundary-BN results are loaded from --pcn-bn-root
# (default: logs/pcn_boundary_bn_mismatch_slurm).
python -m baseline.generate_robustness_reports --mode pcn-plots \
  --top3-plus-baseline --pcn-models P0,P3,PB-R
```

This plot-only option writes PDFs to the paper evidence `plots/pcn_vs_top3_baseline_wrns/` for the default P0/P3 selection, leaving the existing all-WRN PDFs unchanged. An explicit selection is encoded in a separate folder name; for example, `P0,P3,PB-R` writes `plots/p0_p3_pb-r_vs_top3_baseline_wrns/`. Each default view includes the selected PCNs, baseline W0-R and the dataset-specific fixed top three **unfused recalibrated-BN** WRNs selected by severe-range averaging; selection is recalculated whenever the script loads new results. Existing complete-grid and optional AvgPool filters still apply. Additional result directories can be supplied with `--controls-source`; `--output-dir` controls the parent report directory, while plot paths and selected IDs are recorded in the shared registry and report manifest. The four-mode shell wrapper also accepts these options and forwards PCN selection only to `pcn-plots`.

## Adaptive Accepted-Step Reuse

For CIFAR/ImageNet PCN training, export `REUSE_ACCEPTED_STEP_TRAINING=true` before the existing local or SLURM launch. The Python option `--reuse_accepted_step_training true` has the same effect; an explicit CLI value overrides the environment. Training defaults to false. Existing SLURM submissions propagate the environment with `--export=ALL`; their shell/Conda setup is unchanged.

Direct solver callers set `options['reuse_accepted_step_training'] = True`. RK12, RK23, Dopri5 and projected Dopri5 retain accepted candidates and their graphs while keeping error control detached. Inference reuses candidates automatically; `options['reuse_accepted_step_inference'] = False` restores replay for comparison. Both solver entry points support these options. Fixed/predefined grids are unaffected. State reload, graph regeneration, energy observers, unsupported solvers, and RHS functions declaring `accepted_step_reuse_safe=False` retain replay. Capability checks do not use model names, datasets or architecture selections. Fresh-per-RHS noise remains fresh, but removing replay changes RNG consumption; seed-for-seed equality applies to deterministic or fixed-weight-mismatch RHSs.

The trained-model batch benchmark checks maximum absolute logits difference against `1e-5` for both training and inference, then reports median synchronized batch times after warmup. Each measurement restores checkpoint weights/BN buffers, initial SGD state and RNG. Training includes forward, cross-entropy, backward and the optimizer update; data loading, checkpoint loading and restoration are outside the timed region. Diagnostic INFO reductions are guarded in the model/convolution/inference paths, and logging is disabled equally in both benchmark modes.

```bash
# scanbase active; PCN_CHECKPOINT points to a trained no-X PCN best checkpoint.
python -m baseline.benchmark_pcn_step_reuse \
  --checkpoint "$PCN_CHECKPOINT" --dataset cifar100 --batch-size 128 \
  --warmup 2 --repeats 5 --tolerance 1e-5 \
  --output logs/accepted_step_reuse_verification/benchmark.json

python -m unittest tests.test_adaptive_accepted_step_reuse tests.test_adaptive_solver_limits
```

The benchmark fixes no-X ODEXInitFFFB, Dopri5, T=1.75 and tolerance 1e-4; choose a matching checkpoint. Optional `--mismatch-type mul --mismatch-level 0.4` or `--mismatch-type add --mismatch-level 0.1` checks fixed weight mismatch with BN parameters excluded. These are temporary in-memory optimizer steps; source checkpoints are never overwritten. Only small JSON benchmark summaries are saved under the code repository's `logs/accepted_step_reuse_verification/`, with no activation caches.

Verified on trained CIFAR-100 WRN-28-4-like PCNs, RTX 4090, batch 128, FP32, final dropout 0.25, deterministic cuDNN and TF32 disabled. Times below are medians of five measured batches per mode after two warmup rounds; both modes use the diagnostic logging guards. Maximum absolute logits differences were 0.0 for all four comparisons, passing the 1e-5 criterion. The 34 focused solver/model/mismatch/output-path regression tests passed.

| Model | Operation | Reuse OFF (s/batch) | Reuse ON (s/batch) | Speedup |
|---|---|---:|---:|---:|
| PCNetNoBatchNorm | Training including backward/SGD | 2.5681 | 2.0593 | 1.25x |
| PCNetNoBatchNorm | Inference | 1.3674 | 0.8285 | 1.65x |
| PCNetBoundaryBN | Training including backward/SGD | 2.1225 | 1.9227 | 1.10x |
| PCNetBoundaryBN | Frozen-BN inference | 1.2469 | 0.8083 | 1.54x |

Exact checkpoints, hashes, tolerance checks, RHS counts, all timing samples and peak allocated memory: [no-BN JSON](../logs/accepted_step_reuse_verification/pcn_nobn_cifar100_wrn28_4_clean.json) and [Boundary-BN JSON](../logs/accepted_step_reuse_verification/pcn_boundary_bn_cifar100_wrn28_4_clean.json). These measurements concern clean weights and a single fixed batch, not full-training duration or a full-test accuracy evaluation.

# Launching Guide

Work in the `mismatch_analysis` worktree. Commands below are manual launch instructions, not evidence that a job has run. Updated 2026-09-11.

The pinned WRN study is **64 model training runs in eight SLURM jobs plus deferred local row 3**, not the exhaustive 288-combination search. Each SLURM job owns one row and all sizes `16_2,16_4,28_2,28_4` across datasets `cifar10,cifar100` unless filtered.

| Row | Main downsampling | Shortcuts | BN | Conv bias | Location |
|---|---|---|---|---|---|
| 1 | Stride-2 conv | Learned | On | Off | Reuse eight existing models; no new launch |
| 2 | Stride-2 conv | Learned | Off | On | SLURM |
| 3 | Stride-2 conv | Learned | Off | Off | Local, later |
| 4 | Stride-2 conv | AvgPool/padding | On | Off | SLURM |
| 5 | Stride-2 conv | AvgPool/padding | Off | On | SLURM |
| 6 | Stride-2 conv | AvgPool/padding | Off | Off | SLURM |
| 7 | Stride-1 conv then AvgPool | AvgPool/padding | On | Off | SLURM |
| 8 | Stride-1 conv then AvgPool | AvgPool/padding | Off | On | SLURM |
| 9 | Stride-1 conv then AvgPool | AvgPool/padding | Off | Off | SLURM |
| 10 | Standard WRN, as row 1 | Learned | On | Off | SLURM: additional LR/init control |

Rows 2-9: original WRN training recipe, LR=0.1, WRN initialization, **WD=1e-3 and final dropout=0.25**, block dropout=0. Row 10 changes only LR to 0.01 and Conv/Linear initialization to PyTorch default; BN gamma/beta remain 1/0. All have classifier bias. Historical searched BN-free recipes are NOT used. Training is 300 epochs, seed 4096, with existing standard-WRN augmentation and checkpoint-selection settings. LR/init change jointly in row 10.

All new scripts derive the repository from their own location; explicitly setting `REPO_ROOT` is optional. Data defaults to the sibling `../data` locally and remotely. Set `DATA_DIR` only for another location. Activate `scanbase` locally; SLURM workers activate it themselves. Select a new `OUTPUT_ROOT` for a genuinely new experiment.

## 1. SLURM Training

Scripts: [scheduler](../launch_scripts/slurm_run_wrn_controls.sh), [one-GPU worker](../launch_scripts/run_wrn_controls.sbatch), [shared wrapper](../launch_scripts/run_wrn_controls.sh), [controller](../baseline/run_wrn_controls.py).

From the remote mismatch-analysis repository root:

```bash
mkdir -p logs/scheduler_slurm logs/slurm_jobs
DRY_RUN=1 bash launch_scripts/slurm_run_wrn_controls.sh
```

This prints **eight submissions without submitting**: rows 2,4,5,6,7,8,9,10, each containing eight model/dataset pairs. Then launch:

```bash
conda activate scanbase
STAGE=train-test PARALLELISM=4 EVAL_PARALLELISM=8 \
  CONDITIONS=max_additive,multiplicative,rms_additive \
  bash launch_scripts/slurm_run_wrn_controls.sh \
  > logs/scheduler_slurm/wrn_controls_train_test.log 2>&1
```

Each sbatch job runs its eight-model row on one GPU: `ising`, 16 CPUs, 72:10:00, `scanbase`, matching the baseline worker resources. SLURM decides placement and simultaneous GPU count. Submission returns after scheduling; jobs survive terminal logout.

- `train-test` is the SLURM default. Train up to four models concurrently until the row's training phase finishes. Then evaluate successfully trained models, up to eight concurrently, under max-additive; wait for that condition to finish before multiplicative, then RMS. No training/evaluation overlap, second submission or manual trigger. Set `STAGE=train` explicitly for training only.
- `PARALLELISM=4` controls training processes and `EVAL_PARALLELISM=8` controls evaluation processes per GPU, not per model. Each condition uses the exact standalone evaluator command. Frozen/recalibrated BN are paired within that condition. Failures remain recorded, while other models and conditions continue. GPU capacity and completion within the unchanged 72:10:00 allocation limit are not established by simulation.
- The scheduler records all expected tasks in `submissions/*.json` before submitting, then records each returned job ID. Activate Python on the submission host too; the workers still activate `scanbase` themselves.
- Filters: e.g. `ROWS=4 DATASETS=cifar100 SIZES=28_2`. Rows 1 and 3 are rejected by this scheduler.
- Default output: `logs/wrn_controls_slurm/row<row>/<dataset>/<size>/`.
- Training command/recipe, code revision, dirty status and SLURM ID: `manifest.json`; resolved trainer settings: `checkpoints/<dataset>/custom_noresize/<model>/baseline_config.json`.
- Training output: `train.log`; progress/completion/failure: `state.json`; verified completion/checkpoint hash: `train_complete.json`.
- Existing successful outputs under this new root are skipped after hash/config verification. Historical row-4/7 checkpoints are not silently substituted; all eight are intentionally trained anew.

### Executable pipeline simulation

Unlike `DRY_RUN=1` (print only), this executes the worker/controller with fake training and evaluation subprocesses, without `sbatch`, CUDA or real training:

```bash
SIMULATE=1 ROWS=4,6 DATASETS=cifar100 SIZES=16_2 \
  CONDITIONS=max_additive,multiplicative,rms_additive \
  OUTPUT_ROOT="$PWD/logs/wrn_control_pipeline_check" \
  bash launch_scripts/slurm_run_wrn_controls.sh
```

All outputs go under `OUTPUT_ROOT/SIMULATED`, with explicit simulated markers and placeholder accuracies. Fake training emits the expected checkpoint path and completion log; fake evaluations create the expected result files through the same completion checks. It validates orchestration, not numerical accuracy or cluster resources. Omit the row/dataset/size filters to exercise all 64 tasks. Tests also cover failed training, missing/wrong checkpoints, failed/incomplete evaluation, concurrent evaluation, and safe reruns. Never merge simulation results into research tables.

### Existing PCN training

[PCN scheduler](../launch_scripts/slurm_run_rgb_ode_train.sh) -> [RGB worker](../launch_scripts/run_rgb_ode_train.sh) -> [train_ode_cifar.py](../train_ode_cifar.py). Review the scheduler's dataset/architecture and ODE/t_end maps before sourcing it; its current defaults are not a universal eight-model plan.

For one explicit model, export `PCN`, `ODE_BLOCK`, `T_END`, `INP_CHANNELS`, `OUT_CHANNELS`, `MAX_POOL`, `DATASET_NAME`, `WARMUP_EPOCH=5`, `IS_TIMM=true`, `TIMM_SCHED=cosine`, `PCCONV=PCConv`, then:

```bash
export IS_SLURM=1 REPO_ROOT="$PWD"
sbatch --chdir="$REPO_ROOT" --export=ALL --gres=gpu:1 \
  --output="$REPO_ROOT/logs/slurm_jobs/slurm_%j.out" \
  "$REPO_ROOT/launch_scripts/run_rgb_ode_train.sh"
```

Boundary-BN PCN uses `PCN=PCNetBoundaryBN`, `ODE_BLOCK=ODEXInitFFFB`, `T_END=1.75`. These training jobs are separate from the new WRN schedule.

## 2. SLURM Mismatch Testing

After training, use the same WRN `OUTPUT_ROOT`, training seed, and filters:

```bash
STAGE=test bash launch_scripts/slurm_run_wrn_controls.sh \
  > logs/scheduler_slurm/wrn_controls_test.log 2>&1
```

SLURM defaults to max-absolute additive `0:0.01:0.10`, multiplicative `0:0.05:0.40`, then RMS additive `0.25,0.5,0.75,1,1.25`; 10 trials, mismatch seed 123, test batch 128. Conditions execute sequentially, with up to eight models evaluated concurrently per condition. Evaluation reuses the existing [BN](../baseline/run_wrn_bn_recalibration_experiment.py) and [BN-free](../baseline/run_wrn_nobn_mismatch_experiment.py) evaluators, not a new noise implementation.

BN controls produce **unfused frozen-BN and unfused recalibrated-BN** results using the same perturbed weights per pair. Calibration uses 5120 training samples, seed 20240618, batch 128; dropout and BN mismatch are off. BN-free models have one result per trial. Convolution biases are excluded; classifier weight/bias are included. No folding is requested here. Standalone WRN evaluation reads no old PCN comparison CSVs.

Outputs under each model directory: `evaluation/<condition>/run.log`, `full_per_trial.csv`, `full_aggregate.csv`, `full_summary.json`, and verified `complete.json`. Tests require a verified training completion, not merely a best checkpoint from an unfinished/failed run.

Each evaluation verifies the task-specific checkpoint hash before and after testing, model/dataset identity, seed and complete level/trial coverage. `complete.json` stores result hashes; condition-level `state.json` records failures. A model is complete only after all requested conditions succeed. Failed/partial logs are not silently overwritten.

### Pack results for analysis

From the repository root, after the integrated jobs finish:

```bash
python -m baseline.wrn_control_artifacts pack \
  --output-root "$PWD/logs/wrn_controls_slurm" \
  --archive "$PWD/logs/wrn_controls_slurm_results.tar.gz"
```

Download that archive. It contains submission inventories/job IDs, manifests, resolved training configs, training/evaluation logs, completion records, CSV/JSON results and `inventory.json` with each expected task's status. Checkpoints are excluded. External scheduler/SLURM stdout files are unnecessary for normal result analysis because per-task logs are included.

Packing refuses missing/failed tasks or invalid results by default; add `--allow-incomplete` to collect diagnostic evidence with explicit missing-task statuses. Existing archives are never overwritten: choose a new archive name. To test packaging the simulation above, use `--output-root "$PWD/logs/wrn_control_pipeline_check/SIMULATED"` and a separate archive name. The same packer works for local runs.

### Existing PCN mismatch testing

[Scheduler](../launch_scripts/slurm_run_rgb_ode_mismatch_eval.sh) -> [worker](../launch_scripts/run_rgb_ode_mismatch_eval.sh) -> [ode_inference.py](../ode_inference.py). Export these for one checkpoint:

```bash
export MODEL_NAME="<exact checkpoint directory name>"
export MODEL_INDEX="<0-7 comparison index>"
export ARCHITECTURE="WRN_28_2"  # replace with the actual size
export RESULT_TAG="<unique result tag>"
export ODE_BLOCK="ODEXInitFFFB" # must match training dynamics
export PC_CONV="PCConvNoisy" EVAL_MODE="mismatch"
export CONDITIONS="max_additive,multiplicative,rms_additive"
export NOISY_TRIALS=10 BASE_SEED=123
export OUTPUT_ROOT="$PWD/logs/pcn_boundary_bn_mismatch_slurm"
export BN_EVAL_MODE="paired" NOISE_TO_BN="false"
source launch_scripts/slurm_run_rgb_ode_mismatch_eval.sh
```

`paired` is for PCNs with BN: frozen evaluation, recalibration via the WRN routine, then reevaluation; both accuracies appear in `run.log`. Separate `frozen/result.pkl`, `recalibrated/result.pkl`, `trials.json`, `full_aggregate.csv`, `paired_complete.json` live under `<dataset>/<result_tag>/<condition>/`. For historical no-BN PCNs use `BN_EVAL_MODE=legacy` and a separate output root. Dataset/t_end are inferred from the model name; dynamics are explicitly supplied. Default `MODEL_DIR` is this worktree's `saved_ckpt`; a checkpoint symlink or explicit override is needed if files reside elsewhere.

## 3. Local Training

The [shared WRN wrapper](../launch_scripts/run_wrn_controls.sh) defaults to **row 3, all eight pairs, parallelism four**. It uses the active Python unless `PYTHON_BIN` is set. First inspect commands without training:

```bash
conda activate scanbase
DRY_RUN=1 bash launch_scripts/run_wrn_controls.sh
```

When ready to launch independently of the terminal:

```bash
mkdir -p logs/wrn_controls_local
nohup env ROWS=3 STAGE=train PARALLELISM=4 \
  bash launch_scripts/run_wrn_controls.sh \
  > logs/wrn_controls_local/controller.log 2>&1 < /dev/null &
```

Four subprocess slots share the visible GPU; memory capacity is not assumed validated on every machine. Use `CUDA_VISIBLE_DEVICES` and/or lower `PARALLELISM` as needed. Row 3 is not launched by the SLURM scheduler. Default local output is `logs/wrn_controls_local`; old row-3 results remain untouched. Default eight-run execution retrains the three prior successes. To run only the four untrained C10 models set `DATASETS=cifar10`; to retry only C100/16-4 set `DATASETS=cifar100 SIZES=16_4`. Use a new output root after a failed/partial attempt: scripts refuse to overwrite its logs.

For PCN local training, use the same explicit model/channel exports as section 1, activate `scanbase`, set `IS_SLURM=0`, and run `bash launch_scripts/run_rgb_ode_train.sh` instead of `sbatch`. No WRN recipe or PCN dynamics are automatically selected by this instruction.

## 4. Local Mismatch Testing

For the newly trained local row-3 models:

```bash
nohup env ROWS=3 STAGE=test PARALLELISM=4 \
  bash launch_scripts/run_wrn_controls.sh \
  > logs/wrn_controls_local/test_controller.log 2>&1 < /dev/null &
```

Keep the same `OUTPUT_ROOT`/filters/seed as training. Use `STAGE=train-test` during training to have each model's tests follow its successful training automatically. Conditions, exclusions, result paths and completion checks are identical to section 2. A failure is recorded per model; other models continue and the controller ultimately exits nonzero if any failed. No partial log is considered a completed run.

Local evaluation concurrency can reach `PARALLELISM * EVAL_PARALLELISM` processes. With four model slots, use `EVAL_PARALLELISM=1` to cap this at four; the default three allows up to twelve. Choose based on available memory, not the simulation results.

To use the SLURM row-style ordering locally, set `PHASED=1 PARALLELISM=4 EVAL_PARALLELISM=8 CONDITIONS=max_additive,multiplicative,rms_additive`. This replaces the preceding per-model concurrency behavior with a training barrier and sequential conditions, capped at eight evaluation processes total.

For PCN, retain section 2's explicit exports, set `IS_SLURM=0`, and run `bash launch_scripts/run_rgb_ode_mismatch_eval.sh`. This evaluates that selected model sequentially; it does not submit SLURM jobs. Use `nohup` and a unique outer log for logout-safe local execution.

Historical WRN launch paths are indexed in the [paper handoff](../logs/pcn_wrn_robustness_summary/paper_handoff_max_mul.md). The new WRN controller delegates training to [train_baseline_cifar.py](../baseline/train_baseline_cifar.py), with [configs](../baseline/baseline_cifar_configs.py) and shared `TrainerCiFarTimmStyle`; models use [WideResNetCIFAR](../baseline/cifar_resnet.py). Row definitions and override strings are in [wrn_control_specs.py](../baseline/wrn_control_specs.py). This guide does not authorize regenerating or overwriting historical reports.

# Common Launch Commands and SLURM Environment Contract

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

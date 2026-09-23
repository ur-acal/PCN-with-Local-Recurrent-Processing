# MNIST TC study

Run commands from the `ScAN-PCN` repository root. Data lives in
`../data/MNIST/`, next to the CIFAR directories, **not inside the code repository**.

## Models

| `FAMILY` | `VARIANT` | Construction | Parameters |
|---|---|---|---:|
| cnn | small | Stem 1→32; residual block 32→64→64 | 56,234 |
| cnn | deep | Stem 1→32; blocks 32→32→32, 32→64→64 | 74,666 |
| pcn | small | Two existing PCN layers: 1→32, 32→64 | 38,186 |
| pcn | deep | Three existing PCN layers: 1→32, 32→32, 32→64 | 56,650 |

CNN registrations are `mnist_cnn3_avgpool` and `mnist_cnn5_avgpool` in
`baseline/cifar_resnet.py`. Both reuse the existing CIFAR residual block and
physical conversion. PCN construction stays in `mnist_train.py`; `TC_STATE=1`
selects ODEXInitFFFB and `TC_STATE=2` selects S2NoisyIYAsXZAs0.
No BN or convolution biases; classifier bias remains enabled. All models use
one intermediate average pool plus GAP, internal ReLU6 and final dropout 0.25.
CNN adds the shortcut before pooling; PCN pools before channel expansion.

## Recipe and hardware

Native 28×28 grayscale → ToTensor → Normalize(0.1307, 0.3081), identical in
all stages. No augmentation, distillation, label smoothing or input quantization.

Default pretrain: 14 epochs, batch 64, Adadelta LR 1, rho 0.9, eps 1e-6,
weight decay 0; StepLR gamma 0.7 after every epoch. FT uses the same recipe
with LR 0.01 and a fresh optimizer. These are the optimizer/preprocessing
defaults of the [PyTorch MNIST example](https://github.com/pytorch/examples/blob/main/mnist/main.py),
not its architecture. Train health checks use training samples at epochs 5/10;
test accuracy is evaluated only after the last epoch. Save `last` each epoch;
no best-checkpoint selection. FT saves both baked and full-parameter weights.

FT and evaluation source `launch_scripts/tc_hardware_defaults.sh` and reuse
the existing TC classes, noise, physical scaling, measured activation/pooling,
and unrolling. Defaults: 5-bit weights, ENOB=None, 0.1 V, shared/histogram
nonlinear-R training, all TC nonidealities on. FT ReLU is fixed TT MC18;
evaluation samples the full 0906 bank per spin. Evaluation reconstructs a fresh
unrolled realization per trial (10 trials); static defects stay fixed across
the trial, dynamic noise remains enabled. PCN uses QATTester on baked weights;
CNN uses the existing TC physical validator. No toggle or solver code changes.
The MNIST CNN evaluator calls the existing sparse unroller on CPU, then moves
the identical matrix to the target device; this avoids one GPU scalar read per
physical edge during cache construction.

## Local

```bash
conda activate scanbase
FAMILY=cnn VARIANT=small DOWNLOAD=true ./launch_scripts/run_mnist_pipeline.sh
```

Local 24-GiB GPU caveat: CNN unrolled evaluation exceeded `MEM_FRAC=0.9` during
per-coupler curve allocation. With an otherwise free GPU, the tested workaround
is to prefix the command with `MEM_FRAC=1 PYTORCH_ALLOC_CONF=expandable_segments:True`.
This changes memory allocation only, not the nonideality model. CPU evaluation
(`STAGE=eval DEVICE=cpu`) is also supported. Do not use the full GPU allowance
when sharing the GPU with another job.

Use `FAMILY=pcn TC_STATE=1` (or `2`) for PCN. `VARIANT=deep` chooses the
larger pair. `STAGE=pretrain`, `ft`, or `eval` runs only that stage; default
`all` runs pretrain→FT→eval and stops on any failure. `MODEL_CKPT` overrides
the input only for standalone FT/eval. `OUTPUT_DIR` isolates experiment runs.

Recipe overrides: `PRETRAIN_EPOCHS`, `FT_EPOCHS`, `PRETRAIN_LR`, `FT_LR`,
`BATCH_SIZE`, `TEST_BATCH_SIZE` (launcher default 64), `OPTIMIZER`, `RHO`,
`EPS`, `WEIGHT_DECAY`, `SCHEDULER`, `STEP_SIZE`, `GAMMA`, `MIN_LR`, `SEED`.
Every optimizer/scheduler setting can also be prefixed `PRETRAIN_` or `FT_`.
`PRETRAIN_TOL=1e-4`, `FT_TOL=1e-6`, `ONE_SHOT_CONV=false` are the defaults.
`python -m mnist_train_eval.mnist_train --help` lists the complete CLI.
`DRY_RUN=true` prints resolved commands/configuration without training.
`LIMIT_TRAIN_SAMPLES`/`LIMIT_TEST_SAMPLES` are smoke-test-only limits (default 0,
full dataset); leave them unset for experiments.

## Slurm

Use the existing successful module setup **outside** the submit script:

```bash
module swap slurm slurm/24.05.0.b1
cd /scratch/rzeng7/repos/PCN-with-Local-Recurrent-Processing
mkdir -p logs/scheduler_slurm
(
  export FAMILY=cnn VARIANT=small
  source ./launch_scripts/slurm_run_mnist.sh
) > logs/scheduler_slurm/mnist_cnn3.log 2>&1
```

For PCN, replace the export with `export FAMILY=pcn VARIANT=small TC_STATE=1`.
Stage/recipe overrides work identically. Upload `../data/MNIST` before submission
if compute nodes cannot download it. The worker uses the existing PCN resource
directives and `source activate base; conda activate scanbase`; the submit script
inherits the environment via `--export=ALL`, without a new login shell or module
initialization. `DRY_RUN=true` prints the sbatch command without submitting.

## Artifacts

Default root: `saved_ckpt_runs/mnist_tc_<model>/`.
Each `<model>_pretrain/` or `<model>_ft/` contains its `_last_ckpt.pth` and
`mnist_training_result.json` (recipe, final test accuracy, train health/LR history).
FT also contains `_full_param_last_ckpt.pth`. The final evaluation uses ordinary
baked `_last_ckpt.pth`; `results/<model>_tc_trials.json` records every trial/seed
and resolved configuration. Slurm output: `logs/slurm_jobs/slurm_<jobid>.out`.

MNIST-only trainer: `mnist_trainer.py`, a thin subclass reusing the existing
training step, health evaluation and checkpoint serializer. The only shared
trainer edit is an overridable dataset-name validator; CIFAR validation is
unchanged. No scientific dynamics code is modified.

## Verification

`PYTHONPATH=.:tests python -m unittest test_mnist_pipeline` exercises both CNN
sizes and both PCN state counts through real train/FT/checkpoint reloads on a
tiny fixture, train-only health checks, optimizer/scheduler and shell handoffs.
Its CUDA check verifies QAT setup and exact CPU/GPU unrolled-matrix equality.
Local real-MNIST smoke runs also exercise batch-64 GPU pretrain/FT and unrolled
evaluation. These small-sample checks are not accuracy experiments; no complete
training run or remote Slurm execution has been performed.

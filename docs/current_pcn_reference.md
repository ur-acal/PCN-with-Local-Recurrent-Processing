# Current PCN training and evaluation reference

Pinned 2026-09-09. Read this file before constructing launch commands for the
current physical PCN experiments. These are explicit experiment settings, NOT
the generic launchers' defaults. Update this document when the agreed recipe
changes. Historical `scan_test` summaries and `coupler_monte_v2` results are not
the reference for this recipe.

## Model and recipe

The main reference is the CiFAIR-100 PCN with 24/48/96 channels, stage depths
4/5/4, 16 total layers, and average pooling before channel expansion.
The small local experiments use 7/14/28 channels with either 4/5/4 (16 layers)
or 6/6/6 (21 layers). Stage depths exclude the input convolution and the two
channel-transition layers. Do not substitute a small-model checkpoint into the
96-channel command below.

| Setting | Pinned value |
|---|---|
| Dataset | CiFAIR100; IQ12, no input centering |
| Coupler source | `coupler_full_range`, conductance |
| Nominal R / C | 50 kOhm / 500 fF |
| Voltage rail | 0.5 V |
| one_over_q | 5 |
| Toggle mode | ODEXInitFFFB; fixed timing; 5 cycles |
| Base y time / z:y ratio | 10 ns / 1 |
| Weight quantization | 5 bits; weight_quant_factor_bits=1 |
| Training recipe scaling | SCALE_TRAIN_RECIPE=1 |
| Output ENOB | none |
| Measured pooling | enabled; same coupler source as FF/FB |
| ReLU source | 0906_RELU_Voltage |
| FT ReLU | fixed TT_V1_T1 MC18, including validation |
| Pretraining measured ReLU | disabled |

Fixed base timing still uses the quantized weight-factor scaling in the pulse
implementation; 10 ns is not a promise that every layer's realized stage time
is exactly 10 ns. Training recipe compensation is already in the trained
weights; do not add SCALE_TRAIN_RECIPE to the Level-3 evaluation command.

## Recent non-ideality audit

| Feature | Status / launch consequence |
|---|---|
| 0906 supply-dependent offset | Python `adapt_relu_offset=True` by default. V0/V1/V2 subtract 0.588/0.600/0.612 V BEFORE axis scaling. Same offset for all curves at a supply category. Older sources unaffected. |
| Explicit offset override | `ode_inference.py --adapt_relu_offset false` restores fixed 0.6 V subtraction for 0906. Shell launchers and the ablation driver do not expose this override; they inherit true. |
| ReLU rail clamp | Already in activation forward; no new flag. |
| Independent noise RNG streams | Already implemented in ode_pc.py; no new flag. |
| Toggle cycles/split/fast path | Forwarded by test launchers; defaults remain 5/0.5/true. |
| Fast summing and coupler noise | Both enabled; each ASD is 0.6e-12 A/sqrt(Hz). |
| Slow summing and coupler offsets | Supported, but BOTH DISABLED in this reference. Each configured std is 2.47e-9 A when enabled. |
| Spin mismatch and DTC | Enabled in 45-corner evaluation, using the selected corner's existing characterization. |
| Differential mismatch | Disabled. |
| Measured pooling granularity | One sampled curve per channel/pooling window, not per input pixel. Final average pooling precedes out_scale. |
| Patched Gaussian | Not used. Empirical sampling from the full curve bank is requested. |
| New coupler_asd_vs_freq.csv | Not referenced by implementation at this audit; does not automatically modify noise parameters. |

No additional testing-script edit is required to activate the above reference.
The explicit overrides below are required because generic launcher defaults
still include older sources, voltage and timing settings.

Source trail:

- [Search configuration](../launch_scripts/slurm_search_config.sh) forwards to
  [KD/CRD then FT](../launch_scripts/run_kdcrd_then_ft.sbatch).
- [Training Python](../train_ode_cifar.py) passes nonlinear_R_table to both the
  wrapper and configure_measured_pooling; changing NONLINEAR_R_TABLE changes
  both sources, not necessarily their individual random assignments.
- [MC45 SLURM scheduler](../launch_scripts/slurm_run_mc45_toggle_ablation.sh)
  -> [local/worker launcher](../launch_scripts/run_mc45_toggle_ablation.sh)
  -> [ablation driver](../scripts/run_toggle_nonideality_ablation.py)
  -> [inference](../ode_inference.py) -> [toggle/wrapper](../ode_pc.py).
- [ReLU preprocessing](../measured_activation.py),
  [pooling](../measured_pooling.py), [corner lookup](../data_utils.py).

## Main-model SLURM training: pretrain then fixed-ReLU FT

Run from ScAN-PCN with a fresh shell, or clear stale experiment overrides first.
In particular, inherited ACTIVATION_CURVE_PATH or ACTIVATION_CORNER can override
the intended typical curve. Keep the normal environment/PATH; local dataset
initialization requires `uv` as well as scanbase Python.

```bash
mkdir -p logs/scheduler_slurm
(
  export TOGGLE_MODE=odexinit \
         TASK=cifar100 \
         IMG_TYPE=CiFAIR \
         EXP_PREFIX=coupler_full_range_CiFAIR100_qf1_noENOB_fixedTiming_scaledRecipe1_zOvery1_relu0906_fixed \
         TEACHER_CKPT=./checkpoint/efficientnet_v2_l_cifar100_CiFAIR_OldNoTimm_MatchDistill.pth \
         ADAPT_PIL_TEACHER=false \
         DISTILL_METHOD=srrl \
         TIMM_RE_PROB=0.0 \
         NONLINEAR_R_TABLE=coupler_full_range \
         MC_RELU_MONTE_CARLO_SOURCE=0906_RELU_Voltage \
         R_VAL=50e3 \
         C_VAL=500e-15 \
         V_DD=0.5 \
         TOGGLE_ONE_OVER_Q=5 \
         TOGGLE_TIMING_MODE=fixed \
         TOGGLE_Y_TIME=10e-9 \
         Z_OVER_Y_TIME=1 \
         SCALE_TRAIN_RECIPE=1 \
         INPUT_QUANT_BITS=12 \
         CENTER_STUDENT_INPUT=false \
         ENABLE_MEASURED_POOLING=true \
         WEIGHT_QUANT_FACTOR_BITS=1 \
         NUM_COMB_PER_NUM_LAYER=2 \
         ENOB=none
  source ./launch_scripts/slurm_search_config.sh
) > logs/scheduler_slurm/cifair100_coupler_full_range_fixedTiming_scaledRecipe1_zOvery1_iq12_relu0906_fixed.log 2>&1 < /dev/null &
echo Scheduler PID: $!
```

ACTIVATION_CORNER_MODE defaults to fixed in this search launcher. FT selects
`0906_RELU_Voltage/tt_25_1.csv`, MC18. Training-time validation uses the same
fixed curve. The automatic post-FT evaluation is FS_V2_T1, 10 trials, with
**per-model** ReLU sampling from `fs_25_2.csv`. It is not the per-spin 45-corner
evaluation below. MC18 is V1, so adaptive subtraction leaves its 0.600 V offset
unchanged.

## Main-model 45-corner SLURM evaluation

This uses the existing uploaded 96-channel fixed-ReLU checkpoint. For a future
training run, verify MODEL_DIR against its printed checkpoint root. Identical
model names do not imply identical checkpoints.

```bash
MODEL_NAME=TIMMQAT5bNoneaNT0p0mulTIMMPCNetNoBatchNorm_PCConvReLU6_0.0eps_ToggleODEXInitFFFB_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_C100_3K1S96C_0.25Dropout_16Layers4l5l4_2Pool_srrlDistill_a0p3_t2p0_CiFAIR_1REP
mkdir -p logs/scheduler_slurm
(
  export SBATCH_TIMELIMIT=12:00:00 \
         N_TRIALS=10 \
         N_SERVERS=9 \
         OUTPUT_ROOT=results/coupler_full_range_CiFAIR100_qf1_noENOB_zOvery1_relu0906_fixed_default45 \
         MODEL_NAME="${MODEL_NAME}" \
         MODEL_DIR=saved_ckpt_runs/coupler_full_range_CiFAIR100_qf1_noENOB_fixedTiming_scaledRecipe1_zOvery1_relu0906_fixed_iq12_toggle_odexinit \
         WEIGHT_QUANT_FACTOR_BITS=1 \
         FULL_45_CORNER_C=500e-15 \
         MC_COUPLER_NONLINEAR_VARIATION_SOURCE=coupler_full_range \
         MC_COUPLER_NONLINEAR_VARIATION_QUANTITY=conductance \
         MC_COUPLER_NOMINAL_R=50e3 \
         NONLINEAR_R_CURVE_SAMPLING=empirical_with_replacement \
         MC_RELU_MONTE_CARLO_SOURCE=0906_RELU_Voltage \
         ACTIVATION_CURVE_SHARING=per_spin \
         V_DD=0.5 \
         ONE_OVER_Q=5 \
         TOGGLE_TIMING_MODE=fixed \
         TOGGLE_Y_TIME=10e-9 \
         Z_OVER_Y_TIME=1 \
         INPUT_QUANT_BITS=12 \
         CENTER_STUDENT_INPUT=false \
         ENABLE_MEASURED_POOLING=true \
         ENOB=none
  source ./launch_scripts/slurm_run_mc45_toggle_ablation.sh
) > logs/scheduler_slurm/slurm_mc45_CiFAIR100_full_range_zOvery1_relu0906_fixed.log 2>&1 < /dev/null &
echo Scheduler PID: $!
```

ReLU and coupler assignments are sampled from the selected corner and held
fixed for each dataset trial, then resampled for the next trial. Pooling uses
the same selected coupler source. The checkpoint loaded is `_best_ckpt.pth`
(baked quantized weights), not `_full_param_best_ckpt.pth`.

For local evaluation, activate scanbase and use the SAME physical/model/source
exports above, replacing OUTPUT_ROOT with OUTPUT_DIR and the source command
with `bash ./launch_scripts/run_mc45_toggle_ablation.sh`. N_SERVERS and
SBATCH_TIMELIMIT are scheduler-only. Local worker defaults are zero; the local
launcher runs the corners serially. Do not rely on its older dataset/model defaults.

## Small local model: 7/14/28 channels, 6/6/6 depths

```bash
conda activate scanbase
mkdir -p logs/local_runs
env STAGE_DEPTHS=6\ 6\ 6 ACTIVATION_CORNER_MODE=fixed nohup bash launch_scripts/run_local_toggle_pretrain_then_ft.sh > logs/local_runs/cifair100_C7_14_28_666_relu0906_fixed.log 2>&1 < /dev/null &
echo Pipeline PID: $!
```

```bash
tail -f logs/local_runs/cifair100_C7_14_28_666_relu0906_fixed.log
```

The [local pipeline](../launch_scripts/run_local_toggle_pretrain_then_ft.sh)
provides the physical settings above and forces zero workers for both stages.
Unlike the search launcher, its ReLU-mode default is random_per_forward, hence
the explicit fixed override. It runs pretraining then FT, without automatic
post-FT ablation. Checkpoints use a timestamped run root printed in the log.
For 4/5/4 depths, change only STAGE_DEPTHS. The old compatibility script
`local_cifar100_c7_14_28_relu0906.sh` calls this same pipeline.

## Operational notes

- Keep source-code changes on the remote server synchronized before launching;
  old server code will not acquire the new offset default from these commands.
- Both slow noises remain off unless explicitly requested. “All-on” is the
  historical reference label, not a claim that every optional feature is enabled.
- A fresh SSH shell drops manually exported variables, except startup-file
  settings or reattached tmux/screen sessions. Existing jobs retain their env.
- A queue starting is not sufficient validation: check the child reaches actual
  training batches. Do not replace PATH with a minimal list that hides `uv`.
- Result directories are experiment identifiers, not proof of successful runs.
  Keep logs and inspect actual completion/accuracy records.

# Current PCN training and evaluation reference

Pinned 2026-09-09. Read this file before constructing launch commands for the
current physical PCN experiments. These are explicit experiment settings, NOT
the generic launchers' defaults. Update this document when the agreed recipe
changes. Historical `scan_test` summaries and `coupler_monte_v2` results are not
the reference for this recipe.

## TC RGB CIFAR-10 model launches

This section is the copy-paste reference for the 22-layer, 16/32/64-channel
TC PCN with 6/7/6 same-channel layers. It is separate from the toggle
CiFAIR-100 reference below. The launch chain is:

- local FT/evaluation: `run_tc_nonidealities.sh` -> `train_ode_cifar.py` or
  `ode_inference.py`;
- Slurm pipeline: `slurm_search_config.sh` -> `run_kdcrd_then_ft.sbatch` ->
  the same Python entry points.

### Defaults supplied by the TC launchers

Do not repeat these in launch commands unless running an ablation: nonlinear-R,
spin variation, fast summing-current noise, fast coupler noise, measured ReLU,
and measured pooling are enabled; slow-current and differential mismatch are
disabled. R/Rmax/C/VDD are 10 kOhm/150 kOhm/49 fF/0.1 V; weights are 5-bit with
no output ENOB; FT uses shared convolution with histogram curve sampling;
expanded evaluation uses loop convolution with per-coupler nonlinear-R curves;
and seeds default to 4096. FT uses fixed TT/25C/VDD1 MC18 measured ReLU, while
evaluation uses the full measured bank with per-spin assignment.

For local or Slurm FT, set `ACTIVATION_CORNER_MODE=random_per_forward` to sample
a new measured ReLU curve on each forward; `per_layer` is already the sharing
default. This mode selects the full `0906_RELU_Voltage` directory rather than
the fixed `tt_25_1.csv` file. `run_tc_nonidealities.sh ft` forwards both
activation-mode controls.

FT also defaults to batch 128, 140 epochs, LR 0.005, zero warmup, SRRL, the RGB
EfficientNet-V2-L teacher, and final-only evaluation. `FT_TIMM_AUG_LEVEL=none`
is retained below because `none` means normal timm augmentation; the launcher's
default `no_aug` means no augmentation.

The live-QAT weight cache and fused measured ReLU are already on automatically.
The three explicit settings below enable accepted-step reuse, RHS checkpointing,
and automatic selection of the checkpointed layer portion.

### Supported stage matrix

| Environment | Pipeline | Pretrain only | FT only | Eval only | FT then eval |
|---|---|---|---|---|---|
| Local | No exact TC pipeline launcher | No exact TC pretrain launcher | `run_tc_nonidealities.sh ft` | `run_tc_nonidealities.sh eval` | Run the two local stages sequentially after reading the FT model name |
| Slurm | `mode=default` | `mode=pretrain_only` | `mode=ft_only` | Not implemented | `mode=ft_and_eval` |

Do not substitute `run_ode_train.sh` for the missing local pipeline entry: its
generic defaults do not reproduce the TC Slurm pretraining recipe. Likewise,
do not claim that the Slurm worker supports `eval_only`; its accepted modes are
exactly the four shown above.

### TC model and path selection

Set this block first for local FT/evaluation. These are the established CIFAR-10
pretrained 22Layers6l7l6 model names. Set `ft_model` from the final `Model Name`
line printed by the FT log before launching evaluation.

```bash
tc_state=1  # 1 or 2

case "$tc_state" in
  1)
    pretrain_model='TIMMPCNetNoBatchNorm_PCConvReLU6_0.0eps_ODEXInitFFFB_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_3K1S64C_0.25Dropout_22Layers6l7l6_2Pool_srrlDistill_a0p3_t2p0_2REP'
    ;;
  2)
    pretrain_model='TIMMPCNetNoBatchNorm_PCConvReLU6_0.0eps_S2NoisyIYAsXZAs0_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_3K1S64C_0.25Dropout_22Layers6l7l6_2Pool_srrlDistill_a0p3_t2p0_2REP'
    ;;
  *) echo 'tc_state must be 1 or 2' >&2; return 2 ;;
esac

run_name="tc_rgb_cifar10_state${tc_state}_pcn_22L6l7l6_ft_5opt_auto_none"
pretrain_root="./saved_ckpt_runs/tc_rgb_cifar10_state${tc_state}_pcn_resnet_depth_study"
ft_root="./saved_ckpt_runs/${run_name}"
ft_log="./logs/local_runs/${run_name}.log"

# Fill this only after FT completes.
ft_model='<FT Model Name from the log>'
```

### Local FT only

Run from an activated `scanbase` environment. Only non-default settings are
passed. The redirection is part of the `nohup` command; do not leave a trailing
backslash or blank line before it.

```bash
cd /home/rongzeng/_workspce_old/repos/pcn/collaboration/ScAN-PCN
conda activate scanbase
mkdir -p logs/local_runs

nohup env \
  MODEL_NAME="$pretrain_model" \
  MODEL_DIR="$pretrain_root" \
  OUTPUT_DIR="$ft_root" \
  TC_STATE="$tc_state" \
  FT_TIMM_AUG_LEVEL=none \
  REUSE_ACCEPTED_STEP_TRAINING=true \
  CHECKPOINT_ODE_RHS_TRAINING=true \
  CHECKPOINT_ODE_RHS_PORTION=auto \
  bash ./launch_scripts/run_tc_nonidealities.sh ft \
  > "$ft_log" 2>&1 < /dev/null &

echo "PID=$!  LOG=$ft_log"
```

### Local eval only

This is the standard expanded all-on TC evaluation: batch 128, ten trials,
full measured-ReLU bank with per-spin assignment, and per-coupler nonlinear-R.
`last` is already the launcher's default evaluation checkpoint.

```bash
eval_log="./logs/local_runs/${run_name}_eval.log"

nohup env \
  MODEL_NAME="$ft_model" \
  MODEL_DIR="$ft_root" \
  TC_STATE="$tc_state" \
  bash ./launch_scripts/run_tc_nonidealities.sh eval \
  > "$eval_log" 2>&1 < /dev/null &

echo "PID=$!  LOG=$eval_log"
```

For local FT followed by evaluation, run the FT block, copy its final printed
model name into `ft_model`, and then run the evaluation block. This explicit
boundary prevents evaluation from silently selecting a stale or partial FT
checkpoint.

### Slurm pipeline, pretrain only, FT only, or FT then eval

Use a fresh module-configured login shell. Set `pipeline_mode` to one of the
four values in the table. For `default` or `pretrain_only`, point
`pretrain_root` to the intended new pretraining output. For `ft_only` or
`ft_and_eval`, it must contain exactly one matching completed 300-epoch
pretrained model; verified checkpoint discovery rejects zero or multiple
matches. Use a fresh `ft_root` for any mode that runs FT.

```bash
module swap slurm slurm/24.05.0.b1
cd /scratch/rzeng7/repos/PCN-with-Local-Recurrent-Processing
mkdir -p logs/scheduler_slurm logs/slurm_jobs

pipeline_mode=ft_only  # default | pretrain_only | ft_only | ft_and_eval

for tc_state in 1 2; do
  case "$pipeline_mode" in
    default|pretrain_only)
      run_name="tc_rgb_cifar10_state${tc_state}_pcn_22L6l7l6_${pipeline_mode}_5opt_auto_none"
      pretrain_root="./saved_ckpt_runs/${run_name}_pretrain"
      ;;
    ft_only|ft_and_eval)
      run_name="tc_rgb_cifar10_state${tc_state}_pcn_22L6l7l6_${pipeline_mode}_5opt_auto_none"
      pretrain_root="./saved_ckpt_runs/tc_rgb_cifar10_state${tc_state}_pcn_resnet_depth_study"
      ;;
    *)
      echo "Unsupported pipeline_mode: $pipeline_mode" >&2
      break
      ;;
  esac
  ft_root="./saved_ckpt_runs/${run_name}"

  (
    export mode="$pipeline_mode" \
           TC_NONIDEALITIES=true \
           TC_STATE="$tc_state" \
           TASK=cifar10 \
           IMG_TYPE=rgb \
           EXP_PREFIX="$run_name" \
           PRETRAIN_SAVE_PATH="$pretrain_root" \
           FT_OUTPUT_SAVE_PATH="$ft_root" \
           FT_TIMM_AUG_LEVEL=none \
           PCN_CHAN_0_LIST=16 \
           PCN_NUM_LAYERS_LIST=22 \
           COMB_SEL_SET=2 \
           REUSE_ACCEPTED_STEP_TRAINING=true \
           CHECKPOINT_ODE_RHS_TRAINING=true \
           CHECKPOINT_ODE_RHS_PORTION=auto

    source ./launch_scripts/slurm_search_config.sh
  ) > "./logs/scheduler_slurm/${run_name}_${pipeline_mode}.log" 2>&1 < /dev/null &

  echo "${run_name}: scheduler PID $!"
done
```

`COMB_SEL_SET=2` selects exactly `N22_C16_n06_n17_n26`, i.e. the 6/7/6
architecture. `NUM_COMB_PER_NUM_LAYER=3`, `MAX_TASKS_PER_GPU=1`, the teacher,
all TC hardware settings, and fused measured ReLU are defaults and are therefore
not repeated above. The scheduler log prints the submitted experiment and Slurm
job ID.

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

### Measured-ReLU pretraining depth stability (2026-09-30)

For two-stage, no-BN CiFAIR-100 models using the fixed 0906 MC18 ReLU with
`UNITLESS_MEASURED_PULLBACK_MODE=direct`, excessive depth can prevent early
training convergence. Five-epoch LR warmup can mitigate this failure. This does not happen when we use ideal ReLU6 in the pretrain stage.

| Control | Mean CE by epoch | Observation |
|---|---|---|
| 36->72, 9:9 stages, no warmup | 4.5216 -> 4.2132 -> 4.0380 | Converges |
| 36->72, 11:11 stages, no warmup | 4.6842 -> 4.6066 -> 4.6060 | Stays at chance |
| 32->64, 11:11 stages, no warmup | 4.7554 -> 4.6083 -> 4.6074 | Stays at chance |
| 32->64, 11:11 stages, 5-epoch warmup | 4.7681 -> 4.6039 -> 4.4281 -> 4.2881 -> 4.1996 | Converges; 13.94% validation top-1 at epoch 5 |

Other three-epoch C32->64, 11:11 controls isolate the failure mechanism:

| Control | Mean CE by epoch | Observation |
|---|---|---|
| Original baseline | 4.7554 -> 4.6083 -> 4.6074 | Stays at chance |
| Boundary BN | 4.4712 -> 4.1534 -> 3.9779 | Clearly learns |
| No SRRL | 4.7547 -> 4.6063 -> 4.6059 | Stays at chance |
| SRRL weight 0.1 | 4.6713 -> 4.6121 -> 4.6061 | Stays at chance |
| No timm augmentation | 4.6415 -> 4.6063 -> 4.6058 | Stays at chance |

The controlled 36->72 comparison identifies depth as the immediate cause:
9:9 learns, while 11:11 fails with width and recipe unchanged. This establishes
only a tested boundary, not a universal maximum depth. Prefer 9:9 or shallower;
for 11:11, use LR warmup and verify convergence before a full run. Logs and the
reproducible launcher are under `logs/local_runs/c32_pullback_controls_*` and
`diagnostic_scripts/run_c32_pullback_pretrain_controls.sh`.

Removing SRRL, reducing its weight, or removing timm augmentation does not
restore learning, so none is the primary cause. Boundary BN does restore
learning, supporting depth-related activation/gradient propagation as the
failure mechanism; LR warmup provides a second mitigation without changing the
architecture.

#### Activation-shape diagnostic

Four five-epoch runs used the same C32->64, 11:11, no-BN configuration, seed,
data, optimizer, SRRL, augmentation, and zero-warmup recipe. Only the
pretraining activation was changed. The CE values below are epoch means over
all 390 minibatches, not the last progress-bar minibatch.

| Pretraining activation | Mean CE by epoch | Epoch-5 validation | Result |
|---|---|---|---|
| Ideal ReLU5 | 4.5977 -> 4.3277 -> 3.9571 -> 3.7871 -> 3.6310 | 32.66% top-1 / 65.33% top-5 | Learns |
| Endpoint-normalized MC18 | 5.0356 -> 4.6063 -> 4.6058 -> 4.6056 -> 4.6055 | 1.00% / 5.00% | Stays at chance |
| Zero-offset MC18, `phi_m(x) - phi_m(0)` | 4.4237 -> 4.0132 -> 3.8230 -> 3.6898 -> 3.5444 | 34.86% / 67.84% | Learns |
| Zero-offset and rectified MC18, `max(phi_m(x) - phi_m(0), 0)` | 4.6059 -> 4.5987 -> 4.5911 -> 4.5780 -> 4.5469 | 6.58% / 21.50% | Only weak recovery |

For the direct-pullback MC18 curve, `phi_m(0) = 0.2136` in unitless model
coordinates. Endpoint normalization preserves this positive zero-input output
and does not restore training, whereas subtracting only `phi_m(0)` does. Along
with the boundary-BN result, **this strongly identifies `phi_m(0) > 0` as a major cause of the deep no-BN model's failure**. It is not the only
relevant curve property: rectifying the centered curve removes the offset but
also removes its negative-input branch and gives only weak recovery. This might suggest that non-dead negative-input region might be helpful.

These are diagnostics, not changes to the production measured-ReLU behavior.
The reproducible cases are `ideal_relu5`, `endpoint_normalized`, `zero_offset`,
and `zero_offset_relu` in
`diagnostic_scripts/run_c32_pullback_pretrain_controls.sh`. The two transformed
diagnostic tables are generated by
`diagnostic_scripts/generate_mc18_activation_controls.py`. Raw logs are under
`logs/local_runs/c32_pullback_controls_activation_*_20260930_161036/`, and the
service/run manifest is
`logs/local_runs/c32_activation_variants_20260930_161036/services.txt`.

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
**per-spin** ReLU sampling from `fs_25_2.csv`, matching the pinned 45-corner
protocol for that same corner. MC18 is V1, so adaptive subtraction leaves its
0.600 V offset unchanged.

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

Historical, **before the FF expansion fix**: 2-trial "FS_V2_T1" run -> 58.73%
results/coupler_full_range_CiFAIR100_qf1_noENOB_zOvery1_relu0906_fixed_FS_V2_T1_2trials/FS_V2_T1.log

### Level-3 FF expansion correction (2026-09-24)

The historical trials were 58.76% and 58.70%. They did not exercise the
intended per-physical-coupler FF curve sampling: FF's functional convolution
bypassed Validator's module-forward expansion hook. FB was expanded correctly.
The dense FF path used its fallback nonlinear curve instead.

The shape-initialization pass now explicitly triggers both pulse modules'
expansion hooks before constructing pulse matrices, and rejects incomplete
FF/FB expansion. Normal training does not enable this initialization flag.
Regression tests cover ideal-output equivalence, old incomplete-cache repair,
complete-cache reuse, and per-coupler curves on both sides.

The matched two-trial replay is recorded in
`results/ff_unroll_fix_FS_V2_T1_2trials/` (`FS_V2_T1.log`, `command.json`,
`results.json`, and `summary.md`). Numerical CLI arguments are preserved from
the historical log; artifact locations are separate. Both trials completed:

| Trial | Before FF fix (%) | After FF fix (%) | Change (percentage points) |
|---|---:|---:|---:|
| 0 | 58.76 | 57.24 | -1.52 |
| 1 | 58.70 | 56.84 | -1.86 |
| Mean | 58.73 | 57.04 | -1.69 |

This is a measurable reduction, not an accuracy collapse. Two trials in one
corner do not establish the effect across all 45 corners.
Earlier accuracy and final-linear-layer study artifacts remain historical
pre-fix measurements, not validation of the corrected level-3 implementation.

Empirical lookup, signed current calculation and accumulation now use an
inference-only Triton fast path on supported CUDA float32 toggle modules;
no launch changes are needed. A sequential five-batch/128-sample comparison
against the FF-fixed **unoptimized** reference measured 4.46x faster forwards,
640/640 matching predictions, and maximum logit difference 5.9605e-6.
This is accepted numerical roundoff, not bitwise equality: atomic accumulation
order differs. All nonideality sampling and pulse/state-update logic remain
unchanged. Results: `results/toggle_full_fusion/summary.md`.
Details, fallbacks and reproduction commands:
[Toggle optimization audit](toggle_inference_optimization_audit.md).

The two subsequent optional stage-reuse experiments were removed. The retained
fusion was reverified against the original unoptimized FF-fixed path on 1,280
inputs: all predictions matched, maximum logit difference 1.0014e-5, and 4.49x
forward speedup. Details: `results/toggle_reuse_removal/summary.md`.

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

## Final linear layer study

The persistent diagnostic runs one full Level-3 test trial using the actual
inference command saved by the MC45 launcher. Its default reference is the
main model's FS_V2_T1 per-spin run (0906 adaptive offset enabled). It changes
only the trial count to one and adds passive recording. It calls the normal
`ode_inference.run_ode_inference` implementation, including its data loader,
hardware sampling, pooling, and accuracy loop.

```bash
conda activate scanbase
bash diagnostic_scripts/run_study_final_linear.sh
```

Optional shell overrides: `REFERENCE_LOG` selects another MC45 corner log,
`OUTPUT_DIR` selects the output directory, and `BATCH_INDEX` selects the one
saved batch (zero-based, default 0). The log must contain its `COMMAND:` line.
Only use trusted run logs: their arguments are passed to inference.

After a completed run, regenerate both PDFs without inference:

```bash
PLOT_ONLY=true DROP_PP=1 bash diagnostic_scripts/run_study_final_linear.sh
```

`DROP_PP` is the requested potential accuracy loss in percentage points
(default 1). It also works during a full inference run. Plot-only mode reads
`margin_data.npz`: one signed `margin` and Boolean `correct` flag per test
sample, plus the scalar `mean_absolute_logit`. Full-test logits are not saved.

Pinned artifacts under `results/final_linear_study/pinned_FS_V2_T1/`:

Completed on 2026-09-22: 10,000 samples, 58.76% accuracy. Every logged batch
accuracy matches the original uninstrumented trial. The saved batch's NumPy
linear reconstruction has maximum absolute error 3.8146973e-6.
The margin-saving rerun reproduced 58.76%. With `DROP_PP=1`, the marked
threshold is 0.1350955963134766 raw logit units: 100 correct samples are
vulnerable, giving potential remaining accuracy 57.76% (1.701838% of correct
predictions vulnerable).

- [One-batch NumPy archive](../results/final_linear_study/pinned_FS_V2_T1/linear_batch.npz).
- [Five-statistic summary table](../results/final_linear_study/pinned_FS_V2_T1/summary.md).
- [Signed-margin histogram](../results/final_linear_study/pinned_FS_V2_T1/signed_margin_histogram.pdf).
- [Two-curve threshold sweep with accuracy and drop markers](../results/final_linear_study/pinned_FS_V2_T1/threshold_sweep.pdf).
- [1 pp marker version](../results/final_linear_study/pinned_FS_V2_T1/threshold_sweep_drop1pp.pdf), threshold 0.1350955963134766.
- [5 pp marker version](../results/final_linear_study/pinned_FS_V2_T1/threshold_sweep_drop5pp.pdf), threshold 0.684863567352295.
- [Saved full-test margins](../results/final_linear_study/pinned_FS_V2_T1/margin_data.npz).

The archive contains `linear_input` [batch, features], `linear_output`
[batch, classes], `weight` [classes, features], `bias` [classes], `labels`
[batch], and `sample_indices` [batch] (positions in test-loader order).
These are the actual linear-layer input, raw output, and parameters during
the selected batch. No softmax or centering is applied.

```python
import numpy as np
with np.load("results/final_linear_study/pinned_FS_V2_T1/linear_batch.npz") as data:
    logits = data["linear_input"] @ data["weight"].T + data["bias"]
    np.testing.assert_allclose(logits, data["linear_output"], rtol=1e-4, atol=1e-5)
    predictions = data["linear_output"].argmax(axis=1)
    labels = data["labels"]
```

Full-test logits are kept only in memory. The summary reports accuracy, mean
absolute raw logit, and mean signed margins for all/correct/incorrect samples.
Margin is true-class logit minus maximum incorrect-class logit. The sweep retains
two curves, both as percentages of the entire test set: correct predictions
with margin < threshold, and all predictions with absolute margin < threshold.
The first curve measures potential accuracy loss if all vulnerable correct
predictions flip, not expected loss under a specified noise distribution.
The current accuracy is a horizontal reference line. A dot and dashed guides
mark the configurable drop on the first curve, with threshold and drop percentage labeled
at the axes. `threshold_marker.json` records the exact threshold and drop.
Each plot-only call also preserves a `threshold_sweep_drop{DROP_PP}pp.pdf`
and corresponding marker JSON, so different markers can coexist. Numeric
markers appear only at the axes; the legend says "Current accuracy".
The threshold is just above the selected margin (strict inequality); tied
margins can make the attainable drop exceed the request, which is reported.
Exact ties in classification follow the actual argmax prediction.
`run_config.json` and `reference_command.txt` preserve provenance.

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

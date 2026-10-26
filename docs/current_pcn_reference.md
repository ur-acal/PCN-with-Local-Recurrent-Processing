# Current PCN training and evaluation reference

Pinned 2026-10-09. Read this file before constructing launch commands for the
current physical PCN experiments. These are explicit experiment settings, NOT
the generic launchers' defaults. Update this document when the agreed recipe
changes. Historical `scan_test` summaries and `coupler_monte_v2` results are not
the reference for this recipe.

## Rule for deriving commands from this reference

When generating a new launch command, first locate and use the exact pinned
command in this file as the authoritative baseline; do not reconstruct it from
memory. Trace every requested change through the launcher, forwarded arguments,
defaults, checkpoint selection, and Python implementation, then list the
required command-line changes before presenting the command.

Copy the pinned command and modify only those listed lines, plus names and paths
needed to prevent output collisions. Preserve every unspecified line exactly,
including intentionally omitted arguments and default behavior: do not add
explicit defaults, remove options, or reorganize the command. Compare the final
command line-by-line with the pinned command and run a non-submitting dry run
when supported. If the request is ambiguous or conflicts with the pinned
configuration, identify the ambiguity instead of silently selecting a different
configuration. This rule applies to every TC and toggle command in this file.

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
and seeds default to 4096. The pinned Slurm recipe samples the full 0906
measured-ReLU bank once per FT forward with per-spin assignment; evaluation
also uses the full measured bank with per-spin assignment.

For local or Slurm FT, `ACTIVATION_CORNER_MODE=random_per_forward` samples new
measured ReLU curves on each forward. The pinned Slurm recipe also sets
`ACTIVATION_RANDOM_CURVE_SHARING=per_spin`. This mode selects the full
`0906_RELU_Voltage` directory rather than the fixed `tt_25_1.csv` file.
`run_tc_nonidealities.sh ft` forwards both activation-mode controls.

FT also defaults to batch 128, 140 epochs, LR 0.005, zero warmup, SRRL, the RGB
EfficientNet-V2-L teacher, and final-only evaluation. The pinned Slurm recipe
uses the custom `rgb_mid` augmentation level.

The live-QAT weight cache and fused measured ReLU are already on automatically.
The three explicit settings below enable accepted-step reuse, RHS checkpointing,
and automatic selection of the checkpointed layer portion.

### Optional physical compute-time variation (TC PCN and CNN)

Two independent relative timing controls default to zero (unchanged execution):

| Effect | Shell / Python control | Sampling |
|---|---|---|
| Fixed timing variation | `COMPUTE_TIME_FIXED_STD` / `--compute_time_fixed_std` | Independent per compute unit; held for the training run or evaluation trial |
| Forward timing noise | `COMPUTE_TIME_NOISE_STD` / `--compute_time_noise_std` | Independent per unit and forward; held throughout the solve |
| Timing-noise scope | `TIMING_NOISE_SCOPE` / `--timing_noise_scope` | `per_batch` (default, unchanged behavior) or `per_sample` |

The actual duration is `T * (1 + fixed_fraction + noise_fraction)`, with
independent zero-mean Gaussian fractions of the requested STD. Thus `0.2`
means 20%, not 0.2 seconds. Nonpositive durations are resampled (the Gaussian
tail is truncated, not clamped). PCN uses one factor for the entire ODE block;
two-state PCN shares that horizon across both simultaneously integrated states.
CNN uses separate factors for every physical convolution, including the stem.
BN, pooling, digital classifiers and unitless pretraining are not timed here.
Analog classifier-head timing is not wired by this option.

`per_sample` draws independent jitter for every image, including within a batch.
Fixed variation remains shared by the layer. It automatically uses the nominal
solver interval with drift multiplied by each image's factor and diffusion by
its square root; no second mode switch is needed. Both coupled PCN states use
the same image-specific factor. Per-sample timing currently rejects coupler
energy metering rather than reporting energy with an incorrect shared duration.

Apply the factor after physical/QAT timing is resolved. Integration and current
diffusion both use the actual duration; capacitances, weight/output scales,
voltage rails and noise spectral densities are unchanged. No compensation is
applied. Nominal solver options are never overwritten, avoiding cumulative
rescaling. Timing RNGs are separate from other circuit RNGs; evaluation trial
seeds advance normally. Exact recovery and the automatic memory profiler
preserve timing assignments/RNGs. Shape-discovery runs precede timing setup.

The existing TC PCN and CNN local/Slurm FT/evaluation launchers accept these
environment variables; direct Python entry points accept the lower-case
arguments. Timing results are included in TC trial metadata (last forward's
nominal/actual durations, fixed/noise fractions and factors).

Full-dataset CIFAR-10 sensitivity sweep: nine combinations of each STD in
`{0, 0.1, 0.2}`, including the zero/zero sanity case; one hardware trial per
combination. Runner: `diagnostic_scripts/run_compute_time_sweep.py`.
Results: `results/tc_compute_time_sweep_20261017/results.csv`; launch commands
and per-case logs are saved alongside it. Models: 22L state-2 PCN
`tc_c10_s2_22L_nonorm_rgbmid_rpf_pspin` (Dopri5, tolerance 1e-6) and the final
`tc_rgb_cifar10_resnet44_cifar_q32_rgb_mid_ft` checkpoint (Euler-10).
Other nonidealities remain enabled. PCN uses raw RGB; this older CNN uses
normalized RGB. These are within-model timing-sensitivity tests, not a
matched-training-recipe comparison. Do not infer completion from this recipe;
consult completed per-case metadata and the scope-tagged results CSV. The
original all-on 22L PCN `(0.2, 0.2)` trial was completed separately with curve
CPU offload disabled; the original runner's completion marker is absent.
The per-sample replay runner is `diagnostic_scripts/rerun_timing_noise_scope.py`;
it preserves historical rows and processes their original CSV order in reverse,
starting with the CNN BN-recalibration controls. Calibration and testing use
the same scope and fixed assignments, with fresh dynamic-noise samples.

Full expanded-model scope validation (four CIFAR-10 images; both timing STDs
0.2; static nonidealities enabled) compared batched `per_sample` with individual
`per_batch` physical-duration solves. All predicted classes matched.

| Model | Current diffusion / solver-grid comparison | Maximum absolute logit difference |
|---|---|---:|
| ResNet-44 | OFF; native Euler-10 grids | 1.79e-6 |
| ResNet-44 | ON; matched physical grids and Gaussian innovations | 2.62e-6 |
| State-2 PCN, 22L | OFF; independent Dopri5 grids, tolerance 1e-6 | 1.65e-3 |
| State-2 PCN, 22L | OFF; independent Dopri5 grids, tolerance 1e-7 | 1.98e-4 |
| State-2 PCN, 22L | ON; matched physical grids and Gaussian innovations, tolerance 1e-6 | 1.19e-7 |

The PCN native-grid check at production tolerance exceeded the initial 0.1%
relative-RMS gate (0.194%); refinement reduced it to 0.0475%, passing the same
gate. This is convergence evidence, not bit equality between adaptive meshes.
The sweep retains its original tolerance 1e-6. Raw logits and checks are in
`results/tc_timing_scope_validation_20261017/{cnn,pcn_convergence}/validation.json`.
The detached reverse-order replay writes under
`results/tc_compute_time_sweep_20261017/per_sample`; historical per-batch rows
are retained separately from appended per-sample results in the main CSV.

### RGB input-normalization controls

The generic RGB default is unchanged: student and RGB teacher both use the
dataset's CIFAR mean/std. The pinned Slurm TC recipe below deliberately uses
raw, uncentered RGB for both student and teacher. CiFAIR/scanGFI preprocessing
is unchanged.

| Experiment | Teacher training | Student pretrain + FT |
|---|---|---|
| Existing recipe | Existing teacher | Existing defaults |
| Raw RGB student, existing teacher | Keep existing checkpoint | `NORMALIZE_STUDENT_INPUT=false` |
| Raw RGB student and teacher | `NORMALIZE_TEACHER_INPUT=false` | `NORMALIZE_STUDENT_INPUT=false`, select the new `TEACHER_CKPT` |

Teacher training uses `--normalize_input`; student training uses
`--normalize_student_input`. There is no teacher-normalization argument during
student training: the teacher checkpoint supplies it. Old RGB teachers retain
CIFAR normalization; the legacy CiFAIR teacher retains 0.5/0.5 normalization.
The RGB teacher worker appends `_NoNorm` to unnormalized teacher filenames/logs.
Use distinct student `EXP_PREFIX`/output directories for these experiments.

Student normalization is saved in ordinary, full-param, and recovery
checkpoints. FT and `ode_inference.py` inherit it when no override is supplied;
exact recovery rejects a changed setting. All settings are logged. Apply the
same student setting during pretraining and FT, not just when loading an old
normalized-input pretrained model. Keep the existing augmentation recipe and
`TIMM_RE_PROB=0.0`: only normalization changes, including preserving RandAugment
fill colors and shared teacher/student Mixup/CutMix samples.

### Supported stage matrix

| Environment | Pipeline | Pretrain only | FT only | Eval only | FT then eval |
|---|---|---|---|---|---|
| Local | No exact TC pipeline launcher | No exact TC pretrain launcher | `run_tc_nonidealities.sh ft` | `run_tc_nonidealities.sh eval` | Run the two local stages sequentially after reading the FT model name |
| Slurm | `mode=default` | `mode=pretrain_only` | `mode=ft_only` | `mode=eval_only` | `mode=ft_and_eval` |

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

pipeline_mode=default  # default | pretrain_only | ft_only | eval_only | ft_and_eval

for tc_state in 1 2; do
  case "$pipeline_mode" in
    default|pretrain_only)
      run_name="tc_rgb_cifar10_state${tc_state}_pcn_22L6l7l6_${pipeline_mode}_nonorm_rgbmid_random_relu_per_spin"
      pretrain_root="./saved_ckpt_runs/${run_name}_pretrain"
      ;;
    ft_only|ft_and_eval)
      run_name="tc_rgb_cifar10_state${tc_state}_pcn_22L6l7l6_${pipeline_mode}_nonorm_rgbmid_random_relu_per_spin"
      pretrain_root="./saved_ckpt_runs/tc_rgb_cifar10_state${tc_state}_pcn_22L6l7l6_pretrain_student_no_norm"
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
           NORMALIZE_STUDENT_INPUT=false \
           CENTER_STUDENT_INPUT=false \
           EXP_PREFIX="$run_name" \
           PRETRAIN_SAVE_PATH="$pretrain_root" \
           FT_OUTPUT_SAVE_PATH="$ft_root" \
           TEACHER_CKPT=./checkpoint/efficientnet_v2_l_cifar10_rgb_OldNoTimm_MatchDistill_NoNorm.pth \
           TEACHER_ARCH=efficientnet_v2_l \
           TEACHER_ARCH_SOURCE=torchvision \
           FT_TIMM_AUG_LEVEL=rgb_mid \
           ACTIVATION_CORNER_MODE=random_per_forward \
           ACTIVATION_RANDOM_CURVE_SHARING=per_spin \
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
architecture. `NUM_COMB_PER_NUM_LAYER=3`, `MAX_TASKS_PER_GPU=1`, all TC hardware
settings, and fused measured ReLU are defaults and are therefore not repeated
above. For `ft_only` or `ft_and_eval`, the stated pretraining root
must contain a completed matching unnormalized-input checkpoint. The scheduler
log prints the submitted experiment and Slurm job ID.

TC pretraining already receives five warmup epochs from the launcher. FT remains
at zero warmup: current CIFAR-10 TC FT completed successfully with this setting.
The historical division-by-zero failure came from the corrected noise-generation
bug, while the separate batch-128 OOM still occurred with warmup. Consequently,
warmup is neither required for current FT stability nor a remedy for its memory
use.

### Active TC non-idealities: FT and expanded evaluation

This is the active-only audit of the pinned TC commands above against the
[scientific TC definition](../../papers/hardware-native-neural-ode/shared/true_continuous_nonidealities.md).
A *forward* is one full-model batch forward; an *accepted interval* includes its
adaptive-step trials and any checkpoint replay; a *validation pass* or *trial*
is one traversal of the evaluation dataset; a *spin/site* is one
`(channel, row, column)` destination, shared across batch items.

Audit result: no enabled TC non-ideality is missing from the implementation.
There are two deliberate fidelity differences to keep visible:

- FT uses the documented low-cost nonlinear-R method 2a: one
  histogram-selected, Gaussian R(V) curve per FF/FB tensor, rather than the
  scientific exact-training target of one curve for every represented weight
  code. Final expanded evaluation does not use this approximation.
- Evaluation during FT remains on that dense/shared FT implementation. It is
  not the final expanded evaluator with independently sampled physical
  couplers.

`TC_CURVE_SAMPLING=histogram` controls the first approximation; it does not
change expanded evaluation, which samples code-conditioned curves per physical
coupler. The final classifier and the final ideal ReLU remain outside the TC
hardware model by design.

| Non-ideality | Fine-tuning | Final expanded TC evaluation | Evaluation during FT |
|---|---|---|---|
| PC FF/FB weight quantization | Symmetric 5-bit live QAT (`q_hi=15`) on every FF/FB tensor. Codes 1--15 map to nominal `R/abs(W)` from 10 to 150 kOhm; code 0 is open. No scale-factor quantization. | Loads the baked 5-bit weights. Every FF/FB convolution is spatially unrolled; code 0 remains open. | Same live-QAT path as FT. |
| Fast summing-current noise | ASD `0.6e-12 A/sqrt(Hz)`, scaled by `sqrt(50 kOhm/R)`. FF and two-state FB use `J/C*sqrt(dt)` diffusion after an accepted update. One-state FB is band-integrated and added in amperes before the measured ReLU; one sample is held throughout an accepted interval and redrawn after acceptance. | Same equations and lifetimes. Trial seeds make the stochastic sequence reproducible; dynamic samples still advance during the dataset traversal. | Same as FT. |
| Fast coupler-current noise | Same branch placement and lifetime as summing noise, with ASD multiplied by `sqrt(sum(abs(W_code)))`. The sum uses nominal quantized conductance levels; zero/open and padding entries contribute no noise, and nonlinear R(V) does not modulate its amplitude. | Same equations, using the active unrolled physical couplers. | Same as FT. |
| Nonlinear coupler R(V) and static curve variation | Code-specific mean curves from `res_vs_vin_10k_150k.csv` plus the shared absolute covariance from `CU_4500_r_vs_vin.csv`. Method 2a histogram-selects one code, samples one positive-floored Gaussian curve independently for each FF/FB tensor/layer/forward, and shares it across that tensor. The draw is held for the ODE solve and backward replay. | Every FF/FB convolution is unrolled. Each nonzero physical coupler gets an independent positive-floored Gaussian curve from its code-specific mean and the common covariance; zero/padding entries are open and consume no draw. Curves stay fixed for the whole trial. | Dense method 2a. Fresh FF/FB assignments are made at the start of the validation pass and then fixed for that pass. |
| Measured ReLU | Fixed 0906 `tt_25_1.csv`, MC18 curve shared by all measured-activation modules and spins for the pinned FT job. Piecewise-linear input/output is rail-clamped. Optional `random_per_forward` overrides this recipe. | Full 0906 measured bank with independent `per_spin` assignment in each layer, shared across batch items and fixed for the whole trial. | Same fixed curve as pinned FT. |
| Measured average pooling | The 10-kOhm mean plus the common covariance supplies one Gaussian R(V) curve per output channel and pooling window, shared across the window inputs and batch. Redrawn every training forward for both intermediate pooling and GAP. | Same source and granularity, fixed for the whole trial; intermediate pooling and GAP have separate assignments. | Fresh assignments at the start of the validation pass, fixed for that pass. |
| Spin/conversion-gain variation | Gaussian mean 1.0, standard deviation 0.10; independent per spin/site, branch, and layer, shared across batch, redrawn every training forward, and held through the solve. In one state, the FB factor multiplies both FB signal and algebraic FB noise before ReLU; in two state, the independent y/z factors multiply their respective drifts, not the diffusion increments. | Same distribution and granularity, fixed for the whole trial. | Fresh factors at the start of the validation pass, fixed for that pass. |
| Voltage-rail saturation | Each accepted dynamical-state update, including diffusion, is projected to `[-0.1,+0.1] V`; nonlinear-R lookup coordinates and measured-ReLU input/output are also clamped to their characterized/rail ranges. | Same. | Same. |

The common physical point is `R=10 kOhm`, `R_max=150 kOhm`, `C=49 fF`,
and `V_DD=0.1 V`. The selected one-state equation is `ODEXInitFFFB`; the
selected two-state equation is `S2NoisyIYAsXZAs0` with simultaneous y/z
evolution and separate FF/FB capacitance scaling.

Supported but inactive in this reference, and therefore excluded from the
table, are differential/generic mismatch, slow-current offsets, DTC effects,
output ENOB, weight-scale-factor quantization, and the legacy thermal-noise
path.

Source trail checked for this audit:

- Launch/default wiring: [TC hardware defaults](../launch_scripts/tc_hardware_defaults.sh)
  -> [TC argument construction](../launch_scripts/tc_nonideality_args.sh) ->
  [local TC launcher](../launch_scripts/run_tc_nonidealities.sh), or
  [Slurm scheduler](../launch_scripts/slurm_search_config.sh) ->
  [Slurm worker](../launch_scripts/run_kdcrd_then_ft.sbatch).
- Python entry and validation: [training](../train_ode_cifar.py) or
  [expanded inference](../ode_inference.py) -> [TC CLI validation](../tc_cli.py).
- Implementations: [TC dynamics, wrappers, and physical scaling](../ode_pc.py),
  [noise lifecycle and Gaussian R(V) construction](../tc_nonidealities.py),
  [expanded physical-coupler convolution](../validation.py),
  [measured activation](../measured_activation.py),
  [measured pooling](../measured_pooling.py), and
  [adaptive accepted-step/noise handling](../TorchDiffEqPack/odesolver/).

## Model and recipe

The main reference is now the CiFAIR-100 two-stage PCN with 36/72 channels,
stage depths 8/10, 20 total layers, and one average-pooling layer immediately
before channel expansion. Stage depths exclude the input convolution and the
channel-expansion layer. Its pretraining and fine-tuning both use the fixed
0906 MC18 measured ReLU; pretraining applies the direct unitless pullback.

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
| Pretraining measured ReLU | fixed TT_V1_T1 MC18; direct unitless pullback |

Fixed base timing still uses the quantized weight-factor scaling in the pulse
implementation; 10 ns is not a promise that every layer's realized stage time
is exactly 10 ns. Training recipe compensation is already in the trained
weights; do not add SCALE_TRAIN_RECIPE to the Level-3 evaluation command.

### Measured-ReLU pretraining depth stability (2026-09-30)
#### Main conclusion
1. As models become deeper, some can suffer poor FT convergence, producing roughly a 10-point pretrain-to-FT drop.
   - C32 12:10: 65.75% → 55.85%.
   - C32 10:12: 66.57% → 53.53%.
   - This is not universal: C32 11:11 reached 66.59% → 61.97%.

1a. Measured-ReLU pretraining can partly mitigate this mismatch.

- C32 12:10 with measured-ReLU pretraining plus five-epoch warmup reached 64.43% pretrain and 61.75% FT, versus 65.75% and 55.85% previously.
- Because measured ReLU and warmup were introduced together, their individual contributions are not fully isolated.

2. With measured ReLU during pretraining, sufficiently deep no-BN models can stop learning, with CE plateauing near \\(\log(100)\approx4.605\\). This occurred at the tested 22-PC-layer depth for both channel widths.
   - C32→64, 11:11: 4.7554 → 4.6083 → 4.6074.
   - C36→72, 11:11: 4.6842 → 4.6066 → 4.6060.
   - C36→72, 9:9 converged: 4.5216 → 4.2132 → 4.0380.

2a. Diagnostics strongly implicate the measured ReLU’s positive zero-input output.

- Direct MC18 has \\(\phi\_m(0)=0.2136\\).
- Endpoint normalization retained the offset and failed.
- Subtracting \\(\phi\_m(0)\\) restored learning.
- Boundary BN also restored learning.
- Removing/reducing SRRL or removing timm augmentation did not.

2b. Five-epoch LR warmup partly resolves the convergence problem.

2c. Once training converges, measured-ReLU pretraining has little effect on final accuracy for otherwise stable models.

- C36 pretraining accuracy changed by at most 0.18 percentage points.
- C36 fine-tuning accuracy changed by −0.11 to +1.19 points.
- The notable exception is previously unstable models, where measured-ReLU pretraining can substantially improve FT convergence.

3. We also found that stage distribution matters, not only total depth: among equally deep C32 models, 11:11 fine-tuned successfully while 12:10 and 10:12 did not.

#### Detail

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

## Active toggle non-idealities: FT and Level-3 evaluation

This table is the active-only audit for the pinned commands below. “Same” means
the same source, sampling granularity, and resampling lifetime as the FT
column. A *forward* is one full-model batch forward; a *validation pass* or
*trial* is one traversal of the evaluation dataset; a *spin/site* is one
`(channel, row, column)` destination, shared across batch items.

| Non-ideality | Fine-tuning | Final 45-corner Level-3 evaluation | Evaluation during FT |
|---|---|---|---|
| CiFAIR sensor noise | `MyNoiseCIFARDataset` generates noise when each image is read; a new realization is obtained on later reads. | Same. The data RNG is explicitly reseeded with `data_seed + trial`, producing reproducibility| Same as FT. |
| Student-input quantization | Endpoint-inclusive 12-bit quantization after sensor noise: clamp each input element to `[0,1]`, round to one of 4096 codes. | Same. | Same. |
| PC FF/FB weight quantization and scale quantization | Symmetric 5-bit QAT for every PC layer's FF and FB tensor (`q_hi=15`). One scale is computed per tensor and quantized with `WEIGHT_QUANT_FACTOR_BITS=1`; recomputed before each forward as weights change. | Load the baked 5-bit FF/FB weights and their saved per-tensor scales. Level 3 realizes every integer weight as up to 15 pulse slices. | Same as FT. |
| Analog final classifier | One-shot physical FF MVM with the same 5-bit symmetric QAT, scale-factor quantization, nonlinear-coupler, spin-variation, current-noise, and fixed-timing controls as the physical FF path. The pinned classifier has no bias. | One pulse-expanded FF MVM using per-coupler corner curves and the trial's spin/noise/DTC realization; it is not spatially unrolled because it is already a dense MVM. | Dense Level-2 classifier with validation-pass hardware sampling lifetime. |
| Measured ReLU | Fixed 0906 `tt_25_1.csv`, MC18 curve for every measured-activation module and every spin, for the entire FT job. Piecewise-linear interpolation. The V1 output offset is 0.600 V before coordinate scaling, and input/output are rail-clamped. Also support random_per_forward in training, with per_spin and per_layer granularity. | Uses the selected PVT corner's 0906 CSV. `per_spin` granularity. V0/V1/V2 subtract 0.588/0.600/0.612 V, respectively. | Same as FT. |
| Nonlinear coupler R(V) | Exact empirical_with_replacement `coupler_full_range` conductance curves from the configured all-corner training bank for different layers, but sample w/o replacement within a layer. Per-tensor granularity. The pair is redrawn every training forward. | Exact empirical_with_replacement from the selected PVT corner. Per-coupler granularity. Both FF and FB are unrolled. | Per-tensor granularity sample from all-corner bank, fixed for the whole pass. |
| Measured average pooling | Sample from `coupler_full_range` all-corner training bank. One empirical curve per output channel and pooling window, shared by all inputs in that window and all batch items; redrawn each training forward. This applies to the intermediate average pool and GAP. | Same per-window granularity, but curves come only from the selected PVT corner and stay fixed for the whole trial. Intermediate pooling and GAP have separate assignments. | Fresh assignments are drawn at the start of the validation pass and fixed for the whole pass. |
| Spin variation | Mean 1.0, standard deviation 0.10. Per-spin sampled and redrawn every training forward. | Uses the selected PVT corner's characterized mean and standard deviation. Per-spin sampled, but factors remain fixed for the whole trial and are redrawn next trial. | Per-spin sampled and fixed for the whole pass. |
| Fast summing-current noise | ASD `0.6e-12 A/sqrt(Hz)`. Sampled a (B, C, H, W) noise tensor from Gaussian distribution per toggle y or z update step. | The selected corner's temperature rescales the ASD by `sqrt(T/298.15)`. Sampled a (B, C, H, W) noise tensor from Gaussian distribution per pulse slice. | Same as FT. |
| Fast coupler-current noise | ASD `0.6e-12 A/sqrt(Hz)`. Sampled a (B, C, H, W) noise tensor per toggle y or z update step from a Gaussian distribution whose variance is derived from the sum of incoming quantized pulse levels. | Temperature-scaled as above. Sampled a (B, C, H, W) noise tensor per pulse slice from a Gaussian distribution whose variance is determined by the active physical couplers, including fractional DTC duty. | Same as FT. |
| DTC fixed pulse-width PVT variation | Inactive in Level-2 FT. | Active. The selected PVT corner supplies pulse-width mean/std relative to the minimum pulse width. One fixed width sample is drawn per dense FF/FB kernel coefficient; spatially unrolled copies of that coefficient share it. It is fixed over the whole trial. Fixed leading-edge variation is configured as zero. | Same as FT. |
| DTC edge jitter | Inactive in Level-2 FT. | Active with leading- and falling-edge std 0.005 in minimum-pulse-slice units. One sample per dense FF/FB kernel coefficient (shared by its spatial copies), redrawn for each y/z stage invocation. | Same. |
| Voltage-rail saturation | Every Level-2 toggle y/z stage update is clamped to `[-V_DD,+V_DD]`; measured-ReLU coordinates and output are also clamped. | Same rails, but y/z state is clamped after every pulse slice. | Same as FT. |

The physical operating point common to these paths is `V_DD=0.5 V`, five
toggle cycles, nominal `y_time=z_time=10 ns`, `R=50 kohm`, and `C=500 fF`.
Level 3 additionally resolves each 5-bit stage into 15 pulse slices. These are
operating/model settings rather than extra rows of sampled non-ideality.

Supported but inactive in this reference—and therefore deliberately excluded
from the table—are slow summing/coupler offsets, generic/differential weight
mismatch, output ENOB, and the legacy thermal
noise path. Gaussian curve synthesis is also inactive: the pinned evaluation
requests empirical sampling with replacement.

Source trail checked for this audit:

- FT launch: [Search configuration](../launch_scripts/slurm_search_config.sh)
  -> [KD/CRD then FT](../launch_scripts/run_kdcrd_then_ft.sbatch) ->
  [training entry point](../train_ode_cifar.py) -> [trainer](../trainer.py).
- Final evaluation: [MC45 scheduler](../launch_scripts/slurm_run_mc45_toggle_ablation.sh)
  -> [worker launcher](../launch_scripts/run_mc45_toggle_ablation.sh) ->
  [ablation driver](../scripts/run_toggle_nonideality_ablation.py) ->
  [inference](../ode_inference.py).
- Implementations: [toggle dynamics and wrappers](../ode_pc.py),
  [FF/FB unrolling](../validation.py),
  [measured ReLU](../measured_activation.py),
  [measured pooling](../measured_pooling.py), and
  [input quantization](../input_preprocessing.py).

## Main-model SLURM training: measured-ReLU pretrain then corner-coupled FT

Run from ScAN-PCN with a fresh shell, or clear stale experiment overrides first.
In particular, inherited ACTIVATION_CURVE_PATH or ACTIVATION_CORNER can override
the intended typical curve. Keep the normal environment/PATH; local dataset
initialization requires `uv` as well as scanbase Python.

For CiFAIR-10 launches derived from this toggle pipeline, use five warmup epochs
in both stages by setting `WARMUP_PRETRAIN=5` and `WARMUP_FT=5`.

```bash
# Reference architecture:
#   C24->48->96, stage depths 4:5:4, pooling before each expansion.
#   Final linear layer is digital with 0.5-LSB noise and no bias.
#   Pretraining uses the fixed measured 0906 MC18 ReLU.
#   FT samples one hardware corner per forward; nonlinear-R, measured pooling,
#   and random-per-forward per-spin measured ReLUs all use that corner.
#   All ReLUs, including the final ReLU, use measured 0906 curves.
#
# Architecture alternatives:
#
# C36->72 two-stage, stage depths 8:10, pooling before expansion:
#   EXP_PREFIX=...C36_72_8_10...
#   SEARCH_ARCH=two_stage_fixed
#   PCN_CHAN_0_LIST=36
#   PCN_NUM_LAYERS_LIST="8:10"
#   TWO_STAGE_POOL_POSITION=before_expansion
#   NUM_COMB_PER_NUM_LAYER=1
#
# C32->64 with more layers:
#   EXP_PREFIX=...twoStage_C32_64_poolPreExp_pullbackDirect...
#   PCN_CHAN_0_LIST=32
#   PCN_NUM_LAYERS_LIST="12:10"       # or "11:11", "10:12", or all three
#   WARMUP_PRETRAIN=5                 # recommended, otherwise pretrain might fail
#
# C56 one-stage with one intermediate pool, testing two pool positions:
#   SEARCH_ARCH=one_stage_fixed
#   PCN_CHAN_0_LIST=56
#   PCN_NUM_LAYERS_LIST=18
#   ONE_STAGE_POOL_MODE=search # Or no_pool
#   NUM_COMB_PER_NUM_LAYER=2
#
# Use FINAL_HEAD_TYPE="digital" or "analog"
# FINAL_ADC_NOISE_LSB=0.5 for digital head noise

mkdir -p logs/scheduler_slurm

(
  export TOGGLE_MODE=odexinit \
         mode=ft_and_eval \
         TASK=cifar100 \
         IMG_TYPE=CiFAIR \
         EXP_PREFIX=C100_C96L16_cc_randSpin \
         PRETRAIN_SAVE_PATH=saved_ckpt_runs/C100_C96_454_16L_allReLU_digital_biasfalse_iq12_toggle_odexinit \
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
         ENOB=none \
         UNITLESS_MEASURED_PULLBACK_MODE=direct \
         MEASURED_ACTIVATION_SCOPE=all \
         PRETRAIN_ACTIVATION_CORNER_MODE=fixed \
         ACTIVATION_CORNER_MODE=random_per_forward \
         ACTIVATION_RANDOM_CURVE_SHARING=per_spin \
         FT_CORNER_COUPLED_SAMPLING=true \
         SEARCH_ARCH=default \
         PCN_CHAN_0_LIST=24 \
         PCN_NUM_LAYERS_LIST=16 \
         FINAL_HEAD_TYPE="digital" \
         FINAL_ADC_NOISE_LSB=0.5 \
         LINEAR_BIAS=false \
         NUM_COMB_PER_NUM_LAYER=2 \
         COMB_SEL_SET=2

  source ./launch_scripts/slurm_search_config.sh
) > logs/scheduler_slurm/cifair100_C96L16_cc_randSpin.log 2>&1 < /dev/null &

echo "C96L16 corner-coupled randSpin scheduler PID: $!"
```

`UNITLESS_MEASURED_PULLBACK_MODE=direct` automatically enables measured
activation during pretraining; no separate `ENABLE_PRETRAIN_MEASURED_ACTIVATION` export is needed.
Pretraining selects fixed `0906_RELU_Voltage` MC18. During FT, every forward
samples one corner shared by nonlinear-R, measured pooling, and measured ReLU;
the ReLU then assigns curves from that corner per spin. Training-time validation
uses the same random-per-forward corner-coupled mechanism.

Classifier head type also affects pretraining: with this fixed timing, analog
pretraining applies a head gain of `T_y/(RC)=0.4`, while digital pretraining does
not. Reusing an analog-pretrained checkpoint for digital FT loads its weights
correctly but introduces a `2.5x` classifier-logit scale change at the FT boundary.

The automatic post-FT evaluation pins two complementary references with 5 trials
each: FS_V0_T0 as the adverse corner and FS_V2_T1 as the average corner. Reporting
both avoids using the previously pinned FS_V2_T1 result alone as an overly
optimistic representation of corner robustness.

### Provisional severe corner-interaction diagnostic (2026-10-07)

The previous rerun C36→72, 8:10 checkpoint showed a severe interaction between
the selected corner's measured ReLU and nonlinear-R/pooling curves. These are
controlled **four-batch (512-image), one-seed diagnostics**, not completed
10-trial corner accuracies:

| FS_V2_T1 base with the following FS_V0_T0 substitutions | Correct / 512 | Accuracy | Drop from base |
| --- | ---: | ---: | ---: |
| None | 307 | 59.96% | — |
| Nonlinear-R and measured pooling only | 298 | 58.20% | 1.76 points |
| Measured ReLU only | 268 | 52.34% | 7.62 points |
| Nonlinear-R, measured pooling, and measured ReLU | 193 | 37.70% | 22.27 points |
| Complete FS_V0_T0 corner | 174 | 33.98% | 25.98 points |

The two isolated drops would predict 50.59% if they combined additively, but
the measured result is 37.70%, an additional **12.89-point negative
interaction**. Therefore the small isolated nonlinear-R drop does not establish
that nonlinear-R is harmless: its effect becomes large in combination with the
FS_V0_T0 0906 measured-ReLU curve. The combined substitution explains most,
but not all, of the observed four-batch FS_V2_T1-to-FS_V0_T0 gap.

The nonlinear-R substitution was verified through the runtime path. All 100
curves were loaded from `coupler_full_range/fs_-20_0.csv`; empirical curves were
assigned with replacement to every stored physical edge in all 20 expanded FF
and 20 expanded FB matrices (97,176,960 assignments, with no dense PC
convolution remaining). The intermediate pool and GAP used the same FS_V0_T0
bank, and the analog classifier inherited that nonlinear-R package. The ReLU
substitution used `0906_RELU_Voltage/fs_-20_0.csv`. Spin variation, DTC, current
noise, timing, checkpoint, data order, and seed remained at FS_V2_T1 settings in
the controlled substitutions. A full multi-trial study is still required before
assigning a population-level effect size.

## Main-model 45-corner SLURM evaluation

The pinned evaluation checkpoint is the **last** checkpoint of the
C24→48→96, 4:5:4 `randSpin` model. It has a digital final classifier with
0.5-LSB noise, no classifier bias, and all ReLUs measured, including the final
ReLU. These settings are restored from checkpoint metadata when no overrides
are exported; no extra head/activation arguments are needed.

The physical evaluation configuration is unchanged. The evaluator reconstructs
the architecture from `MODEL_NAME`; only `MODEL_NAME`, `MODEL_DIR`, output root,
and scheduler-log filename change when the reference model changes. Verify
`MODEL_DIR` against the path printed by training because identical model names
do not imply identical checkpoints. `UNITLESS_MEASURED_PULLBACK_MODE` is a
pretraining-only transformation and must not be added here.

```bash
MODEL_NAME=TIMMQAT5bNoneaNT0p0mulTIMMPCNetNoBatchNorm_PCConvReLU6_0.0eps_ToggleODEXInitFFFB_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_C100_3K1S96C_0.25Dropout_16Layers4l5l4_2Pool_srrlDistill_a0p3_t2p0_CiFAIR_1REP

mkdir -p logs/scheduler_slurm

(
  export SBATCH_TIMELIMIT=12:00:00 \
         N_TRIALS=10 \
         N_SERVERS=4 \
         OUTPUT_ROOT=results/C100_C96L16_cc_randSpin_last_default45 \
         MODEL_NAME="${MODEL_NAME}" \
         MODEL_DIR=saved_ckpt_runs/C100_C96L16_cc_randSpin_iq12_toggle_odexinit \
         CKPT=last \
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
) > logs/scheduler_slurm/slurm_mc45_C100_C96L16_cc_randSpin_last.log 2>&1 < /dev/null &

echo "randSpin last-checkpoint MC45 scheduler PID: $!"
```

Optional total summing-current limits: add this export inside the subshell
before sourcing the launcher (use a path accessible on the evaluation machine):

```bash
export RHS_CURRENT_SUMMARY=hardware_data/summing_current_limit/current_limits_stage_pooled_p99.md
```

That is the only additional setting required for clamping. This explicit table
already contains the 99% bounds: no percentile setting is needed. Pooled rows
apply the same lower/upper bounds independently to FF and FB; the classifier
keeps its own row. With an original percentile summary instead, select its
columns using `RHS_CURRENT_BOUND_PERCENTILE=95` or `99`. Unset/empty
`RHS_CURRENT_SUMMARY` disables the clamp. `RHS_CURRENT_AUDIT_PATH` is optional
and only saves mapping/clipping-count reports. Use distinct output/log paths
for clamped runs. For the previous two-corner comparison, additionally set
`CORNER_IDS="FS_V2_T1 SF_V1_T1"`, `N_TRIALS=1`, and `BASE_SEED=20260723`.
The committed stage-p99 table records its source provenance: it was derived
from the older non-rerun checkpoint's FS_V2_T1 training-current statistics.
Using it with the previous C36→72 rerun checkpoint was therefore an intentional
cross-checkpoint hardware-limit experiment, not a same-checkpoint limit fit.

Below is the local two-corner subset of the pinned MC45 run. It uses the same
checkpoint, ten trials, base seed, evaluator, and physical settings. Because
corner seeds use each corner's index in the complete 45-corner catalog, its
`FS_V0_T0` and `FS_V2_T1` rows use the corresponding full-run logic. Omit
`CKPT=last` to retain the launcher's default `best` checkpoint.

```bash
conda activate scanbase
cd /home/rongzeng/_workspce_old/repos/pcn/collaboration/ScAN-PCN
mkdir -p logs/local_runs

(
  export MODEL_NAME=TIMMQAT5bNoneaNT0p0mulTIMMPCNetNoBatchNorm_PCConvReLU6_0.0eps_ToggleODEXInitFFFB_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_C100_3K1S96C_0.25Dropout_16Layers4l5l4_2Pool_srrlDistill_a0p3_t2p0_CiFAIR_1REP \
         MODEL_DIR=saved_ckpt_runs/C100_C96L16_cc_randSpin_iq12_toggle_odexinit \
         CKPT=last \
         N_TRIALS=10 \
         BASE_SEED=20260723 \
         CORNER_IDS="FS_V0_T0 FS_V2_T1" \
         OUTPUT_DIR=results/C100_C96L16_cc_randSpin_last_FS_V0_T0_FS_V2_T1 \
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
         ENOB=none \
         IS_SLURM=0 \
         DATALOADER_NUM_WORKERS=2

  nohup bash ./launch_scripts/run_mc45_toggle_ablation.sh
) > logs/local_runs/C100_C96L16_cc_randSpin_last_FS_V0_T0_FS_V2_T1.log 2>&1 < /dev/null &

echo "Evaluation PID: $!"
```


Former 96-channel reference, **before the FF expansion fix**: 2-trial
"FS_V2_T1" run -> 58.73%
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

### Averaged-toggle FT input-correction fusion (opt-in; production parity not established)

`FUSE_TOGGLE_NONLINEAR_R_INPUT=true` (CLI
`--fuse_toggle_nonlinear_R_input true`) enables the training-capable shared
correction kernel from `tc_shared_correction.py` in
`ToggleAveragedPhysicalFFFB._averaged_nonlinear_R_source`. **Default: false**,
because tight production-precision parity is not established. This fuses
only projection, native-grid resistance interpolation and `x * R / R(x)`;
the following convolution and its trainable weights remain ordinary PyTorch.
It is not expanded-inference convolution fusion and does not use grouped
15-level convolutions or `toggle_edge_inference.py`.

The scheduler explicitly forwards the control to the worker environment; the
training parser forwards it through the QAT wrapper to the averaged block.
The standalone mixed-FT launcher also forwards it explicitly. Existing TC
correction callers retain their defaults. Expanded toggle inference has no
training curve package and does not enter this path. No CNN launcher was changed.

Supported fast path: CUDA FP32, native-grid piecewise-linear positive resistance,
and either no query projection or an ordinary, non-inplace symmetric
`nn.Hardtanh`. Hardtanh's **zero derivative at rail endpoints** is preserved;
TC's inclusive clamp-endpoint derivative remains unchanged. No resistance floor
is added. CPU, other dtypes/projections and gradient-bearing constants return
to the **literal original toggle arithmetic**. Curve data is trusted, as in
the reference path; the helper does not validate it again on GPU. First-order
autograd only, as in the reused TC kernel.

FF/FB indices are still chosen by the original sampler, including its common
corner restriction and without-replacement pair selection. Selected constants
are cached per realization/device/dtype and cleared by the existing realization
reset. Neither correction forward nor backward draws random values. The OFF
path does not call the correction helper.

Validation on 2026-10-23, RTX 4090:

| Check | Result |
|---|---|
| Operator CUDA FP32 output, literal reference | Passed `atol=1e-5`, `rtol=1e-4` |
| Operator input/ordinary-convolution weight gradients | Passed the same tolerances |
| Five selected actual full-range bank curves: knots, interval midpoints, rails and outside rails | Output max absolute difference 0; input-gradient max absolute difference `1.1920929e-7`, max relative L2 `5.5598768e-8` |
| RHS checkpoint replay, FF/FB assignments, global/local RNG unchanged | Passed operator-level tests; not complete-model replay |
| CPU fallback; unsupported projection/dtype/gradient-bearing-constant fallbacks; TC/CNN shared callers | Passed regression tests |
| Relevant regression suite | 55 passed, including CUDA tests |

The current pinned `mode=ft_and_eval` command explicitly selects the following
pretraining root. Its `CiFAIR_1REP` best checkpoint is available locally:

```text
saved_ckpt_runs/C100_C96_454_16L_allReLU_digital_biasfalse_iq12_toggle_odexinit/
  <TIMMPCNet...C100_3K1S96C...16Layers4l5l4_2Pool...CiFAIR_<rep>REP>/
  <same_model_name>_best_ckpt.pth
```

Provenance: pinned `PRETRAIN_SAVE_PATH` → `slurm_search_config.sh` →
`run_kdcrd_then_ft.sbatch` → `finetune_one_combo_model` loads its **best**
checkpoint (`FINAL_EVAL_ONLY` is not set). This is the job-2090063 pretrained
source, not a QAT checkpoint. The operator diagnostic now resolves the source
from this pinned command instead of hardcoding an obsolete run root.

Existing complete-model checks (not rerun for the top-N fixes) are recorded in
`results/toggle_input_correction_fusion/guard_removed_20261024/summary.md`.
Batch 128, real SRRL teacher, identical assignments/RNG and suppressed optimizer
updates were used for parity. No layers were checkpointed in this recipe.

| Complete pinned-model deterministic-FP32 parity | OFF/OFF max absolute error | OFF/ON max absolute error |
|---|---|---|
| Final logits | 0 | 0 |
| Post-input-quantization gradients | 0 | 2.42e-8 |
| All model-parameter gradients | 0 | 5.96e-8 |
| Distillation auxiliary gradients | 0 | 0 |
| Prediction changes | 0/128 | 0/128 |

Predeclared acceptance: deterministic FP32 logits `atol=1e-5`, `rtol=1e-4`;
each full-model gradient category must have **both** maximum absolute error
≤`1e-3` and relative L2 error ≤`1e-3` (0.1%). Retain the stricter elementwise
`atol=1e-5`, `rtol=1e-4` result; require identical gradient keys/None status,
finite values and explicitly matching hardware assignments. Prediction equality
alone is insufficient. These deterministic-FP32 checks pass, including gradient
keys/None status. Raw input gradients are zero because of the unchanged input
quantizer; the table reports the nonzero post-quantization gradients separately.
Production precision was also tested: both OFF/OFF and OFF/ON fail the fixed
parity criteria despite matching assignments/RNG and predictions. This does not
isolate a fusion regression; keep fusion opt-in/default OFF.

| Production FT benchmark (batch 128) | Median step time | Peak allocated / reserved |
|---|---|---|
| OFF | 306.453 ms | 2.7856 / 4.3027 GiB |
| ON | 356.199 ms | 1.9185 / 3.7148 GiB |

These historical timings use sequential fresh processes, two warm-ups and five
measured real optimizer steps; data loading is excluded and memory includes
warm-up peaks. Fusion was slower in this short run but used less memory. There
is no general speedup or production-precision equivalence claim. The separate
full-model diagnostic `benchmark_toggle_input_correction.py` captures the actual
pinned command and refuses an existing output directory.
It derives conclusions from the saved validation records and exits nonzero if
either precision check fails, after retaining the reports; this also applies to
`--report-only`. The known production-precision failure is not silently accepted.

Persistent operator/checkpoint report generator and reproduction:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python diagnostic_scripts/check_toggle_input_correction.py \
  --output results/toggle_input_correction_fusion/operator_report.json
python -m pytest -q tests/test_toggle_input_correction.py \
  tests/test_tc_shared_correction.py tests/test_tc_empirical_training_optimizations.py \
  tests/test_tc_dense_training.py tests/test_feedforward_corner_coupling.py
```

The report generator refuses to overwrite an existing report; choose a new
output filename when repeating it. Results and regression log:
`results/toggle_input_correction_fusion/operator_report.json` and
`results/toggle_input_correction_fusion/regression.log`. It does not launch a
training job. All existing production jobs/checkpoints are untouched.

The two subsequent optional stage-reuse experiments were removed. The retained
fusion was reverified against the original unoptimized FF-fixed path on 1,280
inputs: all predictions matched, maximum logit difference 1.0014e-5, and 4.49x
forward speedup. Details: `results/toggle_reuse_removal/summary.md`.

In that historical matched replay, ReLU and coupler assignments were sampled
from the selected corner and held fixed for each dataset trial, then resampled
for the next trial. Pooling used the same selected coupler source. The checkpoint
loaded was `_best_ckpt.pth` (baked quantized weights), not
`_full_param_best_ckpt.pth`.

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

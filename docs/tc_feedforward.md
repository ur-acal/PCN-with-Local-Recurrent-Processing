# Resistance-coded feedforward CNN

Opt in to the existing local stage/combined scripts or the feedforward Slurm
submitter with `TC_FEEDFORWARD=true`. Set `ONE_SHOT_CONV=false` (default) for
the existing TC PCN solver, or `ONE_SHOT_CONV=true` for one endpoint update.
Direct Python arguments are `--tc_feedforward true --one_shot_conv false`.
Existing model, teacher, dataset and checkpoint arguments still apply; no job
is submitted by importing the implementation or running the numerical checks.

The existing toggle block/wrapper classes and PCN blocks
are not edited by the CNN implementation. Launcher defaults and failure handling
are aligned as described below. `TCPhysicalBasicBlock` subclasses the averaged CNN block and
`TCFeedForwardPhysicalWrapper` subclasses its averaged wrapper. The converter
selects the new subclasses only when `tc_options` is supplied. TC reuses the
same forward ordering, spin sampling, QAT machinery, activation/pooling helpers,
trainer and validator. It does not use pulse slicing or integer pulse weights.

## Physical contract

### Gaussian resistance floor (TC and toggle, CNN and PCN)

`RESISTANCE_FLOOR_OHMS` / `--resistance_floor_ohms` optionally sets a
pointwise physical lower bound in ohms. It is unset by default: TC couplers
retain their existing 1e-6-ohm floor; generic Gaussian couplers and pooling
retain their existing native-unit numerical guard. An explicit override
also reaches Gaussian pooling, analog heads and CPU-offloaded expanded curves.
Conductance draws retain their positive numerical guard before inversion;
the physical floor is then applied in ohms. Invalid/nonrepresentable floor
values fail explicitly. No additional draws or resampling are introduced.

Empirical measured curves bypass this guard, including pinned toggle FT and
MC45 commands. Exact resume checks the saved effective floor. Historical
checkpoint metadata is read compatibly, but there is no TC-prefixed CLI alias.

The 26,791.333455-ohm measured minimum belongs to the stronger 50-kOhm
bank experiment: do not apply it to the standard 10-kOhm recipe, where it
would clip legitimate resistance values. TC CNN now accepts other positive
nominal resistances while retaining the existing derived `R_max = 15R`.

### Optional corner-coupled fine-tuning (TC CNN and PCN)

Set `FT_CORNER_COUPLED_SAMPLING=true` (Python: `--ft_corner_coupled_sampling true`);
default is OFF. One MC45 corner is selected per training forward for the whole
network. Coupler and measured-pooling Gaussian draws use covariance fitted from
that corner's 100 resistance curves, retaining the existing nominal mean curves.
Random-per-forward measured ReLU uses the same corner's bank, preserving its
per-layer/per-spin sharing. Fixed ReLU is unchanged. Spin-variation statistics
and current-noise strength are **not** corner-conditioned by this flag.

The 45 covariance factors are shared across compatible layers (about 0.46 MiB
in FP32), fitted once, and transferred once per device/dtype. Solver backward
replay retains the forward's sampled curves. Validation/evaluation remains
uncoupled. The source must contain all 45 labelled corners with 100 curves each;
unlabelled synthetic banks and expanded/empirical TC training are unsupported.
This does not enable the separate stronger-covariance experiment or change the
existing nonpositive-resistance guard.

### Mapping

- Initial support: bias-free ordinary Conv2d WRNs, including stride-2 and
  stride-1-plus-pooling variants; the existing BN support is retained, although
  the recommended controlled architecture is BN-free and convolution-bias-free.
- R=10 kOhm, R_max=150 kOhm, 5-bit normalized programmed conductance, C=49 fF,
  v_dd=0.1 V by default. Other resistance code grids fail explicitly rather
  than silently using an incompatible mapping.
- Each convolution holds its input, curves and spin gains fixed, starts its
  output state at zero, and charges its own output capacitor.
- Current uses TC shared-curve correction in dense FT and independent
  code-aware sampled curves per unrolled coupler in physical evaluation.
- Both convolutions use PCN's FF diffusion coefficients, with separate summing
  and coupler random streams; no algebraic one-state FB noise is added.
- Nominal conductance determines noise variance; zero weights and padding
  contribute no coupler noise. Spin gain multiplies drift, not diffusion.
- Measured activation defaults to fixed 0906 TT MC18 during FT; the pinned
  CNN recipe instead resamples the full 0906 bank per forward and per spin.
  Expanded evaluation assigns curves per spin from that bank, fixed per trial.
  Scope `all` replaces the final activation too; `pc_only` leaves it ideal.
  Measured pooling includes GAP and uses the 10 kOhm mean plus the TC covariance.
- FT nonlinear-R/spin draws and random-mode ReLU assignments are renewed per
  forward; evaluation static samples are fixed per trial. Fast current noise
  is sampled per integration interval. Seeds increment between evaluation trials.
- No scale-factor quantization, ENOB, DTC, extra scalar mismatch or slow noise.

## Integration and scaling

The physical increment is current/(capacitance) times duration plus additive
diffusion. `one_shot_conv=false` calls the exact `TorchDiffEqPack.odesolver`
entry point used by TC PCN with `TCNoiseLifecycle`, additive noise, projection,
dopri5 and rtol=atol=1e-6 by default. The fixed CNN current is
computed once and reused by the RHS; PCN's current changes with its state.

`one_shot_conv=true` explicitly draws the integrated noise for the whole
duration and applies one final projection. It uses the same coefficients and
static defects, not the same complete noise trajectory as a multi-step solve.

Retain the existing CNN scale compensation: with QAT scale s, physical duration
is RC/s in derived mode and T_base/s in fixed mode (with the existing conv2
ratio). T_base is therefore NOT necessarily the actual integration time.
Physical state scaling q and inverse output scaling remain unchanged. FT
recipe scaling is inherited; no new optimizer behavior is introduced. RGB
teacher selection defaults are listed below.
Evaluation records resolved options and actual per-convolution integration
durations in `tc_trial_metadata.json` under each trial's result directory.

Optional physical timing variation uses the shared
[`COMPUTE_TIME_FIXED_STD` / `COMPUTE_TIME_NOISE_STD` controls](current_pcn_reference.md#optional-physical-compute-time-variation-tc-pcn-and-cnn).
Both default to zero. Each CNN convolution gets its own fixed offset and
per-forward jitter; its integration and diffusion use the same perturbed
duration. This does not change PCN timing scope: one horizon per ODE block.
`TIMING_NOISE_SCOPE=per_sample` selects independent jitter per image, using
equivalent drift/diffusion rescaling on a common nominal solver interval.
The default `per_batch` retains the previous shared-duration implementation.
BN recalibration and its subsequent test use the same configured scope.

### Fixed-count projected Euler (FT and evaluation)

Set `TC_METHOD=euler TC_NUM_STEPS=10 ONE_SHOT_CONV=false` through the existing
local or Slurm launch path. Python equivalents are `--tc_method euler
--tc_num_steps 10 --one_shot_conv false`. The default remains Dopri5 with no
step count. A positive integer count selects exactly that many updates for
each convolution; its timestep is recomputed as `duration / tc_num_steps`
after every live-QAT scale refresh. Do not also set `TC_STEP_SIZE`.

This reuses the production `ProjEuler` solver, not a diagnostic loop. Each
update is `clamp(v + dt * drift + sqrt(dt) * diffusion)`, with independent
summing/coupler noise increments and one final projection. The shared
`ProjEuler.step` no longer projects before noise: the common integrator owns
the projection, as for Dopri5. This intentionally changes noisy Euler near
saturation, including non-TC callers; noiseless integrated Euler and Dopri5
behavior are unchanged. Run configuration/recovery records retain the solver
settings, and exact recovery rejects a changed step count or solver.

## Numerical checks and fairness limits

Run:

```bash
python -m unittest discover -s tests -p 'test_tc_feedforward.py'
python scripts/check_tc_feedforward_numerics.py
```

CPU float64 reference (the solver has float32 time bookkeeping):

| Timing | Max output difference, one-shot vs Dopri5 | Max input-gradient difference | Max weight-gradient difference |
|---|---:|---:|---:|
| Derived | 5.38e-11 | 2.70e-8 | 1.31e-9 |
| Fixed | 2.14e-10 | 1.07e-7 | 4.97e-9 |

With 20,000 unsaturated samples, both modes' noise variances differ from the
same analytic variance by less than 1.3%. A one-step Euler solve reproduces the
explicit update with the same random draws up to time-rounding error. Tests
also compare coefficients directly with PCN FF, check real measured components
in a toy WRN forward/backward, and verify unrolled currents against dense ones
for ideal curves. CUDA QAT/noise/shared-curve backward is checked when available.

Repeated clamping with noise is demonstrably NOT identical to one final clamp.
Use solver mode for the primary PCN comparison; label one-shot as a numerical
ablation. Matching hardware coefficients does not prove timestep convergence
or equal latency/energy: report physical durations and test step sensitivity
for BOTH architectures before making such claims. No trained-model accuracy
or full training convergence is established by these numerical checks.

The existing no-padding CSR builder has a multi-channel weight-ordering issue.
TC uses the existing correct trimmed-CSR builder through an optional validator
hook, with a separate cache namespace and no legacy-cache migration. PCN and
toggle retain their prior default path; this finding is not silently patched
into their experiments. Standard padded 3x3 convolutions use the trimmed path
already, so this issue does not establish a defect in those PCN runs.

## TC launcher alignment contract

Scientific reference: `papers/hardware-native-neural-ode/shared/true_continuous_nonidealities.md`
in the parent collaboration workspace. Common opt-in defaults now live in
`launch_scripts/tc_hardware_defaults.sh`, sourced by **both** TC argument builders
before any legacy Slurm defaults. Explicit false flags remain supported for
ablations. Unitless pretraining does not enable these hardware defects.

| Component | PCN and CNN FT default | PCN and CNN evaluation default |
|---|---|---|
| Quantization | Signed 5-bit programmed conductance, zero=open circuit | Same |
| Coupler means | `hardware_data/res_vs_vin_10k_150k.csv` | Same |
| Covariance | Common absolute resistance covariance from all `hardware_data/mc_45_corners/CU_4500_r_vs_vin.csv` curves | Same |
| Nonlinear-R sampling | Histogram-selected resistance level, one Gaussian curve per convolution tensor/layer/forward, held over solve | Independent per nonzero unrolled coupler, fixed per trial |
| Activation | Generic default: fixed `0906_RELU_Voltage/tt_25_1.csv`, MC18, shared. Pinned CNN recipe: full 0906 bank, random per forward and per spin. | Full `0906_RELU_Voltage` bank, independently per spin/layer, fixed per trial |
| Pooling, including GAP | 10-kOhm mean from the coupler means file and same absolute covariance; independent per channel/window, per forward | Same distribution/granularity, fixed per trial |
| Spin/gain variation | N(1,0.1^2), independent per spin/branch/layer, held over solve | Same, fixed per trial |
| Integrated summing noise | 0.6 pA/sqrt(Hz) at 50kOhm, scaled by sqrt(50k/R) | Same |
| Integrated coupler noise | Same ASD reference, multiplied by sqrt(sum of nominal programmed conductances); zero/padding excluded | Same |
| Physical operating point | R=10kOhm, R_max=150kOhm, C=49fF, rails +/-0.1V, one_over_q=1 | Same |
| Exclusions | No scalar/diff mismatch, ENOB, scale-factor quantization, DTC or slow offsets | Same |
| Numerical method | Dopri5, tolerance 1e-6; CNN one_shot_conv=false | Same |

These are generic hardware defaults. The pinned CNN recipe below explicitly
overrides FT activation sampling to random-per-forward/per-spin and
`one_over_q` to 32, and disables RGB normalization for student and teacher.

The shared FT row is the selected **method 2a** approximation linked at the top
of the scientific reference, not its more expensive exact 15-code family.
PCN selects it with `tc_conv_method=shared`; CNN's TC wrapper implements that
same sampler even though its inherited enabling flag is `nonlinear_R_train_mode=exact_curve`.
PCN uses `nonlinear_R_train_mode=none` to bypass the legacy implementation, not
to disable TC nonlinear-R. These differing flag names do not denote different
default curve models. Both use `TC_MEAN_TABLE`/`TC_COVARIANCE_TABLE` overrides.

One-state PCN also has algebraic FB current noise from the integrated
`hardware_data/coupler_asd_vs_freq.csv` spectrum, held across accepted-step
retries. CNN has no algebraic FB path: both convolutions receive the integrated
FF-style diffusion above. Two-state PCN uses capacitor diffusion on both states.
The final activation remains ideal only with scope `pc_only`. Fresh CNN
pretraining records scope `all`, which FT and expanded evaluation inherit;
the pinned full pipeline therefore replaces the final activation with measured
ReLU as well. Older checkpoints retain their saved scope (metadata-free
checkpoints fall back to `pc_only`). The default digital classifier remains ideal.
CNN residual topology and physical integration durations remain architecture-
specific; matching defaults is not a claim of equal energy/latency or dynamics.

Both standalone and search RGB training default to
`checkpoint/efficientnet_v2_l_cifar{10,100}_rgb_OldNoTimm_MatchDistill.pth`;
explicit teacher overrides and non-RGB teacher selection are preserved.
Pretraining LR stays PCN=0.01/CNN=0.1; FT LR stays 0.005 for both.

PCN's Slurm worker uses `pipefail` and checks phase exit statuses before moving
to FT/evaluation; missing checkpoint model names also fail visibly. With
`TC_FEEDFORWARD=true`, CNN's local combined launcher and Slurm worker now run
pretrain -> FT -> unrolled evaluation sequentially. They require the exact FT
`full_param_{best|last}_ckpt.pth` selected by the pipeline (default final-only
evaluation uses `full_param_last_ckpt.pth`), including inferred preprocessing
suffixes, not a fallback/pretraining checkpoint. Evaluation uses the full 0906 bank and
per-coupler sampling, with 10 trials by default (`N_TRIALS` overrides).
Results default to `results/${EXP_PREFIX}_tc_eval`, with `evaluation.log` and
per-trial accuracy/metadata. `RESULT_PATH` and `EVAL_LOG` allow explicit output
locations. Any failed stage or missing checkpoint fails the job and prevents
dependent stages; evaluation failures propagate through `tee`. Non-TC combined
jobs retain their existing pretrain -> FT behavior.
No training jobs were submitted to validate these wiring changes.

## Pinned RGB TC feedforward pipeline

This command launches unitless pretraining, TC fine-tuning, and expanded TC
evaluation for every listed CNN on CIFAR-10 and CIFAR-100. Pretraining retains
its existing augmentation default and ideal ReLU; only FT uses `rgb_mid` and
random-per-forward/per-spin measured ReLU from the full 0906 bank. Student
and teacher both receive raw, uncentered RGB. The teacher's `NoNorm` checkpoint
metadata determines its preprocessing. Expanded evaluation uses fixed per-spin
curve assignments per trial, not per-forward resampling. `one_over_q=32` is
explicit for these CNN runs, as in the successful ResNet-44 fine-tuning run;
the generic mapping default remains 1. Accepted-step reuse
is enabled for Dopri5 training, and the RHS-checkpoint portion is selected by
the automatic memory guard. For just ResNet-44/56 on both datasets, replace
the model list below with `models=(resnet44_cifar resnet56_cifar)`.
Student normalization is saved in checkpoints and inherited by FT/evaluation
when no override is supplied; metadata-free checkpoints retain normalized RGB.
The random activation setting uses independent layer/stage RNG streams and
resamples per-spin assignments on each full-network forward, including FT tests.
Fresh pipeline checkpoints use activation scope `all`: measured ReLU includes
the final activation during FT and expanded evaluation, but is disabled in
the pinned ideal-ReLU pretraining stage.

```bash
module swap slurm slurm/24.05.0.b1

cd /scratch/rzeng7/repos/PCN-with-Local-Recurrent-Processing
mkdir -p ./logs/scheduler_slurm ./logs/slurm_jobs

models=(
  wrn_16_2_cifar
  wrn_16_2_cifar_avgpool
  wrn_16_2_cifar_nobn
  wrn_16_2_cifar_nobn_avgpool
  wrn_16_2_cifar_nobn_no_bias
  wrn_16_2_cifar_nobn_no_bias_avgpool

  resnet44_cifar
  resnet44_cifar_avgpool
  resnet44_cifar_nobn
  resnet44_cifar_nobn_avgpool
  resnet44_cifar_nobn_no_bias
  resnet44_cifar_nobn_no_bias_avgpool

  resnet56_cifar
  resnet56_cifar_avgpool
  resnet56_cifar_nobn
  resnet56_cifar_nobn_avgpool
  resnet56_cifar_nobn_no_bias
  resnet56_cifar_nobn_no_bias_avgpool
)

for model in "${models[@]}"; do
  for task in cifar10 cifar100; do
    run_name="tc_rgb_${task}_${model}_q32_nonorm_rgbmid_rpf_pspin"

    (
      export mode=default \
             TC_FEEDFORWARD=true \
             ONE_SHOT_CONV=false \
             MODEL_NAME="${model}" \
             TASK="${task}" \
             IMG_TYPE=rgb \
             NORMALIZE_STUDENT_INPUT=false \
             CENTER_STUDENT_INPUT=false \
             EXP_PREFIX="${run_name}" \
             TEACHER_CKPT="./checkpoint/efficientnet_v2_l_${task}_rgb_OldNoTimm_MatchDistill_NoNorm.pth" \
             TEACHER_ARCH=efficientnet_v2_l \
             TEACHER_ARCH_SOURCE=torchvision \
             TOGGLE_ONE_OVER_Q=32 \
             FT_TIMM_AUG_LEVEL=rgb_mid \
             ACTIVATION_CORNER_MODE=random_per_forward \
             ACTIVATION_RANDOM_CURVE_SHARING=per_spin \
             REUSE_ACCEPTED_STEP_TRAINING=true \
             CHECKPOINT_ODE_RHS_TRAINING=true \
             CHECKPOINT_ODE_RHS_PORTION=auto

      source ./launch_scripts/slurm_search_feedforward_config.sh
    ) > "./logs/scheduler_slurm/${run_name}.log" 2>&1 < /dev/null &

    echo "${run_name}: scheduler PID $!"
  done
done
```

Regression coverage: `tests/test_tc_launch_alignment.py` captures actual CNN
Slurm-to-stage commands and compares real parsed FT/evaluation defaults against
both PCN state variants; checks RGB teachers, explicit ablations, and simulated
Python/phase failures. `tests/test_tc_cli.py` additionally captures actual PCN
Slurm worker FT/evaluation commands. Physical behavior is exercised separately
by the TC noise, dense-training, unrolled-inference and feedforward suites.
`tests/test_tc_pipeline.py` runs the actual submitter, worker, combined launcher
and stage scripts with fake training/evaluation processes: it verifies automatic
post-FT evaluation, checkpoint suffix resolution, matched PCN evaluation args,
trial-count overrides, failure propagation, and the unchanged non-TC stage count.

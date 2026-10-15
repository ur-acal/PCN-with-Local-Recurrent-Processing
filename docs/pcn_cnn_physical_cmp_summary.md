# PCN versus physical feedforward CNN: experiment summary

## Current focus: single-curve adaptation followed by full-bank evaluation

Goal: compare PCN and feedforward CNN adaptation to one circuit realization,
then accuracy retention across the complete circuit population. Repeat the
CiFAIR-100 study on CiFAIR-10 and extend CiFAIR-100 to ResNet-110. A flatter
curve is not necessarily a representative training curve.

### CiFAIR-100 results

All entries below are absolute top-1 percentages. FT uses 50 epochs and SRRL;
MC45 uses Level 3, expanded convolutions, the entire corresponding family,
45 corners and one trial per corner. Primary WRN results do **not** use BN
recalibration. Digital anchors: PCN 64.44%; narrow WRN 63.87%.

| Model | Family | Flattest TT curve, mcparamset | Flattest FT | Flattest MC45 | Representative TT curve, mcparamset | Representative FT | Representative MC45 |
|---|---|---:|---:|---:|---:|---:|---:|
| PCN | `coupler_monte` (original circuit) | 81 | 60.97 | 38.5458 | 73 | 60.89 | 38.1544 |
| PCN | `coupler_monte_v2` (improved circuit) | 59 | 61.73 | 27.0616 | 34 | 61.38 | 50.9271 |
| Narrow WRN c27 | `coupler_monte` | 81 | 60.44 | 22.1944 | 73 | 59.59 | 20.8229 |
| Narrow WRN c27 | `coupler_monte_v2` | 59 | 59.93 | 10.6609 | 34 | In progress | Pending |

Separate flattest-WRN MC45 controls with 5,120-sample BN recalibration reached
40.4618% (`coupler_monte`) and 46.5036% (`coupler_monte_v2`). Do not mix these
with the unrecalibrated primary results.

Both selections are restricted to `TT_V1_T1`. Flattest minimizes
`max_V |R(V)/R(0)-1|`. Representative minimizes the mean, across all 4,500
family curves, of RMS log-resistance distance on 50 voltage points spanning
approximately -0.096 to +0.1 V. Curve IDs are one-based `mcparamset` values.
The singleton CSV repeats the selected realization twice solely to satisfy
the distinct-index FF/FB sampler; the two curves are identical.

### Pinned commands and operating points

Source of truth: [integrated FT → MC45 launcher](../launch_scripts/run_cifair100_tt_singleton_ft_eval.sh).
It supports `STAGE=ft|eval|all`, dynamically supplied `TT_SINGLETON`, and
explicit `WRN_RECOVERY_CKPT` for epoch recovery. Never resume a completed
experiment from its latest checkpoint implicitly.

Representative-curve reproduction (four pipelines, sequential; **not a request
to relaunch existing jobs**):

```bash
cd /home/rongzeng/_workspce_old/repos/pcn/collaboration/ScAN-PCN
for kind in pcn wrn_c27; do
  for family in coupler_monte coupler_monte_v2; do
    case "${family}" in
      coupler_monte) singleton=coupler_monte_tt_25_1_mc73_singleton.csv ;;
      coupler_monte_v2) singleton=coupler_monte_v2_tt_25_1_mc34_singleton.csv ;;
    esac
    (
      unset WRN_RECOVERY_CKPT RUN_TAG
      export MODEL_KIND="${kind}" COUPLER_FAMILY="${family}" \
             TT_SINGLETON="${singleton}" WRN_ONE_OVER_Q=5 \
             STAGE=all N_TRIALS=1 CORNER_IDS="" \
             FUSE_MEASURED_ACTIVATION=true PREFLIGHT_ONLY=false REHEARSAL_ONLY=false
      bash ./launch_scripts/run_cifair100_tt_singleton_ft_eval.sh
    ) || break 2
  done
done
```

For the earlier flattest selections, substitute `mc81` for `mc73` and `mc59`
for `mc34`, and use `FUSE_MEASURED_ACTIVATION=false` to match those historical
launches. They used separate historical output names; the current launcher's
names include the singleton basename to avoid collisions.

| Setting | PCN | Narrow WRN c27 |
|---|---|---|
| Source | Original pretrained 64.44% checkpoint | Independent weights-divided-by-three, fixed-15-ns source; original PT 63.87% |
| Timing | Derived, five cycles | Fixed, 15 ns per convolution stage; `SCALE_TRAIN_RECIPE=1` |
| Physical scale | VDD=0.1 V, `one_over_q=1` | VDD=0.1 V, `one_over_q=5` |
| Input | CiFAIR-100; no input quantization; no centering | CiFAIR-100; 12-bit input; no centering |

Common FT: LR 0.005, cosine, 50 epochs, no warmup, WD 0.001, BS 128,
`no_aug`, SRRL.

Both use R=50 kΩ, C=500 fF, 5-bit weights/QF1, measured pooling and measured
ReLU. FT activation is `relu_current_0p2uA_finer.csv`, TT; nonlinear-R and
measured pooling use the selected singleton. Spin variation and summing/coupler
current noise are enabled; FT current-noise ASDs are 0.6 pA/√Hz. FT DTC is off.
Evaluation samples the complete family bank per coupler, uses
`relu_monteCarlo` per spin, enables corner-specific spin/DTC effects and current
noise, and retains an `old_ideal` final head. ENOB and slow-current modes are off.
Timing, scale and input differences above are intentional, not identical
PCN/CNN physical mappings.

Representative-run records: [queue](../logs/local_runs/tt_representative_50ep/queue.log),
[resumed WRN queue](../logs/local_runs/tt_representative_50ep/resume_wrns.queue.log).
The original-circuit WRN recovered from its completed-epoch-23 `latest_ckpt`;
all 191 Python command arguments matched the interrupted job except the
checkpoint path. FT completion is checked before evaluation starts.

Representative result summaries:
[PCN original](../results/tt_singleton_50ep_CiFAIR100_pcn_coupler_monte_coupler_monte_tt_25_1_mc73_singleton_mc45/summary.md),
[PCN improved](../results/tt_singleton_50ep_CiFAIR100_pcn_coupler_monte_v2_coupler_monte_v2_tt_25_1_mc34_singleton_mc45/summary.md),
[WRN original](../results/tt_singleton_50ep_CiFAIR100_wrn_c27_coupler_monte_fixedTiming15ns_vdd0p1_qinv5_coupler_monte_tt_25_1_mc73_singleton_mc45/summary.md).

Persistent [percentile plot generator](../diagnostic_scripts/plot_mc45_descending_percentiles.py)
supports result-directory overrides and `--relative-to 63.87`.
Current relative-accuracy plots: [10 dots](../results/mc45_descending_corner_percentiles_tt_representative_pcn_relative_wrn6387/descending_corner_percentiles_10_dots.pdf),
[15 dots](../results/mc45_descending_corner_percentiles_tt_representative_pcn_relative_wrn6387/descending_corner_percentiles_15_dots.pdf).

### Existing black-line references and next experiments

| Dataset | Existing PCN adapted reference | MC45 mean | Evaluation repeats |
|---|---|---:|---|
| CiFAIR-100 | [Original reference](../results/coupler_monte_v2_CiFAIR100_qf1_noENOB_default45/summary.md) | 58.3096% | 10 per corner |
| CiFAIR-10 | [FT LR 0.002, 200-epoch reference](../results/coupler_monte_v2_CiFAIR10_qf1_noENOB_ftlr0p002_e200_default45/summary.md) | **86.9645%** | 10 per corner |

Both existing black lines use the improved `coupler_monte_v2` family; they are
not 50-epoch singleton-trained models. Plot one mean per corner so their ten
trials do not give them more percentile observations than singleton runs.

Remaining work, in order:

1. Finish the representative-34 CiFAIR-100 WRN FT and unrecalibrated MC45;
   record the fourth row and optionally add its improved-circuit curve.
2. The integrated launcher now supports CiFAIR-10/100 and PCN / narrow WRN
   c27 / ResNet-110, with parameterized teacher, source checkpoints, output/cache
   paths and launch/handoff checks. No PCN pretraining is needed.
3. Pretrain exactly three new CNNs from scratch: ResNet-110 on CiFAIR-10,
   ResNet-110 on CiFAIR-100, and narrow WRN c27 on CiFAIR-10. Use the WRN
   pretraining recipe: 300 epochs, base LR 0.1, cosine, no warmup, WD 0.001,
   BS 128, SRRL with the dataset-specific teacher, IQ12, no centering, and
   standard pretraining augmentation with random erasing 0. Use intermediate
   ReLU6 and final dropout 0.25 explicitly for ResNet; keep the final ReLU ideal.
   Measured activation and hardware nonidealities are disabled in pretraining.
4. **Pretrain directly at fixed 15 ns**, `Z_OVER_Y_TIME=1`,
   `SCALE_TRAIN_RECIPE=1`, R=50 kΩ and C=500 fF. Retain the other historical
   WRN pretraining settings (VDD=0.5 V, `one_over_q=5`). The existing scale
   recipe adjusts initialization, convolution LR and weight decay for T/RC;
   leave base LR/WD unchanged. Verify the 5-ns/15-ns reparameterization on a
   short same-seed check before long runs. Load the resulting 15-ns checkpoint
   directly into 15-ns FT: **no intermediate divide-by-three checkpoint**.
   Existing historical checkpoints and their earlier transformations stay
   unchanged. Do not use the old digital ResNet's divide-by-0.6 route for these
   new runs.
5. Verify each new CNN's ideal physical initialization before introducing
   nonidealities; validate `one_over_q` for the new ResNets rather than assume
   WRN's value is suitable. Launch eight 50-epoch FT → MC45 pipelines: two
   families each for CiFAIR-10 PCN, CiFAIR-10 WRN c27, CiFAIR-10 ResNet-110,
   and CiFAIR-100 ResNet-110. Each model/dataset uses a single pretrained
   checkpoint for both TT representatives (73/34). Keep base FT LR 0.005,
   current measured-ReLU/noise protocol, full corresponding evaluation banks,
   expanded Level 3, one trial per corner, and no BN recalibration.
6. Reuse the existing CiFAIR-10 PCN pretrained checkpoint (**89.18%, epoch
   300**) and the existing 200-epoch adapted black-line reference (**86.9645%
   MC45**, FT LR 0.002). Its source root is
   `saved_ckpt_runs/coupler_v2_CiFAIR10_qf1_noENOB_matchdistill_srrl_noRE_ftlr0p002_e200_toggle_odexinit`;
   the pretrained model name has no `TIMMQAT` prefix and no `_C100` token.
   Reuse finished CiFAIR-100 PCN/WRN results; do not retrain their PCNs.
7. Produce CiFAIR-10 and extended CiFAIR-100 percentile plots. Normalize all
   curves in a dataset by that dataset's digital WRN accuracy (CiFAIR-100:
   63.87%; CiFAIR-10: verify first). Report raw accuracies alongside relative
   values. BN-recalibrated CNN runs are optional, separately labeled controls.

The following sections preserve the **earlier fixed-timing/full-range and
single-corner experiments**, not the current singleton protocol.

## Motivation and question

The original BN-equipped WRN achieved higher ideal pretraining accuracy than
the five-cycle PCN, but recovered much less accuracy during hardware-aware
fine-tuning. We therefore test whether **PCN's repeated local processing improves
accuracy retention under the modeled hardware nonidealities**, rather than
assuming that higher ideal CNN accuracy translates into better hardware accuracy.

The experiments examine alternative explanations: weight-range limitations,
BatchNorm statistics, measured pooling, parameter count, and integration time.
The strongest within-architecture control is reducing PCN from five cycles to one.
These are simulation results, not measurements on fabricated hardware.

## Completed results

All entries are **CiFAIR-100 top-1 accuracy (%)**. PT and FT are the best saved
validation scores. L2/L3 entries are arithmetic means, not selected best trials.
L2 uses five trials per condition; L3 uses three trials per condition on **SS_V2_T2**.
“—” means BN recalibration is inapplicable because the model has no BN.

| Model | Parameters | Stage times, ns: PCN y/z; CNN conv1/conv2 | PT best | FT best | L2 no recal | L2 recal | L3 no recal | L3 recal |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **PCN 16L96C, 5 cycles** | **1,028,380** | **5 / 15** | **63.90** | **60.02** | **59.120** | — | **58.480** | — |
| PCN 16L96C, 1 cycle | 1,028,380 | 5 / 15 | 60.98 | 53.84 | 53.374 | — | 50.647 | — |
| PCN 16L96C, 1 cycle | 1,028,380 | 7.5 / 22.5 | 62.28 | 54.67 | 54.086 | — | 51.093 | — |
| PCN 16L96C, 1 cycle | 1,028,380 | 25 / 75 | 61.73 | 47.97 | 47.782 | — | 25.327 | — |
| WRN-28-2 AvgPool, BN | 1,468,612 | 15 / 15 | 66.24 | 35.86 | 26.754 | 39.196 | 1.280 | 7.390 |
| WRN-28 c27, stride-2 + AvgPool shortcuts, BN | 1,048,497 | 15 / 15 | 63.87 | 27.49 | 13.364 | 30.608 | 1.083 | 2.117 |
| WRN-28-2 AvgPool, no BN/no conv bias | 1,464,996 | 15 / 15 | 56.37 | 55.64 | 54.634 | — | 49.933 | — |

Parameter counts follow the training logs' model-parameter totals, excluding
teacher/distillation auxiliary modules and hardware unrolling. Changing PCN
cycle count or timing does not change its parameter count. The c27 WRN has
approximately 1.96% more parameters than PCN.

The three WRNs above were pretrained at 5 ns, then their convolution weights
(and convolution biases where present) were divided by three and stage times
increased to 15 ns before FT. PT scores refer to the original pretraining
checkpoints. The older 5 ns BN-WRN FT/evaluation row is intentionally omitted.

## Variant definitions

- **PCN 16L96C:** BatchNorm-free recurrent baseline; the one-cycle variants retain
  the architecture and change cycle count/timing. Five cycles use a nominal
  total stage duration of 100 ns per block. One-cycle variants use 20, 30, and
  100 ns, respectively. The 30 ns case matches the WRNs' two-stage duration;
  the 100 ns case matches the five-cycle PCN's total stage duration. These are
  not equal-energy or equal-operation-count comparisons.
- **WRN-28-2 AvgPool, BN:** stem width 16; stages 32/64/128; 1,468,612 parameters.
  Transition main paths use stride-1 convolution followed by AvgPool. Shortcuts
  use parameter-free pooling/channel padding rather than learned projections.
- **WRN-28 c27:** depth 28; stem width 16; stages 27/54/108; 1,048,497 parameters.
  Main paths retain stride-2 convolutions; shortcuts use AvgPool/padding.
  This is the approximately PCN-parameter-matched CNN, **not a pooling-only
  ablation**: width and main-path downsampling both differ from the larger WRN.
- **No-BN/no-conv-bias WRN:** same AvgPool topology and stage widths as the larger
  WRN, with BN removed and no convolution biases added; final Linear bias remains.
  It has 1,464,996 parameters. The intended training recipe is unchanged.

## Protocol and comparability

All models use CiFAIR-100 sensor/RGGB inputs, SRRL-assisted training, and the
same hardware-model family. The reported physical runs use R=50 kΩ, C=500 fF,
VDD=0.5 V, ONE_OVER_Q=5, 5-bit weights with QF1, 12-bit input quantization,
no input centering, measured pooling, and no ENOB noise. Slow-current noises
are disabled.

**L2** reproduces training-style averaged evaluation: `coupler_full_range`
with `exact_curve`, measured TT activation shared per model, spin variation
σ=0.10, and summing/coupler current-noise ASDs of 0.6 pA/√Hz; DTC is disabled.
**L3** uses pulse-level SS_V2_T2 evaluation: the corner's `coupler_monte_v2`
curve bank with per-coupler sampling, per-spin measured activation, corner
spin/DTC parameters, and temperature-adjusted current noise.
Consequently, **L2→L3 differences do not isolate pulse discretization alone**.

BN recalibration uses 5,120 training samples, batch size 5,120 at L2 and 256
at L3, with dynamic noise enabled and static defects retained. The new L3
uncalibrated controls reuse the calibrated runs' checkpoints, configurations,
and trial seeds. Calibration advances dynamic-noise streams, so the two
conditions do not use identical dynamic-noise sample sequences. Historical
static-defect identity was not independently recorded by tensor fingerprints.

## ResNet-110 feedforward-toggle operating-point study

This separate CiFAIR-10 diagnostic uses the SRRL-trained ResNet-110 digital
checkpoint (88.24% best full-validation accuracy). Each hardware result below
uses the same `FS_V2_T1` realization and seed (`20260723`) on the first 512 test
samples (four batches of 128); it is therefore a controlled diagnostic, not a
full-validation or 45-corner result.

The ideal Level-3 feedforward path preserves **87–88%** on this subset when it
uses 8-bit weights, exact (unquantized) weight scales
(`WEIGHT_QUANT_FACTOR_BITS=-1`), an 8-bit digital head, derived timing,
12-bit input quantization, and `ONE_OVER_Q=128`. This remains true at both
`V_DD=0.5 V` and `0.1 V` when all modeled nonidealities are disabled. In
contrast, changing only `ONE_OVER_Q` from 128 to 10 at 0.5 V gives **8.9844%**.
Thus, for this unadapted digital checkpoint, selecting an activation scale that
avoids Level-3 rail clipping is essential; lowering `V_DD` alone did not cause
the collapse. These observations do not establish 128 as a universal setting.

Starting from the 0.5 V, `ONE_OVER_Q=128` ideal configuration, the following
one-at-a-time results identify the largest individual degradations:

| Additional effect enabled alone | Four-batch top-1 |
|---|---:|
| Spin variation | 87.5000% |
| DTC nonideality | 87.5000% |
| Measured pooling | 86.1328% |
| Summing-current noise | 71.4844% |
| Nonlinear resistance | 66.4062% |
| Coupler-current noise | 47.6562% |
| Measured activation | 11.5234% |

The two pinned combined-nonideality rows below use the same base configuration plus
spin variation, summing-current noise, coupler-current noise, DTC nonideality,
nonlinear resistance, and measured pooling. Measured activation remains
disabled. The only difference between the rows is the curve family used by
both nonlinear resistance and measured pooling:

| Coupler curve family used by nonlinear-R and pooling | File | Voltage domain | Curves | Full-dataset top-1 |
|---|---|---:|---:|---:|
| Full-range | `coupler_full_range/fs_25_2.csv` | [-0.4933, 0.5] V | 100 | **25.0800%** |
| Legacy narrow-range | `coupler_monte/fsg_25_2.csv` | [-0.096, 0.1] V | 100 | **12.8600%** |

| Shared setting | Value |
|---|---|
| Execution | Level 3; derived timing; `R=50 kΩ`, `C=500 fF`; no BN recalibration |
| Physical scale | `V_DD=0.5 V`; `ONE_OVER_Q=128` |
| Quantization | 8-bit weights; exact weight scales; 12-bit input; no input centering |
| Final head | Digital; 8-bit representation/weights; 32-bit bias/accumulator; quantization and clamp enabled |
| Curve use | Same listed bank for nonlinear-R and measured pooling; per-coupler empirical sampling with replacement |
| Spin variation | Mean 0.9885; sigma 0.02988 |
| Current noise | Summing and coupler ASDs both 0.6 pA/√Hz |
| DTC | Width mean -0.01712; width sigma 0.0058592; leading/falling jitter sigma 0.005 |
| Disabled | Measured activation, ENOB, differential mismatch, and slow-current modes |

Because both nonlinear-R and pooling curve sources change, the 25.0800% versus
12.8600% comparison is a **curve-family comparison**, not an isolated
nonlinear-R ablation.

To remove curve-to-curve sampling variation, the same evaluation was repeated
with every coupler and measured-pooling element using the flattest curve in
each `FS_V2_T1` bank. Flatness is the minimum worst-case fractional deviation
of effective resistance from its zero-voltage value over the stored domain.

These are full-dataset **CiFAIR-10** Level-3 evaluations of `resnet110_cifar`.
Nonlinear resistance, measured pooling, spin variation, summing-current noise,
coupler noise, DTC variation/jitter, and quantization are enabled; measured
ReLU is disabled, so the model's ideal ReLU is used.

| Singleton curve bank | Zero-based curve (`mcparamset`) | Resistance range | Maximum deviation from R(0) | Full-dataset top-1 |
|---|---:|---:|---:|---:|
| `coupler_full_range/fs_25_2.csv` | 87 (88) | 35.49 kΩ–2.834 MΩ | 6799.15% | **17.7800%** |
| `coupler_monte/fsg_25_2.csv` | 42 (43) | 43.08–67.37 kΩ | 35.80% | **26.5800%** |
| `coupler_monte_v2/fs_25_2.csv` | 41 (42) | 43.39–45.87 kΩ | 3.09% | **37.6600%** |

The `coupler_monte_v2` singleton is both substantially flatter and closer to
the nominal 50 kΩ than the other selected curves, and gives the highest
accuracy under otherwise identical settings.

## Evidence so far

1. **Recurrence is beneficial in the tested PCNs.** The five-cycle model exceeds
   the best one-cycle result by 5.35 percentage points in FT and 7.387 points at
   L3. Giving one cycle the same total nominal duration does not recover the
   five-cycle accuracy. This supports repeated processing over merely extending
   one integration interval in these operating conditions.
2. **Higher PT accuracy does not guarantee physical robustness.** The BN WRN's
   PT score is 66.24%, above PCN's 63.90%, but its FT and L3 scores are much lower.
   The parameter-matched c27 WRN also underperforms this PCN.
3. **BN is an important contributor, but recalibration is not sufficient here.**
   L3 recalibration improves the larger BN WRN from 1.280% to 7.390%, and c27
   from 1.083% to 2.117%. The no-BN/no-conv-bias WRN retains substantially more
   accuracy: 55.64% FT and 49.933% L3.
4. **The broader claim remains qualified.** Five-cycle PCN exceeds the tested
   BN-free WRN by 4.38 points in FT and 8.547 points at L3, but their PT accuracy,
   architecture, and parameter count differ. This is not proof that recurrence
   alone explains every CNN–PCN gap, or that PCNs universally outperform CNNs.

## Remaining controlled experiments

| Priority | Experiment | Purpose |
|---|---|---|
| 1 | Complete the no-BN **with-conv-bias** WRN, retaining the recipe and AvgPool topology; evaluate L2/L3 | Isolate the contribution of convolution bias against the no-BN/no-bias model |
| 1 | Diagnose the 1-cycle 25/75 ns L3 loss with matched L2/L3 static assignments, first ideal and then individual nonidealities | Separate pulse/clamping effects from differing resistance/corner configurations and rule out implementation errors |
| 2 | Compare depth-28 WRNs with stem 16 and stages **27/54/108** in both cases: main-path AvgPool versus stride-2, identical shortcuts and recipe | Isolate additional measured pooling while retaining the PCN-like parameter budget |
| 2 | Repeat key PCN and strongest BN-free CNN results over training seeds and all 45 corners | Establish uncertainty and robustness beyond one trained model and one corner |
| 3 | Evaluate a PCN-channel-matched feedforward residual CNN and measure latency/energy | Tighten architectural controls and distinguish accuracy robustness from hardware efficiency |

For new training, fixed 15 ns timing from the start with `SCALE_TRAIN_RECIPE=1`
avoids intermediate checkpoint rescaling. Label it as a new run, not an exact
reproduction of the earlier 5 ns training trajectory.

## Reusable single-curve pipeline

The same stage launcher runs locally and inside the Slurm job. Production jobs
are **not** automatically submitted when this code is installed.

| Job | Pretraining | Then, separately for original and improved circuit |
|---|---|---|
| PCN / CiFAIR-10 | Reuse existing digital checkpoint | 50-epoch singleton FT → MC45 |
| WRN c27 / CiFAIR-10 | Train from scratch | Select `one_over_q` → 50-epoch singleton FT → MC45 |
| ResNet-110 / CiFAIR-10 | Train from scratch | Select `one_over_q` → 50-epoch singleton FT → MC45 |
| ResNet-110 / CiFAIR-100 | Train from scratch | Select `one_over_q` → 50-epoch singleton FT → MC45 |

Existing CiFAIR-100 PCN/WRN experiments are not repeated. Every family branches
from the **same original pretrained checkpoint**, never the other family's FT.
Singleton CSVs are required inputs; curve indices are not hardcoded in launchers.
Current requested TT representatives are `coupler_monte` mcparamset 73 and
`coupler_monte_v2` mcparamset 34.

### Pinned recipe and handoffs

- CNN pretraining: fixed **15 ns**, ratio 1, `scale_train_recipe=1`,
  (R=50\,\mathrm{k\Omega}, C=500\,\mathrm{fF}), historical pretraining
  (V_{DD}=0.5), `one_over_q=5`; **unitless**, no measured activation/pooling
  or hardware defects. 300 epochs, SGD/cosine, LR 0.1, warmup 0, WD 0.001,
  batch 128, seed 4096, SRRL using the dataset's legacy EfficientNet teacher.
  Standard CiFAIR/timm augmentation with random erasing 0. Input quantization
  12 bits, no centering. ReLU6 intermediates; final dropout 0.25. The model's
  ideal final activation/head remains in place. This avoids any intermediate
  weights-÷3 checkpoint. The 5/15-ns initialization and compensated SGD
  equivalence is regression-tested for both residual-block styles.
- Selection: first 1,280 **training** samples, 10 batches of 128, no augmentation,
  BN frozen; identical sample order and seeds (4096) across candidates. First
  compare unitless accuracy against an ideal physically scaled 8-bit/exact-scale
  reference (`one_over_q=128`, tolerance 5 pp). Then test
  `1 2 4 5 8 16 25 32 50 64 128` with the intended singleton nonlinear-R,
  measured ReLU/pooling, spin variation and current noise; 5-bit weights and
  1-bit weight-scale factors, no DTC. Highest subset accuracy wins; ties choose
  smaller `one_over_q`. Any failed candidate fails the pipeline. Candidate
  list/sample count/batch limit are configurable. This is a short starting-point
  selection, not a full-dataset accuracy claim.
- FT: same historical singleton recipe, **50 epochs**, LR 0.005, cosine,
  warmup 0, WD 0.001, batch 128, evaluation every 2 epochs, `no_aug`, SRRL.
  Measured current ReLU: `hardware_data/relu_current_0p2uA_finer.csv`, TT,
  `pc_only`, fused. Nonlinear-R and pooling use the supplied singleton.
  Spin variation σ=0.10, summing/coupler noise each 0.6e-12; DTC off.
  5-bit weights, 1-bit scale factors, ENOB disabled, ideal final head.
- MC45: best FT checkpoint → expanded Level 3, **all 45 corners, one trial
  each**, full corresponding coupler-family bank (not the training singleton),
  corner ReLU bank `relu_monteCarlo` per spin, fused; measured pooling on,
  corner spin/DTC and temperature-scaled current noise on; no BN recalibration.
  PCN: derived timing, 5 cycles, q inverse 1, 5 ns timing argument, ratio 3,
  input quantization disabled. CNN: fixed 15 ns, ratio 1, selected q inverse,
  12-bit input quantization. Both use (V_{DD}=0.1), no input centering,
  5-bit weights/1-bit scales/ENOB disabled/ideal head. Separate expanded caches
  per dataset, architecture and family. Evaluation receives ReLU6/dropout
  construction options explicitly; state dictionaries do not encode them.

Selection records, including checkpoint/curve SHA-256 fingerprints and full
resolved evaluator options, are in `results/${RUN_TAG}_selection/selection.json`.
Actual FT configurations are saved as `baseline_config.json` (CNN) and stage
logs. Every corner log records its actual inference command. Failures propagate
through `pipefail`; a missing FT/pretrain checkpoint fails the handoff.
`latest_ckpt` discovery, completion queues and existing trainer code are untouched.

### Launch locally

Activate `scanbase` and run from the repository root. Example WRN / CiFAIR-10:

```bash
(
  export MODEL_KIND=wrn_c27 TASK=cifar10 \
    COUPLER_MONTE_TT_SINGLETON=hardware_data/mc_45_corners/coupler_monte_tt_25_1_mc73_singleton.csv \
    COUPLER_MONTE_V2_TT_SINGLETON=hardware_data/mc_45_corners/coupler_monte_v2_tt_25_1_mc34_singleton.csv
  bash ./launch_scripts/run_tt_singleton_study.sh
)
```

Use `(MODEL_KIND, TASK)` of `(pcn, cifar10)`, `(wrn_c27, cifar10)`,
`(resnet110, cifar10)`, or `(resnet110, cifar100)` for the four jobs.
Per-stage live logs: `logs/${STUDY_TAG}/pretrain.log`,
`${family}_selection.log`, `${family}_ft_eval.log`, and `status.log`.
The default tag is `tt_representative_50ep_CiFAIR${TASK#cifar}_${MODEL_KIND}_pipeline`.
Use distinct `STUDY_TAG`s for independent repetitions. There is no implicit resume
or automatic skipping of pretraining based on old artifacts. Existing nonempty
study logs are rejected before training to prevent accidental output reuse.
Operating-point selection uses augmentation-free training samples with sensor
noise enabled; BN stays frozen. BN-recalibration callers retain their existing
clean-input loader default. `SCANGEN_DATA_ROOT` is shared with training unless an
explicit `SCAN_TEST_*` dataset-file override is supplied.

### Launch on Slurm

Use the **same exports** as above; replace the last `bash` line with:

```bash
source ./launch_scripts/slurm_search_tt_singleton_config.sh
```

This follows `slurm_search_feedforward_config.sh` and
`feedforward_pretrain_then_ft.sbatch`: `sbatch --parsable --gres=gpu:1
--export=ALL`, login-shell job, partition `ising`, existing scratch repository
path, `source activate base`, `conda activate scanbase`, then the same local
stage launcher. `REPO_ROOT` can override the scratch checkout. Scheduler logs
remain `logs/slurm_jobs/slurm_%j.out`. No new module/login-shell submission
mechanism is introduced. Local tests cannot validate cluster availability,
permissions, installed packages, or Slurm policy.

### Isolated executable rehearsal

```bash
python diagnostic_scripts/rehearse_tt_singleton_studies.py \
  --artifact_root /tmp/scan-singleton-rehearsal-UNIQUE \
  --original_singleton hardware_data/mc_45_corners/coupler_monte_tt_25_1_mc73_singleton.csv \
  --improved_singleton hardware_data/mc_45_corners/coupler_monte_v2_tt_25_1_mc34_singleton.csv
```

Runs all four actual job wrappers locally, mocking **only** `sbatch` and conda
activation. Pretraining/FT execute two batches, validate, and save real
checkpoints; selection executes a training batch for candidates 4 and 5.
Each evaluation loads that newly saved FT checkpoint and executes its first
actual convolution-to-CSR expansion, saves a proof, then exits via a test-only
milestone. Both family branches must pass. No full accuracy is claimed.
The workload limits are injected solely by opt-in rehearsal hooks; production
recipe arguments are unchanged. Logs/checkpoints/configs/expanded-cache paths
are isolated under a **new** `/tmp` directory and cannot satisfy future real
handoffs or cause corner skips. `report.json` records all pass/fail results.

Existing CiFAIR-10 PCN digital source: the `TIMMPCNetNoBatchNorm...CiFAIR_1REP`
`_best_ckpt.pth` under
`saved_ckpt_runs/coupler_v2_CiFAIR10_qf1_noENOB_matchdistill_srrl_noRE_ftlr0p002_e200_toggle_odexinit`.
The existing black-curve reference is
`results/coupler_monte_v2_CiFAIR10_qf1_noENOB_ftlr0p002_e200_default45/summary.md`
(86.9645% mean, 10 trials per corner). It is not overwritten by this pipeline.

### Verification — 2026-10-15

All four final-code wrapper-path rehearsals exited successfully, each with two
FT checkpoint handoffs and two actual CSR expansion proofs. The three CNN jobs
also completed pretraining and both real selection stages. Final reports are
`/tmp/scan-singleton-final-${MODEL_KIND}-${TASK}-20261015/report.json`.
The focused regression suite passed **124 tests**, including direct-15-ns
initialization/SGD equivalence, activation-override compatibility, selection
failure propagation and a PCN tee-pipeline failure with a stale checkpoint.
Trainer/validation algorithms were not edited. These bounded rehearsals do not
claim convergence, full MC45 completion, or an actual Slurm-cluster execution.

## Planned full-FT WRN reruns — existing local/Slurm pipeline

These are **140-epoch full-bank FT**, not the 50-epoch single-curve study:

| Model / dataset | Coupler family | VDD | `one_over_q` |
|---|---|---:|---:|
| Narrow WRN c27 / CiFAIR-100 | `coupler_full_range` | 0.5 V | 20 |
| Narrow WRN c27 / CiFAIR-100 | `coupler_monte_v2` | 0.1 V | 5 |

Both start from the existing independent weights-÷3 / fixed-15-ns pretrained
checkpoint. Keep the original full-FT optimizer/augmentation/distillation,
measured activation/pooling and training nonidealities unchanged. Each is
followed by expanded L3 MC45 using the corresponding full family bank, one
trial per corner and no BN recalibration. Corner DTC is enabled for evaluation,
not FT. ReLU evaluation uses `relu_monteCarlo` per spin. No new launcher is
needed: the existing `run_feedforward_pretrain_then_ft.sh` supports
`mode=ft_only` and `mode=ft_and_eval` for toggle CNNs and honors an explicitly
supplied `MODEL_CKPT` in these modes. Default toggle behavior and TC routing
are unchanged. FT failure or a missing output checkpoint prevents evaluation.

Run from the Slurm checkout (the independent source checkpoint must be present):

```bash
for spec in "coupler_full_range:0.5:20" "coupler_monte_v2:0.1:5"; do
  family="${spec%%:*}"
  rest="${spec#*:}"
  voltage="${rest%%:*}"
  qinv="${rest#*:}"
  (
    unset EXTRA_OVERRIDE TIMM_AUG_LEVEL FT_OUTPUT_DIR RESULT_PATH EVAL_LOG
    unset EXPANDED_WEIGHT_DIR CORNER_IDS MC_COUPLER_NONLINEAR_VARIATION_SOURCE
    unset FIXED_COUPLER_MC_INDEX FIXED_RELU_MC_INDEX
    export mode=ft_and_eval TC_FEEDFORWARD=false \
      MODEL_NAME=wrn_flexible_cifar_avgpool_shortcut \
      WRN_DEPTH=28 WRN_FIRST_STAGE_CHANNELS=27 \
      TASK=cifar100 IMG_TYPE=CiFAIR \
      MODEL_CKPT=saved_ckpt_runs/coupler_full_range_CiFAIR100_wrn28_c27_avgpool_shortcut_fixedTiming15ns_rescaled_pretrain_iq12/cifar100/custom_noresize/wrn_flexible_cifar_avgpool_shortcut/custom_noresize_cifar100_wrn_flexible_cifar_avgpool_shortcut/custom_noresize_cifar100_wrn_flexible_cifar_avgpool_shortcut_best_ckpt.pth \
      EXP_PREFIX="${family}_CiFAIR100_wrn28_c27_fullFT140_15ns_vdd${voltage}_qinv${qinv}" \
      TEACHER_CKPT=./checkpoint/efficientnet_v2_l_cifar100_CiFAIR_OldNoTimm_MatchDistill.pth \
      NONLINEAR_R_TABLE="${family}" \
      R_VAL=50e3 C_VAL=500e-15 V_DD="${voltage}" \
      TOGGLE_ONE_OVER_Q="${qinv}" TOGGLE_TIMING_MODE=fixed \
      TOGGLE_Y_TIME=15e-9 Z_OVER_Y_TIME=1 SCALE_TRAIN_RECIPE=1 \
      INPUT_QUANT_BITS=12 CENTER_STUDENT_INPUT=false \
      ENABLE_MEASURED_POOLING=true WEIGHT_QUANT_FACTOR_BITS=1 ENOB=none \
      N_TRIALS=1 ENABLE_BN_RECALIBRATION=false
    source ./launch_scripts/slurm_search_feedforward_config.sh
  )
done
```

For local execution, activate `scanbase`, retain the same exports and replace
the final `source` line with
`bash ./launch_scripts/run_feedforward_pretrain_then_ft.sh`.
Use `mode=ft_only` to omit MC45. FT logs are in the Slurm job output;
MC45's overall live log is `results/${EXP_PREFIX}_mc45/evaluation.log` and
individual corners are logged beside it. These commands are pinned only;
the production reruns have not been submitted here.

## Result sources

- Five-cycle PCN: [evaluation records](../results/pcn16l96c_full_range_fixed5ns_5cycle_wrn_comparison/).
- One-cycle PCNs: [L2 trials](../results/pcn_onecycle_three_timings_level2/trials.csv),
  [L3 corner runs](../results/pcn_onecycle_three_timings_SS_V2_T2/).
  PT/FT scores come from the supplied Slurm logs for jobs 1866843–1866845.
- BN WRN L2: [32-channel log](../logs/local_runs/wrn28_2_fixedTiming15ns_level2_bn_pair5.log),
  [27-channel log](../logs/local_runs/wrn_flexible_c27_fixedTiming15ns_level2_bn_pair5.log).
- BN WRN L3 calibrated: [32-channel trials](../results/wrn28_2_cifair100_fixedTiming15ns_rescaled_level3_SS_V2_T2_bn_recal3/corner_trials.csv),
  [27-channel trials](../results/wrn_flexible_c27_cifair100_fixedTiming15ns_rescaled_level3_SS_V2_T2_bn_recal3/corner_trials.csv).
- BN WRN L3 uncalibrated: [32-channel trials](../results/wrn_bn_level3_no_recal_pair/wrn28_2/corner_trials.csv),
  [27-channel trials](../results/wrn_bn_level3_no_recal_pair/wrn28_c27/corner_trials.csv).
- BN-free WRN: [L2 run log](../logs/local_runs/wrn28_2_nobn_no_bias_avgpool_fixedTiming15ns_level3_then_level2.log),
  [L3 trials](../results/wrn28_2_nobn_no_bias_avgpool_cifair100_fixedTiming15ns_rescaled_level3_SS_V2_T2_3trials/corner_trials.csv).

Hardware trials are repeated evaluations of fixed checkpoints, not independent
training runs. Report both sources of variability separately in the paper.

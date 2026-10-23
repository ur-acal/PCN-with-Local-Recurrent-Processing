# PCN versus physical feedforward CNN: experiment summary

## TC: physical scaling, current noise, batch normalization and timing

**True-continuous (TC), not toggle.** Unless stated otherwise: CIFAR-10,
5-bit weights, resistance 10 kOhm, capacitance 49 fF and voltage rails ±0.1 V.

### Head-to-head: trained PCN versus the successful ResNet-44

Full 10,000-image test set; expanded inference, per-spin measured ReLU,
per-coupler resistance curves and all other TC nonidealities ON. The CNN is
the successful `one_over_q=32`, `rgb_mid` fine-tuned model, not the collapsed
`one_over_q=1` model. PCN uses `one_over_q=1` and has no batch normalization.

| Evaluation condition | State-2 PCN, 22 layers | ResNet-44 | PCN minus CNN |
|---|---:|---:|---:|
| No added timing variation/noise; original CNN batch-normalization statistics | 95.10% | 90.08% | +5.02 percentage points |
| No added timing variation/noise; recalibrated CNN batch normalization | **95.10%** | **94.13%** | **+0.97 percentage points** |
| 20% fixed timing variation + 20% batch-shared timing noise; recalibrated CNN | 92.72% | 71.34% | +21.38 percentage points |
| 20% fixed timing variation + 20% independent-per-image timing noise; recalibrated CNN | **92.93%** | **76.63%** | **+16.30 percentage points** |

Timing percentages are relative standard deviations. CNN recalibration uses
5,120 training images with the same hardware effects as testing. Each score
is one hardware trial. The checkpoints have different training recipes:
PCN uses raw RGB and random per-spin ReLU during fine-tuning; this CNN uses
normalized RGB and fixed MC18 during fine-tuning. Solvers: PCN Dopri5 at
tolerance 1e-6; CNN Euler with 10 steps.
Sources: [PCN checkpoint/settings](../results/tc_compute_time_sweep_20261017/pcn_s2_22l_fixed0_noise0/trial.jsonl),
[CNN checkpoint/settings](../results/tc_resnet44_q32_last_euler10_expanded/tc_trial_metadata.json),
[all comparison scores](../results/tc_compute_time_sweep_20261017/results.csv).

### Why changing `one_over_q` helped ResNet-44 fine-tuning

Larger `one_over_q` reduces physical voltages at unchanged rails. At value 1,
the failed fine-tuned ResNet-44 had 49.94% outside rails versus PCN's 0.00%
(32 images; median across layers, before clamping).
[Layer statistics and checkpoint paths](../results/pcn_collapse_diagnosis/ft_rail_comparison_mapping.json).

| ResNet-44 evaluation | `one_over_q` | Test accuracy |
|---|---:|---:|
| Pretrained weights; only 5-bit mapping and rails | 1 | 10.00% |
| Same pretrained weights; only mapping and rails | 32 | 62.77% |
| Same pretrained weights; all fine-tuning nonidealities, no updates | 32 | 10.00% |
| After fine-tuning with `rgb_mid`; fixed training ReLU | 32 | 95.46% |
| **Final fine-tuned weights; expanded per-spin ReLU, Euler with 10 steps** | 32 | 90.08% |
| **Same expanded evaluation, with batch-normalization recalibration** | 32 | 94.13% |

Full 10,000-image test set. The successful rerun also changed augmentation;
scaling alone did not restore accuracy. Sources: [mapping at 1](../results/resnet44_pretrain_q1_noise_control/clean_full/tc_trial_metadata.json),
[mapping at 32](../results/resnet44_pretrain_q32_noise_control/clean_full/tc_trial_metadata.json),
[all effects before updates](../results/resnet44_pretrain_q32_all_ft_nonidealities_original_bn/tc_trial_metadata.json),
[fine-tuning log and final checkpoint](../logs/local_runs/tc_rgb_cifar10_resnet44_cifar_q32_rgb_mid.log),
[expanded configuration and checkpoint](../results/tc_resnet44_q32_last_euler10_expanded/tc_trial_metadata.json),
[recalibration controls](../results/tc_compute_time_bn_recal_20261017/results.csv).
Earlier mapping-only recalibration: **90.76%**, reported in discussion;
saved record not located. Distinct from the 94.13% result above.

### Current noise before hardware-aware fine-tuning

Pretrained weights; mapping/rails with summing-current and coupler-current
noise OFF/ON. Other effects and batch-normalization recalibration are off.

| Model | `one_over_q` | Noise OFF | Noise ON |
|---|---:|---:|---:|
| ResNet-44 | 1 | 10.00% | 10.00% |
| ResNet-44 | 32 | 62.77% | 9.50% |
| State-2 PCN, 22 layers | 1 | 77.02% | 58.51% |
| State-2 PCN, 22 layers | 6 | 94.22% | 18.49% |

Full test set; PCN solver tolerance 1e-6. Sources: [ResNet at 1](../results/resnet44_pretrain_q1_noise_control/),
[ResNet at 32](../results/resnet44_pretrain_q32_noise_control/),
[PCN at 1](../results/pcn_pretrain_q_noise_control/q1_tol1e-6.json),
[PCN at 6](../results/pcn_pretrain_q_noise_control/q6_tol1e-6.json).
Noise strongly affects pretrained models. Smaller physical signals make
unchanged current noise relatively stronger.

### Propagated coupler-noise error in the fine-tuned ResNet-44

Same 128 images, complete forwards with coupler noise OFF/ON; frozen batch
normalization. Error includes upstream propagation. Signal-to-error ratio is
`20 log10(RMS(reference) / RMS(noisy-reference))`; RMS means root-mean-square
magnitude. Each row has its **own** reference, not a common ideal reference.

| Other nonidealities | Final-logit signal-to-error ratio | Final-logit relative RMS error | Layer results before/after batch normalization |
|---|---:|---:|---|
| OFF | 11.37 dB | 27.02% | [43 locations](../results/tc_resnet44_q32_last_euler10_coupler_snr_others_off_v2/propagated_snr.csv) |
| ON, matched between forwards | 15.60 dB | 16.59% | [43 locations](../results/tc_resnet44_q32_last_euler10_coupler_snr_others_on_v2/propagated_snr.csv) |

Higher signal-to-error ratio here does not imply higher accuracy.
[OFF configuration/checks](../results/tc_resnet44_q32_last_euler10_coupler_snr_others_off_v2/),
[ON configuration/checks](../results/tc_resnet44_q32_last_euler10_coupler_snr_others_on_v2/).

### Timing noise: independent samples and recalibration

`per_batch` shares timing noise across a batch; `per_sample` draws per image.
Below: ResNet-44, recalibration on 5,120 training images, full test set,
all other effects ON.

| Fixed timing standard deviation | Timing-noise standard deviation | Batch-shared noise accuracy | Independent-per-image noise accuracy |
|---|---|---:|---:|
| 0% | 0% | 94.13% | 94.13% |
| 20% | 0% | 94.22% | 94.22% |
| 0% | 20% | 71.40% | 76.55% |
| 20% | 20% | 71.34% | 76.63% |

Calibration/test share scope and fixed variation; noise resamples. State-2
PCN, 22 layers, 20%/20%, no recalibration: **92.72% → 92.93%** for batch-shared
→ per-image noise. Remaining reruns are ongoing:
[results](../results/tc_compute_time_sweep_20261017/results.csv),
[commands, checkpoints and status](../results/tc_compute_time_sweep_20261017/per_sample/).

Four-image validation against batch-size-1 physical-duration solves: maximum
final-logit difference **2.62e-6** (ResNet) / **1.19e-7** (PCN), with matched
steps/noise. Separately chosen PCN steps: **1.65e-3** at tolerance 1e-6,
**1.98e-4** at 1e-7. Within a batch, samples still share step sizes.
No tested prediction changed. [ResNet logits](../results/tc_timing_scope_validation_20261017/cnn/validation.json),
[PCN logits](../results/tc_timing_scope_validation_20261017/pcn_convergence/validation.json).

### Why PCN tolerates end-time changes: measured mechanisms

Same 32 CIFAR-10 images and checkpoints as above; static nonidealities retained.
Isolated-layer tests hold each layer's input fixed and turn current noise OFF.
Numbers are medians across 22 PCN blocks / 43 CNN convolutions, not image medians.

| Measurement | State-2 PCN | ResNet-44 |
|---|---:|---:|
| Layer-output relative RMS change at 80% nominal duration | 6.59% | 19.77% |
| Layer-output relative RMS change at 120% nominal duration | 6.33% | 19.35% |
| Endpoint derivative × nominal duration / output RMS | 33.72% | 100.88% |
| Same derivative estimate after local rail projection | 32.88% | 98.36% |
| Output-state entries on a rail at nominal duration | 0.82% | 0.24% |

**Not evidence of a full fixed point or special FF null-space alignment.**
PCN's nominal FF response to its activated z-state is **1.00×** the response
to shuffled activations of the same magnitude (also 1.00× with nonlinear-R).
Its z-state still moves: even after projection, derivative × nominal duration
/ z-state RMS is 55.54% (231% before projection).

**Supported explanation:** PCN preserves a nonzero input-derived initial state
and adds a dynamic update; its initial-state RMS is 0.97× its final-state RMS
(not a claim that 97% of the output comes from the input). CNN integrates each
convolution from zero, making duration directly scale its signal. PCN also has
96.56% negative z entries and 80.45% locally flat measured-ReLU entries. Its
output is less time-sensitive, but not generally stationary. Output clamping
alone does not explain the difference, as the derivative/projection rows show.

With current noise ON and matched noise draws/normalized steps, adding 20%
fixed variation plus 20% independent timing noise changed final logits by
**21.39% relative RMS, 1/32 predictions** for PCN versus **102.43%, 17/32** for
CNN. These are subset sensitivity checks, not new full-test accuracies: CNN
was recalibrated at nominal timing and then frozen across the paired passes.
Natural timing scopes are unchanged: one factor per coupled PCN block, one per
CNN convolution. Both use drift × factor and diffusion × square-root factor;
the uniform ±20% tests above do not depend on different random-factor draws.

Sources: [PCN layer/state probes](../results/tc_timing_robustness_20261017/pcn_state_analysis/layer_endpoints.csv),
[CNN layer probes](../results/tc_timing_robustness_20261017/cnn/layer_endpoints.csv),
[PCN propagated errors and controls](../results/tc_timing_robustness_20261017/pcn_checked/),
[CNN propagated errors and controls](../results/tc_timing_robustness_20261017/cnn/).
Runner: `diagnostic_scripts/study_timing_robustness.py`; no production logic changed.

Launches: [TC PCN](current_pcn_reference.md), [TC feedforward](tc_feedforward.md).

## Digital sRGB CIFAR/ciFAIR baseline support — 2026-10-19

`baseline/train_baseline_cifar.py` accepts `dataset=cifair10/cifair100` with
`img_type=rgb`. These names use ordinary CIFAR training data, the same
normalization and class count, and official ciFAIR RGB test data. Existing
`dataset=cifar10/cifar100` and sensor/RGGB `img_type=CiFAIR` are unchanged.
The new aliases are currently restricted to digital RGB baseline training;
they do not select the hardware/sensor pipeline.

With `validation_mode=true`, checkpoint selection uses the same persistent
45k/5k CIFAR training split. After training, the launcher reloads the saved
best checkpoint and evaluates all 10,000 official ciFAIR test images. It
writes `test_results.json` next to `baseline_config.json`; the logged best
training-run accuracy remains **validation**, not ciFAIR test accuracy.
Checkpoint metadata records `evaluation_dataset`; the internal canonical
`dataset_name` remains CIFAR for normalization, losses and top-5 handling.

```bash
for task in cifair10 cifair100; do
  python baseline/train_baseline_cifar.py \
    --model_name wrn_flexible_cifar_avgpool_shortcut \
    --wrn_depth 28 --wrn_first_stage_channels 27 \
    --dataset "${task}" --img_type rgb --case custom_noresize \
    --validation_mode false --final_eval_only true \
    --output_dir "./saved_ckpt_runs/wrn28_c27_${task}_srgb_digital_fulltrain"
done
```

This preferred full-training variant uses the complete **50,000-image**
training pool and performs no test evaluation during epochs. The trainer
evaluates once after the last epoch and immediately saves those same final
weights as `_last_ckpt.pth`; there is no optimizer or BN update between the
test and saving. The launcher records that result in `test_results.json`
without a second test pass. Set `validation_mode=true` and omit
`final_eval_only` to instead use the held-out-validation/best-checkpoint
protocol described above.

This retains the digital default recipe: PT300, SGD LR 0.1, momentum 0.9,
cosine schedule, warmup 5, WD 0.0005, batch 128; RandAugment
`rand-m9-mstd0.5-inc1`, horizontal flip 0.5, Mixup 0.2, CutMix 1.0,
label smoothing 0.1 and random erasing **0.25**. No physical conversion,
quantization or distillation is enabled. Independent ColorJitter is suppressed
by timm when this RandAugment policy is active. Architecture remains stem16,
stages27/54/108, BN, intermediate ReLU6, final dropout0.25, stride-2 main
convolutions and AvgPool shortcuts. This is not the older RGGB/hardware PT
recipe (notably erasing, warmup, WD and input representation differ).

The reusable `baseline/evaluate_baseline_cifar.py --checkpoint PATH --dataset
cifair10` (or `cifair100`, `cifar10`, `cifar100`) rebuilds the architecture from
the saved run config, strictly loads weights and evaluates without training
augmentation. Optional `--max_batches` is for diagnostics only. It rejects
converted/QAT and non-RGB checkpoints rather than guessing their format.
Official ciFAIR classes and release checksums are in `cifar_rgb.py`; no
missing-data fallback to CIFAR test data is allowed. Data root defaults to
`../data`; ciFAIR files remain separate under `ciFAIR-10/ciFAIR-100`.

The narrower WRN digital RGB parameter counts are **1,038,543 / 1,048,353**
for 10/100 classes. Layer-by-layer reports, including the distinct historical
four-channel counts, are in
`results/narrow_wrn_model_summaries/wrn28_c27_cifair{10,100}_torchinfo.txt`.
Regenerate them with `diagnostic_scripts/summarize_narrow_wrn.py`.

## Current focus: single-curve adaptation followed by full-bank evaluation

Goal: compare PCN and feedforward CNN adaptation to one circuit realization,
then accuracy retention across the complete circuit population. Repeat the
CiFAIR-100 study on CiFAIR-10 and extend CiFAIR-100 to ResNet-110. A flatter
curve is not necessarily a representative training curve.

### Submitted BlueHive jobs — reported 2026-10-15

Submission is confirmed by the user's `sbatch --parsable` output. Downloaded
logs/results were checked on **2026-10-18**: PCN completed both families;
both ResNets completed PT but failed the ideal-physical sanity gate before FT.
Jobs are independent Slurm jobs,
not a dependency queue. Each singleton job processes original then improved
circuit sequentially, branching from the same pretrained checkpoint.

| Job ID | Experiment | Expected stages |
|---|---|---|
| 2153379 | PCN / CiFAIR-10 singleton study | Existing PT → FT50 → MC45, for each family |
| 2153381 | ResNet-110 / CiFAIR-10 singleton study | PT300 once → select q inverse → FT50 → MC45, for each family |
| 2153382 | ResNet-110 / CiFAIR-100 singleton study | PT300 once → select q inverse → FT50 → MC45, for each family |

All MC45 stages request expanded Level 3, all 45 corners, one trial per corner,
and no BN recalibration. The singleton FT curves are TT mcparamset **73** for
`coupler_monte` and **34** for `coupler_monte_v2`; evaluation uses the full
corresponding family bank. Measured-ReLU fusion is enabled.

The submitted commands used `RUN_ID="$(date +%Y%m%d)"`. Its expanded value was
not included in the pasted submission output: recover the actual suffix from
the remote paths/logs rather than recomputing the date when downloading later.
Paths below are relative to the remote repository root.

| Job | Submitted tag/prefix (`D` = actual submission `RUN_ID`) |
|---|---|
| PCN / CiFAIR-10 | `tt_representative_pcn_cifar10_D` |
| WRN / CiFAIR-10 | `tt_representative_wrn_c27_cifar10_D` |
| ResNet / CiFAIR-10 | `tt_representative_resnet110_cifar10_D` |
| ResNet / CiFAIR-100 | `tt_representative_resnet110_cifar100_D` |
| Full-range WRN FT140 | `coupler_full_range_CiFAIR100_wrn28_c27_fullFT140_15ns_vdd0.5_qinv20_D` |
| V2 WRN FT140 | `coupler_monte_v2_CiFAIR100_wrn28_c27_fullFT140_15ns_vdd0.1_qinv5_D` |

#### Local counterpart of Slurm job 2153380 — launched 2026-10-16

Use the same `MODEL_KIND=wrn_c27`, `TASK=cifar10`, and representative singleton
CSVs as the submitted WRN/CiFAIR-10 job above. The Slurm worker sources
`run_tt_singleton_study.sh`; the local command below calls that same body,
changing only the repository location, activation/submission wrapper and
artifact tag. The study and FT/eval orchestration scripts are unchanged from
`55d5cc7`; the local checkout at launch is `178db7e`. Actual remote resolved
logs for job 2153380 have not been downloaded, so equivalence is to the pinned
submission and code path, not a verified bitwise replay of remote execution.

Stages: PT300 once (seed 4096, LR 0.1, SGD/cosine, fixed 15 ns from the start,
VDD=0.5, q inverse 5, scale recipe 1, 12-bit uncentered input, SRRL and standard
CiFAIR training augmentation); then, sequentially for `coupler_monte` TT
mcparamset 73 and `coupler_monte_v2` TT mcparamset 34, select q inverse on
1,280 training samples/10 batches, FT50 at LR 0.005, and expanded L3 MC45
(45 corners, one trial each, no BN recalibration). Both FT branches start from
the same newly pretrained checkpoint, use VDD=0.1 and fixed 15/15 ns, measured
TT current-ReLU/pooling, and the selected q inverse. No intermediate weight
division is needed because PT already uses 15 ns. Do not confuse this with
the separate CiFAIR-100 FT140 reruns.

Exact detached local command (already launched; do not paste again to monitor):

```bash
source /home/rongzeng/anaconda3/etc/profile.d/conda.sh
conda activate scanbase
cd /home/rongzeng/_workspce_old/repos/pcn/collaboration/ScAN-PCN
mkdir -p logs/local_runs

(
  export MODEL_KIND=wrn_c27 \
         TASK=cifar10 \
         STUDY_TAG=tt_representative_wrn_c27_cifar10_local_20261016 \
         COUPLER_MONTE_TT_SINGLETON=hardware_data/mc_45_corners/coupler_monte_tt_25_1_mc73_singleton.csv \
         COUPLER_MONTE_V2_TT_SINGLETON=hardware_data/mc_45_corners/coupler_monte_v2_tt_25_1_mc34_singleton.csv

  exec nohup setsid bash ./launch_scripts/run_tt_singleton_study.sh
) > logs/local_runs/tt_representative_wrn_c27_cifar10_local_20261016.runner.log 2>&1 < /dev/null &
disown
```

Overall log: `logs/local_runs/tt_representative_wrn_c27_cifar10_local_20261016.runner.log`.
Live pretraining log: `logs/tt_representative_wrn_c27_cifar10_local_20261016/pretrain.log`.
Startup verified on GPU: detached pipeline PID 320488, pretraining PID 320495,
with actual epoch-1 batch progress. The logged training arguments confirm
CiFAIR-10, depth 28/c27, seed 4096, PT300/LR 0.1, fixed 15 ns, VDD=0.5,
q inverse 5, SRRL, and standard pretraining augmentation. These PIDs are
startup identifiers, not permanent proof of a currently running process.
Later family selection and FT/eval logs are in that same study log directory;
`status.log` records `STUDY_COMPLETE` or the exit status. A failed stage prevents
subsequent stages through the existing pipeline's `set -euo pipefail` behavior.


#### What to download after completion

Keep the relative directory structure. In this checklist, `S` is one of the
four singleton tags above, `F` is `coupler_monte` or `coupler_monte_v2`, and `P`
is one of the two full-FT prefixes above. Substitute real names, not literal
`D`, `S`, `F`, or `P`.

| Artifact | Remote location | Why it is needed |
|---|---|---|
| Six Slurm logs | `logs/slurm_jobs/slurm_<JOB_ID>.out` for the six IDs above | Startup failures, actual training commands, PT/FT best accuracy/epoch, and successful stage handoffs |
| Singleton pipeline logs | `logs/${S}/` for all four jobs | `pretrain.log` for CNNs, family selection/FT/eval logs, and `status.log`; PCN has no new pretraining log |
| PCN FT's separate logs | `logs/${S}_${F}/` for the PCN job, both families | Preserve the stage's own training log as well as its pipeline copy |
| CNN q-selection records | `results/${S}_${F}_selection/` for the three CNN jobs | Six `selection.json` files, `selected_one_over_q.txt`, source/curve hashes, sanity checks and candidate scores |
| Singleton MC45 records | `results/${S}_${F}_mc45/` for all four jobs | Eight result directories, including `summary.md`, `corner_trials.csv`, all corner logs and available resolved configuration/trial outputs |
| Full-FT MC45 records | `results/${P}_mc45/` for both reruns | Two result directories, including overall `evaluation.log`, summary/CSV, all corner logs, `resolved_config.json` and trial outputs |
| New CNN PT checkpoints/configs | `saved_ckpt_runs/${S}_pretrain_iq12/` for the three CNN jobs | Three selected pretrained models, `baseline_config.json` and preprocessing metadata |
| Singleton FT checkpoints/configs | PCN: `saved_ckpt_runs/${S}_${F}_ft/`; CNN: `saved_ckpt_runs/${S}_${F}_ft_iq12/` | Eight FT models; retain best and full-parameter-best checkpoints, configuration files and preprocessing metadata |
| Full-FT checkpoints/configs | `saved_ckpt_runs/${P}_ft_iq12/` | Two FT models, best/full-parameter-best checkpoints, configuration files and preprocessing metadata |

The two existing source roots are the CiFAIR-10 PCN root listed below and the
independent CiFAIR-100 WRN ÷3/15-ns root used in the full-FT launch command.
Their corresponding checkpoints already exist locally. Record their remote
SHA-256 hashes; download them too if a matching local copy cannot be verified.

Also retain the submitted launcher/code revision (`git rev-parse HEAD`), any
uncommitted patch, and the two training singleton CSVs. Record the revision used
at launch, if known: the checkout revision at download time is not proof of the
code used by an earlier job. Resolved runtime logs/configurations take priority
over current defaults. A Slurm accounting export with state/exit code/elapsed
time for these six IDs is useful to distinguish completion from interruption.

**Do not download expanded-weight caches or datasets for initial analysis.**
They are not needed to reconstruct the tables. Keep them on the remote machine
for reuse. If transferring whole checkpoint folders is inconvenient, retain
the best/full-parameter-best `.pth` files and every JSON configuration; retain
`latest_ckpt` and the original failure logs for any interrupted job.

Expected complete analysis inventory: **3 new CNN PT results, 6 q selections,
10 FT results, and 10 MC45 result sets (450 corner trials)**. Verify all 45
corner logs against each CSV before computing a mean; do not silently average
a partial or stale aggregate. No results have been inferred from submission.

#### Local full-FT v2: queued BN-recalibration control — 2026-10-16

This is the **local** pipeline with prefix
`coupler_monte_v2_CiFAIR100_wrn28_c27_fullFT140_15ns_vdd0.1_qinv5_20261015`,
not a dependency on remote Slurm job 2153384. Its FT→unrecalibrated MC45 process
was identified as PID 125149 (start ticks 3613810). A detached user service,
`scan-wrn-fullft-v2-bnrecal-20261015.service`, is queued behind that exact
process identity. The pipeline and its queued control are now **COMPLETE**:
FT best 57.69% (epoch 110), unrecalibrated MC45 mean 35.9664%, recalibrated mean
48.1891% (+12.2227 pp). Both CSVs contain 45 unique corner trials and match their
individual logs. User lingering is enabled, so the service is independent of
terminal lifetime.

- Require predecessor exit, its final FT+MC45 success message, and exactly 45
  unique baseline trial records matching the individual corner logs. A missing
  success marker, partial CSV or log mismatch fails visibly without evaluation.
- Read the predecessor's actual `resolved_config.json`; replay every evaluator
  setting, checkpoint, seed, hardware bank and expanded-cache path unchanged.
  The only recipe changes are `bn_recalibrate=true`, 5,120 training calibration
  samples and calibration batch size 256. Use a separate result directory;
  never overwrite the baseline. BN stays unfused and dynamic hardware noise
  remains enabled during each corner's recalibration.
- Runner: `diagnostic_scripts/queue_feedforward_bn_recal.py`. Its saved queue
  state, dependency identity/runtime-path snapshot and eventual replay command
  are in `logs/local_runs/${P}_bnrecal_queue/`, with `P` equal to this local
  prefix. Live queue/eval log: `logs/local_runs/${P}_bnrecal5120.queue.log`.
  Recalibrated results: `results/${P}_mc45_bnrecal5120/`.
- Validation: 18 queue/BN/pipeline regression tests passed. Rehearsals through
  both the real queue process and the detached user-service launch path loaded
  a copy of this running FT's current checkpoint, initialized real corner
  hardware and stopped after genuine CSR expansion. Dependency-completion
  artifacts were mocked and all rehearsal outputs isolated under
  `/tmp/scan-fullft-bn-queue-rehearsal-ikkfm6_h/`; these are startup checks, not
  full BN-calibration or accuracy results. No running training/evaluation code
  was changed, and rehearsal outputs cannot satisfy the production dependency.

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
| Narrow WRN c27 | `coupler_monte_v2` | 59 | 59.93 | 10.6609 | 34 | 60.86 | 27.3027 |

Separate flattest-WRN MC45 controls with 5,120-sample BN recalibration reached
40.4618% (`coupler_monte`) and 46.5036% (`coupler_monte_v2`). Do not mix these
with the unrecalibrated primary results.

Both selections are restricted to `TT_V1_T1`. Flattest minimizes
`max_V |R(V)/R(0)-1|`. Representative minimizes the mean, across all 4,500
family curves, of RMS log-resistance distance on 50 voltage points spanning
approximately -0.096 to +0.1 V. Curve IDs are one-based `mcparamset` values.
The singleton CSV repeats the selected realization twice solely to satisfy
the distinct-index FF/FB sampler; the two curves are identical.

### CiFAIR-10 completed single-curve results

These are **CiFAIR-10 top-1 percentages**, separate from CiFAIR-100 above.
The local counterpart of Slurm job 2153380 completed with exit status 0:
PT300 once, then q selection → FT50 → MC45 for the original circuit, followed
by the same sequence for the improved circuit. Both FT branches start from
the same pretrained model (**88.83%**, epoch 295), trained directly at fixed
15 ns; no intermediate weight rescaling is used.

| Model | Family | Representative TT curve, mcparamset | PT best | Selected `one_over_q` | Best FT (epoch) | MC45 mean, no BN recal |
|---|---|---:|---:|---:|---:|---:|
| Narrow WRN c27 | `coupler_monte` (original circuit) | 73 | 88.83 | 8 | 85.10 (46) | **58.1549** |
| Narrow WRN c27 | `coupler_monte_v2` (improved circuit) | 34 | 88.83 | 8 | 86.55 (48) | **68.1158** |
| PCN (local) | `coupler_monte` (original circuit) | 73 | 89.18 | 1 (fixed) | 87.42 (48) | **76.3516** |
| PCN (local) | `coupler_monte_v2` (improved circuit) | 34 | 89.18 | 1 (fixed) | 87.40 (50) | **82.6782** |

Both models use VDD=0.1 V, the selected singleton for nonlinear-R
and measured pooling, TT `relu_current_0p2uA_finer.csv`, and SRRL during FT.
WRN uses fixed 15/15 ns timing and the selected `one_over_q=8`;
PCN uses derived timing and fixed `one_over_q=1`. Evaluation
retains the operating point and uses expanded Level 3, the **full corresponding
45-corner bank**, `relu_monteCarlo` per spin, an `old_ideal` final head, and no
BN recalibration. Both result sets contain 45 unique corner trials (one per
corner), verified against their individual logs. Q-selection records are
preserved; the WRN value 8 selected for both families was not imposed beforehand.

Evidence: [pretraining log](../logs/tt_representative_wrn_c27_cifar10_local_20261016/pretrain.log),
[original-circuit FT/eval log](../logs/tt_representative_wrn_c27_cifar10_local_20261016/coupler_monte_ft_eval.log),
[improved-circuit FT/eval log](../logs/tt_representative_wrn_c27_cifar10_local_20261016/coupler_monte_v2_ft_eval.log),
[original q selection](../results/tt_representative_wrn_c27_cifar10_local_20261016_coupler_monte_selection/selection.json),
[improved q selection](../results/tt_representative_wrn_c27_cifar10_local_20261016_coupler_monte_v2_selection/selection.json),
[original MC45](../results/tt_representative_wrn_c27_cifar10_local_20261016_coupler_monte_mc45/summary.md),
[improved MC45](../results/tt_representative_wrn_c27_cifar10_local_20261016_coupler_monte_v2_mc45/summary.md),
[pipeline completion](../logs/tt_representative_wrn_c27_cifar10_local_20261016/status.log).
The exact WRN launch command is pinned in the local-counterpart section above.

### Consolidated representative-curve results — verified 2026-10-18

Absolute top-1 percentages; each MC45 entry averages **45 corners, one trial
per corner**, without BN recalibration. Original = `coupler_monte`, TT
mcparamset **73**; improved = `coupler_monte_v2`, TT mcparamset **34**.
FT trains on that singleton; MC45 samples the full corresponding family.

| Dataset | Model | Digital PT | Original FT50 | Original MC45 | Improved FT50 | Improved MC45 |
|---|---|---:|---:|---:|---:|---:|
| CiFAIR-10 | PCN, local | 89.18 | 87.42 | **76.3516** | 87.40 | **82.6782** |
| CiFAIR-10 | Narrow WRN c27 | 88.83 | 85.10 | **58.1549** | 86.55 | **68.1158** |
| CiFAIR-100 | PCN | 64.44 | 60.89 | **38.1544** | 61.38 | **50.9271** |
| CiFAIR-100 | Narrow WRN c27 | 63.87 | 59.59 | **20.8229** | 60.86 | **27.3027** |

PCN retains derived timing, VDD=0.1, q inverse=1. WRN retains fixed 15 ns,
VDD=0.1, selected q inverse=8 (CiFAIR-10) or 5 (CiFAIR-100). FT uses the TT
current ReLU; MC45 uses `relu_monteCarlo` per spin, expanded per-coupler
nonlinear-R, measured pooling, spin variation, current noise and DTC
nonidealities. These are matched study protocols, not identical architectures
or operating-point mappings. The separate 200-epoch black reference is not
an FT50 result.

#### Downloaded versus local CiFAIR-10 PCN

Archive `results/cifair_singlecurve_pretrained_resnets_20261018.tar.gz` was
extracted into its original `logs/`, `results/` and `saved_ckpt_runs/` paths;
no existing files were overwritten. The remote prefix is
`tt_representative_pcn_cifar10_20261015`; local is
`tt_representative_pcn_cifar10_local_20261018`.

| Circuit | Remote best FT | Local best FT | Remote MC45 | Local MC45 |
|---|---:|---:|---:|---:|
| Original | 87.50 | 87.42 | 75.7873 | 76.3516 |
| Improved | 87.38 | 87.40 | 82.7389 | 82.6782 |

The resolved FT banners, wrapper settings, input preprocessing, checkpoint
architecture metadata and all 50 logged source-weight statistics match.
All **90 corner command argument maps** match after normalizing repository
roots and output/checkpoint namespaces. Both datasets' PCN/WRN result CSVs
contain 45 unique complete corners; CiFAIR-10 CSV values also match the
individual corner logs. The jobs are recipe-equivalent, **not bitwise reruns**:
host execution differs (Slurm memory-checkpoint threshold 80% versus local
65%), and the logs record different teacher evaluation realizations
(93.68% versus 93.33%). Remote input/teacher files were not included for a
byte-for-byte data comparison. The plots below deliberately use **local** PCN
results, not a mixture of remote and local corners.

#### ResNet-110 failure and correction

Both downloaded ResNet PT checkpoints are intact: CiFAIR-10 **88.07%**
(epoch 300), CiFAIR-100 **63.70%** (epoch 295). Neither job reached FT or MC45;
their missing results must not be treated as zero accuracy.

The post-activation ResNet converter scaled the BN state into voltage
coordinates but left ideal ReLU6 capped at **6**, rather than **6q**. This
broke equivalence to the unitless pretrained network. The correction in
`physical_feedforward.py` scales only toggle post-activation ResNet's ideal
ReLU6 caps, including the stem. Ordinary ReLU, measured block activations,
legacy WRN and TC conversion are unchanged.

Replaying the failed **10-batch, 1,280-training-sample** sanity checks through
the production selector after the fix gives:

| Dataset | Unitless reference | Before fix, physical 8-bit | After fix, physical 8-bit |
|---|---:|---:|---:|
| CiFAIR-10 | 93.6719 | 19.3750 | **93.5156** |
| CiFAIR-100 | 79.7656 | 71.0938 | **79.7656** |

These are training-subset diagnostics, **not full-test accuracies**. Sanity
uses fixed 15 ns, VDD=0.1, q inverse=128, exact weight scale, ideal head and
all nonidealities disabled. Both datasets also passed isolated two-batch
FT → saved checkpoint → MC45 startup rehearsals, stopping after the first
real CSR convolution expansion. No production retraining was launched.
Q inverse=5 in these handoff rehearsals was a test value, not a completed
selection sweep; reruns must execute their normal q-selection stage.
Verification: 46 physical-conversion/selector/pipeline/plot regressions passed,
plus all 24 TC feedforward tests. Syntax checks and `git diff --check` passed.

Downloaded source checkpoints, ready to reuse without repeating PT:

- [CiFAIR-10 ResNet PT](../saved_ckpt_runs/tt_representative_resnet110_cifar10_20261015_pretrain_iq12/cifar10/custom_noresize/resnet110_cifar/custom_noresize_cifar10_resnet110_cifar/custom_noresize_cifar10_resnet110_cifar_best_ckpt.pth)
- [CiFAIR-100 ResNet PT](../saved_ckpt_runs/tt_representative_resnet110_cifar100_20261015_pretrain_iq12/cifar100/custom_noresize/resnet110_cifar/custom_noresize_cifar100_resnet110_cifar/custom_noresize_cifar100_resnet110_cifar_best_ckpt.pth)

#### CiFAIR-10 four-curve percentile plots

Digital anchors are verified in the saved best checkpoints: PCN **89.18%**,
WRN **88.83%**. All curves share the latter as the 100% reference. Black is
the existing **86.9645%** MC45, 200-epoch PCN reference (10 trials per corner);
orange/red/dark-red use the local improved PCN/original PCN/original WRN
single-curve MC45 data above. Each series contributes 45 corner means;
10/15 markers only change display density. Labels retain the earlier figure's
“raw” terminology; those three models have received singleton FT50.

[10-marker PDF](../results/mc45_descending_corner_percentiles_CiFAIR10_singlecurve_relative_wrn8883/descending_corner_percentiles_10_dots.pdf),
[15-marker PDF](../results/mc45_descending_corner_percentiles_CiFAIR10_singlecurve_relative_wrn8883/descending_corner_percentiles_15_dots.pdf).
PNG, exact ranked data and `plot_sources.json` are saved beside the PDFs.
Existing CiFAIR-100 plots were not overwritten.

Reproduce with the persistent generator:

```bash
python diagnostic_scripts/plot_mc45_descending_percentiles.py \
  --dataset cifar10 --relative-to 88.83 --pcn-ideal 89.18 --wrn-ideal 88.83 \
  --pcn-adapted-dir results/coupler_monte_v2_CiFAIR10_qf1_noENOB_ftlr0p002_e200_default45 \
  --pcn-improved-dir results/tt_representative_pcn_cifar10_local_20261018_coupler_monte_v2_mc45 \
  --pcn-original-dir results/tt_representative_pcn_cifar10_local_20261018_coupler_monte_mc45 \
  --wrn-original-dir results/tt_representative_wrn_c27_cifar10_local_20261016_coupler_monte_mc45 \
  --output-dir results/mc45_descending_corner_percentiles_CiFAIR10_singlecurve_relative_wrn8883 \
  --dots 10 15
```

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
[WRN original](../results/tt_singleton_50ep_CiFAIR100_wrn_c27_coupler_monte_fixedTiming15ns_vdd0p1_qinv5_coupler_monte_tt_25_1_mc73_singleton_mc45/summary.md),
[WRN improved](../results/tt_singleton_50ep_CiFAIR100_wrn_c27_coupler_monte_v2_fixedTiming15ns_vdd0p1_qinv5_coupler_monte_v2_tt_25_1_mc34_singleton_mc45/summary.md).

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

Study plan and completion status:

1. **Completed:** representative-34 CiFAIR-100 WRN FT **60.86%** and
   unrecalibrated MC45 **27.3027%**; all four CiFAIR-100 representative rows
   are recorded above. Its improved-circuit plot curve remains optional.
2. The integrated launcher now supports CiFAIR-10/100 and PCN / narrow WRN
   c27 / ResNet-110, with parameterized teacher, source checkpoints, output/cache
   paths and launch/handoff checks. No PCN pretraining is needed.
3. The new-CNN pretraining plan comprises ResNet-110 on CiFAIR-10,
   ResNet-110 on CiFAIR-100, and narrow WRN c27 on CiFAIR-10. The latter is
   **completed locally (88.83%)**, with both FT/MC45 branches finished;
   verify the submitted ResNet jobs' remote artifacts before launching duplicates. Use the WRN
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
   WRN's value is suitable. The plan contains eight 50-epoch FT → MC45 pipelines: two
   families each for CiFAIR-10 PCN, CiFAIR-10 WRN c27, CiFAIR-10 ResNet-110,
   and CiFAIR-100 ResNet-110. CiFAIR-10 WRN and PCN pipelines are complete.
   Both ResNet PT stages are complete; after the ReLU6 conversion correction,
   resume at q selection → FT → MC45, reusing the downloaded PT checkpoints.
   Each model/dataset uses a single pretrained
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
   63.87%; CiFAIR-10: **88.83%** for this local WRN). Report raw accuracies alongside relative
   values. BN-recalibrated CNN runs are optional, separately labeled controls.

The following sections preserve the **earlier fixed-timing/full-range and
single-corner experiments and full-bank WRN reruns**, not the current singleton
protocol.

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
validation scores. L2/L3 entries are arithmetic means, not selected best trials,
except explicitly marked single-trial results. L2 uses five trials per condition;
L3 SS uses three trials per condition on **SS_V2_T2**, except the five-cycle
PCN and the two WRN FT140 reruns' one-trial results extracted from MC45. MC45 columns average all 45 corners, one trial per corner;
do not compare them as though they were the SS-only protocol. “—” means BN
recalibration is inapplicable because the model has no BN; “n/r” means no result
is reported for that configuration/protocol here.

| Model | Parameters | Stage times, ns: PCN y/z; CNN conv1/conv2 | PT best | FT best | L2 no recal | L2 recal | L3 SS no recal | L3 SS recal | MC45 no recal | MC45 recal |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **PCN 16L96C, 5 cycles; full-range, VDD=0.5, q inverse 5** | **1,028,380** | **5 / 15** | **63.90** | **60.02** | **59.120** | — | **57.81** (1 trial) | — | **56.9136** | — |
| PCN 16L96C, 1 cycle; FT/L2 full-range, L3 v2 | 1,028,380 | 5 / 15 | 60.98 | 53.84 | 53.374 | — | 50.647 | — | n/r | — |
| PCN 16L96C, 1 cycle; FT/L2 full-range, L3 v2 | 1,028,380 | 7.5 / 22.5 | 62.28 | 54.67 | 54.086 | — | 51.093 | — | n/r | — |
| PCN 16L96C, 1 cycle; FT/L2 full-range, L3 v2 | 1,028,380 | 25 / 75 | 61.73 | 47.97 | 47.782 | — | 25.327 | — | n/r | — |
| WRN-28-2 AvgPool, BN | 1,468,612 | 15 / 15 | 66.24 | 35.86 | 26.754 | 39.196 | 1.280 | 7.390 | n/r | n/r |
| WRN-28 c27, stride-2 + AvgPool shortcuts, BN; full-range, VDD=0.5, q inverse 5 | 1,048,497 | 15 / 15 | 63.87 | 27.49 | 13.364 | 30.608 | 1.083 | 2.117 | n/r | n/r |
| **WRN-28 c27, stride-2 + AvgPool shortcuts, BN; full-range, VDD=0.5, q inverse 20** | **1,048,497** | **15 / 15** | **63.87** | **57.53** | n/r | n/r | **32.79** (1 trial) | **48.59** (1 trial) | **24.9869** | **44.3233** |
| **WRN-28 c27, stride-2 + AvgPool shortcuts, BN; v2, VDD=0.1, q inverse 5** | **1,048,497** | **15 / 15** | **63.87** | **57.69** | n/r | n/r | **35.55** (1 trial) | **50.67** (1 trial) | **35.9664** | **48.1891** |
| **WRN-28 c27, stride-2 + AvgPool shortcuts, BN; v2, VDD=0.1, q inverse 8** | **1,048,497** | **15 / 15** | **63.87** | **56.46** | n/r | n/r | **32.43** (1 trial) | **50.06** (1 trial) | **33.7211** | **47.1720** |
| WRN-28-2 AvgPool, no BN/no conv bias | 1,464,996 | 15 / 15 | 56.37 | 55.64 | 54.634 | — | 49.933 | — | n/r | — |

**PCN hardware settings for the four PCN rows above:** all use VDD=0.5 V
and `one_over_q=5`; timing/cycle differences are listed in the main table.
Measured ReLU is enabled in FT and hardware evaluation, but its source changes
between FT/L2 and L3 as recorded below.

| PCN rows / stage | Coupler bank (nonlinear-R and measured pooling) | Measured ReLU | VDD (V) | `one_over_q` |
|---|---|---|---:|---:|
| All four PCNs: FT and L2 | `coupler_full_range` | `relu_current_0p2uA_finer.csv`, fixed TT, per-model | 0.5 | 5 |
| Five-cycle PCN: L3 SS and MC45 | `coupler_full_range` | `relu_monteCarlo`, corner-specific, per-spin; SS uses `relu_ssg8.csv` | Same | Same |
| Three one-cycle PCNs: recorded L3 SS | `coupler_monte_v2/ss_85_2.csv` | `relu_monteCarlo/relu_ssg8.csv`, per-spin | Same | Same |

The one-cycle L3 records therefore do **not** use the same coupler bank as
their FT/L2 or the updated five-cycle L3 result. Settings were checked against
the [one-cycle L2 commands](../results/pcn_onecycle_three_timings_level2/commands.json)
and [one-cycle L3 logs/configurations](../results/pcn_onecycle_three_timings_SS_V2_T2/).

The five-cycle PCN MC45 evaluation uses `coupler_full_range`, VDD=0.5 V,
q inverse 5, fixed y/z times of 5/15 ns, and the ordinary baked-quantized
`best` checkpoint; all 45 corners completed with one trial each. Its SS entry
is extracted from this same run. The previous 58.480% SS result used
`coupler_monte_v2` and is superseded here because it used the wrong bank for
this experiment. Evidence: [per-corner results](../results/pcn16l96c_fixed5ns_full_range_mc45_1trial/summary.md),
[resolved configuration](../results/pcn16l96c_fixed5ns_full_range_mc45_1trial/run_config.txt).

The new v2 row is the **same architecture and pretrained source** as the
27.49%-FT c27 row, not a different WRN: stem 16, stages 27/54/108, BN enabled,
stride-2 second convolutions at stage transitions, AvgPool/padding shortcuts.
Both retain fixed 15 ns, q inverse 5 and the 140-epoch full-bank FT recipe.
The rerun changes nonlinear-R and pooling from `coupler_full_range` to
`coupler_monte_v2` and VDD from 0.5 to 0.1 V; measured-ReLU GPU fusion is enabled.
Its FT best is epoch 110. The paired MC45 control changes only BN recalibration
(5,120 training samples, batch size 256; dynamic hardware noise remains on)
and output location, improving the mean by **12.2227 percentage points**.

New-row evidence: [FT/pipeline log](../logs/local_runs/coupler_monte_v2_CiFAIR100_wrn28_c27_fullFT140_15ns_vdd0.1_qinv5_20261015.log),
[unrecalibrated MC45](../results/coupler_monte_v2_CiFAIR100_wrn28_c27_fullFT140_15ns_vdd0.1_qinv5_20261015_mc45/summary.md),
[recalibrated MC45](../results/coupler_monte_v2_CiFAIR100_wrn28_c27_fullFT140_15ns_vdd0.1_qinv5_20261015_mc45_bnrecal5120/summary.md).

The **full-range q-inverse-20 rerun** retains the same independent weights-÷3
pretrained source, fixed 15/15 ns timing, VDD=0.5 V and FT140 recipe as the
historical full-range q-inverse-5 row; its intended recipe change is q inverse
5 → 20, using current code/speed optimizations rather than a bitwise historical
replay. Its best FT is **57.53% at epoch 110**. Both unrecalibrated and
recalibrated MC45 evaluations completed all 45 corners, one trial each. Their
resolved configurations differ only in BN recalibration, calibration sample
count/batch size (5,120/256), and output directory; checkpoint, hardware
settings, seeds and expanded-cache path are unchanged. Dynamic hardware noise
remains enabled during calibration. BN recalibration increases the MC45 mean
from **24.9869% to 44.3233%**, a **19.3364 pp** gain. The SS entries in the
table are extracted from these same two MC45 runs, not separate three-trial
experiments. The historical q-inverse-5 row is retained separately.

Full-range rerun evidence: [FT/pipeline log](../logs/local_runs/coupler_full_range_CiFAIR100_wrn28_c27_fullFT140_15ns_vdd0.5_qinv20_20261016.log),
[unrecalibrated MC45](../results/coupler_full_range_CiFAIR100_wrn28_c27_fullFT140_15ns_vdd0.5_qinv20_20261016_mc45/summary.md),
[recalibrated MC45](../results/coupler_full_range_CiFAIR100_wrn28_c27_fullFT140_15ns_vdd0.5_qinv20_20261016_mc45_bnrecal5120/summary.md).

The **v2 q-inverse-8 rerun** completed FT140 (best **56.46%, epoch 132**) and
both MC45 evaluations. Verified on 2026-10-21: each CSV contains exactly 45
unique corners, one trial each, and every value matches its individual log;
the BN-recalibration queue reports `COMPLETE`. Highest corner / mean accuracy
are **49.68% (`FS_V1_T2`) / 33.7211%** without recalibration and
**50.21% (`SS_V1_T2`) / 47.1720%** with recalibration. SS-column entries above
are specifically `SS_V2_T2`, not the highest corners. BN recalibration uses
5,120 training samples, batch size 256, with dynamic noise enabled, and raises
the mean by **13.4509 pp**. Hardware settings, checkpoint, seeds and expanded
cache are shared between the paired evaluations; only BN calibration and
output location differ.

Evidence: [FT/pipeline log](../logs/local_runs/coupler_monte_v2_CiFAIR100_wrn28_c27_fullFT140_15ns_vdd0.1_qinv8_20261021.log),
[unrecalibrated MC45](../results/coupler_monte_v2_CiFAIR100_wrn28_c27_fullFT140_15ns_vdd0.1_qinv8_20261021_mc45/summary.md),
[recalibrated MC45](../results/coupler_monte_v2_CiFAIR100_wrn28_c27_fullFT140_15ns_vdd0.1_qinv8_20261021_mc45_bnrecal5120/summary.md),
[queue completion](../logs/local_runs/coupler_monte_v2_CiFAIR100_wrn28_c27_fullFT140_15ns_vdd0.1_qinv8_20261021_bnrecal_queue/status.json).

Parameter counts follow the training logs' model-parameter totals, excluding
teacher/distillation auxiliary modules and hardware unrolling. Changing PCN
cycle count or timing does not change its parameter count. The c27 WRN has
approximately 1.96% more parameters than PCN.

The three WRN architectures above were pretrained at 5 ns, then their convolution weights
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

1. **FT results support recurrence in the tested PCNs.** The five-cycle model
   exceeds the best one-cycle result by 5.35 percentage points in FT. Giving
   one cycle the same total nominal duration does not recover the five-cycle
   FT accuracy. The current L3 entries are not a matched recurrence comparison:
   five cycles uses `coupler_full_range` with one trial, whereas the one-cycle
   entries use `coupler_monte_v2` with three trials. Matched-bank L3 evaluation
   is needed before attributing an L3 gap to recurrence.
2. **Higher PT accuracy does not guarantee physical robustness.** The BN WRN's
   PT score is 66.24%, above PCN's 63.90%, but its FT and L3 scores are much lower.
   The parameter-matched c27 WRN also underperforms this PCN.
3. **BN is an important contributor, but recalibration is not sufficient here.**
   L3 recalibration improves the larger BN WRN from 1.280% to 7.390%, and c27
   from 1.083% to 2.117%. The no-BN/no-conv-bias WRN retains substantially more
   accuracy: 55.64% FT and 49.933% L3.
4. **The broader claim remains qualified.** Five-cycle PCN exceeds the tested
   BN-free WRN by 4.38 points in FT, but their PT accuracy, architecture, and
   parameter count differ. Their current L3 entries also use different coupler
   banks and trial counts (`coupler_full_range`, one trial, versus
   `coupler_monte_v2`, three trials), so no matched L3 advantage is claimed.
   These results do not prove that recurrence alone explains every CNN–PCN gap,
   or that PCNs universally outperform CNNs.

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

## Full-FT WRN reruns — existing local/Slurm pipeline

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
individual corners are logged beside it. This is the generic recipe example;
the submitted BlueHive job IDs and date-suffixed output prefixes are pinned at
the top of this document. Submission is not evidence of successful execution.

### Optional coupled-corner CNN FT

The existing full-FT launcher now accepts:

```bash
ACTIVATION_CORNER_MODE=random_per_forward
ACTIVATION_CURVE_SHARING=per_spin
FT_CORNER_COUPLED_SAMPLING=true
```

With this combination, unless explicitly overridden, FT loads
`hardware_data/mc_45_corners/relu_monteCarlo/` (the same ReLU bank as MC45).
The initial corner is `RELU_TTG4_MC1`; training draws replace it each forward.
One common corner is chosen per training forward for nonlinear-R, measured
pooling and measured ReLU; each component draws empirical realizations within
that corner, with ReLU sharing set to per spin. Coupling requires physical
Level-2 FT and complete 45-corner empirical banks (100 curves per corner).
An incompatible bank fails explicitly rather than silently falling back.

Coupling defaults to **false**, so historical commands retain their behavior.
Explicit activation paths/corners still take precedence. Non-coupled random
ReLU retains the previous `relu_current_0p2uA_all.csv` default. The flag, bank
path and sharing mode are recorded in `baseline_config.json`. The coupling
hook is inactive in evaluation mode; the standalone MC45 launcher/evaluator
is unchanged. This option does not couple spin variation or current noise.

Validated with the real current-ReLU and full-range coupler banks: all 45
ReLU corner mappings agree with `MC45CornerData`, two WRN forward/backward
passes stay in the selected common corner across all three components, and
the existing shell launcher forwards the options. The actual c27 full-range,
VDD=0.5/q-inverse-20 launcher also passed print-only initialization with the
original weights-÷3 checkpoint and teacher, reporting all three coupling
participants. Its resolved optimizer/augmentation recipe matches the saved
full-FT configuration. Outputs were isolated under `/tmp`; no production job
was launched as part of adding this wiring.

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

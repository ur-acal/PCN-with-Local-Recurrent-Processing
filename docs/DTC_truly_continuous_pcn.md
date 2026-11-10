# Truly continuous DTC PCN

## Experimental full-schedule Dopri5 with selective landings

Branch: `feature/s2-dtc-selective-landing-dopri5`. This is an opt-in,
deterministic evaluation diagnostic, not a change to existing PT/FT/eval defaults.
Use `S2DTCPulseFFFB`, `method=dopri5`, and
`--s2_dtc_landing_slices none` or a comma-separated list such as `1,2,3,4`.
Omitting the argument retains the existing slice-wise solver. Stochastic current
noise and gradient-enabled execution explicitly fail rather than silently
changing the noise or QAT semantics. Existing curve/spin realizations, measured
activation, clipping, physical scales, and dense/expanded currents are reused.

One solve spans normalized time tau in [0,1], with physical time t=H*tau,
H=cycles*T. The RHS is H times the physical joint-state derivative, using the
instantaneous signed pulse masks. Selected slice k forces a landing at
tau=(c+k/15)/cycles in every cycle c. No other nominal slice or cycle boundary
is forced; the final endpoint is always error-controlled. All 15 selected
slices are supported. At an incoming step endpoint the mask uses its left
limit; the next step starts with the right limit. The implementation uses the
existing Dopri5 stages and error controller, but does not restart at landings
or use the legacy unchecked endpoint finishing step.

DTC leading/falling windows are sampled once for each cycle before integration.
Their actual edges, clipped to that cycle, determine the RHS masks. Mandatory
landings still refer to nominal boundaries, not every perturbed edge. This
differs from the old slice-overlap approximation when DTC nonidealities are on.

The existing `diagnostic_scripts/benchmark_toggle_input_correction.py` supports
`--selective-reference DIR` with its numerical pulse worker. It checks saved
weight, input and hardware-assignment hashes before reusing reference outputs,
then tests none, 1, 1–2, 1–3 and 1–4 at tolerances 1e-4 and 1e-6.
The parent `--s2-dtc --selective-reference DIR` mode runs only one numerical
pulse worker for the requested `--scene` and `--representation`, forwarding
`--selective-timing` and `--selective-timing-rk2` when supplied. It does not
launch the ordinary numerical/timing sweep or regenerate its references.
Raw outputs are under `artifacts/s2_selective_landing/`. Saved reference solves
are not rerun. Runtime is measured on the shared GPU and is not an isolated
production-speed benchmark. These small random-model cases are solver checks,
not trained-model accuracy estimates.

### Saved-reference numerical results

Two fixed images, random six-block C24/48/96 model (one repeat per stage),
five 10 ns cycles, QF1, R=50 kOhm, C=500 fF, VDD=0.5 V, inverse-q=5.
Measured cases use fixed 0906 MC18; nonlinear cases add full-range FS_V2_T1
coupler curves. Scene constraints accumulate from left to right below.
Every entry is relative final-logit L2 error (%) versus the saved slice-wise
pulse Dopri5 1e-6 output. Matching initial-weight, input and curve-assignment
hashes were checked. No stochastic current noise or DTC edge variation.

| Forced slices per cycle | Tolerance | Ideal | + clipping | + measured ReLU | + nonlinear-R |
|---|---|---:|---:|---:|---:|
| None | 1e-4 | 70.014 | 69.733 | 63.262 | 61.331 |
| 1 | 1e-4 | 21.549 | 20.551 | 15.589 | 9.079 |
| 1,2 | 1e-4 | 10.853 | 11.736 | 5.335 | 12.105 |
| 1,2,3 | 1e-4 | 20.320 | 18.884 | 16.057 | 14.031 |
| 1,2,3,4 | 1e-4 | 9.075 | 9.075 | 11.725 | 7.883 |
| None | 1e-6 | 50.827 | 51.989 | 51.873 | 61.647 |
| 1 | 1e-6 | 0.469 | 0.600 | 0.387 | 0.211 |
| 1,2 | 1e-6 | 0.356 | 0.658 | 0.169 | 0.313 |
| 1,2,3 | 1e-6 | 0.375 | 0.328 | 0.301 | 0.256 |
| 1,2,3,4 | 1e-6 | 0.394 | 0.204 | 0.312 | 0.115 |

No forced landings can miss pulse activity even at tight tolerance. Additional
landings do not guarantee monotonic improvement, because they change the
adaptive grid. For the final scene at 1e-6, measured seconds / prediction flips
(out of two images) were: none 2.117 / 2; 1: 16.690 / 0; 1,2: 13.623 / 0;
1,2,3: 13.575 / 0; 1,2,3,4: 9.385 / 0. These timings were collected alongside
ongoing training and cannot establish speedup against historical reference
timings. Raw JSON includes accepted/rejected step counts, accepted times,
RHS counts, peak memory, per-layer errors, prediction flips and accuracy.

Bug checks include all requested landing sets plus all 15, an all-landings
slice-solver comparison, explicit rejection of stochastic noise, dense/expanded
trajectory parity with ideal timing, and same-time FF/FB current parity with
fractional DTC width offsets. The latter isolates operator correctness:
independently adaptive dense/sparse grids need not produce identical trajectories.
Saved expanded-model numerical references were not found, so the four-scene
saved-reference numerical sweep above is dense only.

### Matched shared-GPU evaluation-forward timing

The reference was subsequently rerun alongside all five selective methods on
the same currently shared GPU: same dense six-block model, same two images,
same nonlinear scene and fixed hardware assignments, tolerance 1e-6. One warmup
per method is excluded; medians use three timed repeats with alternating method
order. These are plain `model(x)` evaluation forwards under `no_grad`, with CUDA
synchronization and no trajectory-copy hooks, teacher, backward or optimizer.
The historical numerical harness copied frame states only for the slice-wise
path, so its timings were asymmetric in addition to being collected at different
times. Use the matched measurements below for speed comparisons instead.

| Method / forced slices | Median seconds | Min–max seconds | Speedup vs reference | Logit error (%) |
|---|---:|---:|---:|---:|
| Slice-wise reference | 14.618 | 14.512–15.388 | 1.00x | 0 |
| None | 1.574 | 1.553–2.238 | 9.29x | 61.647 |
| 1 | 10.118 | 10.089–10.274 | 1.44x | 0.211 |
| 1,2 | 8.252 | 7.776–8.554 | 1.77x | 0.313 |
| 1,2,3 | 7.816 | 6.827–7.871 | 1.87x | 0.256 |
| 1,2,3,4 | 7.070 | 5.875–7.113 | 2.07x | 0.115 |

The rerun reference logits exactly equal the saved reference. The ongoing GPU
training was not stopped. Results describe this shared-GPU batch-2 experiment,
not expanded evaluation, training throughput or production-batch speedup.
Raw measurements: `artifacts/s2_selective_landing/matched_timing/dense_nonlinear_pulse_selective_timing.json`.
Reproduce via the existing numerical worker with `--selective-reference` and
`--selective-timing`; no separate benchmark script was added.

Adding `--selective-timing-rk2` repeats the matched benchmark for reference
Dopri5, pulse RK2 N=8 and selective Dopri5 slices 1–4. Same model, images,
hardware, shared GPU, warmup and three alternating-order measured repeats:

| Method | Median seconds | Min–max seconds | Logit error vs Dopri5 (%) |
|---|---:|---:|---:|
| Slice-wise Dopri5 1e-6 | 14.847 | 14.679–15.312 | 0 |
| Pulse RK2 N=8 | 3.479 | 3.363–3.634 | 0.005077 |
| Selective Dopri5 1–4, 1e-6 | 7.058 | 6.890–7.322 | 0.115207 |

For this dense batch-2 case, RK2 N=8 is about 2.03x faster than selective
Dopri5 and has about 22.7x less logit error. This is not an expanded benchmark.
Raw results: `artifacts/s2_selective_landing/matched_timing_rk2/dense_nonlinear_pulse_selective_timing.json`.

### Experimental fixed-grid slice dropout and merging

For dense `S2DTCPulseFFFB` Euler/RK2, `--s2_dtc_slice_policy` accepts `full`
(unchanged default), `drop`, or `merge`. `--s2_dtc_keep_slices 1,2,3,4`
sets mandatory one-based slices; `--s2_dtc_random_k K` samples K additional
indices without replacement from the remainder. By default each layer samples
once per forward, reusing that selection across all five cycles. A direct block call
may override K through `pulse_slice_k=K`; callers may also update the block's
`s2_dtc_random_k` before a model forward. Sampling occurs outside checkpointed
RHS calls and is not repeated during backward. Adaptive landings and reduced
fixed-grid schedules cannot be combined. Expanded reduced schedules explicitly
reject execution for now rather than silently falling back to dense operators.

Drop keeps the 50 ns physical horizon but turns off unselected pulse intervals.
With no stochastic/slow currents, both states are constant in an off interval,
so no RHS evaluation is needed. With those effects enabled, off intervals still
evolve through the existing current/noise machinery with zero pulse masks and
zero active-coupler counts; summing noise is not silently removed.

Merge resolves each selected slice individually. Each contiguous unselected
run G is replaced by an interval of duration |G|*T/15 using average signed pulse
weights sum(P_k for k in G)/|G|, with averaged active counts for coupler diffusion.
This preserves integrated exposure at frozen state, not ordering or the exact
nonlinear trajectory. With N substeps, each retained or merged interval gets N
Euler/RK2 steps. No selected slices means five-step averaged dynamics when N=1;
all selected slices reproduces the existing pulse method. Neither policy adds
an amplitude-compensation gain.

Training/inference CLI options for the reduced policy and its sampling settings
inherit checkpoint metadata when omitted. Fresh models and older checkpoints
without these fields keep the original full-pulse defaults. Explicit values,
including `--s2_dtc_slice_policy full` and `--s2_dtc_random_k 0`, override the
saved settings. Switching a reduced checkpoint to an averaged block requires
explicit `full`; the final corner runner always explicitly selects full pulses.

The trained-model study uses the baked last checkpoint from
`C100_C96L16_s2_avg4_ft_qf2_noCP_eval1_iq12_s2_dtc` (epoch 140). Production
`S2DTCQATTester2State` preserves saved FF/FB scales and converts normalized baked
levels to integer pulse levels without requantizing them. Sixteen fixed test
images receive the checkpoint's 12-bit input quantization and are evaluated
densely with a bounded allocator. Fixed 1022 MC18 ReLU
and shared-per-convolution cascode curves use FS_V2_T1. R=67 kOhm, C=500 fF,
VDD=0.6 V, inverse-q=6, five 10 ns cycles; clipping is on. Current noise, spin
variation, DTC variation, measured-pooling distortion, additional head
quantization/clamping/noise are off to isolate dynamics differences.

Existing script entry point `--s2-slice-study` benchmarks reference Dopri5,
RK2 N=8/N=1, all five selective landing configurations, drop/merge of first four
or alternating eight slices, and mandatory first four plus random K=2/4/7.
Each method has one warmup and three measured sixteen-image eval forwards in
alternating order. Random cases use three seeds, paired across drop/merge;
each seed's layer selections are shared across all images, even if microbatching
is needed. Saved weights, scales and curve assignments are checked throughout.
Raw logits, selections, timings, errors and configuration are recorded under
`artifacts/s2_selective_landing/ft16_slice_study_iq12/`. The earlier partial
`ft16_slice_study/` run was stopped after detecting that its cached inputs
lacked 12-bit preprocessing; do not use that partial run for this comparison.

#### Completed trained-model comparison (16 images, batch 16)

All eighteen methods completed one warmup plus three timed evaluation forwards,
with alternating order and the ongoing GPU training left running. Times below
are medians for all sixteen images, with no gradient graph, backward, teacher,
optimizer, or trajectory-copy hooks. All compared methods used identical baked
weights and scales, fixed curves, inputs and clipping. Reference logits repeated
exactly. No OOM occurred; all sixteen images fit in one batch under the 15%
allocator cap. Reference peak PyTorch allocated memory was 74,324,992 bytes
(this excludes the CUDA context/library workspace and other processes).

Random rows show mean +/- population standard deviation of logit error over
three recorded selection seeds, paired between drop/merge. Fixed rows have
identical errors in all repeats. Prediction flips are relative to reference
predictions on the same sixteen images, not differences in correct counts.

| Method | Median seconds / 16 images | Speedup vs reference | Logit error (%) | Flips / 16 |
|---|---:|---:|---:|---:|
| Slice-wise Dopri5, tol 1e-6 | 21.025 | 1.00x | Reference | 0 |
| Pulse RK2 N=8 | 9.988 | 2.11x | 0.02853 | 0 |
| Pulse RK2 N=1 | 2.031 | 10.35x | 1.47386 | 0 |
| Selective Dopri5, none | 62.634 | 0.34x | 1.20228 | 0 |
| Selective Dopri5, 1 | 54.270 | 0.39x | 1.43879 | 0 |
| Selective Dopri5, 1,2 | 46.511 | 0.45x | 1.45580 | 0 |
| Selective Dopri5, 1–3 | 40.286 | 0.52x | 1.14936 | 0 |
| Selective Dopri5, 1–4 | 34.547 | 0.61x | 1.10927 | 0 |
| Drop: keep 1–4 | 1.040 | 20.22x | 65.681 | 11 |
| Drop: keep alternating eight | 1.266 | 16.61x | 87.149 | 12 |
| Drop: keep 1–4 + random 2 | 1.051 | 20.01x | 50.771 +/- 1.015 | 5–6 |
| Drop: keep 1–4 + random 4 | 1.812 | 11.60x | 39.356 +/- 1.667 | 6–7 |
| Drop: keep 1–4 + random 7 | 2.240 | 9.38x | 22.450 +/- 0.441 | 2 |
| Merge: retain 1–4 | 1.396 | 15.06x | 5.204 | 0 |
| Merge: retain alternating eight | 2.130 | 9.87x | 1.47386 | 0 |
| Merge: retain 1–4 + random 2 | 1.482 | 14.18x | 3.069 +/- 0.568 | 0 |
| Merge: retain 1–4 + random 4 | 1.924 | 10.93x | 2.166 +/- 0.455 | 0 |
| Merge: retain 1–4 + random 7 | 2.625 | 8.01x | 1.500 +/- 0.036 | 0 |

All selective Dopri5 rows use tol 1e-6. Drop/merge use RK2 N=1 per retained or
merged interval. Alternating eight means indices 1,3,5,...,15. For merging,
the intervening omitted runs are single slices, so this configuration does not
coarsen the grid: it equals full RK2 N=1 numerically, with extra setup overhead.

For this trained checkpoint, selective Dopri5 does not beat the slice-wise
reference or RK2 N=8 on the time/error tradeoff. This reverses the speed result
on the earlier random six-block model; those old timings should not be mixed
with this table. Dropout substantially changes the network function. Merging
preserves pulse exposure and is much closer: retaining 1–4 is about 31% faster
than full N=1 but raises logit error from 1.47% to 5.20%; adding random two is
about 27% faster with 3.07% mean error. Timing differences near one second remain
subject to shared-GPU variability. All merge methods retained the reference's
9/16 correct predictions here; this small diagnostic does not establish full
dataset accuracy or behavior after retraining with random schedules.

Persistent outputs in the IQ12 directory include `configuration.json`,
`fixed_test_batch.pt`, every method/repeat's logits, `results.json` (72 records,
including warmups and per-layer selections), and `summary.json`. Existing test
files were extended, not duplicated: all-slice output/gradient parity,
integrated-mask/horizon preservation, no-selection merge equivalence to
five-step averaged RK2, dynamic K with checkpointed backward, and off-interval
summing-noise exposure. Existing pulse, adaptive-reuse and pipeline tests also
pass; syntax compilation and diff whitespace checks pass.

#### Independent random selections per cycle

`--s2_dtc_slice_sampling per_cycle` draws a separate additional-K selection for
each of the five cycles in each layer. Default `per_forward` preserves the old
shared-cycle behavior. In both modes the default selection remains shared by FF/FB,
all weights and all images within that layer/cycle. Mandatory slices are always
retained. All five schedules are sampled before integration and reused during
checkpointed backward. Ideal-DTC pulse masks are still built once per forward
and reused; the mask-preparation optimizations discussed separately were not
implemented as part of this extension.

Matched benchmark: same baked QF2 +4 epoch-140 checkpoint, same sixteen IQ12
images, batch 16, same deterministic hardware settings as above. Verified
checkpoint/input/model hashes allowed reuse of the saved Dopri5 reference.
Shared-cycle controls were rerun alongside independent-cycle cases, with one
warmup and three measured forwards in alternating order. No alternating-eight
or random-seven cases were run. Each pair uses the same three seed numbers;
drop/merge also use matching schedules within a given sampling mode.

| Policy, mandatory 1–4 | Shared-cycle median seconds | Shared error (%) | Per-cycle median seconds | Per-cycle error (%) |
|---|---:|---:|---:|---:|
| Drop + random 2 | 1.316 | 50.771 +/- 1.015 | 1.129 | 47.636 +/- 4.076 |
| Drop + random 4 | 1.520 | 39.356 +/- 1.667 | 1.727 | 35.327 +/- 2.235 |
| Merge + random 2 | 1.527 | 3.069 +/- 0.568 | 1.620 | 2.379 +/- 0.076 |
| Merge + random 4 | 1.874 | 2.166 +/- 0.455 | 2.006 | 1.922 +/- 0.089 |

Errors are mean +/- population SD across the three selection seeds, relative
to saved Dopri5 1e-6 logits. Times are whole sixteen-image eval forwards on the
currently shared GPU; small timing differences should not be interpreted as
precise scheduling-overhead measurements. Fresh controls: RK2 N=8 10.069 s,
0.02853% error; RK2 N=1 1.487 s, 1.47386%; Drop first-four-only 1.075 s,
65.681%; Merge first-four-only 1.474 s, 5.204%. K=0 is bitwise identical between
modes. These fresh timings should not be mixed with the previous table's times.

Measured integration work supports comparable cost: Drop K=2 uses exactly 480
RK2 steps over all sixteen layers in either mode. Merge K=2 averaged 681.67
steps shared versus 672.67 independent; K=4 averaged 885 versus 888. Thus actual
merge counts vary with sampled adjacency, without a systematic work increase.
Mean logit error improved for every tested random setting. All merge runs kept
the same sixteen predictions as the reference; no new training was performed.

All sixteen shared-mode output comparisons (four random configurations times
four forwards including warmup) exactly matched the previous saved logits.
Recorded schedules were inspected for cycle sharing/resampling. New unit tests
check mandatory indices/counts, cycle diversity, zero-K bitwise parity,
drop/merge integration budgets, and checkpointed output/gradient/RNG parity.
Together with existing pulse, adaptive-reuse, pipeline and recovery tests:
153 passed. Syntax and whitespace checks passed.

Raw configuration, per-layer/per-cycle selections, logits, results and summary:
`artifacts/s2_selective_landing/ft16_cycle_sampling_iq12/`. The existing script's
`--s2-slice-study --cycle-sampling-study --reference-dir DIR` mode reproduces
the comparison without rerunning reference Dopri5. No new benchmark/test
script was created; ongoing GPU training was not stopped.

New trained-model slice-study configurations also record a hardware identity:
actual sampled curve assignments, physical block/head configuration, nonlinear
curve tensors and source-file hashes. Cached-reference reuse requires matching
identity and settings in addition to checkpoint, images, weights and microbatch.
Older artifacts without this identity are rejected with an explicit error;
they are not automatically rerun, overwritten, or assumed hardware-matched.

#### Branch-independent selection and reduced-schedule optimizations

`--s2_dtc_branch_sampling independent` draws separate additional-K selections
for FF and FB. Default `shared` retains the previous sampling/RNG sequence.
Combine it with `--s2_dtc_slice_sampling per_cycle` for independent draws in
each cycle. The mandatory set and K are common configuration values, but the
selected extra indices differ. Each branch selection remains shared across
that layer's weights and batch images. All selections are made before solving;
checkpointed backward never resamples them. K=0 consumes no selection RNG.
The new argument is forwarded by the training and inference Python CLIs and
saved with nondefault reduced-schedule metadata. Launch scripts are unchanged.

Drop now prepares only retained branch masks, reuses them within the forward,
and omits inactive-branch convolutions and the unused FF activation evaluation.
Merge still needs all pulse masks but caches each branch's averaged kernels.
Both reuse interval diffusion coefficients and zero-noise flags, avoiding
repeated zero-coefficient device synchronizations. Random noise increments are
still drawn at each numerical step, including dropped gaps when noise is on.
DTC variation invalidates mask/average/noise caches at each cycle as before.
No cache survives a top-level forward.

Independent branch schedules use the union of branch interval boundaries.
For Merge, each branch retains the average over its own original merged run;
splitting a run at the other branch's boundary does not recompute that average.
This preserves frozen-state exposure, but the finer joint grid changes the
numerical approximation. For Drop, each branch still has exactly 4+K active
slice updates per cycle when mandatory slices are 1–4. More union intervals
add state-update overhead, not extra active branch convolutions. Merge can
increase both intervals and convolutions because both branches remain active.

Checks: 172 selected regression tests passed. Added tests cover independent
FF/FB selection in per-forward/per-cycle modes, inactive-branch versus explicit
zero-weight output/gradient parity with checkpointing and noise, integrated
charge per branch, and retained-only mask preparation. A direct comparison
against the pre-optimization methods also passed output, input/parameter
gradient and RNG parity in 16 combinations of Drop/Merge, checkpointing,
current noise and DTC variation. Full-schedule/default pulse behavior is not
changed. Reduced schedules remain dense-only Euler/RK2 experiments.

The matched sixteen-image benchmark reuses the same baked QF2 +4 model and
verified saved Dopri5 1e-6 reference. It measures original shared sampling,
optimized shared sampling and optimized independent FF/FB sampling in one
process, using per-cycle random draws, one warmup and three timed forwards.
The pre-optimization methods are loaded from a preserved source snapshot;
weights, hardware realization and preprocessing are identical across methods.
Timing includes the entire eval forward, not backward. Ongoing GPU training
remains active, so small runtime differences are not reliable speedups.

All times below are seconds for sixteen images, batch 16. Errors are mean
relative logit L2 percentages versus the saved Dopri5 reference, across three
selection seeds. Every random case uses independent draws across cycles.
"Shared" and "independent" here refer only to FF versus FB.

| Policy / kept slices | Before, shared (s) | Optimized, shared (s) | Shared error (%) | Optimized, independent (s) | Independent error (%) |
|---|---:|---:|---:|---:|---:|
| Drop 1–4 | 0.946 | 0.542 | 65.681 | — | — |
| Drop 1–4 + random 2 | 1.065 | 0.785 | 47.636 | 0.945 | 47.659 |
| Drop 1–4 + random 4 | 1.645 | 1.508 | 35.327 | 1.605 | 34.563 |
| Drop 1–5 + random 2 | 1.782 | 0.975 | 36.228 | 1.152 | 36.295 |
| Drop 1–6 + random 2 | 1.324 | 1.028 | 25.827 | 1.136 | 27.825 |
| Merge 1–4 | 1.463 | 1.236 | 5.204 | — | — |
| Merge 1–4 + random 2 | 2.021 | 1.356 | 2.379 | 1.374 | 2.418 |
| Merge 1–4 + random 4 | 1.707 | 1.828 | 1.922 | 2.151 | 1.902 |
| Merge 1–5 + random 2 | 2.171 | 1.291 | 1.932 | 1.486 | 2.219 |
| Merge 1–6 + random 2 | 1.751 | 1.426 | 2.130 | 2.066 | 1.998 |

Fresh full-pulse controls: RK2 N=1 median 2.234 s, error 1.47386%; RK2 N=8
median 9.886 s, error 0.02853%. All forty original/optimized shared output
comparisons (ten policies times four forwards including warmup) were bitwise
identical; all 120 records' mandatory sets, K and branch schedules were checked.
All Merge predictions matched the reference on these sixteen images, whereas
Drop changed predictions substantially. Independent branch sampling did not
systematically improve logit error. Drop shared improved in each tested case;
Merge was not uniformly faster (random 4 regressed in this shared-GPU timing).
Peak diagnostic tensor allocation was below 150 MiB; no training was stopped.

The convolution-count test confirms exactly 60 FF and 60 FB calls per layer
for both shared and independent Drop with mandatory 1–4, K=2, five cycles and
RK2 N=1. Independent sampling increases union-grid bookkeeping, but inactive
branches no longer pay convolution cost. These are eval measurements, not
training-throughput or end-to-end accuracy results.

Results and preserved baseline source:
`artifacts/s2_selective_landing/ft16_branch_optimized_iq12/`.
Reproduce using the existing benchmark's `--s2-slice-study
--branch-sampling-study --reference-dir DIR --baseline-ode-file FILE` options.

## Isolated pulse-FT cache prototype (2026-11-05)

Temporary branch: `perf/s2-pulse-ft-cache`, worktree
`ScAN-PCN-s2-pulse-ft-cache`, based on `6bced59`. The original ordering
worktree and its ongoing training were not edited, restarted or stopped.
There are no new production CLI options and no checkpoint-policy changes.

The prototype implements only these three optimizations:

1. Cache selected S2 nonlinear-R curve lengths and native-grid views within
   each forward, removing repeated scalar device-to-host lookups in the RHS.
2. Prepare exact nonzero-noise masks once for the pulse schedule and pass
   Python booleans to `TCNoiseLifecycle`. Zero coefficients still skip the same
   draws; summing and coupler noise remain independent and fresh each slice.
3. Reuse the 15 dense pulse masks, active-coupler counts and diffusion
   coefficients across five frames within one forward. Graphs are not detached,
   and the cache is rebuilt after every optimizer update/forward. DTC
   variation/jitter bypasses this schedule cache; expanded evaluation retains
   its existing path. Legacy toggle/TC callers do not opt into these caches.

### Full sixteen-layer, small-batch benchmark

Actual C24→48→96, `4:5:4`, sixteen-layer model, loaded from the completed
`C100_C96L16_s2_rk2_pt_iq12_s2_dtc` last checkpoint. Both variants use pulse
RK2 N=1 (75 steps/layer), **full RHS checkpointing, 16/16 layers**, batch 4,
the real EfficientNet teacher/SRRL/SGD step, all configured FT nonidealities,
1022 random-per-forward per-spin ReLU with corner coupling, cascode curves,
R=67 kΩ, C=500 fF, VDD=0.6 V and inverse-q=6. Nonlinear-R input fusion remains
off and measured-ReLU fusion remains on. No approximation is introduced.

RTX 4090 **shared with the unchanged ongoing avg4 run**; OMP/MKL threads=4,
per-process GPU memory cap=20%. Sequential fresh-process order was baseline,
optimized, optimized, baseline. Each process used two warmups and five timed
steps on the same cached real training batch. Timing includes teacher/student
forward, SRRL loss, backward and optimizer; data loading and checkpoint saving
are excluded. No trained checkpoint was saved. These are not idle-GPU or
batch-128 Slurm timing predictions.

| Implementation | Median s/batch (10 timed steps) | Range (s) | Peak allocated GiB | Peak reserved GiB |
|---|---:|---:|---:|---:|
| Baseline | 15.403 | 15.216–15.771 | 2.625 | 3.000 |
| Three caches | 8.177 | 7.075–8.501 | 1.520 | 1.887 |

Measured speedup: **1.884×**, or **46.91% less batch time**. All runs fit
without OOM. Initial model/auxiliary weights and cached batch hashes match.

### Correctness and numerical limits

- Strict deterministic FP32: logits, SRRL features, total loss, auxiliary
  gradients, hardware assignments and ending global/module RNG states match
  exactly. Model-gradient max absolute difference is `1.1921e-7`, relative
  L2 difference `7.1802e-8`; all compared tensors are finite.
- Production precision: logits/predictions, hardware assignments and ending
  RNG states match. Model-gradient relative L2 difference is **0.10738%**
  (max absolute `0.0013801`), versus **0.00424%** (max absolute `0.000054434`)
  in a repeated baseline. Consequently production-gradient parity is **not
  established**; this exceeds the existing diagnostic's 0.1%/0.001 limits.
  Do not claim bitwise-equivalent gradients, identical SGD trajectories or
  final-accuracy equivalence. No tolerance or precision setting was relaxed.
- CPU regression suite: 156 passed, 7 GPU-only skips. New checks cover
  noisy/noiseless output, gradient and RNG parity with/without checkpointing
  and N=1/2; zero-noise skip/replay; rebuilding after weight updates; DTC-jitter
  fallback; curve-cache invalidation; and additive/multiplicative mismatch
  cleanup before checkpointed backward. Existing TC/toggle/checkpoint/head
  regressions, syntax checks and `git diff --check` pass.

Results and raw per-process records:
`artifacts/s2_pulse_ft_cache/summary.json`, `*_time*/result.json`,
`baseline_parity_run1/`, `optimized_parity/`, and `*_production*/`.
These diagnostic artifacts are local and Git-ignored. The existing
`diagnostic_scripts/benchmark_toggle_input_correction.py` was extended, not
replaced by another test script. Its `--s2-pulse-ft-cache` mode accepts
`--source-root`, `--ft-args`, `--batch-cache`, `--batch-size`,
`--memory-fraction`, `--output` and `--phase {parity,parity-production,benchmark}`;
each worker records its complete resolved command. Keep this prototype
isolated pending a decision on the production-gradient difference.

## Scope and experimental setup

The new S2 DTC model updates y and z simultaneously while retaining the
weight-dependent pulse masks. Neither state is held fixed while the other
charges, and neither is reset between frames. These experiments compare
physical averaged dynamics (`S2DTCFFFB`) with physical pulse dynamics
(`S2DTCPulseFFFB`), not unitless pretraining with physical evaluation.

- **Model:** `PCNetNoBatchNorm`, channels 24→48→96, stage repeats 1:1:1,
  six PC blocks, pooling before channel expansion. Random initialization,
  seed 1907; this is not a trained checkpoint.
- **Inputs:** a fixed pair of real CiFAIR-100 images, each 4×16×16, for
  numerical errors; a fixed 128-image batch for timing, reduced only after OOM.
- **Physical settings:** V_DD = 0.5 V, one_over_q = 5, hence q = 0.1;
  R = 50 kΩ and C = 500 fF, so RC = 25 ns.
- **Timing:** five simultaneous FF/FB frames of 10 ns each, 50 ns total.
  The constructor retains t_end = 1.75. Both FF and FB initial weights are
  multiplied by 7/8: the fixed-frame unitless duration 0.4 times 7/8 equals
  the reference step 1.75/5 = 0.35. No optimizer updates occur in these
  numerical comparisons.
- **Initialization:** y from the layer input, z = 0; persistent states across
  all five frames.
- **Quantization:** 5-bit weights, quantization-factor bits = 1, giving
  15 pulse slices per frame.
- **Head and pooling:** ideal bias-free linear head and ideal average pooling
  to isolate recurrent dynamics. When enabled, measured activation scope is
  `all`, including the final activation.
- **Disabled throughout:** noise, spin variation, DTC nonidealities, ENOB
  noise and dropout.

The scenarios are cumulative: **ideal without clipping → add clipping → add
measured ReLU → add nonlinear-R**. The measured-ReLU-only rows are deliberately
omitted from the first table below, but **the “+ Nonlinear-R” rows still include
both clipping and measured ReLU**. Enabling clipping does not guarantee that
this particular batch reaches the rails.

Measured ReLU uses `hardware_data/mc_45_corners/0906_RELU_Voltage/fs_25_2.csv`
(FS_V2_T1), piecewise-linear interpolation, endpoint normalization off, and
fixed per-spin curve assignments with seed 4096. Although the constructor
receives `activation_corner=MC18`, per-spin sharing samples from the bank;
it does not use MC18 everywhere.

Nonlinear-R uses `hardware_data/mc_45_corners/coupler_full_range/fs_25_2.csv`
(FS_V2_T1), conductance curves, `empirical_with_replacement`, seed 4096.
Dense convolutions use per-tensor assignments; expanded convolutions use
per-coupler assignments on both FF and FB. Within each representation,
averaged and pulse comparisons use identical weights, inputs and sampled
curve assignments. The nonlinear dense/expanded comparison is therefore
not a strict representation-equivalence test: the assignment granularities differ.

## References and measurement definitions

The pulse reference uses Dopri5 with `atol=rtol=1e-6`, separately integrating
each of the 75 pulse slices per layer. The accurately integrated averaged
reference uses the same tolerances on five consecutive 10 ns frames with
states carried between frames. Both were checked against tolerance 1e-4.
These are numerical references, not analytical solutions.

The averaged reference uses `S2DTCFFFB`, not the old `S2NoisyIYAsXZAs0` class.
Their ideal joint equations have the same structure, but the new implementation
retains fixed timing, physical scaling and the selected nonidealities.

All errors in the tables are **relative logit L2 error against pulse Dopri5**:

\[
100\,\frac{\|\mathrm{logits}_{\mathrm{test}}-\mathrm{logits}_{\mathrm{pulse\ reference}}\|_2}
{\|\mathrm{logits}_{\mathrm{pulse\ reference}}\|_2}.
\]

These percentages are **not classification accuracy drops**.

Timing uses an RTX 4090, strict FP32 with TF32 disabled, measured-activation
fusion off, one warmup and two measured iterations; reported time is their
median with CUDA synchronization. Evaluation is forward-only at batch 128.
Training is model forward + backward, excluding teacher inference, optimizer
steps and data loading. Thus these are diagnostic solver-cost measurements,
not full pinned-pipeline batch times.

`CP` means RHS checkpointing. In the first table, training entries show
**seconds @ batch size**. Batch sizes are halved from 128 after OOM; they are
the largest successful tested sizes, not exact capacity limits. Times at
different batch sizes should not be compared as equal-throughput measurements.

## 1. Pulse Euler/RK2 with \(75N\) steps versus pulse Dopri5

The timing grid below focuses on **RK2**. N is substeps per pulse slice:
N = 1, 2, 4, 8 means 75, 150, 300, 600 total steps per layer respectively.
All forward-only timings use batch 128.

| Scenario | N | Error dense / expanded (%) | Dense forward (s) | Dense training (s @ batch) | Dense training + CP (s @ batch) | Expanded forward (s) |
|---|---:|---:|---:|---:|---:|---:|
| Ideal | 1 | 0.3978 / 0.3978 | 0.403 | 0.923 @ 128 | 1.216 @ 128 | 2.357 |
| Ideal | 2 | 0.1083 / 0.1083 | 0.544 | 1.344 @ 128 | 1.784 @ 128 | 4.464 |
| Ideal | 4 | 0.0280 / 0.0280 | 0.824 | 2.185 @ 128 | 3.039 @ 128 | 8.682 |
| Ideal | 8 | 0.0071 / 0.0071 | 1.380 | 3.862 @ 128 | 4.824 @ 64 | 17.125 |
| + Clipping | 1 | 0.3978 / 0.3978 | 0.416 | 0.999 @ 128 | 1.311 @ 128 | 2.374 |
| + Clipping | 2 | 0.1083 / 0.1083 | 0.577 | 1.493 @ 128 | 1.971 @ 128 | 4.493 |
| + Clipping | 4 | 0.0280 / 0.0280 | 0.883 | 2.087 @ 64 | 3.232 @ 64 | 8.741 |
| + Clipping | 8 | 0.0071 / 0.0071 | 1.490 | 3.497 @ 32 | 5.837 @ 32 | 17.230 |
| + Nonlinear-R | 1 | 0.2962 / 0.2196 | 0.669 | 1.314 @ 64 | 2.149 @ 128 | 0.709 |
| + Nonlinear-R | 2 | 0.0764 / 0.0568 | 1.061 | 2.102 @ 32 | 3.771 @ 128 | 1.174 |
| + Nonlinear-R | 4 | 0.0196 / 0.0144 | 1.834 | 3.491 @ 16 | 6.963 @ 64 | 2.060 |
| + Nonlinear-R | 8 | 0.0051 / 0.0036 | 3.350 | 6.298 @ 8 | 13.503 @ 32 | 3.832 |

The nonlinear expanded path uses different optimized current-computation
kernels from the ideal sparse path. Its lower runtime does not imply that
nonlinear hardware is intrinsically cheaper. RHS checkpointing also does
not universally reduce memory or runtime; its benefit depends on the path.

## 2. Averaged five-step Euler/RK2 versus pulse reference

### Toy model: randomly initialized six-block 24→48→96 network

Only **RK2** rows are retained below. Every timing entry uses batch 128.
Error measurements use two fixed test images, not the timing batch.

The error column is **final-logit relative L2 error against the matching
pulse Dopri5 reference (atol=rtol=1e-6)**, separately for dense and expanded
couplers. It is not a layer-state error or classification-accuracy drop.
These runs carry z across all five cycles; no cycle-boundary reset is used.

| Scenario | Method | Final-logit relative L2 error, dense / expanded (%) | Dense forward (s) | Dense training (s) | Dense training + CP (s) | Expanded forward (s) |
|---|---|---:|---:|---:|---:|---:|
| Ideal | RK2 | 8.8719 / 8.8719 | 0.012 | 0.032 | 0.043 | 0.144 |
| + Clipping | RK2 | 8.8719 / 8.8719 | 0.013 | 0.036 | 0.047 | 0.148 |
| + Measured ReLU | RK2 | 5.8609 / 5.8608 | 0.019 | 0.045 | 0.068 | 0.153 |
| + Nonlinear-R | RK2 | 2.5783 / 3.0922 | 0.029 | 0.061 | 0.103 | 1.527 |

### Real model: epoch-43 averaged-FT checkpoint, 16-layer C24→48→96

This is the same **final-logit relative L2 versus pulse dynamics** measurement,
but **not a directly controlled extension of the toy-model table**. It uses a
QF1 checkpoint trained with five-step averaged RK2 **plus four ordering pairs**,
256 cached training images (two batches of 128), dense convolutions, and a
**pulse RK2 N=8 reference**, rather than pulse Dopri5. Both methods use identical
quantized weights/scales and static hardware within each comparison. States
persist across five 10 ns cycles; no cycle-boundary reset is used.

| Conditions | Averaged RK2 +4 pairs accuracy (%) | Pulse RK2 N=8 accuracy (%) | Final-logit relative L2 error (%) | Prediction flips / 256 |
|---|---:|---:|---:|---:|
| Quantization, clipping, fixed measured ReLU; other hardware effects disabled | 67.9688 | 60.5469 | **18.1598** | 50 |
| Additional hardware nonidealities enabled | 66.4062 | 62.1094 | **20.4756** | 47 |

The real-model settings are R=67 kΩ, C=500 fF, VDD=0.6 V, inverse-q=6,
fixed **1022 TT MC18** ReLU, versus R=50 kΩ, VDD=0.5 V, inverse-q=5,
**0906** ReLU and full-range coupler curves in the toy study. The hardware-enabled
row adds cascode nonlinear-R, measured pooling, spin variation, fast current
noise and digital-head quantization/clamping/ADC noise. Static draws and ADC
noise are paired; current-noise paths differ between integration grids, so
**20.4756% is not purely deterministic dynamics error**. Slow noise, mismatch
and DTC timing variation remain disabled.

The deterministic **18.1598%** result establishes a substantial averaged-to-pulse
gap for this trained checkpoint, despite four ordering pairs. It does not isolate
training or model size as the cause of the difference from the toy results.
Sources: [deterministic comparison](../results/s2_dtc_avg4_latest_train256/summary.md)
and [hardware-enabled comparison](../results/s2_dtc_avg4_epoch043_train256_nonideal/summary.md),
with checkpoint hashes and full settings in their adjacent configuration files.

### Five-step RK4 follow-up (2026-11-05)

This extends the original random-initialization study, not the pretrained or
in-progress FT checkpoint studies. It reuses the same six-layer 24→48→96
model, two cached images, quantized weights, physical scales and sampled
curves. Clipping, 0906 measured ReLU and full-range nonlinear-R are enabled;
noise, spin variation and DTC timing variation remain disabled. There are
five 10 ns frames. Both references are the original saved Dopri5 solutions
with `atol=rtol=1e-6`, for their respective averaged and pulse equations.

The following table uses **expanded couplers**. Both error columns are final
logit relative L2 percentages, each normalized by its own reference logits.

| Averaged method | Error against accurate averaged solution (%) | Error against pulse reference (%) |
|---|---:|---:|
| Euler, 5 steps | 9.268590 | 9.330954 |
| RK2, 5 steps | 0.479205 | 3.092249 |
| RK2, 10 steps | 0.128335 | 3.165473 |
| RK2, 20 steps | 0.033196 | 3.192203 |
| RK2, 40 steps | 0.008535 | 3.199531 |
| **RK4, 5 steps** | **0.003071** | **3.201418** |
| Accurate averaged, Dopri5 1e-6 | 0 | 3.201913 |

The dense follow-up agrees qualitatively: five-step RK4 has **0.003660%**
error against the accurate averaged solution and **2.707785%** against the
pulse reference, versus **0.513096%** and **2.578313%** for five-step RK2.
Thus RK4 integrates the averaged equation substantially more accurately,
but is slightly farther from pulse dynamics than five-step RK2 in both
representations. This is evidence for partial error cancellation on this
batch, not a universal ranking of RK2 and RK4.

RK4 was tested **only in an isolated diagnostic process**, using the existing
`TorchDiffEqPack` classical RK4 step on the unmodified S2 joint RHS at its
solver boundary. It takes exactly one RK4 step per frame: 30 frame updates
and 120 RHS evaluations across six layers. State projection is retained
inside the RHS and after each frame. Native and substituted Euler/RK2
controls reproduce the original full trajectories within `atol=1e-6,
rtol=1e-4`; control-logit maximum absolute differences are at most
`2.24e-8`. Initialization, input and sampled-assignment hashes match the
original study. No production RK4 support, launcher changes or changes to
ongoing training were introduced.

[Follow-up tables](../results/s2_dtc_rk4_five_step_comparison/summary.md),
[raw results and control checks](../results/s2_dtc_rk4_five_step_comparison/results.json),
and [configuration](../results/s2_dtc_rk4_five_step_comparison/configuration.json).

## Interpretation and provisional choices

- Refining the averaged solver does not consistently reduce its discrepancy
  from pulse dynamics. Five-step RK2 is slightly closer than the accurate
  averaged reference in the ideal/clipped/nonlinear-R cases, but slightly
  farther in the measured-ReLU-only case (about 5.861% versus 5.839%).
  This is partial error cancellation, not a general advantage of fewer steps.
- Five-step averaged RK2 outperformed five-step averaged Euler in every
  tested scenario; the Euler data remain in the source artifacts.
- **Pulse evaluation candidate: RK2, N=4.** Worst tested logit error is about
  0.028%, versus 0.108% at N=2. N=8 costs more for a smaller residual error.
- **Pulse training candidate, if needed: RK2, N=2 with RHS checkpointing.**
  With nonlinear-R this fits batch 128 at 3.77 s; N=4 requires batch 64 at
  6.96 s. Cheap averaged training remains five-step RK2, with its averaging
  discrepancy acknowledged.
- These are **candidates for trained-model validation, not production-validated
  defaults**. The numerical errors use two images and random weights. Compare
  N=2/4/8 on trained-model logits and validation accuracy before fixing N.

## Source code and local artifacts

- Existing diagnostic: [benchmark_toggle_input_correction.py](../diagnostic_scripts/benchmark_toggle_input_correction.py), `--s2-dtc` mode.
- [Complete accuracy/timing CSV, including peak GPU memory](../artifacts/s2_dtc_validation_20261104/numerical/production_grid/accuracy_timing.csv).
- [Benchmark records, including OOM attempts](../artifacts/s2_dtc_validation_20261104/numerical/production_grid/results.json).
- [Full numerical summary](../artifacts/s2_dtc_validation_20261104/numerical/summary.md).
- [Per-layer/frame comparison data](../artifacts/s2_dtc_validation_20261104/numerical/comparisons.json).

The artifact links are local and ignored by Git; this document pins the
selected results independently of those files. This documentation change
does not modify training or evaluation behavior.

## Opt-in RK2 ordering correction

Implemented on `feature/s2-dtc-ordering-correction`, in the separate
`ScAN-PCN-s2-dtc-ordering` worktree. The original `colab_all` worktree and its
uncommitted files were retained. Existing uncommitted work was copied into the
new worktree before the implementation.

### Definition and scope

Let `u=(y,z)`, frame duration `H=10 ns`, and `m=15` pulse slices. `f_k(u)` is
the deterministic joint physical drift during slice `k`; `Df_k(u)v` denotes
its Jacobian applied to the direction `v`. With

\[
\bar f(u)=\frac1m\sum_{k=0}^{m-1}f_k(u),\qquad
C(u)=\frac1{2m^2}\sum_{k>j}\left[Df_k(u)f_j(u)-Df_j(u)f_k(u)\right],
\]

the corrected frame is

\[
u_{n+1}=u_n+H\bar f\!\left(u_n+\tfrac H2\bar f(u_n)\right)+H^2 C_r(u_n).
\]

The skew matrix `S[k,j]=sign(k-j)/(2m²)` has seven nonzero real skew pairs.
`C_r` retains its largest `r` pairs; `r=7` reproduces the full second-order
ordering term. All candidates are compared with pulse Dopri5, **not** with
the seven-pair correction as ground truth. Smooth second-order analysis does
not guarantee accuracy at clipping boundaries or activation knots.

Weighted temporal masks collapse into dense convolution kernels. Pointwise
activation/current slopes and convolutions evaluate the directional terms;
there are no 15-slice ODE solves inside a corrected frame. The sampled
physical curves and spin factors remain unchanged throughout the correction.
The correction draws no additional noise. Existing frame noise is added
after the corrected drift, followed by the existing state projection.

The correction's local slope calculation uses the existing unfused arithmetic:
the fused ReLU/current kernels support only first derivatives. Original RK2
updates retain their configured backends. This preserves gradients through
nonlinear-R slopes rather than detaching the correction.

### Launch settings

Minimal additions to the S2 averaged pipeline:

```bash
TOGGLE_MODE=s2_dtc
S2_DTC_METHOD=rk2
S2_DTC_FT_SUBSTEPS=1
S2_DTC_ORDERING_PAIRS=2       # 0 (default), 1, 2, 4, or 7
```

Use a distinct `EXP_PREFIX` for each correction setting; the long model
filename is not extended. The Python argument is `--s2_dtc_ordering_pairs`.
It is physical-averaged-only: unitless pretraining is unchanged. Pipeline
pulse evaluation does not apply the correction. Checkpoint metadata preserves
the setting for averaged loading, and full recovery rejects a different
setting. Old metadata without this field resolves to zero.

Supported correction path: bias-free dense FF/FB convolutions, 5-bit weights,
one RK2 step per frame. Pulse mode, other solvers/substep counts, expanded
couplers and slow-current noise are rejected when correction is active.
Default-off toggle/TC paths are unchanged.
Corrected averaged evaluation uses `torch.no_grad()`, as the current
validation pipeline does; `torch.inference_mode()` is outside this validation.

### Limitation: ordering correction during full-precision pretraining

The current ordering correction is implemented only for physical averaged
dynamics. Its masks are constructed from the normalized quantized weights
provided by the physical/QAT wrapper. Enabling `S2_DTC_ORDERING_PAIRS=4` does
**not** add correction to unitless pretraining; that path remains unchanged.

This is not a mathematical restriction of unitless coordinates or measured
ReLU. The correction can be transformed consistently with the dynamics under
`v=q*u` and `tau=t/(RC)`. However, full-precision pretraining also needs an
explicit weight-to-pulse representation: either quantize weights as in FT,
or define continuous pulse widths with fractional slice occupancy. The latter
must preserve the averaged RHS, including weight normalization and branch
scales, and needs its own numerical and gradient validation. A coordinate
conversion alone is therefore insufficient. Constructing these masks does
not require running a reference pulse solver.

Likewise, the current `S2DTCPulseFFFB` block requires physical mode; it is not
a unitless full-precision pretraining block. `TOGGLE_MODE=s2_dtc_pulse` selects
pulse **FT**, but the pipeline still pretrains with averaged `S2DTCFFFB`.

### Training experiment plan (PT launched; diagnostic and FT pending)

1. Pretrain one model with five-step averaged RK2, fixed measured ReLU and the
   current pinned fixed-timing recipe. Omit ordering-corrected pretraining.
2. From that checkpoint, compare physical averaged RK2, averaged RK2 plus four
   ordering pairs, and pulse RK2 N=1 against a common pulse RK2 N=8 reference.
   Keep quantized weights/pulse levels, physical scaling, fixed measured ReLU,
   data and clipping identical. Disable noise, variation, nonlinear-R,
   measured-pooling distortion and head nonidealities. Here the shared fixed
   measured activation is the activation definition, not a nonideality.
   Report relative logit L2, prediction-flip rate and accuracy.
3. Branch the same pretrained checkpoint into three independent FT runs:
   five-step averaged RK2; five-step averaged RK2 plus four ordering pairs;
   and pulse RK2 N=1 with RHS checkpointing as needed. Keep the pinned FT
   recipe otherwise unchanged and use distinct output roots.
4. Evaluate each last FT checkpoint using expanded pulse RK2 N=8 and the same
   corner settings, trial counts and seeds. The current pin uses five trials
   each at FS_V0_T0 and FS_V2_T1.

The user clarified that pretraining means averaged S2 DTC, not pulse
pretraining. This plan is an experiment proposal, not evidence of training
convergence. The deterministic comparison uses N=8 as a practical pulse
reference, not an analytic solution.

Local launch commands were derived from the updated main-worktree
`ScAN-PCN/docs/current_pcn_reference.md` pin: CiFAIR-100, 24→48→96 with 4:5:4
stage depths, fixed 1022 MC18 ReLU in PT, cascode couplers, R=67 kohm,
C=500 fF, VDD=0.6 V, one_over_q=6, digital head with 0.5-LSB noise and no
bias. These differ from the older 0906/full-range numerical-study settings
below. The pinned PT/FT learning rates, epochs, warmup and corner-coupled
random-per-spin FT activation recipe are unchanged.

Augmentation is unchanged from that pin. These settings apply to packed
4-channel 16×16 CiFAIR inputs; label smoothing is included for completeness
as a loss setting, not an image transform.

| Setting | Pretraining | FT (all three branches) |
|---|---|---|
| Random crop | 16×16, padding 2 | Same |
| Horizontal flip | Probability 0.5 | Same |
| Random affine | Rotation ±9°, translation up to 12.5%, shear ±5° | Disabled |
| Mixup / CutMix | Alpha 0.2 / 1.0 | Disabled |
| Label smoothing | 0.1 | 0 |
| Random erasing | Disabled (`TIMM_RE_PROB=0.0`) | Disabled |
| Color jitter / RandAugment | Disabled for RGGB | Disabled |

PT's default `timm_aug_level=none` means no preset override, not no
augmentation. FT defaults to `FT_TIMM_AUG_LEVEL=no_aug`, which retains
random crop and horizontal flip. Sources: `train_ode_cifar.py` config
resolution, `baseline/baseline_cifar_configs.py` RGGB presets and
`trainer_timm.py` non-RGB transforms/Mixup implementation.

- [Pretrain-only local command](../artifacts/s2_dtc_training_plan/pretrain_local.sh).
- [FT-and-evaluation local commands](../artifacts/s2_dtc_training_plan/ft_and_eval_local.sh):
  pass `avg`, `avg4` or `pulse1`. Pulse FT enables full RHS checkpointing.

These are local, Git-ignored launch artifacts, not changes to the main
launchers. Activate `scanbase` before invoking them with Bash. They call the
existing pipeline worker with `IS_SLURM=0`; the architecture record was
verified against the pinned scheduler's generated combination. FT discovery
requires the common PT root to contain one matching completed 300-epoch last
checkpoint. Each FT variant has a distinct output root and evaluates its
last checkpoint. Run the deterministic diagnostic before starting FT.

Bash syntax and four mocked-Python worker dry runs passed: PT-only, averaged
FT, four-pair FT and pulse FT. Checks verified argument forwarding, shared
last-PT discovery, per-spin corner-coupled FT, and two-corner five-trial pulse
RK2 N=8 evaluation. No training was launched and full-model GPU memory fit
was not measured by these command checks.

#### Actual PT launch verification (2026-11-05)

The user subsequently launched PT. Inspected the live host process (PID
1315810), [launch log](../logs/local_runs/C100_C96L16_s2_rk2_pt_launch.log),
and its saved `_latest_ckpt.pth` metadata. Confirmed:

- `S2DTCFFFB`, `method=rk2`, five cycles, `s2_dtc_substeps=1`; checkpoint
  metadata also records `y_init_with=x`, `z_init_with=0`, normalized weight
  encoding. No ordering correction is enabled. The class's forward uses
  joint y/z midpoint RK2, not alternating toggle updates.
- Fixed 10 ns frames; R=67 kohm, C=500 fF; logged FF/FB relative timing
  scales 0.852878 and initial-weight compensation factors 1.1725.
- Fixed `1022_RELU_Voltage`, `TT_25_1_MC18`, direct unitless pullback with
  q=0.1, VDD=0.6, scope `all`; digital classifier, no bias.
- Requested 300 epochs, batch 128, base LR 0.01, zero warmup, cosine schedule,
  SRRL and Mixup/CutMix enabled, matching the pinned CiFAIR-100 PT recipe.
- The checkpoint inspected had completed two epochs, with recorded total
  training losses 12.7904 and 11.8247. The log had subsequently entered
  zero-based epoch 3. No traceback or training failure was found at inspection.
  Checkpoint validation history was empty: `acc=0` at this stage is not a
  measured zero accuracy (`FINAL_EVAL_ONLY=true`).

The early printed model contains ideal ReLU/ReLU6 modules because that print
precedes measured-activation/ODE conversion; the later conversion and pullback
messages, live arguments and S2 checkpoint metadata identify the actual
training configuration. Verification was read-only; the running process and
implementation were not modified. This confirms launch/configuration, not
eventual convergence or a fresh in-process numerical instrumentation test.

#### Queued post-pretrain diagnostic (2026-11-05)

Queued locally at 00:38 America/New_York in persistent execution session
`73783`, queue PID `1331146` (verified alive after launch). The
[queue command](../artifacts/s2_dtc_training_plan/queue_post_pretrain_validation.sh)
waits for PT launcher PID `1315741` with its recorded process start identity,
then requires the successful TRAIN completion marker. The diagnostic reuses
`find_tc_pretrain` to require a matching completed epoch-300 **last** checkpoint;
it does not accept an interim/latest checkpoint or start FT. A file lock
prevents duplicate queues. Poll interval is 30 seconds.

The existing `diagnostic_scripts/benchmark_toggle_input_correction.py` now
accepts `--s2-pretrained` for this experiment. It caches the first 128 test
images once (12-bit production preprocessing), uses microbatches of 16, and
compares dense physical averaged RK2, averaged RK2 + four pairs, and pulse
RK2 N=1 against the common pulse RK2 N=8 reference. All methods use the same
fixed 1022 TT MC18 curve, state clipping and quantized PC weights. It asserts
matching actual pulse levels, branch scales, classifier weights and activation
buffers. Noise/variation, nonlinear-R, measured-pooling distortion and
classifier quantization/clipping/noise are disabled. Logits are restored to
model coordinates before comparison. Cached input tensors are identical
across methods, including any input realization from the production loader.

- [Queue/progress log](../artifacts/s2_dtc_training_plan/post_pretrain_validation.log).
- Planned results: `results/s2_dtc_pretrained_C100_C96L16_last/summary.md`,
  `results.json`, configuration/checkpoint hashes, fixed test batch and logits.
- This is a one-batch dynamics diagnostic, not full-test or corner accuracy.
- Bash/Python syntax checks passed. A one-image, end-to-end smoke test on an
  immutable copy of an interim checkpoint passed all four forwards, identical
  quantization/curve signatures, data loading and report generation. Only the
  smoke test bypassed completed-checkpoint discovery; the queued command does
  not. Missing-success-marker rejection and successful-marker command dispatch
  were tested separately. No training implementation or live PT state changed.

### Verification records

- Seven-pair skew reconstruction matches the literal matrix.
- The implemented correction and FF/FB gradients match an independent sum of
  all 105 ordered pairs, including a smooth nonlinear-current/activation case.
- Ranks 1/2/4/7 have finite FF/FB gradients; RHS checkpointing preserves outputs
  and gradients. The fused measured-ReLU GPU backward regression also passes.
- Unitless pretraining outputs are bit-identical with the setting enabled or
  disabled. Checkpoint save/load preserves corrected averaged logits and
  disables correction when converting to pulse dynamics.
- Existing toggle/TC, physical-head, pullback, launch-routing and recovery
  regression tests pass. See [test log](../artifacts/s2_ordering_validation/bug_checks.log).
- The real shortened pipeline passes pretrain, corrected FT, and expanded
  pulse evaluation on both `FS_V0_T0` and `FS_V2_T1`. Runtime probes observe
  150 correction calls during FT, finite gradients in all six layers, and
  zero correction calls in pretrain/evaluation. See
  [rehearsal log](../artifacts/s2_ordering_validation/rehearsal_pairs2_fixed.log).
  This is one epoch on four training/four validation images, followed by one
  two-image batch per evaluation corner: a wiring/gradient check, not accuracy
  or convergence evidence.
- A second rehearsal uses four pairs with RHS checkpointing enabled. All
  six FT layers have finite gradients; both expanded pulse corner evaluations
  pass with correction disabled. See
  [checkpointed rehearsal](../artifacts/s2_ordering_validation/rehearsal_pairs4_cp_fixed.log).
- That rehearsal exposed a pre-existing S2 checkpoint issue, reproduced with
  **zero correction pairs**: the QAT wrapper clears forward-pass weight
  mismatch before backward. Recomputed S2 RHS calls consequently omitted
  those multipliers. Each S2 interval now retains its own mismatch realization
  for recomputation. Multiplicative and additive mismatch regression tests
  verify output/gradient parity after wrapper cleanup, with and without
  correction. Existing toggle and TC classes are not changed by this fix.

Numerical checks use the same six-block 24→48→96 model, two fixed images and
five simultaneous 10 ns frames described above. Curves, initialization and
batch hashes are checked against the common pulse reference. Benchmarks are
**dense only**, starting at batch 128 and halving after OOM, in fresh processes.
Forward/backward timings exclude the teacher, optimizer and data loading.
This is not a training-convergence experiment.

Full numerical output and timing/memory records are in
[ordering_summary.md](../artifacts/s2_ordering_validation/numerical_final/ordering_summary.md)
and [ordering_results.json](../artifacts/s2_ordering_validation/numerical_final/ordering_results.json).
The JSON includes per-layer/frame errors, peak allocated/reserved memory, and
failed batch-size attempts. These artifact files are local and Git-ignored.

### Results (final implementation)

All errors below are **relative logit L2 (%) versus pulse Dopri5**, with
`atol=rtol=1e-6`, on the same two images:
`100 * ||logits_method - logits_reference||_2 / ||logits_reference||_2`.
Within each scenario, every method uses the **same common pulse solution**,
with identical weights, inputs, sampled curves and physical settings. This is
not an accurately integrated averaged-ODE reference. Each scenario has its
own matching pulse reference; scenarios after ideal add nonidealities
cumulatively. Dopri5 supplies a numerical reference, not an analytic solution.
The final rerun's 28 saved trajectories are bit-identical to the first
numerical pass; the mismatch-lifetime fix does not change these deterministic
forward calculations.

| Scenario | Pulse RK2 N=1 | Averaged RK2 | +1 pair | +2 pairs | +4 pairs | +7 pairs |
|---|---:|---:|---:|---:|---:|---:|
| Ideal, no clipping | 0.39780 | 8.87194 | 6.23113 | 4.26400 | 2.59634 | 2.25381 |
| + Clipping | 0.39780 | 8.87195 | 6.23113 | 4.26400 | 2.59634 | 2.25381 |
| + Measured ReLU | 0.38171 | 5.86086 | 3.83048 | 2.31737 | 1.08671 | 0.93109 |
| + Nonlinear-R | 0.29618 | 2.57831 | 1.59899 | 0.94148 | 0.59097 | 0.60444 |

Clipping has negligible impact on this particular batch; this is not evidence
that clipping is generally irrelevant.

Dense timings with **clipping + measured ReLU + nonlinear-R**, on RTX 4090:
entries are seconds per batch. All batches are 128 except the marked pulse
F+B entry. CP means RHS checkpointing.

| Method | Logit error (%) | Eval | F+B without CP | F+B with CP | Peak allocated GiB without / with CP |
|---|---:|---:|---:|---:|---:|
| Pulse RK2 N=1 | 0.29618 | 0.6624 | 1.3582 (batch 64) | 2.1400 | 12.101 (batch 64) / 8.245 |
| Averaged RK2 | 2.57831 | 0.0294 | 0.0612 | 0.1038 | 1.550 / 0.641 |
| RK2 + 1 pair | 1.59899 | 0.0668 | 0.1667 | 0.3052 | 2.917 / 1.335 |
| RK2 + 2 pairs | 0.94148 | 0.0781 | 0.2132 | 0.3534 | 3.270 / 1.344 |
| RK2 + 4 pairs | 0.59097 | 0.1102 | 0.2945 | 0.4631 | 3.977 / 1.370 |
| RK2 + 7 pairs | 0.60444 | 0.1435 | 0.4135 | 0.6324 | 5.041 / 1.460 |

The logit-error column uses the two-image accuracy batch above, independently
of the larger timing batches. Averaged methods take five RK2 steps total;
pulse RK2 N=1 takes 75 steps (15 slices per frame, five frames).

Pulse nonlinear-R F+B without CP exhausted memory at batch 128 and was retried
at 64. Every other tested configuration fit batch 128. The full linked summary
also lists timings for ideal, clipped, and measured-ReLU-only scenarios.

Interpretation: two/four pairs are promising cost/error trade-offs for a future
training experiment. Seven pairs is an algebraic full-correction reference,
not a guaranteed smallest network-logit error: four pairs is slightly better
on the nonlinear-R logit metric, while seven has smaller concatenated state
error (0.88805% versus 0.90946%). Even seven pairs was faster than pulse RK2
N=1 here; weighted kernels reuse the pointwise corrections within each frame.
**No convergence or trained-model accuracy claim follows from these tests.**

Final checks: 135 selected regression tests passed, Python/Bash syntax and
`git diff --check` passed. Reuse the existing diagnostic to reproduce the study:

```bash
python diagnostic_scripts/benchmark_toggle_input_correction.py \
  --s2-dtc --ordering-study \
  --batch artifacts/s2_ordering_validation/batch128.pt \
  --output artifacts/s2_ordering_validation/reproduction
```

## Zero-z cycle reset: toy-model benchmark (2026-11-09)

Randomly initialized six-block C24→48→96, seed 1907; same cached two error images and 128 timing images as the original persistent-z study. Five 10 ns cycles, R=50 kΩ, C=500 fF, VDD=0.5 V, inverse-q=5, QF1, 5-bit weights. Only z is zeroed between complete cycles, four resets per layer; y is retained. No reset occurs between PWM slices or integration substeps. No trained checkpoint.

Errors are final-logit relative L2 percentages. References are freshly generated reset-enabled slice-wise pulse Dopri5 at atol=rtol=1e-6. Dense and expanded each use matching weights, scales and sampled curves within their comparisons. Scenarios cumulatively add clipping, fixed per-spin 0906 FS_V2_T1 measured ReLU, then full-range FS_V2_T1 nonlinear-R (dense per-tensor; expanded per-coupler). Noise, spin and DTC variation are disabled. Pooling and final head are ideal.

Timings: strict FP32/TF32 off, one warmup and two measured forwards or forward/backward passes, no teacher/optimizer/data-loading. Forward timings use batch 128; training starts at 128 and halves after OOM in fresh processes. Training cells report seconds @ batch. CP means RHS checkpointing.

### 1. Pulse RK2 versus pulse Dopri5

| Scenario | N | Final-logit relative L2 error dense / expanded (%) | Dense forward (s) | Dense training (s @ batch) | Dense training + CP (s @ batch) | Expanded forward (s) |
|---|---:|---:|---:|---:|---:|---:|
| Ideal | 1 | 0.0593 / 0.0593 | 0.1567 | 0.4601 @ 128 | 0.6559 @ 128 | 2.3577 |
| Ideal | 2 | 0.0242 / 0.0242 | 0.2428 | 0.8198 @ 128 | 1.1503 @ 128 | 4.4616 |
| Ideal | 4 | 0.0077 / 0.0077 | 0.4236 | 1.5242 @ 128 | 2.2340 @ 128 | 8.6830 |
| Ideal | 8 | 0.0022 / 0.0022 | 0.7666 | 2.9175 @ 128 | 3.7800 @ 64 | 17.1975 |
| + Clipping | 1 | 0.0593 / 0.0593 | 0.1684 | 0.5186 @ 128 | 0.7469 @ 128 | 2.3847 |
| + Clipping | 2 | 0.0242 / 0.0242 | 0.2673 | 0.9362 @ 128 | 1.3350 @ 128 | 4.5226 |
| + Clipping | 4 | 0.0077 / 0.0077 | 0.4751 | 1.4057 @ 64 | 2.5846 @ 128 | 8.7838 |
| + Clipping | 8 | 0.0022 / 0.0022 | 0.8593 | 2.5805 @ 32 | 4.8323 @ 64 | 17.3212 |
| + Nonlinear-R | 1 | 0.3467 / 0.2296 | 0.3565 | 0.7678 @ 64 | 1.5392 @ 128 | 0.7116 |
| + Nonlinear-R | 2 | 0.0835 / 0.0560 | 0.6564 | 1.4556 @ 32 | 3.0773 @ 128 | 1.1617 |
| + Nonlinear-R | 4 | 0.0208 / 0.0136 | 1.2315 | 2.7004 @ 16 | 6.0497 @ 128 | 2.0847 |
| + Nonlinear-R | 8 | 0.0052 / 0.0033 | 2.3901 | 5.1996 @ 8 | 12.0497 @ 64 | 3.8415 |

### 2. Averaged five-step RK2 versus pulse reference

| Scenario | Method | Final-logit relative L2 error dense / expanded (%) | Dense forward (s) | Dense training (s @ batch) | Dense training + CP (s @ batch) | Expanded forward (s) |
|---|---|---:|---:|---:|---:|---:|
| Ideal | RK2 | 8.5061 / 8.5061 | 0.0127 | 0.0261 @ 128 | 0.0378 @ 128 | 0.1484 |
| + Clipping | RK2 | 8.5061 / 8.5061 | 0.0137 | 0.0283 @ 128 | 0.0435 @ 128 | 0.1488 |
| + Measured ReLU | RK2 | 3.9600 / 3.9598 | 0.0195 | 0.0363 @ 128 | 0.0622 @ 128 | 0.1536 |
| + Nonlinear-R | RK2 | 3.5921 / 3.9407 | 0.0274 | 0.0486 @ 128 | 0.0892 @ 128 | 1.5229 |

### 3. Averaged refinement: expanded, clipping + measured ReLU + nonlinear-R

| Averaged method | Error against accurate reset-averaged solution (%) | Error against reset-pulse reference (%) |
|---|---:|---:|
| EULER, 5 steps | 10.254670 | 9.515387 |
| RK2, 5 steps | 0.477842 | 3.940672 |
| RK2, 10 steps | 0.118399 | 3.919443 |
| RK2, 20 steps | 0.030172 | 3.932992 |
| RK2, 40 steps | 0.007525 | 3.937806 |
| RK4, 5 steps | 0.010152 | 3.939899 |
| Accurate averaged, Dopri5 1e-6 | 0.000000 | 3.939597 |

Python training/inference flag: `--s2_dtc_reset_z_per_cycle true`.
The training parser also reads `S2_DTC_RESET_Z_PER_CYCLE`, following the
existing `S2_FB_USE_TOGGLE_SCALE` pattern so the pipeline needs no shell changes.
Omitting the flag inherits checkpoint metadata; fresh models and old checkpoints
without that metadata default to false. Explicit false overrides saved true.
No wrapper arithmetic, physical gains, training-scale recipe or launch scripts
were changed. The reset is applied after a complete cycle (and any ordering
correction), not after each solver substep; y and its gradient are retained.
Adaptive full-schedule evaluation restarts at exact cycle boundaries.

All 16 original-study weight/input/curve-assignment identity checks passed.
The largest reference-logit difference between Dopri5 tolerances 1e-4 and 1e-6
was 0.002196%. Reset placement, first-cycle equivalence, default-off behavior,
gradients with/without RHS checkpointing, saved-setting inheritance/override,
and dense/expanded operators (including DTC variation) were checked.
Selected regression suite: 173 passed. Python syntax and `git diff --check` passed.

Interpretation: zero-reset reduces the ideal and measured-ReLU gaps, but
increases the nonlinear-R gap relative to persistent z (dense 2.5783→3.5921%;
expanded 3.0922→3.9407%). It does not universally improve averaged/pulse agreement.
The reset-averaged integration refinement converges to about 3.9396% error
against reset-pulse dynamics; higher integration accuracy does not eliminate
that averaging discrepancy. This zero-reset experiment is distinct from the
earlier auxiliary mean-FB-reference reset, and does not establish FT accuracy.

[Complete raw results](../artifacts/s2_cycle_reset_validation/results.json),
[benchmark tables](../artifacts/s2_cycle_reset_validation/summary.md),
[bug-check log](../artifacts/s2_cycle_reset_validation/bug_checks.log).
Reproduce using the existing diagnostic:

```bash
python diagnostic_scripts/benchmark_toggle_input_correction.py \
  --s2-dtc --reset-study --reset-z-per-cycle \
  --batch artifacts/s2_dtc_validation_20261104/batch128.pt \
  --output artifacts/s2_cycle_reset_validation_reproduction
```

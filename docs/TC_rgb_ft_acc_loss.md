# TC RGB fine-tuning accuracy gap

PT = pretrained accuracy; pre-FT = PT checkpoint evaluated after hardware
mapping, before optimization; FT = accuracy after fine-tuning. Differences
below are percentage points (pp), not relative percentages.

## Reference observations

| Experiment | PT | After FT | Observation |
|---|---:|---:|---|
| Pinned CiFAIR-100 toggle, no input centering | 64.12% | 60.36% | 3.76 pp gap; fixed 0906 ReLU, stronger coupler nonlinearity than current TC. |
| Centered toggle, current ReLU and improved coupler curves | ~60% | ~56% | ~4 pp gap; user-reported reference, exact run/result not linked yet. |
| Old scanGFI TC, ideal clamped ReLU, no input centering | 67.39% | 65.12%; 63.16% with differential mismatch | Recent full-test reproduction; 2.27 pp without additional mismatch. |
| New RGB TC, state 1, 22L/64C | 76.22% | 68.28% | 7.94 pp gap. |
| New RGB TC, state 2, 22L/64C | 75.42% | 68.31% | 7.11 pp gap. |

Old TC therefore has a **4.23 pp** gap with differential mismatch
(67.39% → 63.16%), or **2.27 pp** without additional mismatch.
The reproduced PT is the old launcher's selected 6REP, FT is QAT8a 1REP;
historical parentage is not established by the saved metadata.

Sources: [pinned toggle recipe](current_pcn_reference.md),
[old TC full-test results](../results/legacy_scangfi_recovery/summary.md),
[RGB TC validation ablations](../results/tc_ft_validation_ablations/summary.md).

## Initial mapping versus recovery

- Prior comparisons place toggle's PT→pre-FT collapse as the largest and old
  TC's as the smallest, despite toggle recovering to a smaller final gap.
  This is a cross-experiment observation, not a matched dataset/architecture test.
- Old TC: 67.39% FP → **47.09% pre-FT** → 65.12% FT, on the full test set.
- RGB state 1: **38.92%** with basic 5-bit physical mapping, **26.32%** after
  adding the fixed measured ReLU, **20.46%** with all FT effects. These are
  2,048-image diagnostics, not full-test results.
- Therefore the size of the initial collapse alone does not explain how much
  accuracy FT ultimately recovers.

## Current TC coupler variation is substantially milder

Current TC uses `res_vs_vin_10k_150k.csv` means and the common absolute
covariance from `mc_45_corners/CU_4500_r_vs_vin.csv`. Its fixed FT activation is
`0906_RELU_Voltage/tt_25_1.csv`, column `MC18`, including FT validation.

The rough effective multiplicative **conductance** standard deviation from
the covariance is **~2.5%**: state 1 **2.47%**, state 2 **2.62%**. This is the
RMS standard deviation of mean-resistance/sample-resistance, weighted by each
model's nonzero weight-code histogram and uniformly over the voltage grid
within ±0.1 V; it is not an activation-weighted output-error measurement.

For comparison, old TC's differential-mismatch dictionary gives **22.59%**
under its actual weight-code histogram; its separate training mismatch
setting was **25%**. Including deterministic mean-curve distortion gives a
different TC metric, RMS conductance-gain error, of ~7%, not ~2.5%.
TC FT shares a curve across a tensor, unlike independent parameter mismatch;
smaller scalar variation does not establish weaker regularization overall.
[Calculation details](../results/tc_ft_validation_ablations/recovery_diagnosis.md).

## What the completed ablations establish

Full-test checks at tolerance **1e-6**, using the same final RGB FT weights:

| Evaluation | State 1 | State 2 |
|---|---:|---:|
| All FT effects reproduced | 68.11% | 68.23% |
| Only 5-bit weights + fixed measured ReLU retained | 69.53% | 69.24% |
| Also replace measured ReLU with ideal rail-clamped ReLU | 64.23% | 66.81% |

- Removing the other inference defects jointly recovers only ~1–1.4 pp;
  most of the residual gap remains. Removing nonlinear-R alone does not help.
- Replacing measured ReLU after FT worsens accuracy: the weights have adapted
  to it. This does not establish the result of training with ideal ReLU instead.
- Old TC already used the 0.1 V rail, q=0.1 and adaptive solver. Neither
  clamping nor adaptive solving alone is a demonstrated cause of the new gap.
- Old TC did not train with measured ReLU; toggle has different state headroom
  and dynamics. These references do not isolate their interaction in RGB TC.
- **The cause remains unresolved.** Inference ablations cannot determine
  which effects impaired optimization. Priority matched FT controls, on both
  states: basic quantized/clamped hardware only; then add fixed measured ReLU;
  separately repeat all-on FT without dynamic noise.

## Completed controlled FT results

### Train–test gap and next augmentation control

Full 50,000-image training-split evaluation, with validation preprocessing,
dropout disabled and no gradients, using each model's dense FT hardware
configuration (not unrolled evaluation):

| Final checkpoint, 22Layers6l7l6 | Train top-1 | Final test top-1 | Gap |
|---|---:|---:|---:|
| State 1, all-on | 92.294% | 68.28% | 24.014 pp |
| State 2, all-on | 94.190% | 68.31% | 25.880 pp |
| State 1, mapping-only | 98.032% | 68.03% | 30.002 pp |

Sources: `logs/train_accuracy/{state1,state2,mapping_only_state1}.log`;
original final-test accuracies are checkpoint metadata, and mapping-only is
recorded in `logs/tc_rgb_ft_studies/mapping_only_state1_launcher.log`.
These large gaps support substantial overfitting/generalization limitations:
mapping-only fits training images better without improving test accuracy.
They do not isolate the cause of the entire PT→FT drop or when overfitting began;
all-on evaluation also has stochastic hardware variation.

Next control: restart state-1 `mapping_only` and `all_on` from their same
pretrained 2REP checkpoint, changing **only `--timm_aug_level no_aug` to
`--timm_aug_level none`** (plus distinct output/log paths). Despite its name,
`none` selects the default timm augmentation recipe, not disabled augmentation.
Keep `--timm_re_prob 0.0`, SRRL, 140 epochs, LR 0.005, zero warmup and all
other settings unchanged. Studies below: `mapping_only_timm_aug`, `all_on_timm_aug`.

After the all-on augmentation run hit the 90% process-memory cap, both
augmentation studies now use `--mem_frac 1.0` instead of `0.9`. This changes
the GPU memory allowance only, not the training recipe; it does not guarantee
that physical GPU memory is sufficient. Other study defaults remain unchanged.

### Priority 6: five-step Euler, state 1 — completed

Recorded 2026-09-25 from the user-provided final training log. RGB CIFAR-100,
22Layers6l7l6, 0.6533 M parameters; same pretrained checkpoint and all-on FT
recipe, changing only Dopri5 to five-step Euler. Final epoch: 140.
These are dense FT final-validation results, not unrolled hardware evaluation.

| Configuration | Top-1 | Top-5 | Gap from pretrained top-1 |
|---|---:|---:|---:|
| Pretrained | 76.22% | 95.07% | — |
| Original Dopri5 FT | 68.28% | 88.06% | 7.94 pp |
| Five-step Euler FT | **68.08%** | **88.13%** | **8.14 pp** |

Changing the solver alone did **not** recover the accuracy gap. The 0.20 pp
top-1 difference is inconclusive from one run with stochastic validation;
this weakens a solver-only explanation, but does not establish equivalence
with toggle hardware/dynamics. State 2 remains untested in this FT study.

Checkpoint (relative to the ScAN-PCN repository root):

```text
saved_ckpt_runs/tc_rgb_cifar100_state1_pcn_resnet_depth_study_ft_euler5/TIMMQAT5bNoneaNT0p0mulTIMMPCNetNoBatchNorm_PCConvReLU6_0.0eps_ODEXInitFFFB_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_C100_3K1S64C_0.25Dropout_22Layers6l7l6_2Pool_srrlDistill_a0p3_t2p0_1REP/TIMMQAT5bNoneaNT0p0mulTIMMPCNetNoBatchNorm_PCConvReLU6_0.0eps_ODEXInitFFFB_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_C100_3K1S64C_0.25Dropout_22Layers6l7l6_2Pool_srrlDistill_a0p3_t2p0_1REP_full_param_last_ckpt.pth
```

The name retains the pretrained `dopri5Solver` token; the resolved run used
`--method euler --n_steps 5`. This completed run used the earlier
`_ft_euler5` directory, not the generic `_ft_study_euler5` directory below.

## Controlled FT studies: priority and commands

Restart each study from the same **pretrained 2REP, 22Layers6l7l6** checkpoint
for its state, not from the completed FT checkpoint. Retain SRRL, RGB CIFAR-100,
5-bit weights, ENOB=None, no augmentation, batch 128, LR 0.005, cosine schedule,
140 epochs, zero warmup and Dopri5/tolerance 1e-6 unless changed below.

| Priority | Study name | Only intended change from all-on FT |
|---|---|---|
| 1 | `mapping_only` | Disable measured ReLU, nonlinear-R, measured pooling, spin variation and both noise sources; retain quantization and physical clamps. |
| 2 | `mapping_relu` | Same as priority 1, but retain fixed 0906 TT MC18 measured ReLU. |
| 3 | `no_dynamic_noise` | Disable coupler and summing-current noise, in both FF/FB; retain all other effects. |
| 4 | `cap500fF` | C=500 fF; keep derived timing. This is a different hardware configuration. |
| 5 | `warmup5_epochs200` | Warmup=5, epochs=200; a combined recipe test, not an isolated warmup test. |
| 6 | `euler5` | Euler with five steps; **state 1 completed above**, state 2 pending. Two-state simultaneous Euler is not toggle. |
| 7a | `no_spin` | Disable spin variation only. |
| 7b | `no_pooling` | Disable measured pooling only. |
| 7c | `no_nonlinear_R` | Disable nonlinear-R only. |

Start with priorities 1–3 on both states (six runs). The existing all-on FT is
the reference; optional `all_on` below reproduces its recipe in a new directory.
For each checkpoint, compare its own FT configuration and the original all-on
target configuration, using paired evaluation seeds. Keep final-only test
evaluation/checkpoint selection; repeat promising comparisons with more training
seeds. Inference ablations are not substitutes for these FT controls.

### Local setup (paste once in zsh or Bash)

The function below copies the checked local Euler launcher's training recipe,
but selects Dopri5 by default and gives every study a separate output directory.
It refuses to reuse an existing output directory. The launch helper below uses
`nohup` to survive terminal/SSH closure, runs the selected states sequentially, and
submits no Slurm jobs. Your terminal stays in zsh or Bash; the detached worker
uses non-login Bash with the current environment. No `export -f` is needed.

```bash
cd /home/rongzeng/_workspce_old/repos/pcn/collaboration/ScAN-PCN
conda activate scanbase

tc_ft_study() (
  set -euo pipefail
  cd /home/rongzeng/_workspce_old/repos/pcn/collaboration/ScAN-PCN
  state="$1"
  study="$2"
  case "$state" in
    1) block=ODEXInitFFFB ;;
    2) block=S2NoisyIYAsXZAs0 ;;
    *) echo 'State must be 1 or 2' >&2; exit 2 ;;
  esac
  export TC_NONIDEALITIES=true TC_STATE="$state" TOGGLE_MODE=none SWITCH_INF=false
  export ENABLE_NONLINEAR_R=true ENABLE_MEASURED_ACTIVATION=true
  export ENABLE_MEASURED_POOLING=true ENABLE_SPIN_VARIATION=true
  export ENABLE_SUMMING_CURRENT_NOISE=true ENABLE_COUPLER_NOISE=true
  export TC_NOISE_STAGES=both C_VAL=49e-15 TC_CONV_METHOD=shared
  export TC_CURVE_SAMPLING=histogram TC_EMPIRICAL_CURVE_BANK=""
  method=dopri5; epochs=140; warmup=0; aug_level=no_aug; mem_frac=0.9
  case "$study" in
    mapping_only_timm_aug|all_on_timm_aug) aug_level=none; mem_frac=1.0 ;;
  esac
  case "$study" in
    mapping_only|mapping_only_timm_aug|mapping_relu)
      export ENABLE_NONLINEAR_R=false ENABLE_MEASURED_POOLING=false
      export ENABLE_SPIN_VARIATION=false
      export ENABLE_SUMMING_CURRENT_NOISE=false ENABLE_COUPLER_NOISE=false
      if [[ "$study" != mapping_relu ]]; then export ENABLE_MEASURED_ACTIVATION=false; fi ;;
    no_dynamic_noise) export ENABLE_SUMMING_CURRENT_NOISE=false ENABLE_COUPLER_NOISE=false ;;
    cap500fF) export C_VAL=500e-15 ;;
    warmup5_epochs200) warmup=5; epochs=200 ;;
    euler5) method=euler ;;
    no_spin) export ENABLE_SPIN_VARIATION=false ;;
    no_pooling) export ENABLE_MEASURED_POOLING=false ;;
    no_nonlinear_R) export ENABLE_NONLINEAR_R=false ;;
    all_on|all_on_timm_aug) ;;
    *) echo "Unknown study: $study" >&2; exit 2 ;;
  esac
  source ./launch_scripts/tc_nonideality_args.sh ft
  model="TIMMPCNetNoBatchNorm_PCConvReLU6_0.0eps_${block}_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_C100_3K1S64C_0.25Dropout_22Layers6l7l6_2Pool_srrlDistill_a0p3_t2p0_2REP"
  root="./saved_ckpt_runs/tc_rgb_cifar100_state${state}_pcn_resnet_depth_study"
  output="${root}_ft_study_${study}"
  teacher=./checkpoint/efficientnet_v2_l_cifar100_rgb_OldNoTimm_MatchDistill.pth
  for checkpoint in "${root}/${model}/${model}_last_ckpt.pth" "$teacher"; do
    [[ -s "$checkpoint" ]] || { echo "Missing checkpoint: $checkpoint" >&2; exit 2; }
  done
  [[ ! -e "$output" ]] || { echo "Refusing to reuse: $output" >&2; exit 2; }
  cmd=(python -u train_ode_cifar.py
    --model_name "$model" --save_path "$root" --output_save_path "$output"
    --dataset cifar100 --num_classes 100 --img_type rgb --ckpt last
    --timm_trainer true --timm_sched cosine --timm_aug_level "$aug_level" --timm_re_prob 0.0
    --optim SGD --learning_rate 0.005 --num_epochs "$epochs" --warmup_epoch "$warmup"
    --eval_every 2 --final_eval_only true
    --health_check_epochs 20,50 --health_check_batches 4 --health_check_seed 4096
    --input_quant_bits none --center_student_input false
    --offset_eps 0.0 --dropout 0.25 --avg_pooling true
    --tie_weights false --tie_bp false --bypass false
    --batch_size 128 --tol 1e-6 --t_end 1.75 --scale_train_recipe false
    --patch_node "" --patch_stride "" --patch_cycle "" --patch_pad "" --fold_scalar ""
    --tie_cap false --qat_cls SymQuantizeWeight --pcn PCNetNoBatchNorm --pc_conv PCConvReLU6
    --noise_type mul --pulse_mismatch_training_mode post_quant_amplitude
    --activation_corner_mode fixed --activation_random_curve_sharing per_layer
    --activation_interpolation piecewise_linear --activation_spline_parameters 10
    --activation_fit_constraint auto --activation_normalize_positive_endpoint false
    --slow_summing_current 2.47e-9 --slow_coupler_noise 2.47e-9
    --nonlinear_R_mc_quantity conductance --train_conv_expanded false --nonlinear_R_corner_range all
    --teacher_ckpt "$teacher" --teacher_arch efficientnet_v2_l --teacher_arch_source torchvision
    --teacher_input_size 224 --teacher_center_crop true --adapt_PIL_teacher false
    --distill_method srrl --distill_alpha 0.3 --distill_temperature 2.0
    --reviewkd_weight 1.0 --reviewkd_warmup_epochs 20 --reviewkd_num_stages 4
    --contrast_method memory --test_only false --mem_frac "$mem_frac" --num_workers 2
    "${TC_ARGS[@]}" --method "$method" --n_steps 5)
  if [[ "${TC_STUDY_DRY_RUN:-false}" == true ]]; then
    printf '%q ' "${cmd[@]}"; printf '\n'
  else
    mkdir -p ./logs/tc_rgb_ft_studies
    "${cmd[@]}" 2>&1 | tee "./logs/tc_rgb_ft_studies/state${state}_${study}.log"
  fi
)

tc_ft_detached() {
  local study="$1"
  local selection="${2:-both}"
  case "$selection" in
    1|2|both) ;;
    *) echo 'Usage: tc_ft_detached STUDY [1|2|both]' >&2; return 2 ;;
  esac
  local definition
  definition=$(typeset -f tc_ft_study) || return 2
  local repo=/home/rongzeng/_workspce_old/repos/pcn/collaboration/ScAN-PCN
  local log="$repo/logs/tc_rgb_ft_studies/${study}_state${selection}_launcher.log"
  mkdir -p "$repo/logs/tc_rgb_ft_studies" || return
  nohup bash -c "$definition"$'\n''
    set -e
    case "$2" in
      1|2) tc_ft_study "$2" "$1" ;;
      both)
        tc_ft_study 1 "$1"
        tc_ft_study 2 "$1" ;;
    esac
    echo "Completed state selection $2 for $1"
  ' _ "$study" "$selection" > "$log" 2>&1 < /dev/null &
  echo "$study (state=$selection): launcher PID $!; log: $log"
}
```

### Select states and launch each study

The optional second argument is `1`, `2`, or `both` (default). Every launch
returns immediately. `both` runs state 1 then state 2 sequentially; failure
stops the pair. Separate launch calls run concurrently, so select only the
desired command rather than pasting all launches on one GPU. Do not launch
overlapping state/study combinations simultaneously.
For a command-only check (printed to the launcher log, without training), use
`TC_STUDY_DRY_RUN=true tc_ft_detached mapping_only 1`.
Use `tc_ft_detached` from zsh; the inner `tc_ft_study` sources Bash scripts
and is run by the Bash worker.

```bash
# Alternatives: choose ONE of these.
tc_ft_detached mapping_only 1     # State 1 only, detached
tc_ft_detached mapping_only 2     # State 2 only, detached
tc_ft_detached mapping_only both  # State 1 then state 2, detached
```

For any study below, append `1` or `2` to run only that state; without a
second argument both states run sequentially.

```bash
# Priority 1
tc_ft_detached mapping_only
# Priority 2
tc_ft_detached mapping_relu
# Priority 3
tc_ft_detached no_dynamic_noise
# Priority 4
tc_ft_detached cap500fF
# Priority 5
tc_ft_detached warmup5_epochs200
# Priority 6
tc_ft_detached euler5
# Priority 7: three separate studies
tc_ft_detached no_spin
tc_ft_detached no_pooling
tc_ft_detached no_nonlinear_R
```

Track a selected study (Ctrl-C stops only `tail`, not training):

```bash
tail -f ./logs/tc_rgb_ft_studies/mapping_only_stateboth_launcher.log
```

For a single-state launch, replace `stateboth` with `state1` or `state2`.

Checkpoints: `saved_ckpt_runs/tc_rgb_cifar100_state{1,2}_pcn_resnet_depth_study_ft_study_<study>/`.
Logs: `logs/tc_rgb_ft_studies/state{1,2}_<study>.log`.
These are fresh FT runs, not optimizer-state resumes; pretraining is not rerun.
Validation: Bash syntax and all 20 resolved command comparisons (nine studies
plus the reference, both states) passed against the existing local launcher,
with Dopri5 as reference. Only the listed settings and output directory differ.
This is command validation, not a completed training test; no jobs were launched.

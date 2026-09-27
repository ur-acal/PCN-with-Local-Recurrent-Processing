# Current code issues

Last audited: 2026-09-26. This is the single list of known, intentionally
deferred issues in the current PCN/TC code. Remove an entry when it is fixed;
do not duplicate it across experiment documents.

## Open

1. **Legacy wrapped-inference ENOB fallback.**
   `run_ode_wrapped_inference.sh` now defaults to `enob=none` and extracts a
   numeric ENOB from names such as `TIMMQAT5b8a...`. Older QAT names without an
   encoded `...bNa...` field previously inherited ENOB 8 and now inherit no
   output quantization. Encoded numeric names are unaffected. A future fix
   should preserve the legacy fallback while explicitly parsing both numeric
   and `Nonea` names.

2. **TC empirical-bank inference uses the reference edge path.**
   The optional finite-bank, empirical-with-replacement TC mode is numerically
   correct but does not use the fused Gaussian TC kernel. This is a performance
   limitation only; the default Gaussian per-coupler evaluation remains fused.

3. **Standalone downstream launches do not always prove upstream completion.**
   A separately launched student can accept a validation teacher's best
   checkpoint even if teacher training stopped before its planned final epoch;
   the validation-split metadata proves which split was used, not that the run
   completed. Likewise, CNN `ft_only` and `ft_and_eval` require the exact
   expected pretraining checkpoint path, but do not prove that the originating
   pretraining run completed. These checkpoints are not inherently invalid,
   but they may represent fewer epochs than the intended recipe. Normal
   pipelined pretrain-to-FT execution is unaffected: a failed pretraining
   process prevents FT from starting. This completion guard is deferred.

## Audited and not open

- `sde_noise_type=add` in `run_ode_wrapped_inference.sh` is an intentional bug
  fix and should not be reverted to `mul`.
- `MODEL_DIR`, `BASE_LOGDIR`, `MODEL_NAMES_STR`, and explicit `EVAL_MODE`
  overrides are safe additions.

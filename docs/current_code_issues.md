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

3. **CNN stage-only checkpoint verification is path-based.**
   `ft_only` and `ft_and_eval` require the exact expected checkpoint path and
   reject a missing file, but do not validate saved run metadata or completion
   as thoroughly as `find_tc_pretrain.py` does for PCN. The complete default
   pipeline is unaffected. Do not move an unrelated checkpoint into the
   expected CNN path.

## Audited and not open

- `sde_noise_type=add` in `run_ode_wrapped_inference.sh` is an intentional bug
  fix and should not be reverted to `mul`.
- `MODEL_DIR`, `BASE_LOGDIR`, `MODEL_NAMES_STR`, and explicit `EVAL_MODE`
  overrides are safe additions.

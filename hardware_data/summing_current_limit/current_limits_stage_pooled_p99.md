# Custom combined current limits

Explicit signed total-current bounds. Percentile selection is not used by the evaluator.

Values are written in amperes to preserve parsed bounds exactly (no unit-roundtrip rounding).

Source summary: `results/toggle_summing_current_distribution/C100_C36_72_8_10_analog_nobias_stage_FS_V2_T1_1trial/total/combined/summary.md`
Source summary SHA-256: `dcf2676e8caea6bb3fa948294222d4885f55b23f61dffe7414fcc9cdb2f3e75e`
Source model_name: `TIMMQAT5bNoneaNT0p0mulTIMMPCNetNoBatchNorm_PCConvReLU6_0.0eps_ToggleODEXInitFFFB_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_C100_3K1S72C_0.25Dropout_20Layers8l10l0_1Pool9_srrlDistill_a0p3_t2p0_CiFAIR_1REP`
Source model_dir: `saved_ckpt_runs/C100_C36_72_8_10_allReLU_analog_biasfalse_adc05_iq12_toggle_odexinit`
Source dataset_split: `train`
Source corners: `['FS_V2_T1']`
Source git_commit: `f92b28906390076f59813622bd27e305b84d4585`

| Layer | Branch | Lower (A) | Upper (A) |
|---|---|---:|---:|
| final_linear | COMBINED | -3.0882899999999995e-05 | 1.19796e-05 |
| layer_01 | COMBINED | -2.5089799999999998e-05 | 2.0132999999999998e-05 |
| layer_02 | COMBINED | -2.5089799999999998e-05 | 2.0132999999999998e-05 |
| layer_03 | COMBINED | -2.5089799999999998e-05 | 2.0132999999999998e-05 |
| layer_04 | COMBINED | -2.5089799999999998e-05 | 2.0132999999999998e-05 |
| layer_05 | COMBINED | -2.5089799999999998e-05 | 2.0132999999999998e-05 |
| layer_06 | COMBINED | -2.5089799999999998e-05 | 2.0132999999999998e-05 |
| layer_07 | COMBINED | -2.5089799999999998e-05 | 2.0132999999999998e-05 |
| layer_08 | COMBINED | -2.5089799999999998e-05 | 2.0132999999999998e-05 |
| layer_09 | COMBINED | -2.5089799999999998e-05 | 2.0132999999999998e-05 |
| layer_10 | COMBINED | -4.0671499999999998e-05 | 3.2489400000000001e-05 |
| layer_11 | COMBINED | -4.0671499999999998e-05 | 3.2489400000000001e-05 |
| layer_12 | COMBINED | -4.0671499999999998e-05 | 3.2489400000000001e-05 |
| layer_13 | COMBINED | -4.0671499999999998e-05 | 3.2489400000000001e-05 |
| layer_14 | COMBINED | -4.0671499999999998e-05 | 3.2489400000000001e-05 |
| layer_15 | COMBINED | -4.0671499999999998e-05 | 3.2489400000000001e-05 |
| layer_16 | COMBINED | -4.0671499999999998e-05 | 3.2489400000000001e-05 |
| layer_17 | COMBINED | -4.0671499999999998e-05 | 3.2489400000000001e-05 |
| layer_18 | COMBINED | -4.0671499999999998e-05 | 3.2489400000000001e-05 |
| layer_19 | COMBINED | -4.0671499999999998e-05 | 3.2489400000000001e-05 |
| layer_20 | COMBINED | -4.0671499999999998e-05 | 3.2489400000000001e-05 |

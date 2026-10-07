# Custom combined current limits

Explicit signed total-current bounds. Percentile selection is not used by the evaluator.

Values are written in amperes to preserve parsed bounds exactly (no unit-roundtrip rounding).

Source summary: `results/toggle_summing_current_distribution/C100_C36_72_8_10_allReLU_analog_biasfalse_adc05_iq12_toggle_odexinit_FS_V2_T1_1trials/total/combined/summary.md`
Source summary SHA-256: `41c04070273ebd2663b270ed31bbd42fc5168d48461eb20d8cdda8bd8d3d6e43`
Source model_name: `TIMMQAT5bNoneaNT0p0mulTIMMPCNetNoBatchNorm_PCConvReLU6_0.0eps_ToggleODEXInitFFFB_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_C100_3K1S72C_0.25Dropout_20Layers8l10l0_1Pool9_srrlDistill_a0p3_t2p0_CiFAIR_1REP`
Source model_dir: `saved_ckpt_runs/C100_C36_72_8_10_allReLU_analog_biasfalse_adc05_iq12_toggle_odexinit`
Source dataset_split: `train`
Source corners: `['FS_V2_T1']`
Source git_commit: `f92b28906390076f59813622bd27e305b84d4585`

| Layer | Branch | Lower (A) | Upper (A) |
|---|---|---:|---:|
| final_linear | COMBINED | -2.5758799999999998e-05 | 6.8554899999999991e-06 |
| layer_01 | COMBINED | -7.2837199999999991e-06 | 6.2785099999999991e-06 |
| layer_02 | COMBINED | -2.2957699999999999e-05 | 1.6861e-05 |
| layer_03 | COMBINED | -1.7564900000000002e-05 | 1.3510199999999998e-05 |
| layer_04 | COMBINED | -1.9309199999999998e-05 | 1.47331e-05 |
| layer_05 | COMBINED | -2.0445699999999996e-05 | 1.5276899999999999e-05 |
| layer_06 | COMBINED | -1.9785299999999998e-05 | 1.47033e-05 |
| layer_07 | COMBINED | -2.0945399999999998e-05 | 1.5546400000000001e-05 |
| layer_08 | COMBINED | -2.0704499999999998e-05 | 1.5074399999999999e-05 |
| layer_09 | COMBINED | -2.0191099999999997e-05 | 1.43492e-05 |
| layer_10 | COMBINED | -2.7797399999999999e-05 | 2.0396099999999999e-05 |
| layer_11 | COMBINED | -2.4749899999999998e-05 | 1.89749e-05 |
| layer_12 | COMBINED | -2.4381499999999997e-05 | 1.8649299999999998e-05 |
| layer_13 | COMBINED | -3.4517899999999997e-05 | 2.3130299999999998e-05 |
| layer_14 | COMBINED | -2.5172099999999999e-05 | 1.8980799999999996e-05 |
| layer_15 | COMBINED | -3.7254800000000001e-05 | 2.4504399999999999e-05 |
| layer_16 | COMBINED | -2.7569600000000001e-05 | 2.0713499999999997e-05 |
| layer_17 | COMBINED | -2.9177199999999999e-05 | 2.2135300000000001e-05 |
| layer_18 | COMBINED | -2.9732299999999997e-05 | 2.3176699999999998e-05 |
| layer_19 | COMBINED | -3.6007399999999997e-05 | 2.8519899999999998e-05 |
| layer_20 | COMBINED | -4.6664999999999995e-05 | 3.4035699999999997e-05 |

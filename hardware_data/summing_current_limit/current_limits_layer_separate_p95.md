# Custom separate current limits

Explicit signed total-current bounds. Percentile selection is not used by the evaluator.

Values are written in amperes to preserve parsed bounds exactly (no unit-roundtrip rounding).

Source summary: `results/toggle_summing_current_distribution/C100_C36_72_8_10_allReLU_analog_biasfalse_adc05_iq12_toggle_odexinit_FS_V2_T1_1trials/total/separate/summary.md`
Source summary SHA-256: `ab583d841cc407f8b35dd3c422273d9a3a789929c4389b3d75e76c083bc834b4`
Source model_name: `TIMMQAT5bNoneaNT0p0mulTIMMPCNetNoBatchNorm_PCConvReLU6_0.0eps_ToggleODEXInitFFFB_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_C100_3K1S72C_0.25Dropout_20Layers8l10l0_1Pool9_srrlDistill_a0p3_t2p0_CiFAIR_1REP`
Source model_dir: `saved_ckpt_runs/C100_C36_72_8_10_allReLU_analog_biasfalse_adc05_iq12_toggle_odexinit`
Source dataset_split: `train`
Source corners: `['FS_V2_T1']`
Source git_commit: `f92b28906390076f59813622bd27e305b84d4585`

| Layer | Branch | Lower (A) | Upper (A) |
|---|---|---:|---:|
| final_linear | FF | -2.5758799999999998e-05 | 6.8554899999999991e-06 |
| layer_01 | FB | -2.39516e-05 | 1.4761499999999999e-05 |
| layer_01 | FF | -1.2804099999999999e-06 | 1.1846399999999999e-06 |
| layer_02 | FB | -3.2762999999999998e-05 | 2.0491899999999999e-05 |
| layer_02 | FF | -3.2043200000000001e-06 | 3.2820699999999999e-06 |
| layer_03 | FB | -2.5103499999999998e-05 | 1.7140299999999999e-05 |
| layer_03 | FF | -2.7852e-06 | 2.6389700000000001e-06 |
| layer_04 | FB | -2.76369e-05 | 1.8663000000000001e-05 |
| layer_04 | FF | -2.6135799999999997e-06 | 2.4352399999999998e-06 |
| layer_05 | FB | -2.9266200000000001e-05 | 1.9144200000000002e-05 |
| layer_05 | FF | -2.35241e-06 | 2.1368299999999996e-06 |
| layer_06 | FB | -2.8278799999999998e-05 | 1.8232799999999999e-05 |
| layer_06 | FF | -2.6220799999999997e-06 | 2.5041899999999998e-06 |
| layer_07 | FB | -3.00058e-05 | 1.9286499999999999e-05 |
| layer_07 | FF | -2.0449600000000001e-06 | 1.96628e-06 |
| layer_08 | FB | -2.9565899999999998e-05 | 1.8270999999999998e-05 |
| layer_08 | FF | -2.4997899999999999e-06 | 2.53453e-06 |
| layer_09 | FB | -2.8665199999999998e-05 | 1.6944700000000001e-05 |
| layer_09 | FF | -3.1554700000000001e-06 | 3.1921099999999997e-06 |
| layer_10 | FB | -4.8783499999999991e-05 | 2.6574699999999998e-05 |
| layer_10 | FF | -1.7863899999999999e-06 | 1.78884e-06 |
| layer_11 | FB | -3.5557199999999999e-05 | 2.4014299999999998e-05 |
| layer_11 | FF | -2.2040399999999998e-06 | 2.1968899999999998e-06 |
| layer_12 | FB | -3.5040299999999997e-05 | 2.3583300000000002e-05 |
| layer_12 | FF | -1.9257699999999999e-06 | 1.91832e-06 |
| layer_13 | FB | -4.89078e-05 | 2.6112500000000001e-05 |
| layer_13 | FF | -2.1557299999999999e-06 | 2.1758999999999998e-06 |
| layer_14 | FB | -3.6148800000000002e-05 | 2.3768599999999998e-05 |
| layer_14 | FF | -1.91048e-06 | 1.9081500000000001e-06 |
| layer_15 | FB | -5.2636599999999997e-05 | 2.7140899999999996e-05 |
| layer_15 | FF | -1.9961899999999998e-06 | 1.9911599999999998e-06 |
| layer_16 | FB | -3.9569800000000002e-05 | 2.5860999999999999e-05 |
| layer_16 | FF | -2.2688499999999999e-06 | 2.2652899999999999e-06 |
| layer_17 | FB | -4.1879099999999997e-05 | 2.78271e-05 |
| layer_17 | FF | -2.7280199999999997e-06 | 2.6962199999999998e-06 |
| layer_18 | FB | -4.2645599999999997e-05 | 2.9513e-05 |
| layer_18 | FF | -3.84635e-06 | 3.8677499999999995e-06 |
| layer_19 | FB | -5.1629399999999997e-05 | 3.6594399999999996e-05 |
| layer_19 | FF | -5.1279999999999999e-06 | 5.1881199999999992e-06 |
| layer_20 | FB | -6.649089999999999e-05 | 4.0759099999999994e-05 |
| layer_20 | FF | -6.8995300000000002e-06 | 7.3727299999999993e-06 |

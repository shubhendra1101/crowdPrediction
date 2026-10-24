# Forecasting on Jülich (T6.1–T6.4, Jülich part)

Generated 2026-10-24 17:23. Test = ['class_120_door_closes', 'corr_surge', 'corr_narrow_0p8', 'hall_700_A_closes', 'bidir_2p0'] (5 runs); conformal calibration on val = ['class_90', 'corr_2p5', 'hall_500_A_closes', 'bidir_1p0'] (3 runs). Context 30 s, a forecast every 5 s. Density in persons/m².

- `mae`: error of the median forecast; `mae_dense`: same, only where the true density ≥ 2.
- `below_p90`: share of true values under the raw upper bound; `below_p90_cal`: after conformal calibration (target 90%); `q`: offset added to the bound, fitted on val.
- Baselines are point forecasts (p10 = p50 = p90); their calibrated bound is p50 + q.

| model | horizon_s | n | mae | rmse | n_dense | mae_dense | cover_p10_p90 | below_p90 | n_target_critical | below_p90_cal | mean_p90_cal_minus_p50 | q | n_val |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| persistence | 10 | 8300 | 0.165 | 0.361 | 2221 | 0.324 | 0.645 | 0.851 | 1229 | 0.881 | 0.15 | 0.1499999999999999 | 4180 |
| persistence | 30 | 7660 | 0.354 | 0.756 | 2090 | 0.632 | 0.618 | 0.852 | 1155 | 0.872 | 0.15 | 0.15000000000000002 | 3700 |
| persistence | 60 | 6800 | 0.599 | 1.209 | 1868 | 1.027 | 0.596 | 0.86 | 1041 | 0.86 | 0.0 | 0.0 | 2980 |
| persistence | 120 | 5120 | 0.868 | 1.7 | 1321 | 1.37 | 0.575 | 0.909 | 759 | 0.909 | 0.0 | 0.0 | 1580 |
| physics | 10 | 8300 | 0.248 | 0.7 | 2221 | 0.581 | 0.65 | 0.844 | 1229 | 0.906 | 0.25 | 0.25 | 4180 |
| physics | 30 | 7660 | 0.63 | 1.951 | 2090 | 1.586 | 0.644 | 0.861 | 1155 | 0.889 | 0.25 | 0.25 | 3700 |
| physics | 60 | 6800 | 1.225 | 3.898 | 1868 | 3.073 | 0.649 | 0.876 | 1041 | 0.876 | 0.0 | 0.0 | 2980 |
| physics | 120 | 5120 | 2.381 | 7.746 | 1321 | 5.925 | 0.661 | 0.914 | 759 | 0.914 | 0.0 | 0.0 | 1580 |

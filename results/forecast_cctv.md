# Forecasting: forecast_cctv (T6.1–T6.4)

Generated 2026-10-26 07:06. Test = ['cam02', 'cam04'] (2 runs); conformal calibration on val = ['cam01', 'cam03'] (2 runs). Context 30 s, a forecast every 5 s. Density in persons/m².

- `mae`: error of the median forecast; `mae_dense`: same, only where the true density ≥ 2.
- `below_p90`: share of true values under the raw upper bound; `below_p90_cal`: after conformal calibration (target 90%); `q`: offset added to the bound, fitted on val.
- Baselines are point forecasts (p10 = p50 = p90); their calibrated bound is p50 + q.

| model | horizon_s | n | mae | rmse | n_dense | mae_dense | cover_p10_p90 | below_p90 | n_target_critical | below_p90_cal | mean_p90_cal_minus_p50 | q | n_val |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| chronos2_joint | 10 | 6190 | 0.064 | 0.102 | 0 | nan | 0.795 | 0.892 | 0 | 0.903 | 0.114 | 0.001105162878620801 | 12540 |
| chronos2_joint | 30 | 6110 | 0.08 | 0.132 | 0 | nan | 0.762 | 0.874 | 0 | 0.907 | 0.137 | 0.011511338062775406 | 12460 |
| chronos2_joint | 60 | 5990 | 0.097 | 0.164 | 0 | nan | 0.733 | 0.855 | 0 | 0.891 | 0.16 | 0.02002249092891284 | 12340 |
| chronos2_per_zone | 10 | 6190 | 0.067 | 0.106 | 0 | nan | 0.8 | 0.901 | 0 | 0.905 | 0.131 | 1.3400313311526406e-05 | 12540 |
| chronos2_per_zone | 30 | 6110 | 0.081 | 0.133 | 0 | nan | 0.821 | 0.911 | 0 | 0.904 | 0.179 | -3.1191751064784947e-07 | 12460 |
| chronos2_per_zone | 60 | 5990 | 0.099 | 0.166 | 0 | nan | 0.835 | 0.913 | 0 | 0.895 | 0.243 | -2.1583025201785596e-06 | 12340 |
| persistence | 10 | 6190 | 0.074 | 0.115 | 0 | nan | 0.0 | 0.495 | 0 | 0.919 | 0.141 | 0.1413125416328167 | 12540 |
| persistence | 30 | 6110 | 0.087 | 0.14 | 0 | nan | 0.0 | 0.491 | 0 | 0.925 | 0.166 | 0.16605801716867297 | 12460 |
| persistence | 60 | 5990 | 0.105 | 0.171 | 0 | nan | 0.0 | 0.483 | 0 | 0.911 | 0.182 | 0.18150485478914702 | 12340 |

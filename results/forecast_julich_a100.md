# Forecasting on Jülich (T6.1–T6.4, Jülich part)

Generated 2026-09-30 12:18. Test = ['julich_corridor_uni_2013', 'julich_entrance_semicircle'] (10 runs); conformal calibration on val = ['julich_bottleneck2', 'julich_entrance_corridor', 'julich_corridor_bi_2009'] (39 runs). Context 30 s, a forecast every 5 s. Density in persons/m².

- `mae`: error of the median forecast; `mae_dense`: same, only where the true density ≥ 2.
- `below_p90`: share of true values under the raw upper bound; `below_p90_cal`: after conformal calibration (target 90%); `q`: offset added to the bound, fitted on val.
- Baselines are point forecasts (p10 = p50 = p90); their calibrated bound is p50 + q.

| model | horizon_s | n | mae | rmse | n_dense | mae_dense | cover_p10_p90 | below_p90 | n_target_critical | below_p90_cal | mean_p90_cal_minus_p50 | q | n_val |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| chronos2_joint | 10 | 3339 | 0.372 | 0.54 | 1580 | 0.349 | 0.735 | 0.904 | 85 | 0.861 | 0.441 | -0.08072542399168015 | 1526 |
| chronos2_joint | 30 | 2867 | 0.569 | 0.856 | 1371 | 0.466 | 0.663 | 0.899 | 64 | 0.835 | 0.466 | -0.13001295924186707 | 756 |
| chronos2_joint | 60 | 2159 | 0.741 | 1.056 | 1026 | 0.611 | 0.602 | 0.893 | 33 | 0.88 | 0.72 | -0.03696900233626366 | 176 |
| chronos2_per_zone | 10 | 3339 | 0.376 | 0.553 | 1580 | 0.351 | 0.745 | 0.916 | 85 | 0.892 | 0.49 | -0.05872111767530441 | 1526 |
| chronos2_per_zone | 30 | 2867 | 0.558 | 0.849 | 1371 | 0.447 | 0.698 | 0.922 | 64 | 0.875 | 0.55 | -0.12114404141902924 | 756 |
| chronos2_per_zone | 60 | 2159 | 0.727 | 1.037 | 1026 | 0.588 | 0.665 | 0.932 | 33 | 0.925 | 0.864 | -0.03541150316596031 | 176 |
| persistence | 10 | 3339 | 0.387 | 0.547 | 1580 | 0.373 | 0.095 | 0.601 | 85 | 0.85 | 0.35 | 0.3500000000000001 | 1526 |
| persistence | 30 | 2867 | 0.568 | 0.845 | 1371 | 0.448 | 0.054 | 0.596 | 64 | 0.66 | 0.1 | 0.10000000000000009 | 756 |
| persistence | 60 | 2159 | 0.711 | 1.022 | 1026 | 0.544 | 0.053 | 0.624 | 33 | 0.676 | 0.1 | 0.1 | 176 |
| physics | 10 | 3339 | 1.049 | 1.723 | 1580 | 0.828 | 0.081 | 0.577 | 85 | 0.798 | 0.65 | 0.6499999999999999 | 1526 |
| physics | 30 | 2867 | 2.507 | 4.743 | 1371 | 1.62 | 0.057 | 0.571 | 64 | 0.7 | 0.65 | 0.65 | 756 |
| physics | 60 | 2159 | 4.449 | 9.298 | 1026 | 2.578 | 0.064 | 0.573 | 33 | 0.751 | 1.25 | 1.25 | 176 |

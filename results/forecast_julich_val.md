# Forecasting on Jülich — val split

Generated 2026-09-30 10:40 · datasets ['julich_bottleneck2', 'julich_entrance_corridor', 'julich_corridor_bi_2009'] · 39 runs · context 30 s · origin every 5 s. Density in persons/m². `mae_dense` = MAE where the true density ≥ 2. Baselines are point forecasts (p10 = p90), so their coverage is not meaningful until conformal calibration (T6.3).

| model | horizon_s | n | mae | rmse | n_dense | mae_dense | cover_p10_p90 | below_p90 | n_target_critical |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| persistence | 10 | 1526 | 0.499 | 0.703 | 316 | 0.491 | 0.08 | 0.71 | 5 |
| persistence | 30 | 756 | 0.909 | 1.214 | 117 | 0.664 | 0.094 | 0.865 | 5 |
| persistence | 60 | 176 | 0.581 | 1.018 | 47 | 0.809 | 0.312 | 0.835 | 2 |
| physics | 10 | 1526 | 0.591 | 0.82 | 316 | 0.752 | 0.117 | 0.64 | 5 |
| physics | 30 | 756 | 1.379 | 2.048 | 117 | 1.573 | 0.163 | 0.771 | 5 |
| physics | 60 | 176 | 1.128 | 1.851 | 47 | 2.41 | 0.318 | 0.744 | 2 |

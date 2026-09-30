# Forecasting on Jülich — test split

Generated 2026-09-30 10:40 · datasets ['julich_corridor_uni_2013', 'julich_entrance_semicircle'] · 10 runs · context 30 s · origin every 5 s. Density in persons/m². `mae_dense` = MAE where the true density ≥ 2. Baselines are point forecasts (p10 = p90), so their coverage is not meaningful until conformal calibration (T6.3).

| model | horizon_s | n | mae | rmse | n_dense | mae_dense | cover_p10_p90 | below_p90 | n_target_critical |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| persistence | 10 | 3339 | 0.387 | 0.547 | 1580 | 0.373 | 0.095 | 0.601 | 85 |
| persistence | 30 | 2867 | 0.568 | 0.845 | 1371 | 0.448 | 0.054 | 0.596 | 64 |
| persistence | 60 | 2159 | 0.711 | 1.022 | 1026 | 0.544 | 0.053 | 0.624 | 33 |
| physics | 10 | 3339 | 1.049 | 1.723 | 1580 | 0.828 | 0.081 | 0.577 | 85 |
| physics | 30 | 2867 | 2.507 | 4.743 | 1371 | 1.62 | 0.057 | 0.571 | 64 |
| physics | 60 | 2159 | 4.449 | 9.298 | 1026 | 2.578 | 0.064 | 0.573 | 33 |

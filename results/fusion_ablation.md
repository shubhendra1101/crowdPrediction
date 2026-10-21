# Fusion ablation on the classroom CCTV (T3.7 / T3.8)

Generated 2026-10-21 22:03. Detector = `yolo11l`, density = `clipebc`. Split: **val** = first 50% of cam01 by time (43 frames) — θ and τ fitted here only; **test** = rest of cam01 + other recordings (57 frames). Single camera: this is a *time* hold-out, not a held-out camera. Frame-level counts (zones need calibration). Hard switch = detector if density count < τ = -inf, else density (re-implemented, D2). `average_50_50` = plain mean of both counts; `constant_weight` = one detector weight w = 0.43 fitted on val; `fusion` = learned per-frame weight from reliability features (architecture.md §2.2).

## Test split

| Variant | Band | n | MAE | RMSE | Bias |
| --- | --- | --- | --- | --- | --- |
| detector_only | all | 57 | 6.25 | 7.38 | -5.65 |
| detector_only | >=40 | 43 | 7.72 | 8.41 | -7.72 |
| detector_only | 20-40 | 7 | 2.14 | 2.59 | 0.43 |
| detector_only | <20 | 7 | 1.29 | 1.56 | 1.0 |
| density_only | all | 57 | 5.3 | 6.11 | 5.29 |
| density_only | >=40 | 43 | 6.4 | 6.93 | 6.4 |
| density_only | 20-40 | 7 | 2.68 | 2.74 | 2.68 |
| density_only | <20 | 7 | 1.15 | 1.43 | 1.08 |
| hard_switch | all | 57 | 5.3 | 6.11 | 5.29 |
| hard_switch | >=40 | 43 | 6.4 | 6.93 | 6.4 |
| hard_switch | 20-40 | 7 | 2.68 | 2.74 | 2.68 |
| hard_switch | <20 | 7 | 1.15 | 1.43 | 1.08 |
| average_50_50 | all | 57 | 1.66 | 2.14 | -0.18 |
| average_50_50 | >=40 | 43 | 1.74 | 2.27 | -0.66 |
| average_50_50 | 20-40 | 7 | 1.56 | 1.95 | 1.56 |
| average_50_50 | <20 | 7 | 1.22 | 1.46 | 1.04 |
| constant_weight | all | 57 | 1.59 | 2.06 | 0.63 |
| constant_weight | >=40 | 43 | 1.63 | 2.15 | 0.38 |
| constant_weight | 20-40 | 7 | 1.72 | 1.98 | 1.72 |
| constant_weight | <20 | 7 | 1.21 | 1.45 | 1.05 |
| fusion | all | 57 | 2.09 | 2.44 | 0.12 |
| fusion | >=40 | 43 | 2.15 | 2.53 | -0.46 |
| fusion | 20-40 | 7 | 2.63 | 2.69 | 2.63 |
| fusion | <20 | 7 | 1.22 | 1.44 | 1.15 |

## Test split, per camera (all bands)

| Camera | Variant | n | MAE | RMSE | Bias |
| --- | --- | --- | --- | --- | --- |
| cam01 | detector_only | 43 | 6.86 | 7.65 | -6.44 |
| cam01 | density_only | 43 | 6.09 | 6.71 | 6.09 |
| cam01 | hard_switch | 43 | 6.09 | 6.71 | 6.09 |
| cam01 | average_50_50 | 43 | 1.63 | 2.04 | -0.18 |
| cam01 | constant_weight | 43 | 1.58 | 2.05 | 0.74 |
| cam01 | fusion | 43 | 2.1 | 2.46 | -0.33 |
| cam02 | detector_only | 14 | 4.36 | 6.5 | -3.21 |
| cam02 | density_only | 14 | 2.89 | 3.7 | 2.86 |
| cam02 | hard_switch | 14 | 2.89 | 3.7 | 2.86 |
| cam02 | average_50_50 | 14 | 1.73 | 2.44 | -0.18 |
| cam02 | constant_weight | 14 | 1.62 | 2.08 | 0.27 |
| cam02 | fusion | 14 | 2.09 | 2.39 | 1.49 |

Mean fusion weight on test (share given to the detector): 0.40

Validation-split numbers are in the CSV (in-sample for θ/τ, so not a result).

# Fusion ablation on the classroom CCTV (T3.7 / T3.8)

Generated 2026-10-12 13:59. Detector = `yolo11l`, density = `clipebc`. Split: **val** = first 50% of cam01 by time (43 frames) — θ and τ fitted here only; **test** = rest of cam01 + other recordings (57 frames). Single camera: this is a *time* hold-out, not a held-out camera — generalisation to new cameras is not shown yet. Frame-level counts (zones need calibration). Hard switch = detector if density count < τ = -inf, else density (re-implemented, D2). `average_50_50` = plain mean of both counts; `constant_weight` = one detector weight w = 0.43 fitted on val; `fusion` = learned per-frame weight from reliability features (architecture.md §2.2).

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

Mean fusion weight on test (share given to the detector): 0.40

Validation-split numbers are in the CSV (in-sample for θ/τ, so not a result).

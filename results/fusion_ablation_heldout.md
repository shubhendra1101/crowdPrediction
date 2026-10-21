# Fusion ablation on the classroom CCTV (T3.7 / T3.8)

Generated 2026-10-21 22:28. Detector = `yolo11l`, density = `clipebc`. Split: **fit** = all frames of `counting_zeroshot_cctv_per_frame.csv` (100 frames, cameras ['cam01', 'cam02']) — θ, τ and w fitted here only; **test** = `counting_zeroshot_newcams_per_frame.csv` (60 frames, cameras ['cam03', 'cam04']) — **cameras never used for fitting**. Frame-level counts (zones need calibration). Hard switch = detector if density count < τ = 26.3, else density (re-implemented, D2). `average_50_50` = plain mean of both counts; `constant_weight` = one detector weight w = 0.44 fitted on val; `fusion` = learned per-frame weight from reliability features (architecture.md §2.2).

## Test split

| Variant | Band | n | MAE | RMSE | Bias |
| --- | --- | --- | --- | --- | --- |
| detector_only | all | 60 | 12.08 | 15.4 | -11.35 |
| detector_only | >=40 | 39 | 17.51 | 18.97 | -17.51 |
| detector_only | 20-40 | 4 | 4.0 | 4.42 | -3.0 |
| detector_only | <20 | 17 | 1.53 | 2.5 | 0.82 |
| density_only | all | 60 | 2.27 | 2.98 | 1.67 |
| density_only | >=40 | 39 | 2.38 | 3.01 | 1.65 |
| density_only | 20-40 | 4 | 3.26 | 4.41 | 3.26 |
| density_only | <20 | 17 | 1.79 | 2.47 | 1.36 |
| hard_switch | all | 60 | 2.2 | 2.99 | 1.52 |
| hard_switch | >=40 | 39 | 2.38 | 3.01 | 1.65 |
| hard_switch | 20-40 | 4 | 3.26 | 4.41 | 3.26 |
| hard_switch | <20 | 17 | 1.53 | 2.5 | 0.82 |
| average_50_50 | all | 60 | 5.88 | 7.4 | -4.84 |
| average_50_50 | >=40 | 39 | 8.11 | 9.01 | -7.93 |
| average_50_50 | 20-40 | 4 | 2.3 | 2.44 | 0.13 |
| average_50_50 | <20 | 17 | 1.6 | 2.33 | 1.09 |
| constant_weight | all | 60 | 5.14 | 6.43 | -4.02 |
| constant_weight | >=40 | 39 | 6.96 | 7.79 | -6.72 |
| constant_weight | 20-40 | 4 | 2.34 | 2.48 | 0.53 |
| constant_weight | <20 | 17 | 1.62 | 2.33 | 1.13 |
| fusion | all | 60 | 5.21 | 6.67 | -3.92 |
| fusion | >=40 | 39 | 7.04 | 8.07 | -6.69 |
| fusion | 20-40 | 4 | 2.12 | 2.37 | 1.04 |
| fusion | <20 | 17 | 1.74 | 2.49 | 1.25 |

## Test split, per camera (all bands)

| Camera | Variant | n | MAE | RMSE | Bias |
| --- | --- | --- | --- | --- | --- |
| cam03 | detector_only | 30 | 6.53 | 8.33 | -5.07 |
| cam03 | density_only | 30 | 2.58 | 3.3 | 1.95 |
| cam03 | hard_switch | 30 | 2.51 | 3.32 | 1.7 |
| cam03 | average_50_50 | 30 | 3.45 | 4.16 | -1.56 |
| cam03 | constant_weight | 30 | 3.11 | 3.74 | -1.11 |
| cam03 | fusion | 30 | 2.77 | 3.33 | -0.43 |
| cam04 | detector_only | 30 | 17.63 | 20.12 | -17.63 |
| cam04 | density_only | 30 | 1.97 | 2.63 | 1.39 |
| cam04 | hard_switch | 30 | 1.89 | 2.62 | 1.34 |
| cam04 | average_50_50 | 30 | 8.31 | 9.6 | -8.12 |
| cam04 | constant_weight | 30 | 7.17 | 8.29 | -6.92 |
| cam04 | fusion | 30 | 7.66 | 8.82 | -7.42 |

Mean fusion weight on test (share given to the detector): 0.43

Validation-split numbers are in the CSV (in-sample for θ/τ, so not a result).

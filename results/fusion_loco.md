# Leave-one-room-out counting evaluation (T3.8)

Generated 2026-10-21 22:28. Each room is held out in turn; θ / τ / w are fitted on the other two rooms only. Frames: {'H103': 30, 'H705': 100, 'H706': 30}. Ground truth = user-corrected CVAT heads (H103 glass-window area excluded for labels and models alike, `configs/roi.yaml`). Error in people per frame.

## Per held-out room

| Held-out room | Variant | n | MAE | RMSE | Bias |
| --- | --- | --- | --- | --- | --- |
| H103 | detector_only | 30 | 17.63 | 20.12 | -17.63 |
| H103 | density_only | 30 | 1.97 | 2.63 | 1.39 |
| H103 | hard_switch | 30 | 1.89 | 2.62 | 1.34 |
| H103 | average_50_50 | 30 | 8.31 | 9.6 | -8.12 |
| H103 | constant_weight | 30 | 6.79 | 7.86 | -6.52 |
| H103 | fusion | 30 | 8.29 | 9.59 | -7.96 |
| H705 | detector_only | 100 | 6.36 | 7.36 | -6.02 |
| H705 | density_only | 100 | 5.1 | 5.92 | 5.09 |
| H705 | hard_switch | 100 | 5.08 | 5.92 | 4.99 |
| H705 | average_50_50 | 100 | 1.68 | 2.16 | -0.47 |
| H705 | constant_weight | 100 | 4.17 | 4.9 | 4.12 |
| H705 | fusion | 100 | 4.3 | 5.07 | 4.17 |
| H706 | detector_only | 30 | 6.53 | 8.33 | -5.07 |
| H706 | density_only | 30 | 2.58 | 3.3 | 1.95 |
| H706 | hard_switch | 30 | 2.51 | 3.32 | 1.7 |
| H706 | average_50_50 | 30 | 3.45 | 4.16 | -1.56 |
| H706 | constant_weight | 30 | 2.72 | 3.26 | -0.51 |
| H706 | fusion | 30 | 3.08 | 3.67 | -1.18 |

## Mean over the three held-out rooms

| Variant | mean MAE | mean RMSE |
| --- | --- | --- |
| detector_only | 10.17 | 11.94 |
| density_only | 3.22 | 3.95 |
| hard_switch | 3.16 | 3.95 |
| average_50_50 | 4.48 | 5.31 |
| constant_weight | 4.56 | 5.34 |
| fusion | 5.22 | 6.11 |

Fitted parameters per held-out room: H103: τ=31.6, w=0.416; H705: τ=31.6, w=0.087; H706: τ=26.3, w=0.351

# Zero-shot counting on the classroom CCTV (T3.5, CCTV part)

Generated 2026-10-12 13:58. Ground truth: user-corrected CVAT head points, 100 frames ({'cam01': 86, 'cam02': 14}), 4379 heads. Same camera for all frames (H705 Front) — no held-out camera. ONNX on CPU; YOLO11l at 1280 px, conf 0.25; CrowdHuman conf 0.2; CLIP-EBC 224 px windows. Bands = heads per frame (no floor calibration yet, so not persons/m²). `mape_%` = mean absolute error as % of the true count.

| Model | Band | n | MAE | RMSE | Bias | MAPE % |
| --- | --- | --- | --- | --- | --- | --- |
| yolo11l | all | 100 | 6.36 | 7.36 | -6.02 | 15.1 |
| yolo11l | >=40 | 68 | 7.68 | 8.41 | -7.68 |  |
| yolo11l | 20-40 | 25 | 4.2 | 4.87 | -3.48 |  |
| yolo11l | <20 | 7 | 1.29 | 1.56 | 1.0 |  |
| phd_heads | all | 100 | 8.04 | 9.29 | -8.0 | 19.4 |
| phd_heads | >=40 | 68 | 9.46 | 10.37 | -9.46 |  |
| phd_heads | 20-40 | 25 | 6.2 | 7.21 | -6.04 |  |
| phd_heads | <20 | 7 | 0.86 | 1.2 | -0.86 |  |
| phd_persons | all | 100 | 14.1 | 15.97 | -14.04 | 31.2 |
| phd_persons | >=40 | 68 | 17.37 | 18.35 | -17.37 |  |
| phd_persons | 20-40 | 25 | 8.88 | 10.17 | -8.88 |  |
| phd_persons | <20 | 7 | 1.0 | 1.65 | -0.14 |  |
| clipebc | all | 100 | 5.1 | 5.92 | 5.09 | 12.6 |
| clipebc | >=40 | 68 | 6.36 | 6.89 | 6.36 |  |
| clipebc | 20-40 | 25 | 2.8 | 3.24 | 2.76 |  |
| clipebc | <20 | 7 | 1.15 | 1.43 | 1.08 |  |

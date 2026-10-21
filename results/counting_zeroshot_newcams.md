# Zero-shot counting on the classroom CCTV (T3.5, CCTV part)

Generated 2026-10-21 22:27. Ground truth: user-corrected CVAT head points, 60 frames ({'cam03': 30, 'cam04': 30}), 2336 heads. Same camera for all frames (H705 Front) — no held-out camera. ONNX on CPU; YOLO11l at 1280 px, conf 0.25; CrowdHuman conf 0.2; CLIP-EBC 224 px windows. Bands = heads per frame (no floor calibration yet, so not persons/m²). `mape_%` = mean absolute error as % of the true count. Exclusion regions from `configs\roi.yaml` removed 16 labelled heads; the same regions are removed from every model.

| Model | Band | n | MAE | RMSE | Bias | MAPE % |
| --- | --- | --- | --- | --- | --- | --- |
| yolo11l | all | 60 | 12.08 | 15.4 | -11.35 | 38.6 |
| yolo11l | >=40 | 39 | 17.51 | 18.97 | -17.51 |  |
| yolo11l | 20-40 | 4 | 4.0 | 4.42 | -3.0 |  |
| yolo11l | <20 | 17 | 1.53 | 2.5 | 0.82 |  |
| phd_heads | all | 60 | 17.23 | 22.98 | -17.17 | 41.2 |
| phd_heads | >=40 | 39 | 24.33 | 28.14 | -24.28 |  |
| phd_heads | 20-40 | 4 | 10.0 | 12.31 | -10.0 |  |
| phd_heads | <20 | 17 | 2.65 | 3.35 | -2.53 |  |
| phd_persons | all | 60 | 19.97 | 25.12 | -19.9 | 51.4 |
| phd_persons | >=40 | 39 | 27.92 | 30.76 | -27.92 |  |
| phd_persons | 20-40 | 4 | 10.25 | 11.57 | -10.25 |  |
| phd_persons | <20 | 17 | 4.0 | 4.95 | -3.76 |  |
| clipebc | all | 60 | 2.27 | 2.98 | 1.67 | 22.5 |
| clipebc | >=40 | 39 | 2.38 | 3.01 | 1.65 |  |
| clipebc | 20-40 | 4 | 3.26 | 4.41 | 3.26 |  |
| clipebc | <20 | 17 | 1.79 | 2.47 | 1.36 |  |

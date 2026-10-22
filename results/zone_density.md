# Zone density in persons/m² (T2.4, T3 counting → density)

Generated 2026-10-22 18:11 by `scripts/zone_density.py`. Ground truth = user's corrected CVAT head points placed on the floor plan through the calibrated camera at head height; prediction = CLIP-EBC density map summed in each zone's image mask. Density = count / visible zone area; zones with < 2.0 m² visible are skipped. Main head height 1.2 m (seated). Calibration and zone boundaries are approximate (`results/calibration.md`).

Share of labelled heads that fall inside a zone (the rest are outside the zoned floor area, e.g. beyond the last row): H103 97.9 %, H705 96.4 %, H706 94.3 %

## Per room

| Room | zone×frames | mean true density | max true density | MAE density (persons/m²) | bias | MAE count per zone |
| --- | --- | --- | --- | --- | --- | --- |
| H103 | 300 | 0.391 | 1.294 | 0.078 | 0.057 | 0.89 |
| H705 | 1000 | 0.543 | 1.67 | 0.101 | 0.059 | 0.8 |
| H706 | 300 | 0.424 | 1.399 | 0.081 | 0.025 | 0.56 |

## Per zone

| Room | Zone | n | mean true | max true | MAE | bias |
| --- | --- | --- | --- | --- | --- | --- |
| H103 | back_centre | 30 | 0.463 | 0.777 | 0.104 | 0.096 |
| H103 | back_left | 30 | 0.178 | 0.399 | 0.034 | 0.014 |
| H103 | back_right | 30 | 0.476 | 0.752 | 0.056 | 0.025 |
| H103 | front_aisle | 30 | 0.088 | 0.293 | 0.048 | 0.037 |
| H103 | front_centre | 30 | 0.46 | 0.874 | 0.103 | 0.084 |
| H103 | front_left | 30 | 0.361 | 0.565 | 0.066 | 0.058 |
| H103 | front_right | 30 | 0.276 | 0.476 | 0.075 | 0.06 |
| H103 | middle_centre | 30 | 0.748 | 1.294 | 0.128 | 0.086 |
| H103 | middle_left | 30 | 0.276 | 0.478 | 0.054 | 0.037 |
| H103 | middle_right | 30 | 0.58 | 0.94 | 0.112 | 0.073 |
| H705 | back_centre | 100 | 0.118 | 0.315 | 0.043 | 0.022 |
| H705 | back_left | 100 | 0.15 | 0.401 | 0.054 | 0.05 |
| H705 | back_right | 100 | 0.259 | 0.466 | 0.047 | 0.029 |
| H705 | front_aisle | 100 | 0.033 | 0.203 | 0.026 | 0.008 |
| H705 | front_centre | 100 | 0.877 | 1.485 | 0.146 | 0.132 |
| H705 | front_left | 100 | 0.637 | 1.078 | 0.1 | 0.095 |
| H705 | front_right | 100 | 1.063 | 1.67 | 0.222 | 0.159 |
| H705 | middle_centre | 100 | 0.627 | 1.166 | 0.095 | 0.059 |
| H705 | middle_left | 100 | 0.783 | 1.293 | 0.124 | 0.108 |
| H705 | middle_right | 100 | 0.885 | 1.253 | 0.153 | -0.073 |
| H706 | back_centre | 30 | 0.629 | 1.316 | 0.133 | 0.127 |
| H706 | back_left | 30 | 0.751 | 1.301 | 0.153 | 0.041 |
| H706 | back_right | 30 | 0.678 | 1.333 | 0.183 | -0.088 |
| H706 | front_aisle | 30 | 0.036 | 0.203 | 0.021 | 0.01 |
| H706 | front_centre | 30 | 0.155 | 0.583 | 0.021 | -0.002 |
| H706 | front_left | 30 | 0.296 | 0.601 | 0.051 | 0.016 |
| H706 | front_right | 30 | 0.015 | 0.452 | 0.017 | 0.017 |
| H706 | middle_centre | 30 | 0.672 | 1.399 | 0.059 | 0.005 |
| H706 | middle_left | 30 | 0.765 | 1.322 | 0.141 | 0.102 |
| H706 | middle_right | 30 | 0.241 | 0.452 | 0.025 | 0.023 |

## Sensitivity to the assumed head height

| Room | head height (m) | MAE density | mean true density |
| --- | --- | --- | --- |
| H103 | 1.1 | 0.067 | 0.374 |
| H103 | 1.2 | 0.078 | 0.391 |
| H103 | 1.3 | 0.093 | 0.392 |
| H103 | 1.6 | 0.044 | 0.385 |
| H705 | 1.1 | 0.127 | 0.501 |
| H705 | 1.2 | 0.101 | 0.543 |
| H705 | 1.3 | 0.103 | 0.55 |
| H705 | 1.6 | 0.108 | 0.549 |
| H706 | 1.1 | 0.086 | 0.39 |
| H706 | 1.2 | 0.081 | 0.424 |
| H706 | 1.3 | 0.094 | 0.428 |
| H706 | 1.6 | 0.088 | 0.47 |

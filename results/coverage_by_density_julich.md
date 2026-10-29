# Coverage of the calibrated 90 % bound by true density: julich

From stored test forecast points `weights/crowdsafe_nb04_report/forecast_julich_points.parquet` (no model re-run). `cover_all` = share of test points with target <= p90_cal; `cover_geX` = same, only points whose true future density is >= X persons/m²; `n_*` = number of points.

| model | horizon_s | n_all | cover_all | n_ge2 | cover_ge2 | n_ge4 | cover_ge4 | n_ge5 | cover_ge5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| chronos2_joint | 10 | 3339 | 0.861 | 1580 | 0.792 | 180 | 0.85 | 85 | 0.929 |
| chronos2_joint | 30 | 2867 | 0.835 | 1371 | 0.748 | 148 | 0.784 | 64 | 0.938 |
| chronos2_joint | 60 | 2159 | 0.88 | 1026 | 0.802 | 101 | 0.842 | 33 | 0.909 |
| chronos2_per_zone | 10 | 3339 | 0.892 | 1580 | 0.836 | 180 | 0.85 | 85 | 0.906 |
| chronos2_per_zone | 30 | 2867 | 0.875 | 1371 | 0.807 | 148 | 0.764 | 64 | 0.922 |
| chronos2_per_zone | 60 | 2159 | 0.925 | 1026 | 0.884 | 101 | 0.881 | 33 | 0.97 |
| persistence | 10 | 3339 | 0.85 | 1580 | 0.763 | 180 | 0.783 | 85 | 0.882 |
| persistence | 30 | 2867 | 0.66 | 1371 | 0.52 | 148 | 0.561 | 64 | 0.75 |
| persistence | 60 | 2159 | 0.676 | 1026 | 0.518 | 101 | 0.564 | 33 | 0.879 |
| physics | 10 | 3339 | 0.798 | 1580 | 0.717 | 180 | 0.794 | 85 | 0.776 |
| physics | 30 | 2867 | 0.7 | 1371 | 0.582 | 148 | 0.622 | 64 | 0.672 |
| physics | 60 | 2159 | 0.751 | 1026 | 0.616 | 101 | 0.644 | 33 | 0.727 |

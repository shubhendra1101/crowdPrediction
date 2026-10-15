# Camera audit (T1.4)

Generated 2026-10-15 18:18. Frames sampled every 10 s. Counts/head sizes from the CrowdHuman YOLO11s person+head model (zero-shot, conf 0.2) — **indicative only**, not ground truth. No calibration yet, so nothing here is in persons/m².

## cam03 (cam03.asf)

| Metric | Value |
| --- | --- |
| resolution | 1920x1080 |
| fps_claimed | 25.0 |
| fps_measured | None |
| duration_min | 33.5 |
| codec | h264 |
| samples | 202 |
| brightness_median | 114.1 |
| contrast_median | 56.2 |
| sharpness_median | 1871.8 |
| blockiness_median | 1.075 |
| frames_distinct | 170 |
| shake_px_median | 0.06 |
| shake_px_max | 0.29 |
| persons_max | 48 |
| persons_median | 33.5 |
| heads_max | 52 |
| head_px_median | 45.8 |
| head_px_p10 | 31.9 |
| audit_seconds | 188.6 |

**Recommendations**

- Frame rate 25 fps: fine for flow (≥ 5).
- Smallest heads ≈ 31.9 px (10th pct): detectors should work; fusion can use YOLO.
- Lighting OK.
- Compression blocking low.
- Camera steady (shift < 2 px over 1 s).
- Needs floor calibration (4+ measured floor points) before any persons/m² result.

## cam04 (cam04.asf)

| Metric | Value |
| --- | --- |
| resolution | 1920x1080 |
| fps_claimed | 25.0 |
| fps_measured | None |
| duration_min | 37.7 |
| codec | h264 |
| samples | 227 |
| brightness_median | 99.0 |
| contrast_median | 62.8 |
| sharpness_median | 1655.8 |
| blockiness_median | 1.1 |
| frames_distinct | 198 |
| shake_px_median | 0.08 |
| shake_px_max | 0.28 |
| persons_max | 34 |
| persons_median | 15.0 |
| heads_max | 36 |
| head_px_median | 35.4 |
| head_px_p10 | 25.4 |
| audit_seconds | 203.0 |

**Recommendations**

- Frame rate 25 fps: fine for flow (≥ 5).
- Smallest heads ≈ 25.4 px (10th pct): detectors should work; fusion can use YOLO.
- Lighting OK.
- Compression blocking low.
- Camera steady (shift < 2 px over 1 s).
- Needs floor calibration (4+ measured floor points) before any persons/m² result.

# Camera audit (T1.4)

Generated 2026-09-30 18:56. Frames sampled every 10 s. Counts/head sizes from the CrowdHuman YOLO11s person+head model (zero-shot, conf 0.2) — **indicative only**, not ground truth. No calibration yet, so nothing here is in persons/m².

## cam01 (cam01.asf)

| Metric | Value |
| --- | --- |
| resolution | 1920x1080 |
| fps_claimed | 25.0 |
| fps_measured | None |
| duration_min | 72.2 |
| codec | h264 |
| samples | 434 |
| brightness_median | 117.0 |
| contrast_median | 57.5 |
| sharpness_median | 2501.6 |
| blockiness_median | 1.043 |
| frames_distinct | 299 |
| shake_px_median | 0.09 |
| shake_px_max | 0.33 |
| persons_max | 51 |
| persons_median | 33.0 |
| heads_max | 59 |
| head_px_median | 39.9 |
| head_px_p10 | 30.7 |
| audit_seconds | 382.2 |

**Recommendations**

- Frame rate 25 fps: fine for flow (≥ 5).
- Smallest heads ≈ 30.7 px (10th pct): detectors should work; fusion can use YOLO.
- Lighting OK.
- Compression blocking low.
- Camera steady (shift < 2 px over 1 s).
- Needs floor calibration (4+ measured floor points) before any persons/m² result.

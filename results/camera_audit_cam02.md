# Camera audit (T1.4)

Generated 2026-09-30 18:49. Frames sampled every 5 s. Counts/head sizes from the CrowdHuman YOLO11s person+head model (zero-shot, conf 0.2) — **indicative only**, not ground truth. No calibration yet, so nothing here is in persons/m².

## cam02 (cam02.asf)

| Metric | Value |
| --- | --- |
| resolution | 1920x1080 |
| fps_claimed | 25.0 |
| fps_measured | None |
| duration_min | 15.1 |
| codec | h264 |
| samples | 182 |
| brightness_median | 118.0 |
| contrast_median | 53.0 |
| sharpness_median | 1970.2 |
| blockiness_median | 1.079 |
| frames_distinct | 161 |
| shake_px_median | 0.1 |
| shake_px_max | 0.64 |
| persons_max | 45 |
| persons_median | 9.5 |
| heads_max | 48 |
| head_px_median | 53.3 |
| head_px_p10 | 40.8 |
| audit_seconds | 109.1 |

**Recommendations**

- Frame rate 25 fps: fine for flow (≥ 5).
- Smallest heads ≈ 40.8 px (10th pct): detectors should work; fusion can use YOLO.
- Lighting OK.
- Compression blocking low.
- Camera steady (shift < 2 px over 1 s).
- Needs floor calibration (4+ measured floor points) before any persons/m² result.

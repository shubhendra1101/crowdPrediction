# CrowdSafe — computational cost, model parameters and data flow

*2026-10-30. Sources are given per row. **Measured** values come from logs and result files in this repository. **Model card** values come from the official model page. **≈ from file size** means the parameter count was estimated as ONNX fp32 file size / 4 bytes (the weights dominate the file).*

---

## 1. Models in the pipeline

| Model (role) | Parameters | Input it receives | Output it passes on | Settings used | Time per call (A100) |
|---|---|---|---|---|---|
| **CLIP-EBC ViT-B/16, NWPU weights** (people density) | ≈ 97 M (≈ from file size, 389 MB) | 224×224 RGB windows. A 1080×1920 frame is zero-padded to 1120×2016 → **45 windows**, batched (≤ 64) | 28×28 density per window, stitched into a **135×240 map** (1 cell = 8×8 px; values = people; sum = count) | count bins 0 / 1 / 2 / 3 / ≥ 4 with anchor 4.22; 32 deep visual-prompt tokens; zero-shot | **103–159 ms** per frame in live runs (measured, `results/latency_*.md`) |
| **RAFT-large** (optical flow) | ≈ 5.3 M (≈ from file size, 21 MB) | 2 frames 0.2 s apart, resized to **544×960** | flow **2×544×960** px, rescaled to full frame, then averaged to the same 135×240 cells | 12 flow updates; `Raft_Large_Weights.DEFAULT`, SHA-256 verified | **41 ms** benchmark (nb01 `timing.log`); **58–103 ms** in live runs |
| **Chronos-2** (forecast) | **120 M** (model card) | per zone: 30 steps × (density + 4 covariates: pressure, inflow, outflow, mean speed); or joint: 30 steps × all zones of a camera | 10 / 50 / 90 % quantiles for the next 60 (or 120) steps; the 10, 30, 60 (120) s values are kept | zero-shot; context 30 s (model maximum 8,192); called every 5 s for all zones at once | **54 ms** per call live, **40 ms** unpaced |
| YOLO11l, COCO (counting **baseline**) | ≈ 25 M (≈ from file size, 102 MB) | frame letterboxed to **1280** px | person boxes → frame count | conf 0.25, NMS IoU 0.7, class person | **15.5 ms** at 1080p (nb01 `timing.log`) |
| YOLO11s CrowdHuman person+head (counting **baseline**) | ≈ 9.5 M (≈ from file size, 38 MB) | 640×640 letterbox | person and head boxes → head count | conf 0.2, NMS IoU 0.6 | not separately measured on GPU |

- **Deployed path:** CLIP-EBC + RAFT + Chronos-2 ≈ **222 M parameters** in total.
- **Not in the deployed path:** the detectors, used for counting evaluation only.
- **No model was trained or fine-tuned.**

**Non-neural steps** (negligible cost; live medians from `results/latency_live_chronos.md`):

| Step | Median time |
|---|---|
| Zone aggregation | 5 ms |
| Risk + alerts | 8 ms |
| Database log | 9 ms |
| Persistence forecast | 6 ms |
| Conformal offset | one addition |

The camera-model fit runs once per camera, offline, in seconds on a laptop.

---

## 2. Computational cost

### 2.1 Per second of video (live, one camera, one shared A100; `results/latency_*.md`)

| Stage | Median |
|---|---|
| Wait for the second frame of the motion pair | 203 ms |
| CLIP-EBC density | 159 ms |
| RAFT flow | 103 ms |
| Zones + risk + log | ≈ 22 ms |
| Chronos-2 (every 5th second) | 54 ms |
| **End-to-end latency** | **0.49 s** (p95 1.6 s); 94 % of 1-s cycles on time |
| GPU memory | **2.0 GB** (1.5 GB with persistence instead of Chronos-2) |
| Throughput, unpaced | **4.6× real time** (908 s of video in 196 s) |

**Capacity:** one A100 could serve about 4 cameras (*extrapolated* from the throughput, not tested).

**Laptop CPU** (no GPU):

| Stage | Time per second of video |
|---|---|
| CLIP-EBC (ONNX) | ≈ 18.7 s |
| Farneback flow | 0.3 s |
| **Total** | **≈ 19 s** → live use needs a GPU |

### 2.2 Whole project (measured, from notebook logs)

| Job | Data | Wall time | Hardware |
|---|---|---|---|
| CCTV feature extraction (CLIP-EBC + RAFT + zones) | 9,513 s of video (2.64 h), 4 cameras | **33.5 min** (4.6–4.7 sampled frames/s) | A100 (nb05 `cctv_features_log.txt`) |
| JuPedSim simulations | 16 scenarios, 30–500 s each | **9.7 min** total (582.5 s) | CPU (nb05 `simulation_log.txt`) |
| Live latency runs | 10 + 5 min real time + 15 min of video unpaced | ≈ 18 min | A100 (nb06) |
| Chronos-2 forecasts (Jülich, CCTV, sims) | val + test splits | not logged | A100 (nb04, nb05) |
| Counting on 160 labelled frames, ShanghaiTech, all baselines | — | not logged | CPU (ONNX) / A100 |

**Total GPU time** for the measured jobs is **under one hour**; the forecasting jobs were not timed. **Training cost is zero:** no model was trained.

---

## 3. Settings used in each step (all from `configs/` or script defaults)

| Step | Setting | Value | Where |
|---|---|---|---|
| Sampling | frames analysed | 1 per second (+ partner frame 0.2 s later) | `cctv_features.py --every-s 1.0 --pair-gap-s 0.2` |
| | source video | 1080p, 25 fps | camera audit |
| Calibration | desk-top length | 1.00 m | `configs/calibration/*.yaml` |
| | head height for mapping | 1.2 m (seated; 1.1–1.6 m tested) | `configs/zones/` |
| | lab camera-height prior | 2.58 m, weight 200 | `configs/calibration/H103.yaml` |
| | lens distortion | k1, k2 (k2 ≥ 0) | `camera_model.py` |
| Zones | layout | 3×3 seat blocks + front aisle, 4–16 m² | `configs/zones/` |
| | minimum visible area | 2 m² | `--min-area 2.0` |
| | exclusion masks | glass partition (H103) | `configs/roi.yaml` |
| CLIP-EBC | window / reduction / batch | 224 px / 8 / 64 | model JSON, `--batch 64` |
| RAFT | width / updates | 960 px (544×960) / 12 | `--flow-width 960`, `flow.py` |
| Motion | cells used | density ≥ 0.02 people per cell | `zone_motion.py` |
| | speed limits | ≥ 0.1 m/s counts as moving; > 4 m/s discarded | `zone_motion.py` |
| | direction histogram | 8 bins | `zone_motion.py` |
| Trajectory zones | cell size | 2 m × 2 m (Jülich: kept if ≥ 90 % walkable) | `configs/julich.yaml` |
| Forecasting | context / stride | 30 s / every 5 s | `configs/forecast_*.yaml` |
| | horizons | 10, 30, 60 s (+120 s simulation) | same |
| | quantiles | 0.1, 0.5, 0.9 | `chronos.py` |
| | physics flux window | 10 s | same |
| Conformal | α | 0.1 (90 % bound) | `eval_forecast.py` |
| Risk | bands | 2 / 4 / 5 persons/m² | `configs/risk.yaml` |
| | precursor thresholds | none active (none passed) | same |
| | precursor tuning rule | event 4 persons/m² within 60 s; precision ≥ 2× base rate | same |
| Alerts | raise / clear | level ≥ High for 10 s / below for 30 s | same |
| Evaluation | in-time window | longest horizon + 10 s; not in first 15 s of monitoring | `eval_alerts.py` |
| | false-alarm window | no event within 120 s | same |
| All | random seed | 42 | — |

---

## 4. Data flow from start to end (one camera, one second; example H705, 10 zones)

1. **Video frame:** 1080×1920×3 (uint8), plus the frame 0.2 s later.
2. **CLIP-EBC:** frame → 45 windows of 3×224×224 → 45 × 28×28 → **density map D_t, 135×240** (exclusion mask applied). Sum ≈ people in view.
3. **RAFT:** frame pair → 2×544×960 flow → full-frame flow → averaged to **135×240 cells** (same grid as D_t).
4. **Camera model** (fitted once): for each cell, its position on the head-height plane (m); for each zone, a mask **M_z** on the 135×240 grid and a visible area **A_z** (m²).
5. **Zone aggregation:** for each of the 10 zones:
   - count c = Σ D_t·M_z;
   - density ρ = c / A_z;
   - cell velocities (m/s) = floor distance travelled / 0.2 s, weighted by D_t → mean speed, speed variance, pressure, direction entropy, counter-flow.
6. **Feature table:** **10 rows per second**: timestamp, camera, zone, count, density, fusion weight (empty), mean speed, speed variance, pressure, direction entropy, counter-flow, inflow (empty for CCTV), outflow (empty for CCTV), source. 14 columns.
7. **Forecast window (every 5 s):**
   - per zone: 10 series × 30 steps (+ 4 covariates);
   - or joint: 1 series × 30 steps × 10 zones.
8. **Chronos-2** → for each zone, 3 quantiles × 60 future steps; keep h = 10, 30, 60 → **10 × 3 × 3 numbers**.
9. **Conformal step:** add the fixed offsets q_10, q_30, q_60 → upper bounds **U, 10 × 3**.
10. **Risk step:** per zone, W = max(max_h U, ρ_now) → level (Low / Medium / High / Critical), time to critical, reason text.
11. **Alert manager:** per-zone timer (10 s raise, 30 s clear) → **alert events** (zone, level, since, reason, time to critical).
12. **Dashboard message (1 per second):** for each zone, density, speed, pressure, forecast {h, p10, p50, p90}, level and time to critical, plus the active alerts.

**Same table, other sources:** for Jülich and simulations, steps 1–5 are replaced by trajectories → 2-m cells. Inflow and outflow come from boundary crossings. Steps 6–11 are identical.

---

## 5. What could go into the paper

- **Model table** (one column-width table): model, role, parameters, input, output, time per call. This gives reviewers the computational cost at a glance.
- **One sentence on total cost**, for example: "No model is trained; the deployed path has about 222 M parameters (CLIP-EBC ≈ 97 M, RAFT ≈ 5 M, Chronos-2 120 M) and needs 2.0 GB of GPU memory. Extracting features from all 2.6 h of CCTV took 33.5 min on one A100."
- **The data-flow list (§4):** already reflected in Fig. 1 and the methodology. The tensor sizes (45 windows, 135×240 map) could be added as one sentence each in the Counting and Motion steps.

**Caveat:** the parameter counts for CLIP-EBC, RAFT and the detectors are estimates from file size. An exact count needs the `onnx` package (≈ 15 MB install) or the PyTorch models.

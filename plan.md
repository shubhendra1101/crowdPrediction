# plan.md — CrowdSafe 7-week plan

Goal: a working demo and paper results showing that CrowdSafe forecasts zone density and stampede risk and raises alerts **minutes before** a zone reaches 5 persons/m², with few false alarms.

Principle: reuse pretrained models, measure on the user's data, fine-tune only where the numbers demand it. Counting and fusion come first because everything downstream depends on them.

## Timeline

| Week | Phase | Steps | Main output |
| --- | --- | --- | --- |
| 1 | Foundation | 0 Setup & baseline · 1 Data & CCTV audit · 2 Calibration & zones | Clean repo, baseline numbers, camera configs, ~100 test frames |
| 2 | Counting | 3a Detector · 3b Density model | Zero-shot error of each model on user's data |
| 3 | Fusion + motion | 3c Fusion · 3d Ablation · 4 Motion features | Fusion ablation table; six motion features per zone |
| 4 | Dataset | 5 Logging, Jülich features, simulation | Feature dataset (real + Jülich + sim), split |
| 5 | Forecasting | 6 Baselines, Chronos-2, conformal | Forecast table per horizon + coverage |
| 6 | Risk + demo | 7 Risk engine & alerts · 8 Backend & dashboard | Alert replays with lead times; live replay demo |
| 7 | Results | 9 Evaluation & write-up | All tables, figures, final demo |

## Phase details

### Week 1 — Foundation

**Step 0. Setup and baseline.** Environment, repo restructure (see `CLAUDE.md` §4), run old system on 3–4 clips, record MAE/FPS/GPU memory.
Done when: `results/baseline.md` exists and old pipeline runs from the new layout.

**Step 1. Data and CCTV quality.** Start dataset downloads (some need forms). Audit each camera (resolution, head size, FPS, compression, light, angle, shake). Sample ~100 CCTV frames, pre-label with CLIP-EBC, user corrects in CVAT. Split by camera.
Done when: `results/camera_audit.md` exists and labelled frames are split train/val/test by camera.

**Step 2. Calibration and zones.** Homography per camera from user's floor measurements; zones via PolygonZone tool; areas via Shapely; adjacency and boundaries; validate one unused distance (<10% error).
Done when: every camera has a complete YAML.

### Weeks 2–3 — Counting, fusion, motion

**Step 3a/3b. Zero-shot counting.** YOLO11l (COCO) vs CrowdHuman person+head model; CLIP-EBC ShanghaiTech-A vs NWPU checkpoints; optional APGCC; tiled inference on low-res cameras. Measure MAE per density band on user's test frames and on Jülich videos.
Done when: a table of zero-shot errors exists. If CLIP-EBC error is poor → ask user before fine-tuning.

**Step 3c/3d. Fusion and ablation.** Fit fusion weights on validation frames; EMA smoothing; ablation of 4 variants.
Done when: fusion beats both single models on the held-out camera (or the result is documented honestly if it doesn't).

**Step 4. Motion features.** RAFT → m/s via H → six features per zone per second; sanity check vs ByteTrack speeds and Jülich trajectories.
Done when: every camera and Jülich video outputs the full feature set.

### Week 4 — Dataset

**Step 5.** Feature logger (Parquet). Jülich experiments → PedPy features. Mall/UCSD/FDST/CCTV → pipeline features. JuPedSim: 3 geometries, 15–20 runs incl. surges and exit closures, beyond 5 persons/m².
Done when: train/val/test feature sets exist, split by experiment, geometry and camera.

### Week 5 — Forecasting

**Step 6.** Persistence and physics baselines; Chronos-2 zero-shot with zones as a group and motion features as covariates; conformal calibration; optional GRU; optional Chronos-2 fine-tune (ask first).
Done when: MAE per horizon + 90% coverage table on held-out data.

### Week 6 — Risk engine and demo

**Step 7.** Density bands on calibrated upper bound; precursor escalation; thresholds tuned on Jülich + sim; Isolation Forest; hysteresis; alert reasons.
Done when: replay of any clip/run produces an alert timeline with lead times.

**Step 8.** Multiprocess runtime; FastAPI WebSocket; dashboard heatmap, forecast chart, alert feed, replay mode.
Done when: a replayed clip updates the dashboard live end to end.

### Week 7 — Evaluation and write-up

**Step 9.** Fill every table in `architecture.md` §6; figures (heatmap frame, forecast with band, alert timeline); runtime numbers; methodology text from `results/`.
Done when: all tables filled and the demo runs end to end.

## Fallbacks (cut in this order if behind)

1. Anomaly detector (Isolation Forest) → rules only.
2. GRU comparison → Chronos-2 vs baselines only.
3. 5–10 minute horizons → 30–120 s only.
4. Chronos-2 fine-tuning → zero-shot only; physics model as alert source if Chronos-2 is weak.
5. APGCC and CrowdHuman YOLO comparisons → one detector, one density model.

Never cut: calibration in persons/m², the fusion ablation, lead-time evaluation, the replay demo.

## Risks

| Risk | Early sign | Response |
| --- | --- | --- |
| CCTV too poor for YOLO | heads < ~10 px, low confidence | tiled inference; fusion leans on CLIP-EBC; report as finding |
| Density model over-counts user's cameras | held-out error ≫ benchmark | ask user; fine-tune CLIP-EBC with ~100 more corrected frames |
| Chronos-2 zero-shot weak | barely beats persistence | ask user; fine-tune via AutoGluon; physics model drives alerts |
| Lab/sim ≠ real crowds | good on Jülich/sim, poor on CCTV | report per source; tune thresholds on real clips |
| Missing user inputs (measurements, footage) | task blocked > 2 days | work on non-blocked tasks; use Jülich data; flag clearly |
| Time slip | any step > 3 days late | apply fallbacks above, after user agrees |

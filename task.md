# task.md — CrowdSafe task checklist

Source of truth for what to do next. The agent ticks tasks and writes the real result on the `Result:` line.

**Owner:** `AGENT` = the coding agent does it · `USER` = only the user can do it · `BOTH` = agent prepares, user supplies or decides.
**Rule:** if a task needs a USER input that isn't there, the agent stops, asks (see `CLAUDE.md` §6), and moves to another unblocked task.

---

## Week 1 — Foundation

### T0 Setup and baseline

- [x] **T0.1 (AGENT)** Inspect the existing repo; summarise what exists and what can be reused.
  Result: 2026-09-30 — folder contained planning docs only; no earlier code. New project (`docs/decisions.md` D2).
- [x] **T0.2 (AGENT)** Check environment: Python, CUDA, GPU model/memory, free disk (need ~150 GB). Report gaps.
  Result: 2026-09-30 — laptop: Python 3.11.9, torch 2.0.1+cpu, no CUDA, 7.8 GB RAM, 34 GB free. A100 = college, via Kubeflow notebooks (D3); A100 specs pending notebook 01 (T0.6).
- [x] **T0.3 (USER)** Create accounts and set tokens as environment variables (not in chat): Hugging Face (`HF_TOKEN`), Kaggle (`~/.kaggle/kaggle.json`), CVAT (app.cvat.ai or local Docker).
  Result: 2026-09-30 — user confirmed all accounts exist.
- [x] **T0.4 (AGENT)** Restructure repo to the layout in `CLAUDE.md` §4; add `.gitignore` for `data/`, `weights/`; `requirements.txt`; `pytest` setup.
  Result: 2026-09-30 — git repo + GitHub remote; modules in `crowdsafe/` package (D1); `.gitignore`, `requirements.txt`, `pyproject.toml`, camera template; tests pass.
- [x] ~~**T0.5 (AGENT)** Run the old YOLO11m + CSRNet pipeline on 3–4 clips; write `results/baseline.md` (MAE, FPS, GPU memory).~~
  Result: 2026-09-30 — DROPPED by user: new project, old implementation not reused (D2). Hard-switch ablation row will be re-implemented.
- [x] **T0.6 (BOTH)** Run `notebooks/01_env_and_onnx_export.ipynb` on the A100: environment report; ONNX of YOLO11l, RAFT-large, CLIP-EBC ViT-B/16 NWPU (HF); smoke test of `Sharath33/Person`.
  Result: 2026-09-30 — A100 run OK (commit 8bc8672): yolo11l 102 MB (4/4 persons, IoU 0.9999), raft_large 21 MB (shift 6.00/3.02 px, max diff 1.4e-4 px), clipebc_vitb16_nwpu 389 MB (927.54 vs 927.54), CrowdHuman 38 MB (decoder fixed for its 3-output layout: 4 persons / 3 heads on bus.jpg). GPU: YOLO11l 1080p 15.5 ms, RAFT 544×960 40.8 ms, 0.94 GB. Models in `weights/`, checksums match.

### T1 Data and CCTV quality

- [x] **T1.1 (BOTH)** Start downloads. Agent gives links and exact commands; user completes any registration/request forms.
  Must: Jülich archive (bottleneck, corridor, entrance, platform experiments), ShanghaiTech A/B.
  Should: JHU-Crowd++, Mall, UCSD. Nice: FDST, NWPU-Crowd. Agent asks before any single download > 5 GB.
  Result: 2026-09-30 — A100: 13 datasets ok (~23 GB): Jülich traj+videos, JHU 2272/500/1600, Mall 2000, UCSD, ShanghaiTech HF 300/182/400/316; Kaggle ShanghaiTech has every file twice (600/364/800/632) — nb03 reads one copy. Laptop: Jülich trajectories.
- [ ] **T1.2 (USER)** Collect CCTV: 3–5 cameras, 30–60 min each incl. peak periods; written permission to use for research; for each camera note resolution, FPS, mounting height and angle.
  Result: 2026-09-30 — 1 camera (classroom H705 Front, corner, ~2–2.5 m high, permission granted): cam01 72 min (lecture), cam02 15 min (lecture → mass exit → empty). More cameras still needed (≥1 held-out camera; corridor/door/stairs preferred).
- [ ] **T1.3 (USER, optional)** Own recordings: tripod at a high point looking down, 1080p, 25–30 fps, no zoom/pan, crowded times.
  Result:
- [x] **T1.4 (AGENT)** Camera audit script → `results/camera_audit.md` (resolution, head size in px, FPS, compression, brightness, angle, shake) with per-camera recommendations.
  Needs: T1.2.
  Result: 2026-09-30 — cam01/cam02: 1920×1080, 25 fps, h264 ASF (no seeking → sequential reads); brightness ~117, blockiness 1.04–1.08, shake ≤ 0.64 px/s; heads median 40–53 px (p10 31–41 px) → detectors usable; max persons 51 (cam01), 45 (cam02); cam02 exit at ~min 7. `results/camera_audit_cam0*.md`.
- [ ] **T1.5 (AGENT)** Sample ~100 frames (≥ 5 s apart, varied density/time/camera); pre-label with CLIP-EBC; export to CVAT point format; write a 5-step guide for the user.
  Needs: T1.2, T3.3 weights.
  Result:
- [ ] **T1.6 (USER)** Correct pre-labels in CVAT (~5–8 h) and export.
  Result:
- [ ] **T1.7 (AGENT)** Import labels; split train/val/test **by camera** (≥ 1 camera fully held out); write split file.
  Result:

### T2 Calibration and zones

- [ ] **T2.1 (USER)** For each camera: one clear frame; 4+ floor points with real distances (tiles, marked rectangle, or taped markers); photos of the markers; 1 extra known distance for validation.
  Agent gives exact instructions per camera before the user goes on site.
  Result:
- [ ] **T2.2 (USER)** Venue layout sketch: walkable areas, entrances/exits, which areas connect; widths of exits, gates, corridors, stairs in metres.
  Result:
- [ ] **T2.3 (AGENT)** Calibration tool (click points → homography) using the `ViewTransformer` pattern; save H to YAML.
  Result:
- [ ] **T2.4 (BOTH)** Draw zones with the Roboflow PolygonZone tool (agent explains; user or agent draws); compute areas with Shapely; add adjacency and boundaries from T2.2.
  Result:
- [ ] **T2.5 (AGENT)** Validate each camera with the unused distance (target < 10% error). Mark approximate calibrations as approximate.
  Result:

---

## Weeks 2–3 — Counting, fusion, motion

### T3 Counting and fusion

- [ ] **T3.1 (AGENT)** Detector wrapper: YOLO11l (COCO, person) + ByteTrack; option for `InferenceSlicer` tiling.
  Result: 2026-09-30 — wrapper `crowdsafe/counting/yolo_track.py` (YOLO + InferenceSlicer tiling + Ultralytics ByteTrack, reliability features); unit-tested; A100 run = notebook 03.
- [ ] **T3.2 (AGENT)** Try the CrowdHuman YOLO11 person+head model from Hugging Face (`Sharath33/Person`, ONNX); ask user before choosing if results are close.
  Result: 2026-09-30 — `PersonHeadOnnx` wrapper (model-card preprocessing, NMS) unit-tested; benchmark in notebook 03.
- [ ] **T3.3 (AGENT)** Density wrapper: CLIP-EBC ViT-B/16 NWPU weights from the official Hugging Face repo `Yiming-M/CLIP-EBC` (`huggingface_hub.snapshot_download`), adapting model loading and sliding-window inference from its `app.py`. Optionally compare the ShanghaiTech-A checkpoint from GitHub releases.
  Result: 2026-09-30 — `crowdsafe/counting/density_model.py` (torch from HF repo or ONNX; window / whole-image modes); unit-tested; benchmark in notebook 03.
- [ ] **T3.4 (AGENT, optional)** PET-Finetuned point counter: weights from Hugging Face (`Awiros/crowd-counting-and-localization`), inference with the official PET repo.
  Result:
- [ ] **T3.5 (AGENT)** Zero-shot benchmark on ShanghaiTech, Jülich videos and user test frames: MAE/RMSE per density band → `results/counting_zeroshot.md`.
  Result: 2026-09-30 — ShanghaiTech test, zero-shot (A100, commit a0326a2), MAE A/B: CLIP-EBC window 66.95/11.66, whole 66.0/11.98; CrowdHuman heads 374.2/70.2, persons 377.1/58.8; YOLO11l 418.3/92.8; YOLO tiled 421.7/95.4 (tiling did not help). CLIP-EBC best in every band incl. <50 on B (1.95 vs YOLO 8.95). `results/counting_zeroshot_shanghaitech.md`. Jülich-video + CCTV parts pending.
- [ ] **T3.6 (BOTH)** Decision: fine-tune CLIP-EBC or YOLO? Agent presents numbers and GPU-time estimate; user decides.
  Result:
- [ ] **T3.7 (AGENT)** Fusion: reliability features, fit weights on val frames, EMA smoothing, density = count / area.
  Result:
- [ ] **T3.8 (AGENT)** Ablation: YOLO only / CLIP-EBC only / hard switch / weighted fusion on held-out camera → `results/fusion_ablation.md`.
  Result:

### T4 Motion features

- [ ] **T4.1 (AGENT)** RAFT wrapper (`raft_large`, 3–5 FPS, ~960 px); Farneback fallback.
  Result:
- [ ] **T4.2 (AGENT)** Zone motion: flow → m/s via H; density mask; mean_speed, speed_var, pressure, dir_entropy, counterflow, inflow/outflow.
  Result:
- [ ] **T4.3 (AGENT)** Validate speeds against Jülich trajectories and ByteTrack tracks → `results/motion_validation.md`.
  Result:

---

## Week 4 — Dataset

### T5 Feature logging and simulation

- [x] **T5.1 (AGENT)** Feature schema + Parquet logger (`architecture.md` §2.4).
  Result: 2026-09-30 — `crowdsafe/features/schema.py` (14 columns, validate/conform) + `logger.py` (Parquet, SQLite); tests pass.
- [x] **T5.2 (AGENT)** Jülich loader: trajectories → PedPy → feature table, per zone.
  Result: 2026-09-30 — 144 runs / 8 experiments → `data/features/julich/`; 2 m zones; max density 9.0 (entrance_semicircle), 5.65 (entrance_corridor), 5.3 (corridor_uni_2013); 643 zone-s ≥ 5 p/m²; bottleneck1 has no 2 m zone (area 0.6 m wide); PedPy cross-check diff 0.0000 on 6/6. `results/julich_features_summary.md`.
- [ ] **T5.3 (AGENT)** Run the pipeline over Mall/UCSD/FDST/CCTV → feature tables.
  Result:
- [ ] **T5.4 (BOTH)** JuPedSim: agent proposes 3 geometries from the user's venue layout (T2.2); user approves; agent runs 15–20 scenarios incl. surges, exit closures, > 5 persons/m².
  Result:
- [ ] **T5.5 (AGENT)** Split train/val/test by experiment, geometry and camera; write `results/dataset_summary.md`.
  Result: 2026-09-30 — Jülich split proposed in `configs/forecast_julich.yaml` (test: corridor_uni_2013 + entrance_semicircle; val: bottleneck2, entrance_corridor, corridor_bi_2009; train: corridor_uni_2009, train_platform) — awaiting user OK; sim/CCTV splits later.

---

## Week 5 — Forecasting

### T6 Forecasting

- [x] **T6.1 (AGENT)** Persistence and physics fill-rate baselines + time-to-critical.
  Result: 2026-09-30 — `forecasting/baselines.py` (persistence, physics fill-rate + time-to-critical). Jülich test MAE (p/m²) 10/30/60 s: persistence 0.39/0.57/0.71, physics 1.05/2.51/4.45 (noisy 2 m-zone flux). `results/forecast_julich.md`.
- [x] **T6.2 (AGENT)** Chronos-2 zero-shot wrapper: zones as a group, motion features as covariates, quantiles 0.1/0.5/0.9.
  Result: 2026-09-30 — Chronos-2 zero-shot on Jülich test (A100): MAE 10/30/60 s per-zone 0.376/0.558/0.727, joint 0.372/0.569/0.741 vs persistence 0.387/0.568/0.711.
- [x] **T6.3 (AGENT)** Split conformal calibration of the 90% bound.
  Result: 2026-09-30 — split conformal, val→test coverage of calibrated p90 (target 0.90): Chronos-2 per-zone 0.892/0.875/0.925, joint 0.861/0.835/0.880; persistence 0.850/0.660/0.676; physics 0.798/0.700/0.751.
- [ ] **T6.4 (AGENT)** Evaluate on held-out data: MAE per horizon, coverage → `results/forecasting.md`.
  Result: 2026-09-30 — Jülich part done: `results/forecast_julich_a100.md` (Chronos-2 ≈ persistence on MAE, but only Chronos-2 gives a near-90% calibrated upper bound). Sim + CCTV parts pending (T5.3/T5.4).
- [ ] **T6.5 (BOTH, optional)** If Chronos-2 is weak: agent proposes fine-tuning via AutoGluon with time estimate; user decides.
  Result:
- [ ] **T6.6 (AGENT, optional)** Per-zone GRU comparison.
  Result:

---

## Week 6 — Risk engine and demo

### T7 Risk engine and alerts

- [ ] **T7.1 (AGENT)** Density bands on calibrated upper bound.
  Result:
- [ ] **T7.2 (AGENT)** Precursor rules; tune thresholds on Jülich + sim; write them to camera YAMLs.
  Result:
- [ ] **T7.3 (AGENT, optional)** Isolation Forest anomaly detector.
  Result:
- [ ] **T7.4 (AGENT)** Hysteresis, alert reasons, time-to-critical; replay tool → alert timelines.
  Result:
- [ ] **T7.5 (USER)** Review default thresholds for the venue context (Indian crowd densities may justify different limits); confirm or adjust.
  Result:

### T8 Backend and dashboard

- [ ] **T8.1 (AGENT)** Multiprocess runtime (decoder, counting, motion, features/forecast/risk).
  Result:
- [ ] **T8.2 (AGENT)** FastAPI WebSocket with the message schema in `architecture.md` §2.7.
  Result:
- [ ] **T8.3 (AGENT)** Dashboard: heatmap, forecast chart with band, alert feed, replay mode.
  Result:
- [ ] **T8.4 (USER)** Try the demo; list what looks wrong or unclear.
  Result:

---

## Week 7 — Evaluation and write-up

### T9 Evaluation

- [ ] **T9.1 (AGENT)** Fill all metric tables (`architecture.md` §6) on held-out data.
  Result:
- [ ] **T9.2 (AGENT)** Runtime: latency, FPS per camera, GPU memory.
  Result:
- [ ] **T9.3 (AGENT)** Figures: heatmap frame, forecast chart, alert timeline from a run that turns critical.
  Result:
- [ ] **T9.4 (AGENT)** Draft methodology and results text from `results/` only; list citations for every model and dataset.
  Result:
- [ ] **T9.5 (USER)** Final review; rehearse the replay demo.
  Result:

---

## Blocked / waiting on user

The agent keeps this list current.

| Task | Waiting for | Asked on |
| --- | --- | --- |
| T0.6 | User runs notebook 01 on the A100 and returns the two zips | 2026-09-30 |
| T1.2, T2.1, T2.2 | CCTV footage, floor measurements, venue layout | 2026-09-30 |

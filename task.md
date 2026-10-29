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
- [x] **T1.2 (USER)** Collect CCTV: 3–5 cameras, 30–60 min each incl. peak periods; written permission to use for research; for each camera note resolution, FPS, mounting height and angle.
  Result: 2026-10-15 — 4 recordings, 3 rooms (permission granted): cam01/cam02 H705 classroom (72 + 15 min), cam03 H706 classroom, other-side view (33.5 min: studying → break exit), cam04 H103 computer lab (37.7 min: empty → students enter). Room L×B×H in `configs/rooms.yaml` (user, feet → m). No corridor/door camera yet.
  Result: 2026-10-25 — 4 cameras collected (H705 ×2 clips, H706, H103 lab; 2.7 h total) — enough for the paper.
- [ ] **T1.3 (USER, optional)** Own recordings: tripod at a high point looking down, 1080p, 25–30 fps, no zoom/pan, crowded times.
  Result:
- [x] **T1.4 (AGENT)** Camera audit script → `results/camera_audit.md` (resolution, head size in px, FPS, compression, brightness, angle, shake) with per-camera recommendations.
  Needs: T1.2.
  Result: 2026-10-15 — all 4 cameras 1920×1080, 25 fps, steady (≤ 0.64 px/s). Heads p10/median px: cam01 31/40, cam02 41/53, cam03 32/46, cam04 25/35 (lab, darker: brightness 99). Max persons 51/45/48/34. cam04 empty frame: 0 false heads on monitors at conf 0.2; people visible through a glass window must be excluded. `results/camera_audit_*.md`.
- [x] **T1.5 (AGENT)** Sample ~100 frames (≥ 5 s apart, varied density/time/camera); pre-label with CLIP-EBC; export to CVAT point format; write a 5-step guide for the user.
  Needs: T1.2, T3.3 weights.
  Result: 2026-09-30 — 100 frames (cam01 86, cam02 14; ≥ 5 s apart, stratified 0–55 heads) pre-labelled with CrowdHuman head centres (3,562 points; CLIP-EBC gives no points) → `data/cctv/labels/images.zip` + `cvat_prelabels.zip`; guide `docs/cvat_guide.md`. Only one camera, so no camera-level held-out split yet.
- [x] **T1.6 (USER)** Correct pre-labels in CVAT (~5–8 h) and export.
  Result: 2026-10-12 — user corrected 100 frames in CVAT: 4,379 heads (pre-labels 3,562; +817 added, 96 frames changed); 0 out-of-frame points, 2 near-duplicate pairs kept as labelled. User confirmed final labels are reliable (weak part was the model pre-labels).
- [x] **T1.7 (AGENT)** Import labels; split train/val/test **by camera** (≥ 1 camera fully held out); write split file.
  Result: 2026-10-12 — imported (`crowdsafe/datasets/cvat.py`). Single camera → time-block split: val = first 50% of cam01 by time (43 frames), test = rest of cam01 + cam02 (57). No held-out camera yet.

### T2 Calibration and zones

- [x] **T2.1 (USER)** For each camera: one clear frame; 4+ floor points with real distances (tiles, marked rectangle, or taped markers); photos of the markers; 1 extra known distance for validation.
  Agent gives exact instructions per camera before the user goes on site.
  Result: 2026-10-22 — desk/table sizes, gaps, row spacing, doors, camera heights and room photos (`configs/calibration_measurements.yaml`, `data/cctv/calibration/`). H705 and H103 numbers add up (0.8 %, 1 %); H706 text is a copy of H705 and its row spacing disagrees with the image (0.87 vs 0.96 m) → re-measure when convenient.
- [ ] **T2.2 (USER)** Venue layout sketch: walkable areas, entrances/exits, which areas connect; widths of exits, gates, corridors, stairs in metres.
  Result:
- [x] **T2.3 (AGENT)** Calibration tool (click points → homography) using the `ViewTransformer` pattern; save H to YAML.
  Result: 2026-10-15 — `crowdsafe/calibration/homography.py` (plain homography; kept for simple cases). 2026-10-22 — replaced for the CCTV by a full camera model with lens distortion (D6): `crowdsafe/calibration/camera_model.py` (desk-rectangle fit, plumb lines, edge fit), `edges.py`, `scripts/calibrate_cameras.py` → `configs/cameras/{H705,H706,H103}.yaml`. Reprojection 4.2 / 3.6 / 3.9 px. cam01 = cam02 (same fixed camera).
- [ ] **T2.4 (BOTH)** Draw zones with the Roboflow PolygonZone tool (agent explains; user or agent draws); compute areas with Shapely; add adjacency and boundaries from T2.2.
  Result: 2026-10-22 — agent-drawn floor zones (3×3 seating blocks + front aisle per room, 4–16 m²) in `configs/zones/`; `crowdsafe/calibration/zones.py`, `scripts/zone_density.py`. On all 160 labelled frames, CLIP-EBC zone density vs labels: MAE 0.101 (H705) / 0.081 (H706) / 0.078 (H103) persons/m² at true densities up to 1.3–1.7; 94–98 % of labelled heads fall in a zone; head-height assumption 1.1–1.6 m changes MAE by ≤ 0.03. `results/zone_density.md`, `results/figures/zones_*.jpg`. Adjacency/exits wait for T2.2 (user check of zones still open).
- [x] **T2.5 (AGENT)** Validate each camera with the unused distance (target < 10% error). Mark approximate calibrations as approximate.
  Result: 2026-10-22 — all within 10 % except where noted (`results/calibration.md`). H705: row spacing 0.99 vs 0.96 m (+3 %), column pitch 1.69 vs 1.77 m mean (−5 %), desk height 0.78 m (from camera 2.23 m). H706: column pitch 1.78 vs 1.77 m (+1 %), door width 1.19 vs 1.12 m (+6 %; 2nd door partly hidden: 0.92 m), front wall 10.5–10.9 m from camera vs room 10.36 m; camera ~2.6–2.7 m (not 2.13 m as stated). H103: hidden columns 3–4 found within 0.10–0.22 m (2–3 %), table height 0.73 m, camera 2.51 vs 2.58 m (−3 %). All marked approximate.


## Status 2026-10-25 — next step and remaining work

**Done 2026-10-26:** notebook 05 ran (CCTV features + Chronos-2 on sims/CCTV). ~~Next step (user, in the lab): run notebook 05 (`notebooks/05_cctv_features_and_sim_forecasts.ipynb`): upload it and the
videos `cam01–cam04.asf` into `~/crowdsafe_work/data/cctv/`, Run All, download `crowdsafe_nb05_report.zip` into `weights/`.~~ **Next:** T8 dashboard.

**Remaining work (agent unless noted):**
1. ~~Import notebook 05 results~~ done 2026-10-26 (see T6.4/T7.4).
2. T4.3: check CCTV speeds against simple person tracks (and Jülich), `results/motion_validation.md`.
3. ~~Real-data replay timelines~~ done 2026-10-26 (`scripts/alert_timeline.py`, `results/figures/timeline_*.png`).
4. T8: replay dashboard (FastAPI + Next.js): camera view with coloured zones, forecast chart with 90 % band, alert list (T8.1–T8.3); T8.4 user tries it.
5. T5.5 dataset summary; T9: final tables, figures, runtime numbers, methods/results draft (T9.1–T9.4); T9.5 user review + rehearsal.
6. Optional (user): record a short staged crowding video with the calibrated H705 camera for a stronger real demo.

---

## Weeks 2–3 — Counting, fusion, motion

### T3 Counting and fusion

- [x] **T3.1 (AGENT)** Detector wrapper: YOLO11l (COCO, person) + ByteTrack; option for `InferenceSlicer` tiling.
  Result: 2026-09-30 — wrapper `crowdsafe/counting/yolo_track.py` (YOLO + InferenceSlicer tiling + Ultralytics ByteTrack, reliability features); unit-tested; A100 run = notebook 03.
- [x] **T3.2 (AGENT)** Try the CrowdHuman YOLO11 person+head model from Hugging Face (`Sharath33/Person`, ONNX); ask user before choosing if results are close.
  Result: 2026-09-30 — `PersonHeadOnnx` wrapper (model-card preprocessing, NMS) unit-tested; benchmark in notebook 03.
- [x] **T3.3 (AGENT)** Density wrapper: CLIP-EBC ViT-B/16 NWPU weights from the official Hugging Face repo `Yiming-M/CLIP-EBC` (`huggingface_hub.snapshot_download`), adapting model loading and sliding-window inference from its `app.py`. Optionally compare the ShanghaiTech-A checkpoint from GitHub releases.
  Result: 2026-09-30 — `crowdsafe/counting/density_model.py` (torch from HF repo or ONNX; window / whole-image modes); unit-tested; benchmark in notebook 03.
- [ ] **T3.4 (AGENT, optional)** PET-Finetuned point counter: weights from Hugging Face (`Awiros/crowd-counting-and-localization`), inference with the official PET repo.
  Result:
- [x] **T3.5 (AGENT)** Zero-shot benchmark on ShanghaiTech, Jülich videos and user test frames: MAE/RMSE per density band → `results/counting_zeroshot.md`.
  Result: 2026-10-21 — ShanghaiTech MAE A/B: CLIP-EBC 66.95/11.66, YOLO11l 418.3/92.8, CrowdHuman heads 374.2/70.2. Own CCTV (user-labelled, zero-shot, CPU ONNX): H705 (100 frames) CLIP-EBC 5.10 / YOLO11l 6.36 / heads 8.04; H706 (30) 2.58 / 6.53 / 6.23; H103 lab (30, glass window excluded via `configs/roi.yaml`) 1.97 / 17.63 / 28.23. `results/counting_zeroshot_cctv.md`, `results/counting_zeroshot_newcams.md`. Jülich-video part pending.
  Result: 2026-10-25 — closed: CCTV + ShanghaiTech done; Jülich has trajectories, not suitable labelled video, so that part is dropped.
- [x] **T3.6 (BOTH)** Decision: fine-tune CLIP-EBC or YOLO? Agent presents numbers and GPU-time estimate; user decides.
  Result: 2026-10-12 — no fine-tuning proposed: zero-shot error 12–15% and combining models removes most of it (see T3.8). Revisit only if a second camera is much worse.
  Result: 2026-10-25 — no fine-tuning (D7, option A): CLIP-EBC zero-shot is accurate on all rooms (MAE 2–5 people/frame).
- [x] **T3.7 (AGENT)** Fusion: reliability features, fit weights on val frames, EMA smoothing, density = count / area.
  Result: 2026-10-12 — `crowdsafe/counting/fusion.py` (learned w = sigmoid(θ·f), hard switch, constant weight). Detector chosen by measured error: YOLO11l (6.36) over CrowdHuman heads (8.04).
- [x] **T3.8 (AGENT)** Ablation: YOLO only / CLIP-EBC only / hard switch / weighted fusion on held-out camera → `results/fusion_ablation.md`.
  Result: 2026-10-21 — single-camera time split: constant weight 1.59 best, learned 2.09. **Held-out rooms (leave-one-room-out, mean MAE):** hard switch 3.16 · CLIP-EBC only 3.22 · 50/50 4.48 · constant weight 4.56 · learned fusion 5.22 · YOLO only 10.17. Learned/constant weights do not transfer across rooms (YOLO undercounts 18/frame in the lab). `results/fusion_ablation_heldout.md`, `results/fusion_loco.md`. Contribution framing needs a user decision.

### T4 Motion features

- [x] **T4.1 (AGENT)** RAFT wrapper (`raft_large`, 3–5 FPS, ~960 px); Farneback fallback.
  Result:
  Result: 2026-10-25 — code done (`crowdsafe/motion/flow.py`: RAFT-large + Farneback fallback, tested). Full run on the A100 in notebook 05 (user, pending).
  Result: 2026-10-26 — A100 run done (notebook 05, RAFT-large via PyAV video reader).
- [x] **T4.2 (AGENT)** Zone motion: flow → m/s via H; density mask; mean_speed, speed_var, pressure, dir_entropy, counterflow, inflow/outflow.
  Result:
  Result: 2026-10-25 — code done (`crowdsafe/motion/zone_motion.py`, `scripts/cctv_features.py`: density + m/s motion per zone per second, resumable; laptop test on cam02 exit: walking zones 0.16–0.30 m/s). Waiting for notebook 05 results.
  Result: 2026-10-26 — A100 run done: all 4 cameras, 9,513 s of per-zone density + motion (`data/features/cctv/`, from `weights/crowdsafe_nb05_report/`). Peak zone density 1.7–3.2 persons/m² (cam04–cam01); classrooms never reach 4.
- [ ] **T4.3 (AGENT)** Validate speeds against Jülich trajectories and ByteTrack tracks → `results/motion_validation.md`.
  Result:

---

## Week 4 — Dataset

### T5 Feature logging and simulation

- [x] **T5.1 (AGENT)** Feature schema + Parquet logger (`architecture.md` §2.4).
  Result: 2026-09-30 — `crowdsafe/features/schema.py` (14 columns, validate/conform) + `logger.py` (Parquet, SQLite); tests pass.
- [x] **T5.2 (AGENT)** Jülich loader: trajectories → PedPy → feature table, per zone.
  Result: 2026-09-30 — 144 runs / 8 experiments → `data/features/julich/`; 2 m zones; max density 9.0 (entrance_semicircle), 5.65 (entrance_corridor), 5.3 (corridor_uni_2013); 643 zone-s ≥ 5 p/m²; bottleneck1 has no 2 m zone (area 0.6 m wide); PedPy cross-check diff 0.0000 on 6/6. `results/julich_features_summary.md`.
- [x] **T5.3 (AGENT)** Run the pipeline over Mall/UCSD/FDST/CCTV → feature tables.
  Result:
  Result: 2026-10-25 — CCTV part = notebook 05 (pending). Mall/UCSD/FDST dropped (own CCTV + Jülich + sim are enough).
  Result: 2026-10-26 — CCTV feature tables done (4 cameras).
- [x] **T5.4 (BOTH)** JuPedSim: agent proposes 3 geometries from the user's venue layout (T2.2); user approves; agent runs 15–20 scenarios incl. surges, exit closures, > 5 persons/m².
  Result: 2026-10-24 — user approved ("OK sims"). 16 scenarios in 4 layouts (H706-like classroom, corridor + bottleneck, hall with two exits, bidirectional corridor) incl. surges, exit closures, opposing flows; 10 exceed 5 persons/m² (max 8.0), classrooms stay < 4. `crowdsafe/simulation/scenarios.py`, `scripts/run_simulations.py`, `results/simulation_summary.md`. Sim alerts (5 held-out scenarios, 29 events ≥ 4): physics fill-rate upper bound alerts 86 % of events early (median 25 s, 1.6 false alarms/zone-h); current density and persistence 0 % early. Chronos-2 on sims pending (A100). `results/alerts_sim.md`, `results/forecast_sim.md`.
  Result:
- [ ] **T5.5 (AGENT)** Split train/val/test by experiment, geometry and camera; write `results/dataset_summary.md`.
  Result: 2026-09-30 — Jülich split proposed in `configs/forecast_julich.yaml` (test: corridor_uni_2013 + entrance_semicircle; val: bottleneck2, entrance_corridor, corridor_bi_2009; train: corridor_uni_2009, train_platform) — awaiting user OK; sim/CCTV splits later.
  Result: 2026-10-25 — Jülich split (configs/forecast_julich.yaml), sim split by scenario (configs/simulation.yaml), CCTV split by camera (configs/forecast_cctv.yaml). Summary file still to write.

---

## Week 5 — Forecasting

### T6 Forecasting

- [x] **T6.1 (AGENT)** Persistence and physics fill-rate baselines + time-to-critical.
  Result: 2026-09-30 — `forecasting/baselines.py` (persistence, physics fill-rate + time-to-critical). Jülich test MAE (p/m²) 10/30/60 s: persistence 0.39/0.57/0.71, physics 1.05/2.51/4.45 (noisy 2 m-zone flux). `results/forecast_julich.md`.
- [x] **T6.2 (AGENT)** Chronos-2 zero-shot wrapper: zones as a group, motion features as covariates, quantiles 0.1/0.5/0.9.
  Result: 2026-09-30 — Chronos-2 zero-shot on Jülich test (A100): MAE 10/30/60 s per-zone 0.376/0.558/0.727, joint 0.372/0.569/0.741 vs persistence 0.387/0.568/0.711.
- [x] **T6.3 (AGENT)** Split conformal calibration of the 90% bound.
  Result: 2026-09-30 — split conformal, val→test coverage of calibrated p90 (target 0.90): Chronos-2 per-zone 0.892/0.875/0.925, joint 0.861/0.835/0.880; persistence 0.850/0.660/0.676; physics 0.798/0.700/0.751.
- [x] **T6.4 (AGENT)** Evaluate on held-out data: MAE per horizon, coverage → `results/forecasting.md`.
  Result: 2026-09-30 — Jülich part done: `results/forecast_julich_a100.md` (Chronos-2 ≈ persistence on MAE, but only Chronos-2 gives a near-90% calibrated upper bound). Sim + CCTV parts pending (T5.3/T5.4).
  Result: 2026-10-25 — Jülich done (A100); sim: persistence/physics done on laptop (`results/forecast_sim.md`), Chronos-2 in notebook 05; CCTV in notebook 05.
  Result: 2026-10-26 — Done. Sim (A100, server's own sim run): Chronos-2 joint 90 % bound coverage 0.91–0.96 vs persistence 0.87–0.93, MAE similar (`results/forecast_sim_a100.md`). Real CCTV (test cam02+cam04): Chronos-2 lowest MAE 0.064–0.097 vs persistence 0.074–0.105 persons/m², coverage 0.89–0.91 (`results/forecast_cctv.md`).
- [ ] **T6.5 (BOTH, optional)** If Chronos-2 is weak: agent proposes fine-tuning via AutoGluon with time estimate; user decides.
  Result:
- [ ] **T6.6 (AGENT, optional)** Per-zone GRU comparison.
  Result:

---

## Week 6 — Risk engine and demo

### T7 Risk engine and alerts

- [x] **T7.1 (AGENT)** Density bands on calibrated upper bound.
  Result: 2026-10-24 — `crowdsafe/risk/rules.py` (RiskEngine: band of the worst conformal-calibrated 90 % bound over horizons, time-to-critical, reasons); tested.
- [ ] **T7.2 (AGENT)** Precursor rules; tune thresholds on Jülich + sim; write them to camera YAMLs.
  Result: 2026-10-24 — Jülich part done (sim pending, T5.4). Tuned on val only, kept only if precision ≥ 2× base rate: **crowd slowing** (speed ≤ 0.21 m/s at Medium density; P 0.47 vs base 0.145) and direction entropy ≥ 0.78 (P 0.29) kept; **pressure, counter-flow and flow collapse rejected** (do not precede filling-up at Jülich densities). Thresholds in `configs/risk.yaml`. Re-tuned on pooled Jülich + sim val: **no precursor passes the 2× test** (crowd slowing does not transfer to simulated crowds) → all switched off in `configs/risk.yaml`; Jülich-only tuning result kept above for the paper.
- [ ] **T7.3 (AGENT, optional)** Isolation Forest anomaly detector.
  Result:
- [x] **T7.4 (AGENT)** Hysteresis, alert reasons, time-to-critical; replay tool → alert timelines.
  Result: 2026-10-24 — `crowdsafe/risk/alerts.py` (raise 10 s / clear 30 s), `scripts/eval_alerts.py`. Held-out Jülich (9 zone events ≥ 4 persons/m², small sample): alerted *before* the event — current density 0 %, persistence 22 %, physics 22 % (11.9 false alarms/zone-h), **Chronos-2 per-zone 56 % (median lead 19 s, mean 96 s; 1.8 false alarms/zone-h)**; precursors lift density-only to 33 % and persistence to 44 % but add nothing to Chronos-2. esults/alerts_julich.md\.
  Result: 2026-10-26 — **Metric correction:** many alerts were raised in the first seconds of monitoring (first forecast extrapolates the start-up ramp) and simply stayed on, which counted as early warnings with long leads. Added a strict metric (`in window %`: alert ≤ longest horizon + 10 s before the event, not in the first 15 s). Strict results, event = 4 persons/m²: Jülich (9 events) density-only 0 %, persistence/physics/Chronos-2 11 % each (1 event); sim (47 events) density-only 0 %, persistence 0 %, physics 32 % (median 25 s), Chronos-2 per-zone 34 % (7.5 s), joint 30 %. The earlier 56 % / 86 % figures used the loose metric and must not be quoted. Real CCTV: no zone reaches 4 → no events, no alerts, 0 false alarms in 8.6 zone-hours. Figures: `results/figures/timeline_cam02.png`, `timeline_julich_corridor.png`, `timeline_julich_entrance.png`.
  Result: 2026-10-26 — User kept setting A (D8); ablation of shorter wait / rate-of-rise / shorter history in `results/alert_experiment.md`. One-page results for the mentor: `docs/FINAL_RESULTS.md`.
- [ ] **T7.5 (USER)** Review default thresholds for the venue context (Indian crowd densities may justify different limits); confirm or adjust.
  Result:

### T8 Backend and dashboard

- [ ] **T8.1 (AGENT)** Multiprocess runtime (decoder, counting, motion, features/forecast/risk).
  Result:
  Result: 2026-10-26 — Replay mode only for now: stored features + forecasts re-run through the risk engine (`crowdsafe/server/replay.py`). A live multiprocess runtime (decoder → counting → motion → forecast) is not built; optional if time allows.
- [x] **T8.2 (AGENT)** FastAPI WebSocket with the message schema in `architecture.md` §2.7.
  Result:
  Result: 2026-10-26 — `crowdsafe/server/app.py`: REST (runs, layout, frame, summary) + WebSocket `/ws/replay/{id}` streaming the architecture §2.7 messages; smoke test `tests/test_server.py`.
- [x] **T8.3 (AGENT)** Dashboard: heatmap, forecast chart with band, alert feed, replay mode.
  Result:
  Result: 2026-10-26 — `dashboard/` (Next.js): run picker (3 CCTV rooms, 2 Jülich crowds, 2 simulations), camera view with zones coloured by level and live density, time bar with peak density + alert marks (click to seek), forecast chart with 10–90 % band, alert feed with reasons, speed 1–60×. Checked in the browser on cam02 and the Jülich corridor run.
- [ ] **T8.4 (USER)** Try the demo; list what looks wrong or unclear.
  Result:

---

## Week 7 — Evaluation and write-up

### T9 Evaluation

- [ ] **T9.1 (AGENT)** Fill all metric tables (`architecture.md` §6) on held-out data.
  Result:
- [x] **T9.2 (AGENT)** Runtime: latency, FPS per camera, GPU memory.
  Result: 2026-10-26 — A100 (notebook 06, cam02 at real speed): live Chronos-2 median latency 0.49 s (p95 1.6 s), density 159 ms, RAFT 103 ms, Chronos-2 54 ms every 5 s, 94 % of 1-s cycles on time, GPU 2.0 GB; max throughput 4.6x real time. esults/latency_*.md\.
  Result: 2026-10-26 — live pipeline built: \crowdsafe/datasets/live.py\ (real-speed reader, drops frames when slow), \scripts/live_pipeline.py\ (same models/settings, per-stage timing, SQLite log, forecast every 5 s, risk + alerts). Laptop CPU test (cam02, ONNX CLIP-EBC + Farneback): ~19 s per cycle (density 18.7 s; flow 0.3 s; zones/forecast/risk 10–30 ms) → laptop cannot run live. A100 measurement = notebook 06 (user, pending).
  Result:
- [x] **T9.3 (AGENT)** Figures: heatmap frame, forecast chart, alert timeline from a run that turns critical.
  Result: 2026-10-27 — `scripts/paper_figures.py`: pipeline diagram, zone-density heatmaps on the busiest frame of cam02/03/04 (max 1.87 / 1.94 / 1.18 p/m²), forecast fan on sim corridor surge (Chronos-2 bound 2.33→3.09 vs persistence flat 1.8, both < 4 at 30 s lead); alert timelines already exist. Dashboard screenshot added by user (`results/figures/paper_dashboard.png`).
- [ ] **T9.4 (AGENT)** Draft methodology and results text from `results/` only; list citations for every model and dataset.
  Result: 2026-10-26 — first full draft `docs/paper_draft.md` (abstract → conclusion, 5 tables, figure list, references marked [CHECK]); every number from `results/`.
  Result: 2026-10-27 — IEEE conference version `docs/paper/main.tex` (IEEEtran, 3 authors, 6 tables, 6 figures) + `refs.bib` (14 refs, verified; 3 still CHECK: Jülich per-experiment DOIs, JuPedSim citation, PedPy year). Not compiled locally (no LaTeX) — compile on Overleaf. Open: check page limit of the chosen IEEE conference.
  Result: 2026-10-29 — revision per `Claude_CrowdSafe_Research_Paper_Revision_Prompt.md`: new title and section structure; vector pipeline Fig. 1 (`paper_pipeline.pdf`), real dashboard Fig. 2, real alert timeline Fig. 3 (`paper_alert_timeline.pdf`, Jülich uni_corr_500_09: c3_2 warned 49 s ahead); Table I research gap; 5-p/m² alert results added; coverage by true density (`scripts/coverage_by_density.py`, `results/coverage_by_density_*.md`: Jülich ≥ 4 p/m² Chronos-2 per zone 0.76–0.88 vs persistence 0.56–0.78). Corrections: Jülich coverage is over all test points (not "critical targets"); slowing precursor precision 0.54 (3.75×), not 0.47/3.2×; hard switch is frame-level only, zone path = CLIP-EBC map; LOO H706 learned fusion 3.08. 23 references, all cited; static checks pass; NOT compiled (no LaTeX here).
- [ ] **T9.5 (USER)** Final review; rehearse the replay demo.
  Result:

---

## Blocked / waiting on user

The agent keeps this list current.

| Task | Waiting for | Asked on |
| --- | --- | --- |
| T0.6 | User runs notebook 01 on the A100 and returns the two zips | 2026-09-30 |
| T1.2, T2.1, T2.2 | CCTV footage, floor measurements, venue layout | 2026-09-30 |

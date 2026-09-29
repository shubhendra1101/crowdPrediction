# CrowdSafe Implementation Guide

Sep 29, 2026 · @Shruti Mishra

## Overview

The goal is a system that forecasts each zone's crowd density and stampede risk minutes ahead and raises tiered alerts before conditions turn critical. The plan below builds it in 10 steps over 7 weeks on one A100 (40 GB), reusing the existing CrowdSafe repo (YOLO11 + Next.js dashboard).

The core rule: the vision layer turns video into one feature row per zone per second, and everything after that (forecasting, risk, alerts) only reads those rows. That lets you train the forecaster on simulated crowds, where critical situations have ground truth, and still run it on real CCTV.

| Step | What it produces | Prebuilt model / tool | What you build |
| --- | --- | --- | --- |
| 0. Setup & baseline | Clean repo, baseline metrics | Ultralytics, PyTorch | Refactor only |
| 1. Data & CCTV quality | Audit, \~100 labelled test frames | Public datasets, Jülich archive; CLIP-EBC pre-labels in CVAT | Correct the pre-labels |
| 2. Calibration & zones | Zones in real m² | Supervision (PolygonZone, LineZone, ViewTransformer), OpenCV | Measure 4 floor points per camera |
| 3. Counting & fusion | Persons/m² per zone | YOLO11 (COCO or CrowdHuman weights) + ByteTrack; CLIP-EBC released weights; tiled inference | Fusion weights (a small logistic regression) |
| 4. Motion features | Speed, variance, pressure, flow | RAFT, pretrained in torchvision | Zone feature code |
| 5. Dataset | Feature sequences | Jülich trajectories + PedPy; JuPedSim | 15–20 simulation runs |
| 6. Forecasting | Forecasts with bounds | Chronos-2, zero-shot | Nothing at first; optional fine-tune |
| 7. Risk engine & alerts | Levels per zone | Rules + scikit-learn Isolation Forest | Threshold tuning |
| 8. Backend & dashboard | Live demo | FastAPI, Supervision annotators, existing Next.js | Wiring |
| 9. Evaluation | Paper results | — | Metric scripts |

A pipeline diagram of how the steps connect follows at the end of this section.

&#91;embedded content: CrowdSafe pipeline · Steps 1–8\]

Only the feature table connects the vision layer to prediction, so the forecaster can train on simulated crowds and run on real cameras.

## Reuse first: prebuilt models and datasets

Every model in this plan has open-source pretrained weights, so you start with inference and fine-tune only where your own footage shows large errors. You build just three small things yourself: camera calibration, about 100 labelled CCTV test frames (pre-labelled by a model, then corrected), and 15–20 simulation runs.

| Need | Use off the shelf | Pretrained on / source | Fine-tune later if needed |
| --- | --- | --- | --- |
| Zones, line crossing, pixel-to-metre | [Roboflow Supervision](https://supervision.roboflow.com/latest/how_to/count_in_zone/): `PolygonZone`, `LineZone`, and the `ViewTransformer` pattern from its [speed-estimation example](https://github.com/roboflow/supervision/tree/develop/examples/speed_estimation) | — | — |
| Person detection | [YOLO11](https://huggingface.co/Ultralytics/YOLO11) (Ultralytics) | COCO | CrowdHuman, then your CCTV |
| Person + head detection in crowds | Community CrowdHuman-trained YOLO, e.g. [a YOLO11 person + head model](https://huggingface.co/Sharath33/Person/blob/main/README.md) or [a YOLOv8 head detector](https://github.com/Owen718/Head-Detection-Yolov8) | CrowdHuman | Your CCTV |
| Tracking | ByteTrack (built into Ultralytics and Supervision) | — | — |
| Density counting | [CLIP-EBC](https://github.com/Yiming-M/CLIP-EBC): weights on its releases page, including ViT-B/16 | ShanghaiTech, UCF-QNRF, NWPU-Crowd | Your CCTV, with its own `trainer.py` |
| Optical flow | RAFT in `torchvision.models.optical_flow` | Ships pretrained | Rarely needed |
| Real high-density data with ground truth | [Jülich Pedestrian Dynamics Data Archive](https://www.re3data.org/repository/r3d100013370): videos plus every person's trajectory | Lab experiments (bottleneck, corridor, entrance, platform) | — |
| Density, speed and flow from trajectories | PedPy (Jülich's analysis library) | — | — |
| Rare critical scenarios | JuPedSim | — | — |
| Forecasting | [Chronos-2](https://huggingface.co/amazon/chronos-2) (Amazon, 120M parameters) | Large real and synthetic time-series corpus | Your feature sequences, via AutoGluon |
| Anomaly detection | scikit-learn Isolation Forest | Fits on your normal footage in seconds | — |

Reusing these doesn't weaken the research: your contribution is the reliability-aware fusion, the calibrated motion-risk indicators, and the end-to-end early-warning evaluation.

## Requirements: what to download, collect and set up

Most models download automatically in code; the real work on your side is collecting CCTV clips, floor measurements and venue layouts. Not everything is on Hugging Face: CLIP-EBC and APGCC weights come from GitHub, and RAFT ships inside torchvision.

### 1. Accounts and setup

- [ ] Hugging Face account and access token (for Chronos-2 and community YOLO models)
- [ ] Kaggle account (ShanghaiTech download)
- [ ] CVAT: a free account on app.cvat.ai, or run it locally with Docker
- [ ] A100 environment with CUDA working (`torch.cuda.is_available()` returns True)
- [ ] Disk space: reserve roughly 150 GB for datasets, videos and checkpoints

### 2. Pretrained models

| Model | Where | How you get it |
| --- | --- | --- |
| YOLO11l (person) | Ultralytics, also on Hugging Face | Auto-downloads: `YOLO('yolo11l.pt')` |
| CrowdHuman person + head YOLO | Hugging Face / GitHub (community) | Download weights from the model page |
| CLIP-EBC ViT-B/16 | GitHub releases page of Yiming-M/CLIP-EBC | Download the `.pth` checkpoints manually |
| APGCC (optional) | Its GitHub repository | Download the ShanghaiTech checkpoint |
| RAFT optical flow | torchvision | Auto-downloads: `raft_large(weights=Raft_Large_Weights.DEFAULT)` |
| Chronos-2 | Hugging Face `amazon/chronos-2` | Auto-downloads through `chronos-forecasting` |

### 3. Public datasets

Some of these ask you to fill in a request or registration form, so start the downloads on day 1.

| Dataset | Used for | Priority |
| --- | --- | --- |
| Jülich Pedestrian Dynamics Data Archive (bottleneck, corridor, entrance, platform experiments) | Ground-truth density, speed and flow; forecasting data | Must |
| ShanghaiTech A/B | Benchmarking counting weights | Must |
| JHU-Crowd++ | Harder counting benchmark, fine-tuning | Should |
| Mall, UCSD Pedestrian | Video sequences for motion and forecasting | Should |
| FDST | More video sequences | Nice to have |
| NWPU-Crowd | Large-scale counting benchmark | Nice to have |
| CrowdHuman | Only if you fine-tune YOLO later | Later |

### 4. What you collect yourself

**CCTV and your own recordings**

- [ ] 3–5 cameras, 30–60 minutes each, including the busiest periods (peak hours, event entry and exit)
- [ ] For each camera: resolution, frame rate, and approximate mounting height and angle
- [ ] Written permission from the venue or authority to use the footage for research
- [ ] Your own recordings: tripod at a high point looking down, 1080p, 25–30 fps, no zoom or panning during a clip

**Calibration (per camera)**

- [ ] One clear frame of the empty or light floor
- [ ] 4 or more floor points with real distances between them: tile sizes, marked rectangles, or tape-measured markers
- [ ] Photos of the markers so the points can be found in the frame again

**Venue layout (for zones and simulation)**

- [ ] Rough floor plan or sketch: walkable areas, entrances, exits and which areas connect
- [ ] Widths of exits, gates, corridors and stairs, in metres

**Labels**

- [ ] About 100 frames from your CCTV, corrected in CVAT after model pre-labelling (at least one whole camera kept for testing)

### 5. Python packages

```
torch torchvision            # CUDA build matching your A100 driver
ultralytics supervision      # detection, tracking, zones, tiled inference
opencv-python shapely        # homography, zone areas
chronos-forecasting          # Chronos-2
autogluon.timeseries         # optional: fine-tuning Chronos-2
jupedsim pedpy               # simulation, trajectory analysis
scikit-learn                 # Isolation Forest, fusion regression
pandas pyarrow pyyaml        # feature logging, configs
fastapi uvicorn websockets   # backend for the dashboard
huggingface_hub              # model downloads
# plus CLIP-EBC's own requirements.txt, installed from its repo
```

## Step 0 — Setup and baseline (Week 1, days 1–2)

Freeze the current system's numbers before changing anything; every later result is compared against them.

1. Create one environment: Python 3.11, PyTorch 2.x with CUDA, `ultralytics`, `opencv-python`, `shapely`, `torch-geometric-temporal`, `jupedsim`, `fastapi`, `pandas`, `pyarrow`.
2. Restructure the repo into modules:

```
crowdsafe/
├── configs/          # one YAML per camera: homography, zones, adjacency
├── calibration/      # click-points tool, zone editor
├── counting/         # yolo_track.py, density_model.py, fusion.py
├── motion/           # flow.py, zone_motion.py
├── features/         # schema.py, logger.py
├── simulation/       # jupedsim scenarios -> same feature schema
├── forecasting/      # baselines.py, gru.py, stgnn.py, conformal.py
├── risk/             # rules.py, anomaly.py, alerts.py
├── server/           # FastAPI + WebSocket
├── dashboard/        # existing Next.js app
└── eval/             # metrics + plots for the paper
```

3. Run the current YOLO11m + CSRNet system on 3–4 test clips. Record count MAE, FPS and GPU memory in a `results/baseline.md` table.

**Done when:** the old system runs from the new layout and the baseline table exists.

## Step 1 — Data and CCTV quality (Week 1, days 2–5)

Your CCTV footage decides whether the system works in the real world, so audit it first and build a small labelled set from it.

**1a. Collect three kinds of data**

- Public image datasets (ShanghaiTech A/B, JHU-Crowd++, NWPU-Crowd): only to benchmark pretrained weights, not to train from scratch.
- Jülich Pedestrian Dynamics Data Archive: videos plus every person's trajectory, so counts, densities and speeds have ground truth without any annotation.
- Other video datasets for motion and forecasting: Mall, UCSD Pedestrian, FDST.
- Your own CCTV and recorded clips: the real-world test set.

**1b. Audit each CCTV camera.** Fill one row per camera:

| Check | How to measure | If it's bad |
| --- | --- | --- |
| Resolution | Frame size | Below 720p: rely on the density model, not YOLO |
| Head size | Measure a few heads in pixels | Under \~10–15 px (rule of thumb): YOLO misses them; use density model |
| Frame rate | Metadata | Below 5 fps: compute flow between frames 1 apart, not 2+ |
| Compression blocking | Look at moving areas | Light denoise (`cv2.fastNlMeansDenoisingColored`) before flow |
| Low light / night / IR | Mean brightness | CLAHE on the luminance channel |
| Camera angle | Visual check | Near-vertical is best for density; very shallow angles hurt calibration |
| Camera shake | Stabilise a clip | Skip that camera for motion features |

Avoid heavy super-resolution: it adds latency and can invent heads that are not there.

**1c. Annotate your own footage**

1. Sample 200–300 frames, at least 5 seconds apart, across different crowd levels, times of day and cameras. Start with about 100 as a test set; label the rest only if you end up fine-tuning.
2. Pre-label instead of labelling from scratch: run CLIP-EBC (or APGCC) to get head points, import them into CVAT (point mode), and only fix the mistakes. That should take roughly 5–8 hours per 100 frames.
3. Split by camera, not by frame: keep at least one whole camera out as the test set, so results show generalisation.

**Done when:** a camera audit table exists and annotated frames are split into train / validation / test.

## Step 2 — Camera calibration and zones (Week 1, days 5–7)

Each camera gets a homography, a 3×3 matrix that maps image pixels to floor coordinates in metres, so density comes out in persons/m².

1. **Measure 4+ floor points per camera.** Use floor tiles of known size, a marked rectangle, or tape-measured markers. For public datasets, estimate from typical objects (door width ≈ 0.9 m) and note it as approximate. Jülich experiments already come with real-world coordinates.
2. **Draw zones with Roboflow's PolygonZone web tool:** upload a saved frame, click the polygon, and paste the coordinates into the config.
3. **Reuse Supervision's `ViewTransformer` pattern** from its speed-estimation example: it wraps `cv2.getPerspectiveTransform` and `cv2.perspectiveTransform` to map pixels to metres. Get each zone's real area with `shapely.Polygon(...).area`. Keep zones at about 10–50 m².
4. **Boundaries between zones:** use `sv.LineZone` for track-based in/out counts in sparse zones; dense zones use the flow-based flux from Step 4. Record which zones are adjacent.
5. **Check it:** measure one known distance you did not use for calibration. Error under \~10% is fine.

One detail: counting models locate heads, which sit above the floor. Assign each head to a zone using the image-space polygon, and only use the homography for areas and speeds.

```yaml
# configs/cam01.yaml
camera_id: cam01
homography: [[...], [...], [...]]
zones:
  - id: Z1
    polygon_px: [[120, 400], [620, 390], [700, 710], [60, 715]]
    area_m2: 24.5
adjacency: [[Z1, Z2], [Z2, Z3]]
boundaries:
  - between: [Z1, Z2]
    line_px: [[620, 390], [700, 710]]
```

**Done when:** every camera has a YAML config and one validation distance checked.

## Step 3 — Counting and fusion (Weeks 2–3)

Two counters run side by side, and a learned per-zone weight decides how much to trust each. This fusion is your main research contribution, so the ablation here matters most.

**3a. Detector + tracker (sparse zones, and speeds for checking motion)**

- Start with pretrained weights and no training: YOLO11l on COCO, person class only.
- Also try a community CrowdHuman-trained person + head model. In dense, top-down views the head class survives occlusion better. Keep whichever has lower error on your \~100 test frames.
- For small heads in low-resolution CCTV, tile the frame with Supervision's `InferenceSlicer` (SAHI-style) instead of retraining.
- Tracker: ByteTrack via Ultralytics `model.track(..., tracker='bytetrack.yaml')`.
- Fine-tune later, on CrowdHuman then your frames, only if the error stays high.

**3b. Density model (dense zones)**

- Primary: **CLIP-EBC released weights** (ViT-B/16) from the repo's releases page. Wrap its model loading and sliding-window inference as `counting/density_model.py`.
- Compare its ShanghaiTech-A and NWPU checkpoints zero-shot on your test frames and keep the better one.
- Alternative: **APGCC** with its ShanghaiTech weights, if you want head point locations. It over-counts sparse scenes without fine-tuning.
- Fine-tune later with CLIP-EBC's own `trainer.py`: export your CVAT points in ShanghaiTech layout (images plus point files) and train from the released checkpoint.
- Zone count = sum of the density map inside the zone polygon.

**3c. Confidence-weighted fusion**

For each zone and frame, compute a few reliability features: mean YOLO confidence, number of detections, share of boxes overlapping another box (IoU > 0.3), mean box height in pixels, and the density-model count.

```latex
\hat{N}_z = w_z \, N^{\text{YOLO}}_z + (1 - w_z) \, N^{\text{density}}_z, \qquad w_z = \sigma(\theta^\top f_z)
```

- Fit θ (a logistic regression, or a tiny 2-layer MLP) on the validation frames by minimising the absolute error of the fused count against ground truth.
- Smooth each zone's count over about 3 seconds with an exponential moving average.
- Density = fused count ÷ zone area.

**3d. Ablation table for the paper**

Report MAE and RMSE, split by density band, for: YOLO only · CLIP-EBC only · hard switch (your old method) · weighted fusion.

**Done when:** the fusion beats both single models on the held-out camera, and the pipeline outputs persons/m² per zone at 2–5 FPS.

## Step 4 — Motion features (Week 3)

Dense optical flow gives how the crowd moves in every zone, which is where stampede precursors show up; tracking individuals fails once the crowd is packed.

1. **Flow model:** **RAFT** from `torchvision.models.optical_flow` (`raft_large` on the A100; `raft_small` if you need speed). Run it at 3–5 FPS on frames resized to about 960 px wide. Fallback: OpenCV Farneback on CPU.
2. **Convert to metres per second:** for each pixel p with flow d, project both p and p + d through the homography, then divide the floor distance by the time between frames.
3. **Mask out empty floor:** only keep flow where the density map shows people, so static background noise does not count.
4. **Compute per zone, once per second:**

| Feature | How | Why it matters |
| --- | --- | --- |
| Mean speed (m/s) | Density-weighted mean of speed | Flow slowing down is an early precursor |
| Speed variance | Density-weighted variance of velocity | Input to crowd pressure |
| Crowd pressure | Density × speed variance | Turbulence indicator (Helbing) |
| Direction entropy | Entropy of an 8-bin direction histogram | Chaotic, disordered motion |
| Counter-flow ratio | Share of flow opposite the zone's main direction | Colliding streams raise risk |
| Inflow / outflow (persons/s) | Density × velocity component across each boundary line, summed along the line | Net fill rate of each zone |

```latex
P_z = \rho_z \cdot \operatorname{Var}(\mathbf{v})_z
```

5. **Sanity check:** in sparse zones, compare the flow-based mean speed with ByteTrack track speeds. They should agree within roughly 20%.

**Done when:** every camera outputs all six features per zone per second.

## Step 5 — Feature logging and simulation dataset (Week 4)

Real high-density data with ground truth already exists in the Jülich archive, so simulation only fills the gaps it doesn't cover: sustained density above 5 persons/m², surges and exit closures.

**5a. One feature schema for everything**

```
timestamp, camera_id, zone_id, count, density, fusion_weight,
mean_speed, speed_var, pressure, dir_entropy, counterflow,
inflow, outflow, source   # source = real | sim
```

Write to Parquet files (one per clip) for experiments; SQLite is enough for the live demo.

**5b. Real sequences, from two sources**

- **Jülich archive experiments** (bottleneck, corridor, entrance, train platform): compute the features straight from their trajectories with PedPy, which gives density, speed and flow. Also run your own video pipeline on their videos and compare it with the trajectory ground truth. That measures your counting and speed error without labelling anything.
- **Mall, UCSD, FDST and your CCTV**: run Steps 2–4 over them.

**5c. Simulated sequences with** **JuPedSim**

1. Build 3 geometries in JuPedSim: a bottleneck exit, a corridor merge and a two-way corridor (counter-flow).
2. Run 15–20 simulations of 10–30 minutes each, varying arrival rate and exit width. Include runs that pass 5 persons/m², runs that stay safe, and surge events such as a sudden jump in arrivals or an exit closing.
3. Compute the same six features with PedPy from the simulated trajectories, written with `source = sim`.

Hold out whole experiments, geometries and cameras for testing.

**Done when:** you have a training set, a validation set and a test set of feature sequences, split by scenario and camera.

## Step 6 — Forecasting (Week 5)

Start with Chronos-2, a pretrained forecasting model that works zero-shot, so there is nothing to train on day one. Trained models become optional comparisons.

**Setup**

- Input: the last 60–120 seconds of all features for every zone.
- Output: density and pressure per zone at several horizons, each with a 10th, 50th and 90th percentile.
- Horizons: 30 s, 1 min and 2 min on real clips (they are short); add 5 and 10 min on simulated data.

**Models, in build order**

1. **Persistence:** the future equals the latest value. Any model that can't beat this is useless.
2. **Physics fill-rate model:** future density = current density + (inflow − outflow) ÷ area × horizon. This also gives time-to-critical, the seconds until the zone reaches 5 persons/m².
3. **Chronos-2, zero-shot (main model):** a 120M-parameter pretrained model (`amazon/chronos-2` on Hugging Face) that returns quantile forecasts directly. Give it each zone's density as a target, all zones of one camera as one multivariate group, and pressure, inflow and outflow as covariates. Run it with the `chronos-forecasting` package or AutoGluon TimeSeries.
4. **Optional, if time allows:** fine-tune Chronos-2 on your feature sequences through AutoGluon (check its docs for Chronos-2 support), and train the per-zone GRU as a comparison. The spatio-temporal GNN becomes future work.

**Training details**

- Context: 60–120 seconds of history at 1 Hz; try longer contexts too, since Chronos-2 handles them.
- Use its 10th, 50th and 90th percentile outputs directly.
- **Conformal calibration:** on the validation set, check how often the true value lands above the 90th percentile and widen the bound until that is 10% or less. This makes the upper bound trustworthy enough to alert on.
- Evaluate everything on held-out experiments, geometries and cameras.

**Done when:** a table shows MAE per horizon for all four models, plus the actual coverage of the 90% bound, on held-out scenarios and cameras.

## Step 7 — Risk engine and alerts (Week 6, days 1–3)

Risk is decided by explainable rules on the forecast and on precursor signals, not by a black-box classifier, because no large labelled stampede dataset exists.

**7a. Base level from the forecast upper bound (90th percentile)**

| Forecast density (persons/m²) | Level |
| --- | --- |
| Below 2 | Low |
| 2 to 4 | Medium |
| 4 to 5 | High |
| Above 5 | Critical |

**7b. Escalate one level when any precursor fires**

- Crowd pressure above its threshold.
- Flow collapse: mean speed falling while density keeps rising, for 30+ seconds.
- Counter-flow ratio or direction entropy above its threshold.

Set each threshold from the simulated data, for example at the value reached in the minutes before safe runs turned critical. Don't copy numbers from papers; your pixel-to-metre scale and scenes differ.

**7c. Sudden-disturbance detector (simple version)**

- Train an **Isolation Forest** (scikit-learn) on the six features from normal footage.
- Flag a zone when the anomaly score spikes together with a jump in mean speed and direction entropy, such as people suddenly running or scattering.
- Stretch goal only: a CNN-LSTM on flow maps trained on UMN and UCSD anomaly clips.

**7d. Alert logic**

- A level must hold for N seconds (start with 10) before it is raised, and stay below for M seconds (start with 30) before it is cleared. This stops alerts flickering.
- Every alert carries: zone, level, time-to-critical, and the reason, for example "inflow > outflow, pressure rising".
- Keep venue-specific threshold profiles in the camera config.

**Done when:** you can replay any clip or simulation run and see a timeline of alerts with lead times.

## Step 8 — Backend and dashboard (Week 6, days 3–7)

The live system runs as separate processes joined by queues, and a FastAPI server pushes zone state to your existing Next.js dashboard.

**Runtime layout (per camera)**

| Process | Rate | Work |
| --- | --- | --- |
| Decoder | Camera FPS | Read the stream, drop frames to the rates below |
| Counting | 2–5 FPS | YOLO11l + ByteTrack, CLIP-EBC, fusion |
| Motion | 3–5 FPS | RAFT flow, zone motion features |
| Features + forecast + risk | 1 Hz | Merge features, run the GNN, apply rules, emit alerts |

Crowd state changes over seconds, so there is no need for 30 FPS inference. One A100 should handle several cameras at these rates; measure it in Step 9.

**Backend:** FastAPI with one WebSocket endpoint that pushes a JSON message per second: zone densities, forecasts with bounds, levels and active alerts.

**Dashboard views**

- Live zone heatmap coloured by persons/m², drawn over the camera frame.
- Per-zone forecast chart: history, median forecast, the 90% band and a line at 5 persons/m².
- Alert feed: level, zone, time-to-critical, reason.
- Replay mode: play a recorded clip or a simulation run through the whole system. Use this for your demo, since a live crowd won't appear on demand.

**Done when:** a replayed clip shows the heatmap, forecasts and alerts updating live.

## Step 9 — Evaluation and paper results (Week 7)

Alert lead time is the headline number: how many minutes before a zone crossed 5 persons/m² the system raised High or Critical.

| Question | Metric | Compared against |
| --- | --- | --- |
| Is counting better? | MAE and RMSE per density band | YOLO only, CLIP-EBC only, hard switch, fusion |
| Is forecasting better? | MAE per horizon; actual coverage of the 90% bound | Persistence, physics, GRU, GNN |
| Are the alerts useful? | Mean lead time; precision and recall of alerts; false alarms per hour | Density-only rule vs density + precursors |
| Does it run in real time? | End-to-end latency; FPS per camera; GPU memory | Old baseline from Step 0 |
| Does it generalise? | All the above on the held-out camera and held-out simulation geometry | Training cameras and geometries |

Also include a few qualitative figures: a heatmap frame, one forecast chart with its band, and an alert timeline from a run that turns critical.

**Done when:** every table is filled and the replay demo runs end to end.

## Timeline, risks and fallbacks

Seven weeks leaves no slack for the stretch goals, so the order matters: counting and fusion first, because everything downstream depends on them. A drawn week-by-week timeline follows this section.

&#91;embedded content: 7-week plan · Steps 0–9\]

Week 3 runs fusion and motion in parallel; weeks 4–7 each depend on the step before.

**Risks and fallbacks**

| Risk | Early sign | Fallback |
| --- | --- | --- |
| CCTV too poor for YOLO | Heads under \~10 px, low detection confidence | Tiled inference; fusion leans on CLIP-EBC; report it as a finding |
| Pretrained density model over-counts your cameras | Held-out error much worse than on benchmarks | Fine-tune CLIP-EBC with \~100 more corrected pre-labelled frames; recalibrate fusion |
| Chronos-2 zero-shot is weak on crowd data | Barely beats persistence | Fine-tune it through AutoGluon; keep the physics model as the alert source |
| Lab and simulated data differ from real crowds | Good on Jülich/sim, poor on real clips | Report results per source; tune thresholds on real clips |
| Time slips | Any step more than 3 days late | Drop the anomaly detector, the GRU comparison and the 5–10 min horizons first |

**Checklist**

- [ ] Step 0: repo refactored, baseline table saved
- [ ] Step 1: CCTV audit table, annotated frames split by camera
- [ ] Step 2: camera YAML configs, validation distance checked
- [ ] Step 3: fusion beats both single models; ablation table
- [ ] Step 4: six motion features per zone per second
- [ ] Step 5: Jülich + CCTV + simulated feature dataset
- [ ] Step 6: Chronos-2 vs baselines table with bound coverage
- [ ] Step 7: alert replays with lead times
- [ ] Step 8: live dashboard in replay mode
- [ ] Step 9: all evaluation tables and figures

## Sources

- [CLIP-EBC official repository (weights on the releases page)](https://github.com/Yiming-M/CLIP-EBC)
- [Chronos-2 model card, Hugging Face](https://huggingface.co/amazon/chronos-2)
- [Introducing Chronos-2, Amazon Science](https://www.amazon.science/blog/introducing-chronos-2-from-univariate-to-universal-forecasting)
- [Pedestrian Dynamics Data Archive, re3data](https://www.re3data.org/repository/r3d100013370)
- [Supervision: count in zone (PolygonZone, LineZone)](https://supervision.roboflow.com/latest/how_to/count_in_zone/)
- [Supervision speed-estimation example (ViewTransformer)](https://github.com/roboflow/supervision/tree/develop/examples/speed_estimation)
- [YOLO11, Ultralytics on Hugging Face](https://huggingface.co/Ultralytics/YOLO11)
- [YOLO11 person + head model trained on CrowdHuman](https://huggingface.co/Sharath33/Person/blob/main/README.md)
- [YOLOv8 head detector trained on CrowdHuman](https://github.com/Owen718/Head-Detection-Yolov8)

# CrowdSafe — project and research report

*2026-10-28. Authors: Nidhi Singh, Shubhendra Singh, Shruti Mishra (KIET Deemed to be University).*

This report explains what the system uses, how each part works, why it was chosen over the alternatives, what was measured and what we learned. **Every number comes from a file in `results/`** (named in each section). Nothing is estimated by hand. Values marked *approximate* or *extrapolated* are labelled as such.

Related documents:
- `docs/paper/main.tex`: the IEEE paper;
- `docs/FINAL_RESULTS.md`: a one-page summary;
- `docs/decisions.md`: every design decision, with date and reason;
- `task.md`: task status.

---

## 1. Summary

**Goal.** Warn staff *before* a zone of a room or venue becomes dangerously crowded (≥ 4 persons/m²), using the CCTV cameras that are already installed.

**What we built.**
1. A pipeline that turns each CCTV frame into **people per m² for every zone** of the floor.
2. A forecast of that density **10–120 s ahead**, with a calibrated worst case.
3. Tiered **alerts** (Low / Medium / High / Critical).
4. A **replay dashboard**.
5. A **live version**, measured on an A100 GPU.

**Main principle.** Reuse pretrained models zero-shot; **nothing is trained from scratch or fine-tuned**.

**Headline results:**

| Question | Answer | Source |
|---|---|---|
| How well do we count people on our CCTV? | CLIP-EBC: 2.0–5.1 people per frame error; detectors 6.4–28.2 | `results/counting_zeroshot_*.md` |
| How accurate is density per zone? | 0.08–0.10 persons/m² error | `results/zone_density.md` |
| Is the calibration right? | Independent checks within 3–10 % | `results/calibration.md` |
| Can we forecast density? | Chronos-2: 0.064–0.097 persons/m² error on CCTV; worst-case bound holds 84–96 % (target 90 %) | `results/forecast_*.md` |
| Do alerts come early enough? | Current density alone: never in time. With forecasts: about 1 in 3 simulated events, seconds ahead | `results/alerts_*.md` |
| Does it run live? | Yes, on one A100: 0.49 s median latency, 2 GB GPU memory per camera | `results/latency_*.md` |

**Most important finding.** The system measures density well, and its forecast gives a trustworthy worst case. But in our data zones go from safe to dangerous in 10–20 s, so a single camera can warn only **seconds** ahead, not minutes. Longer warnings need information from outside the camera's view, such as entrance cameras, gate counters or schedules.

---

## 2. The problem

- **Density, not head count.** Crowd disasters happen when *local* density becomes too high. Typical bands:
  - below 2 persons/m²: safe;
  - 2–4: crowded;
  - 4–5: high risk;
  - above 5: critical.

  Most CCTV analytics report "people in the frame", which says nothing about density on the floor.
- **No calibration.** Turning pixels into m² needs a camera calibration. Existing installations never have one.
- **No training data.** Dangerous moments are rare, so a forecaster cannot be trained on the venue's own data. It has to work zero-shot.

---

## 3. System overview

Figure: `results/figures/paper_pipeline.png`.

```
CCTV frame (1 per second)
  ├─ CLIP-EBC density map ─────┐
  ├─ RAFT optical flow ────────┤
  └─ camera model (calibration)┴─> zone features: density (p/m²), speed (m/s), pressure, entropy, counter-flow
                                       └─> Chronos-2 forecast 10/30/60 s + conformal worst case (every 5 s)
                                              └─> risk level (2/4/5 p/m² bands) ─> alert manager (10 s raise, 30 s clear)
                                                     └─> FastAPI + WebSocket ─> Next.js dashboard
```

Code: `crowdsafe/` package (calibration, counting, motion, features, forecasting, risk, server), scripts in `scripts/`, A100 notebooks in `notebooks/`.

---

## 4. Data

| Source | What it is | Size | Used for |
|---|---|---|---|
| **Own CCTV** | 4 fixed cameras in 3 rooms: classroom H705 (2 views), classroom H706, computer lab H103 | 2.6 h of video; 160 frames with **6,715 hand-labelled heads** (CVAT) | counting, density, real forecasting, live latency |
| **Jülich archive** (Forschungszentrum Jülich) | measured trajectories of controlled crowd experiments: corridor, bottleneck, entrance | test = uni-directional corridor 2013 + entrance semicircle (10 runs) | forecasting and alerts on **real dangerous crowds** (up to about 9 persons/m²) |
| **JuPedSim simulations** | 16 scenarios in 4 layouts: classroom, corridor + bottleneck, hall with 2 exits, two-way corridor | 5 held-out test scenarios; 10 of 16 exceed 5 persons/m², up to 8.0 | rare dangerous events |
| **ShanghaiTech A/B** | public crowd-counting benchmark | 182 + 316 test images | sanity check of the counting models |

**Why three sources:**
- Our classrooms are real but never get dangerous; the highest zone density measured was 3.2 persons/m².
- Jülich is real and dangerous, but recorded as trajectories rather than CCTV.
- Simulation provides many dangerous events, but they are synthetic.

All three are converted into **the same per-zone feature table**, so forecasting and alerting are evaluated the same way on each. The Jülich conversion matches PedPy, the standard pedestrian-analysis tool, to 0.0000 persons/m² (`results/julich_features_summary.md`).

**Splits.** Data is always split by **whole camera, experiment or scenario**, never by time inside one recording. Thresholds and calibration offsets are fitted only on validation data.

---

## 5. Components: what, how, why, and what else was tried

### 5.1 Counting people

**Used: CLIP-EBC (ViT-B/16, NWPU-Crowd weights)**, from the official Hugging Face repository `Yiming-M/CLIP-EBC`. It runs on 224-px sliding windows and outputs a density map: each pixel holds a fraction of a person, and the map sums to the head count.

**How it works.** CLIP-EBC is built on CLIP, a vision–language model. It treats counting as classification ("how many people are in this block: 0, 1, 2, …?"), which is more robust than direct regression. It is used **zero-shot**, with no training on our rooms.

**Compared with** (all zero-shot):

| Model | H705 (100 frames) | H706 (30) | H103 lab (30) | ShanghaiTech A / B |
|---|---|---|---|---|
| **CLIP-EBC** | **5.10** | **2.58** | **1.97** | **66.95 / 11.66** |
| YOLO11l person detector (COCO) | 6.36 | 6.53 | 17.63 | 418.3 / 92.8 |
| YOLO11l with tiling (640-px tiles) | — | — | — | 421.7 / 95.4 |
| CrowdHuman head detector (`Sharath33/Person`) | 8.04 | 6.23 | 28.23 | 374.2 / 70.2 |
| CrowdHuman person boxes, same model | 14.10 | — | — | — |

The table shows mean absolute error in people per frame (`results/counting_zeroshot_cctv.md`, `counting_zeroshot_newcams.md`, `counting_zeroshot_shanghaitech.md`). H706 and H103 are reported separately in the paper; the combined H706 + H103 error is 2.27 for CLIP-EBC and 12.08 for YOLO11l.

**Why not the detectors:**
- Detectors only count people they can see as a box. In packed rows and behind lab monitors, most bodies are hidden. In the lab, YOLO11l misses about 18 people per frame.
- They always undercount: the bias is −6 to −17 people.
- Tiling (cutting the image into smaller pieces, `InferenceSlicer`) did not help on ShanghaiTech (421.7 vs 418.3), so it was not used.
- Detectors **are** slightly better in nearly empty scenes. With fewer than 20 people in H705, the CrowdHuman heads model errs by 0.86 against 1.15 for CLIP-EBC. This is why the final design switches to the detector when the room is almost empty (the "hard switch").

**Detector + density fusion (tested, rejected; decision D7).** We tested four ways to combine YOLO and CLIP-EBC: a learned per-frame weight, a constant weight, a 50/50 average and a hard switch.
- **Within one room** (H705, first half used for fitting, second half for testing), fusion looked excellent: constant weight 1.59 and learned 2.09, against 5.30 for CLIP-EBC alone (`results/fusion_ablation.md`). This happened only because the two models made opposite errors there, +5.1 and −6.0 people.
- **Leave-one-room-out** (fit on two rooms, test on the third; `results/fusion_loco.md`):

  | Method | Mean error on the unseen room |
  |---|---|
  | Hard switch | **3.16** |
  | CLIP-EBC alone | 3.22 |
  | 50/50 average | 4.48 |
  | Constant weight | 4.56 |
  | Learned fusion | 5.22 |
  | YOLO11l alone | 10.17 |

- **Finding:** learned fusion **does not transfer** to new rooms. It is reported as a negative result, and the system uses CLIP-EBC with the hard switch.

**Not used:**
- **The old YOLO11m + CSRNet project:** replaced by this design from scratch (D2).
- **PET point counter (optional, T3.4):** not run, because CLIP-EBC was already accurate enough and time went to forecasting and alerts.
- **The ShanghaiTech-A checkpoint of CLIP-EBC:** only available on GitHub. The project rule prefers Hugging Face weights (D4), and the NWPU weights worked well.
- **Fine-tuning:** not needed. The measured error (2–5 people per frame, 0.1 persons/m²) is small compared with the 2 persons/m² width of the risk bands.

### 5.2 Camera calibration (pixels → metres)

**Used:** a **full camera model** (decision D6):
- pinhole camera with radial lens distortion (k1, k2), focal length, position and tilt;
- fitted by robust least squares (`crowdsafe/calibration/camera_model.py`).

**How it is fitted:**
- **Classrooms:**
  - Every desk top is a rectangle **1.00 m long** (measured), and all desk tops are parallel and on one plane.
  - Each desk's floor position is left free, so row and column spacing are *not* assumed and can be used as independent checks.
  - Straight edges (door frames, ceiling duct, stage edge) must look straight after removing lens distortion ("plumb lines"). These pin down the distortion.
- **Computer lab:** long table edges at known positions plus vertical edges, with a soft prior from the measured camera height.
- **Heads:** traced through the camera ray down to **seated head height (1.2 m)**. Heights from 1.1 to 1.6 m were tested as a check.

**Why not a plain floor homography** (the usual approach, and the original plan):
- The CCTV lenses are wide-angle with strong barrel distortion: straight edges bend by tens of pixels.
- The floor is hidden under desks, so floor points cannot be clicked.
- Heads are about 1 m above the floor. A floor homography would misplace people at the back of the room by metres.

**Results** (`results/calibration.md`, figures `results/figures/calibration_*.jpg`):

| Room | Reprojection error | Independent checks (not used in the fit) |
|---|---|---|
| H705 | 4.2 px | row spacing 0.99 m vs 0.96 m measured (+3 %); mean desk + gap 1.69 vs 1.77 m (−5 %; single gaps −1 to −10 %) |
| H706 | 3.6 px | desk + gap 1.78 vs 1.77 m; door width 1.23 vs 1.12 m (+10 %) |
| H103 lab | 4.0 px | table edges of the 2 columns left out of the fit found within 0.03–0.18 m |

All calibrations are labelled **approximate (about ±5 %)**.

### 5.3 Zones and density per m²

- **Zones:** 3×3 seating blocks plus the front aisle in each room, 4–16 m² each, drawn as floor polygons in metres (`configs/zones/`).
- **Density:** the CLIP-EBC count inside the zone's image area divided by the visible floor area of the zone.

**Results** (`results/zone_density.md`, figures `results/figures/zones_*.jpg`, `paper_heatmap_cam0*.jpg`):

| Room | Error (persons/m²) | Highest true zone density in labelled frames |
|---|---|---|
| H705 | 0.101 | 1.67 |
| H706 | 0.081 | 1.40 |
| H103 | 0.078 | 1.29 |

- 94–98 % of labelled heads fall inside a zone.
- Changing the head height from 1.1 to 1.6 m changes the error by at most 0.03 persons/m².
- Over the full videos the busiest zone reached 3.2 persons/m² (H705 lecture). **No classroom zone ever reached the danger level of 4.**

### 5.4 Motion

**Used: RAFT-large optical flow** (`torchvision`, pretrained) at 960 px, between the current frame and the frame 0.2 s later.
- The flow is converted to **m/s on the floor** through the camera model.
- It is weighted by the density map, so empty floor counts for nothing.
- From it we compute mean speed, speed variance, crowd pressure (density × speed variance, Helbing), direction entropy and counter-flow share.

**Why RAFT:**
- It is the most accurate widely available pretrained optical-flow model.
- It needs no tracking, so it works in dense crowds where people cannot be tracked individually.
- It stores no identities, which suits the privacy rule.

**Alternatives:**
- **Farneback** (classical OpenCV flow) is kept as a CPU fallback. It was used only in the laptop live test and was not compared for accuracy.
- **ByteTrack** person tracking is implemented in the detector wrapper. It is not used in the reported results, because detectors fail in the crowded rows.

**Open item:** CCTV speeds have not yet been validated against ground truth (T4.3). On Jülich and in simulation, motion features come from measured or simulated trajectories.

### 5.5 Forecasting

**Used: Chronos-2** (`amazon/chronos-2`), a pretrained time-series foundation model, **zero-shot**. Each forecaster receives the last 30 s of zone density at 1 Hz and forecasts 10/30/60 s ahead (120 s in simulation), every 5 s.

*Zero-shot forecasting (concept):* Chronos-2 was pretrained on a very large collection of time series. It forecasts a new series directly from its recent history, without being trained on it. It outputs quantiles (10 %, 50 %, 90 %), not just one value. The "joint" variant forecasts all zones of a camera together, so neighbouring zones can inform each other.

**Compared with:**
- **Persistence:** "density stays as it is now". This is the standard baseline and is surprisingly hard to beat over short horizons.
- **Physics fill-rate:** density + (people in − people out) / area × time.
- **Chronos-2 per zone vs joint.**

**Results** (MAE in persons/m², then the share of true values below the calibrated 90 % bound; `results/forecast_cctv.md`, `forecast_julich_a100.md`, `forecast_sim_a100.md`):

| Data | Model | MAE 10 / 30 / 60 (/120) s | Bound held |
|---|---|---|---|
| Own CCTV, test cameras | Chronos-2 joint | **0.064 / 0.080 / 0.097** | 0.90 / 0.91 / 0.89 |
| | Chronos-2 per zone | 0.067 / 0.081 / 0.099 | 0.905 / 0.904 / 0.895 |
| | Persistence | 0.074 / 0.087 / 0.105 | 0.92 / 0.93 / 0.91 |
| Jülich test (bound measured at critical moments) | Chronos-2 per zone | 0.38 / 0.56 / 0.73 | **0.89 / 0.88 / 0.93** |
| | Persistence | 0.39 / 0.57 / 0.71 | 0.85 / 0.66 / 0.68 |
| Simulation test | Chronos-2 joint | 0.16 / 0.36 / 0.57 / 0.76 | 0.94 / 0.92 / 0.91 / 0.96 |
| | Persistence | 0.17 / 0.34 / 0.56 / 0.75 | 0.89 / 0.88 / 0.87 / 0.93 |
| | Physics | 0.26 / 0.66 / 1.25 / 2.32 | 0.90 / 0.87 / 0.86 / 0.91 |

**Findings:**
- **On average error,** Chronos-2 is about equal to persistence; it is slightly better on our CCTV.
- **On the worst case,** which is what matters for safety, Chronos-2 is clearly better. At dangerous moments in real Jülich crowds its bound holds 88–93 % of the time, while persistence drops to 66–68 %.
- **Physics** needs measured inflow and outflow. Its error grows fast with the horizon, so it is not used on CCTV.
- **Example** (`results/figures/paper_forecast_sim_corridor.png`): 30 s before a corridor zone reaches 4 persons/m², the Chronos-2 bound rises from 2.33 to 3.09 while persistence stays flat (1.65–1.80). The model sees the rise coming but not how high it goes.

**Not used:**
- **GRU** (a small trained neural forecaster, optional T6.6): not done. It would need training data from dangerous events, which we lack, and Chronos-2 already matches the baselines.
- **Fine-tuning Chronos-2** (optional T6.5): not needed, because Chronos-2 was not weak.

### 5.6 Uncertainty: split conformal calibration

**Used:** split conformal prediction on the 90 % upper bound.

*Concept:* On validation data only, we measure how often the true value exceeds the model's 90 % bound and by how much. We then shift the bound by just enough that 90 % of the validation values fall below it. Under mild assumptions this gives a guaranteed coverage level on new data, whatever the model.

**Why:** the raw quantiles of a model are not guaranteed to be honest. Persistence has no spread at all: its raw bound holds only 48–50 % of the time on CCTV. After calibration it holds 91–93 %. For Chronos-2 the needed shift was tiny (0.001–0.02 persons/m² on CCTV), meaning its raw quantiles were already close to honest.

### 5.7 Risk levels and alerts

**Risk level** comes from the **calibrated worst case** over all horizons:

| Level | Worst-case density (persons/m²) |
|---|---|
| Low | below 2 |
| Medium | 2–4 |
| High | 4–5 |
| Critical | 5 or more |

**Alert rule** (hysteresis): an alert is raised when a zone stays High or above for **10 s**, and cleared after **30 s** below. This stops alerts flickering on and off with every small spike.

**Scoring** (strict, our own design):
- An *event* is a zone first reaching 4 persons/m².
- An alert is "in time" only if it comes within (longest horizon + 10 s) before the event and not in the first 15 s of monitoring.
- False alarms are alerts with no event within 120 s.

**Results** (share of events warned in time; `results/alerts_sim.md`, `alerts_julich.md`, `alerts_cctv.md`):

| Test data | Current density only | Persistence | Physics | Chronos-2 per zone | Chronos-2 joint |
|---|---|---|---|---|---|
| Simulation (47 events) | 0 % | 0 % | 32 % (median 25 s ahead; 2.1 false alarms per zone-hour) | **34 %** (7.5 s; 1.6) | 30 % (7.5 s; 1.5) |
| Jülich, real (9 events) | 0 % | 11 % | 11 % (11.9 false alarms per zone-hour) | 11 % (1.8) | 11 % (0.45) |
| Own CCTV (8.6 zone-hours) | no events, **0 false alarms** | | | | |

**Crowd-motion warning signs ("precursors"), tested and rejected:**
- Each sign is kept only if it predicts "reaches 4 persons/m² within 60 s" at least **twice as well as chance**.
- On Jülich validation alone:
  - slowing down (≤ 0.21 m/s) reached 3.2× the base rate;
  - direction entropy reached 2×;
  - crowd pressure, counter-flow and flow collapse failed.
- Pooled with simulation, **no sign passed**, so none is used.
- **Why:** crowd pressure and turbulence are known from real disasters at more than 6–7 persons/m². At the densities in our data they do not appear before a zone fills.

**Alert-timing ablation** (`results/alert_experiment.md`; decision D8): variants with a shorter wait (5 s), a rate-of-rise warning, a shorter history (15 s), or all three. Each one helped one method and hurt another. For example, on the 29 simulated events of the ablation, all three together raised persistence from 0 to 31 % (0.17 false alarms per zone-hour) but lowered physics from 59 to 24 %. So the original setting is kept.

**Not used: Isolation Forest anomaly detection** (optional T7.3). It was not done because the precursor tests already showed that motion signals do not anticipate filling at these densities.

### 5.8 Live pipeline, latency and hardware

- **Live pipeline** (`scripts/live_pipeline.py`):
  - A reader thread keeps only the newest frames; when processing is slow it drops frames instead of falling behind.
  - A 1-Hz loop runs every step and logs each step's time.
- **Measured on the A100**, cam02 played at real speed (`results/latency_*.md`):

| Run | Median / 95 % latency | CLIP-EBC | RAFT | Forecast | Cycles on time | GPU memory |
|---|---|---|---|---|---|---|
| Live, Chronos-2 (10 min) | **0.49 s** / 1.6 s | 159 ms | 103 ms | 54 ms every 5 s | 94 % | 2.0 GB |
| Live, persistence (5 min) | 0.39 s / 1.5 s | 103 ms | 61 ms | 6 ms | 94 % | 1.5 GB |
| As fast as possible | 0.36 s / 0.45 s | 105 ms | 58 ms | 40 ms | 99.8 % | 2.0 GB |

- **Throughput:** 4.6× real time (908 s of video processed in 196 s).
- **Capacity:** about 4 cameras per A100. This is *extrapolated* and has not been tested.
- **Laptop CPU:** about 19 s per second of video, so a **GPU is required** for live use.
- **Hardware guidance** (estimates): an NVIDIA GPU with about 2 GB of video memory per camera.

### 5.9 Dashboard

- **Stack:** a FastAPI + WebSocket backend and a Next.js frontend.
- **What it shows:**
  - the camera video with zones coloured by risk level and labelled with the current density;
  - a time bar of peak density;
  - the forecast band for the selected zone;
  - the alert feed.
- **Runs available:** 3 real CCTV recordings, 2 Jülich experiments (shown as 2 m grids) and 2 simulations, each clearly labelled with its data source.
- **Replay only:** it replays processed results and is **not a live camera feed**. A live view on a laptop would update only every ~20 s.

Screenshot: `results/figures/paper_dashboard.png`.

---

## 6. Key findings (for the paper and the viva)

1. **Zero-shot density counting works on real classroom CCTV;** detectors do not. CLIP-EBC errs by 2–5 people per frame, while detectors miss up to 18 people per frame behind furniture.
2. **Learned detector–density fusion is a trap.** It looks best within one room but is worst on new rooms. A held-out-room test is essential.
3. **Calibration from furniture works.** A camera model fitted to desks gives density within 0.1 persons/m², checked against measurements it did not use.
4. **A calibrated zero-shot forecaster gives an honest worst case.** At dangerous moments in real crowds, Chronos-2's bound holds about 90 % of the time; persistence's holds about 66 %.
5. **Warnings come seconds ahead, not minutes.** Zones fill in 10–20 s. Forecasts warn for about 1 in 3 simulated events (median 7.5 s; physics 25 s). Alerts on current density never warn in time.
6. **Crowd-motion "precursors" did not help** at the densities available (up to 8–9 persons/m²).
7. **How you score alerts matters.** A loose metric gave 56 % (Jülich) and 86 % (simulation) "early warnings". The strict metric gives 11 % and about 33 %. The loose numbers were withdrawn.
8. **It runs live on one GPU:** 0.49 s median latency and about 2 GB per camera.

---

## 7. Negative results and corrections (kept on record)

| Earlier claim | What happened | Status |
|---|---|---|
| Learned fusion is the main contribution (1.59–2.09 error) | Fails on unseen rooms (5.22) | Reported as a negative result (D7) |
| 56 % / 86 % of events warned early | Alerts raised at the start of monitoring were counted as warnings | Replaced by the strict metric |
| Precursors improve alerts | None passed the 2× chance test when pooled | Not used |
| Shorter alert hold or history improves alerts | Helps one method, hurts another | Setting A kept (D8) |
| H706 door check 1.19 m (+6 %) | Recomputed after the door-height correction: 1.23 m (+10 %) | Corrected in all documents |

---

## 8. Limitations

- **Calibration is approximate** (about ±5 %). Head height is assumed (seated, 1.2 m).
- **Our CCTV never shows dangerous density.** Danger-level results rely on **9 real Jülich events** (so percentages are rough) and on **simulation**, which follows model rules rather than real behaviour.
- **CCTV motion speeds are not validated** against ground truth (T4.3 is open).
- **Inflow and outflow are not measured from CCTV,** so the physics forecaster is tested only on trajectories and simulation.
- **Latency was measured with one camera on a shared GPU;** the multi-camera capacity is extrapolated.
- **Single-camera view:** a surge that starts outside the frame cannot be foreseen.

---

## 9. Not done / open items

| Item | Status | Reason |
|---|---|---|
| T4.3: validate CCTV speeds | open | No ground-truth trajectories for our rooms; would need tracked labels |
| T3.4: PET point counter | optional, not run | CLIP-EBC was accurate enough |
| T6.5 / T6.6: Chronos-2 fine-tuning, GRU | optional, not run | Chronos-2 was not weak; no dangerous training data |
| T7.3: Isolation Forest | optional, not run | Motion signals showed no warning value |
| T8.1: multi-process runtime | open | The single-process live loop already meets the 1-s budget 94 % of the time |
| Live mode in the dashboard | not built | Would need the A100 reachable from the laptop; the CPU is too slow |
| T2.2 / T2.4 / T7.5: venue layout, real-venue zones, threshold review | open (USER) | Need a real venue beyond classrooms |
| Paper: page limit, 3 reference details | open | Waiting for the conference choice and the Jülich/JuPedSim citations |

---

## 10. How to reproduce

```bash
pip install -r requirements.txt
pytest -q                                        # unit and smoke tests
python scripts/calibrate_cameras.py              # camera models -> configs/cameras/, results/calibration.md
python scripts/zone_density.py --set LABELS IMAGES FRAMES_CSV   # per label set; zone density vs labels -> results/zone_density.md
python scripts/eval_forecast.py --config configs/forecast_cctv.yaml
python scripts/eval_alerts.py                    # strict alert scoring -> results/alerts_*.md
python scripts/paper_figures.py                  # paper figures -> results/figures/paper_*
python -m uvicorn crowdsafe.server.app:app --port 8000   # dashboard backend
npm --prefix dashboard run dev                   # dashboard at http://localhost:3000
```

- **GPU steps** (counting, RAFT flow, Chronos-2 forecasts, latency) run as notebooks on the A100: `notebooks/`, built from `notebooks/src/`.
- **Large files** (videos, features, weights) are not in git. The dashboard needs about 256 MB of them, at the same paths:
  - `data/features/`;
  - `data/cctv/audit/`;
  - `data/cctv/web/`;
  - `weights/crowdsafe_nb05_report/forecast_{cctv,sim}_points.parquet`;
  - `weights/crowdsafe_nb04_report/forecast_julich_points.parquet`.

---

## 11. Decision log (short)

| ID | Decision | Why |
|---|---|---|
| D1 | Python code in a `crowdsafe/` package | Avoid clashing with Hugging Face's `datasets` library |
| D2 | New project from scratch (old YOLO11m + CSRNet not reused) | New design |
| D3 | GPU work as A100 notebooks; models returned as ONNX | The A100 is reachable only through Kubeflow; the laptop has no GPU |
| D4 | Prefer Hugging Face weights (CLIP-EBC NWPU, CrowdHuman `Sharath33/Person`) | Reproducible downloads |
| D5 | Mirrors for blocked downloads (RAFT weights checked by SHA-256) | College network blocks some hosts |
| D6 | Full camera model instead of a floor homography | Lens distortion, hidden floor, heads above the floor |
| D7 | CLIP-EBC + hard switch; learned fusion reported as negative | Fails on unseen rooms |
| D8 | Keep alert setting A | No variant better everywhere |

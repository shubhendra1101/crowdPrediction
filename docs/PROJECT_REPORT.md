# CrowdSafe — project and research report

*Version for discussion, 2026-10-28. Authors: Nidhi Singh, Shubhendra Singh, Shruti Mishra (KIET Deemed to be University).*

This report explains what the system uses, how each part works, why it was chosen over the alternatives, what was measured and what we learned. **Every number for the new system comes from a file in `results/`** (named in each section). Numbers about the old system come from its repository and are tagged with their source (§2A). Nothing is estimated by hand. Values marked *approximate* or *extrapolated* are labelled as such.

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
| Why not the old YOLO11m + CSRNet system? | It gave people per frame only (no m², zones, motion or forecast); by its own README, counts swing 1.8× with resolution and carry about ±50 %. Alarms on the current state warned 0 % of events in time; forecast alarms 34 % (sim) | §2A |
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

## 2A. Why we switched from the old YOLO11m + CSRNet system

**Source of the old system:** the repository https://github.com/Sidharth260905/CrowdSafe (last commit 93b60d4, 27 Aug 2026). It was read and its code inspected, not re-run (see "Limits of this comparison" below).

**How the old numbers are labelled in this section:**
- **[old README]:** measured by the old project and written in its README, on its own test video, a webcam and ShanghaiTech. Not measured on our CCTV.
- **[old code]:** follows directly from the old code and settings (`backend/config.py`, `backend/fusion/engine.py`, `backend/alerts/alert.py`).
- **[ours]:** measured in this project (`results/`).
- **[published]:** taken from a paper.

### What the old system did

```
video → YOLO11m (imgsz 1920, conf 0.25) + ByteTrack ─┐
      → CSRNet (official ShanghaiTech-A weights) ────┴→ gate: YOLO only, or max(CSRNet, YOLO)
                                                      → 10-frame average → people per frame
                                                      → level: normal 0–50 / medium 51–100 / high 101+ people
                                                      → MJPEG video + WebSocket → Next.js dashboard
```

- **CSRNet** runs on the first frame and on every 5th frame.
- **The gate** opens if any of these holds:
  - YOLO counts more than 50 people;
  - boxes overlap a lot (with at least 12 boxes);
  - CSRNet's own last count is at least 66.
- **The count is trusted from CSRNet only when CSRNet reads 66 or more** ("trust floor"). Otherwise YOLO carries the count alone. [old code]
- **Engineering strengths:** the old system was a solid live system. It had a supervised background thread, a stale-reading watchdog, camera reconnect and MJPEG streaming. These are good engineering, and the new replay dashboard does not have a live camera mode.

### What the old system could not do, and why that blocked the project goal

The research question is: *"Will this zone become dangerous in the next minute?"* That needs density in **persons/m² per zone**, **movement** and a **forecast with a trustworthy worst case**.

| Need | Old YOLO11m + CSRNet | New CrowdSafe | Evidence |
|---|---|---|---|
| Unit of output | people per frame (whole image) [old code] | **persons/m² per floor zone** | zone error 0.08–0.10 persons/m² [ours] (`results/zone_density.md`) |
| Danger levels | fixed head counts: 50 / 100 people [old code]; the same count means very different danger in a small room or a big hall | 2 / 4 / 5 persons/m², the standard crowd-safety bands | `configs/risk.yaml` |
| Pixels → metres | none | full camera model with lens distortion | independent checks within 3–10 % [ours] (`results/calibration.md`) |
| Where in the room | no | 3×3 seat blocks + aisle | `results/figures/paper_heatmap_*.jpg` |
| Movement | ByteTrack tracks, used only for drawing and counting | RAFT flow → m/s, pressure, entropy, counter-flow | tested as warning signs [ours] (`results/alerts_julich.md`) |
| Prediction | none: reports the present only [old code] | Chronos-2, 10–120 s ahead | `results/forecast_*.md` |
| Worst case / uncertainty | none; the README itself says counts carry **about ±50 %** on footage without labels [old README] | conformal 90 % bound: holds 84–93 % of Jülich test forecasts; 76–88 % where the true density is ≥ 4 persons/m² (persistence 56–78 %) [ours] | `results/forecast_julich_a100.md`, `results/coverage_by_density_julich.md` |
| Tested on dangerous crowds | no | Jülich real experiments + 16 simulations | 9 + 47 events [ours] |

### Old vs new: numbers

| Measure | Old system | New system |
|---|---|---|
| Density model on ShanghaiTech A, MAE | CSRNet **66.09** at the deployed input size, 66.46 at native size [old README]; 68.2 [published, CSRNet paper] | CLIP-EBC **66.95** [ours], zero-shot: never trained on ShanghaiTech |
| Near-empty scene | **one person read as 11–12 people** by CSRNet [old README]; handled with the 66-person trust floor | below 20 people the head detector errs by **0.86 people** on H705 [ours]; a hard switch to the detector was tested at frame level (best on unseen rooms, 3.16 vs 3.22) but the zone pipeline uses the CLIP-EBC density map only |
| Per-image reliability | CSRNet predictions range **0.59× to 1.73×** the true count on ShanghaiTech A [old README] | per-frame MAE 2.0–5.1 people on our labelled CCTV [ours] |
| Dependence on camera resolution | the same frame read **108 → 197 people** as the input grew from 640×360 to 2560×1440: a **1.8× swing with no plateau** [old README]; fixed by resizing to 1024×768, which makes counts *repeatable but not calibrated* | the camera model converts the count to m² per zone using the real geometry, checked against tape-measured distances [ours]; CLIP-EBC's resolution sensitivity was *not* tested |
| Dense-scene detector | YOLO11m reads **about 27 where CSRNet reads about 185** [old README] | YOLO11l (larger, same family) misses 6–18 people per frame in our rooms [ours]; detectors are used only near empty |
| How fresh the count is | **1–3 s behind live**, because CSRNet runs every 5th frame and a 10-frame average is applied [old README] | 0.49 s median from frame to risk level [ours] |
| Speed / hardware | 3–6 FPS on an RTX 3050 4 GB; about 1.2 GB VRAM peak [old README] | 1 analysis per second (all steps) on an A100, 94 % on time, 2.0 GB VRAM, 4.6× real-time throughput [ours]. *Different hardware and workload, so not directly comparable* |
| Early warning | none; the alert fires when the current count passes 100 [old code] | an alarm on *current* density (our best version of that idea, already calibrated per zone) warned **0 %** of events in time; forecast alarms warned **34 %** (simulation, median 7.5 s ahead) and **11 %** (real Jülich) [ours] |

### What the old system would do on our classrooms [old code]

- Our labelled frames contain **at most 57 people (H705) and 62 (H706 + H103)** [ours: `results/counting_zeroshot_*_per_frame.csv`]. That is below the old 66-person trust floor.
- So the old system's count on our rooms is one of two things:
  - **the YOLO11m count**, which on our footage behaves like the YOLO11l we measured: it undercounts by 6–18 people per frame;
  - **or a CSRNet count of 66 or more**, which is too high for *every* labelled frame.
- With the old levels (high = 101+ people), **none of our rooms could ever raise an alert**, however packed a corner became. The old system measured the room as a whole, never one zone.
- This is an argument from the code, not a measurement.

### Summary: why we switched
1. **Different question.** The old system answers "how many people are in the picture now?" The project needs "which zone becomes dangerous next?" That requires calibrated m² per zone, motion and forecasting, which the old system did not have.
2. **Counting reliability.** By the old project's own measurements:
   - CSRNet's count swings 1.8× with resolution;
   - it reads 11–12 people in a one-person room;
   - it carries about ±50 % uncertainty on unlabelled footage.

   CLIP-EBC matches CSRNet's accuracy on ShanghaiTech A (66.95 vs 66.09) *without* being trained on it, and is measured at 2–5 people per frame on our labelled classrooms.
3. **No early warning.** Our results show that any alarm on the *current* state never warns in time. Only forecast-driven alarms did.
4. **The old code's design survives where it was right.** The hard switch (detector near empty, density model when crowded) was re-implemented with CLIP-EBC and is the best frame-level counting method on unseen rooms (3.16 vs 3.22 people per frame, `results/fusion_loco.md`). Zone density needs a spatial map, so the zone pipeline uses the CLIP-EBC density map alone.

### Limits of this comparison
- **Different data.** Old-system numbers come from its own README: its test video, a webcam and ShanghaiTech A. Only the ShanghaiTech A row (66.09 vs 66.95) compares the two systems on the same data.
- **Not run on our CCTV.** The old repository does not contain its CSRNet weights (`csrnet_official.pth.tar`, 124 MB), so it was not run on our 160 labelled frames. The classroom analysis above is derived from the old code and our measured detector results.
- **Different hardware for speed.** RTX 3050 vs A100; speeds are listed side by side but are not a like-for-like test.
- **Follow-up (optional):** if the CSRNet weights are found, the old counting can be measured on our frames on CPU. Nothing else in this report depends on it.

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
- Detectors **are** slightly better in nearly empty scenes. With fewer than 20 people in H705, the CrowdHuman heads model errs by 0.86 against 1.15 for CLIP-EBC. This motivated the "hard switch" (detector count when the density count is below τ ≈ 26–32 people). It is evaluated at frame level only; the zone pipeline uses the CLIP-EBC density map, because zone density needs a spatial map.

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

- **Finding:** learned fusion **does not transfer** to new rooms. It is reported as a negative result. The hard switch is marginally best at frame level (3.16 vs 3.22); the zone pipeline uses the CLIP-EBC density map.

**Not used:**
- **The old YOLO11m + CSRNet project:** replaced by this design from scratch (D2). See §2A for why, and for an old-vs-new comparison.
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
| Jülich test (all test forecasts) | Chronos-2 per zone | 0.38 / 0.56 / 0.73 | **0.89 / 0.88 / 0.93** |
| | Persistence | 0.39 / 0.57 / 0.71 | 0.85 / 0.66 / 0.68 |
| Simulation test | Chronos-2 joint | 0.16 / 0.36 / 0.57 / 0.76 | 0.94 / 0.92 / 0.91 / 0.96 |
| | Persistence | 0.17 / 0.34 / 0.56 / 0.75 | 0.89 / 0.88 / 0.87 / 0.93 |
| | Physics | 0.26 / 0.66 / 1.25 / 2.32 | 0.90 / 0.87 / 0.86 / 0.91 |

**Findings:**
- **On average error,** Chronos-2 is about equal to persistence; it is slightly better on our CCTV.
- **On the worst case,** which is what matters for safety, Chronos-2 is clearly better. Over all Jülich test forecasts its bound holds 88–93 % (per zone), while persistence drops to 66–68 % at 30–60 s. Where the true density is ≥ 4 persons/m², every model falls below 90 %, but Chronos-2 still holds 76–88 % against 56–78 % for persistence (`results/coverage_by_density_julich.md`, computed from the stored forecasts).
- **Correction (2026-10-29):** earlier versions called the 88–93 % figure "coverage at dangerous moments"; it is coverage over all test forecasts.
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
  - slowing down (≤ 0.21 m/s) reached precision 0.54 against a base rate of 0.145 (3.75×; an earlier "3.2×" mixed up F1 and precision);
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
4. **A calibrated zero-shot forecaster gives an honest worst case.** On Jülich test runs Chronos-2's bound holds 88–93 % of forecasts (persistence 66–85 %); where density is ≥ 4 persons/m², 76–88 % (persistence 56–78 %).
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
| Measured old-system baseline on our CCTV (YOLO11m + CSRNet) | not done | Old repo (github.com/Sidharth260905/CrowdSafe) has no weights; needs `csrnet_official.pth.tar` (124 MB). Its README results are used in §2A |
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
| D2 | New project from scratch (old YOLO11m + CSRNet not reused) | Old system answered "how many people now", not "which zone becomes dangerous next"; no m², zones, motion or forecast (§2A) |
| D3 | GPU work as A100 notebooks; models returned as ONNX | The A100 is reachable only through Kubeflow; the laptop has no GPU |
| D4 | Prefer Hugging Face weights (CLIP-EBC NWPU, CrowdHuman `Sharath33/Person`) | Reproducible downloads |
| D5 | Mirrors for blocked downloads (RAFT weights checked by SHA-256) | College network blocks some hosts |
| D6 | Full camera model instead of a floor homography | Lens distortion, hidden floor, heads above the floor |
| D7 | CLIP-EBC density map for zones; hard switch best at frame level; learned fusion reported as negative | Fails on unseen rooms |
| D8 | Keep alert setting A | No variant better everywhere |

---

## 12. Points for discussion

1. **Is "seconds ahead" acceptable for the paper's claim?**
   - The data says single-camera warnings come 7.5–25 s early (median), not minutes.
   - Options:
     - (a) present it honestly as the main finding (current draft);
     - (b) add upstream information, such as an entrance camera or a gate counter, as future work.
2. **Old vs new comparison depth.**
   - §2A compares on documented numbers and the code.
   - Is that enough for the panel, or should we try to recover the old CSRNet weights and measure it on our frames?
3. **Real dangerous densities.**
   - Our CCTV never exceeds 3.2 persons/m², so danger-level results rest on 9 real Jülich events plus simulation.
   - A short staged crowding recording in H705 (already calibrated) would strengthen the real-world evidence.
4. **Live demo.**
   - The dashboard replays recorded runs.
   - A live view needs a GPU reachable from the demo machine; the old system's live-streaming code (reconnect, watchdog) could be reused for that.
5. **Venue beyond classrooms.**
   - Zones and thresholds were set for classrooms (T2.2, T7.5 open).
   - Is a corridor or entrance at the college available for one more recording?
6. **Target venue for the paper.**
   - IEEE conference format is ready (`docs/paper/main.tex`).
   - Its page limit decides which figures and tables stay.

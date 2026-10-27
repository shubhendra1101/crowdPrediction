# Zone-level crowd-density forecasting and early warning from uncalibrated classroom CCTV with zero-shot models

*Draft v1, 2026-10-26.*

- **Source of every number:** a file in `results/` (named in brackets). Nothing is estimated by hand.
- **`[CHECK]`:** a reference detail to verify before submission.
- **`[TODO]`:** text the authors still have to write (venue, affiliations, acknowledgements).

---

## Abstract

Crowd crushes develop within tens of seconds, yet most camera-based crowd monitoring reports only a head count for the whole frame. We present CrowdSafe, a pipeline that turns ordinary fixed CCTV into per-zone crowd density in persons/m², forecasts it 10–120 s ahead, and raises tiered alerts with hysteresis. **No component is trained.**

**Pipeline:**
- **Counting:** a zero-shot density model, CLIP-EBC.
- **Calibration:** a full camera model with lens distortion, fitted from furniture of known size.
- **Motion:** RAFT optical flow converted to m/s on the floor plan.
- **Forecasting:** the zero-shot foundation model Chronos-2, whose upper quantile is calibrated by split conformal prediction.

**Evaluation data:**
- four CCTV cameras in three rooms (2.6 h, 6,715 hand-labelled heads);
- held-out Jülich pedestrian experiments;
- 16 JuPedSim scenarios.

**Results:**
- **Counting:** CLIP-EBC counts with a mean absolute error of 2.0–5.1 people per frame. A learned per-frame fusion with a person detector does not generalise to unseen rooms (5.22 vs 3.22 for the density model alone).
- **Density:** zone density is estimated within 0.08–0.10 persons/m² of the labels.
- **Forecast bound:** the calibrated 90 % bound holds 89–96 % of the time on CCTV and simulations, and 84–93 % at dangerous moments in the Jülich data, where a persistence bound holds only 66–85 %.
- **Early warning:** alerts driven by the current density never warn before a zone reaches 4 persons/m². Forecast-driven alerts warn in time for about one third of events in simulation (Chronos-2: 34 %, median 7.5 s ahead). On the real Jülich test set, one of nine events was warned in time.
- **Latency:** the full pipeline runs live on one A100 with a median latency of 0.49 s and 4.6× real-time throughput.

We report these mixed early-warning results openly and argue that minute-level warning requires upstream information beyond a single camera view.

---

## 1. Introduction

**The problem.** Crowd disasters are caused by local density, not by the total number of people present. Fruin's levels of service and later analyses put the onset of dangerous conditions at roughly 4–5 persons/m², with turbulent "crowd quakes" above that [Fruin 1971; Helbing et al. 2007]. Monitoring therefore needs three things:
1. density in physical units, per area of floor;
2. awareness of where in the scene density is building up;
3. enough warning time for staff to act.

**What existing CCTV analytics lack.** Most deployed analytics, and most crowd-counting research, report people per frame. Converting this to persons/m² requires camera calibration, which is rarely available for existing installations. Predicting dangerous density, rather than detecting it, needs a forecaster that works without venue-specific training data, because dangerous events are, fortunately, rare.

**Our approach.** CrowdSafe is built entirely from pretrained, zero-shot components and evaluated on our own classroom CCTV, public trajectory experiments and simulation.

**Contributions:**
1. **Calibration from furniture.** A full camera model with radial distortion, fitted from desk tops of known length and straight scene edges. It converts zero-shot density maps into persons/m² per zone. It was validated against independent measurements: row spacing within 3 %, mean desk spacing within 1–5 % (single gaps up to 10 %), door width within 10 %, and lab table positions within 0.03–0.18 m.
2. **A held-out evaluation of counting for real CCTV.** It includes a negative result: learned detector–density fusion that looks best within one room fails on unseen rooms.
3. **Zero-shot, conformally calibrated forecasting of zone density.** Chronos-2 gives a reliable worst-case bound where persistence does not.
4. **An honest early-warning evaluation.** It uses a strict lead-time metric and tests motion "precursors" (crowd pressure, counter-flow, flow collapse, slowing), which mostly fail to beat chance at the densities available. It also includes an ablation of alert-timing choices.
5. **A live implementation** with measured per-stage latency on one GPU, plus a replay dashboard.

---

## 2. Related work [TODO: expand with 1–2 sentences per item and recent surveys]

- **Crowd counting.**
  - Density-map regression since MCNN on ShanghaiTech [Zhang et al. 2016].
  - Vision–language counting, CLIP-EBC [Ma et al. 2024 `[CHECK]`], which we use zero-shot.
  - Person and head detectors (YOLO family [Ultralytics YOLO11 2024]; CrowdHuman-trained heads) undercount under occlusion.
- **Camera calibration for crowd analysis.** Homography from floor points (common in counting work) assumes a pinhole camera and points on the floor. We fit lens distortion and use the head-height plane, because CCTV lenses are wide-angle and floors are occluded by furniture.
- **Pedestrian dynamics.**
  - Crowd pressure (density × velocity variance) and turbulence before disasters [Helbing et al. 2007].
  - The Jülich Pedestrian Dynamics Data Archive [Forschungszentrum Jülich `[CHECK]`].
  - Analysis with PedPy [PedPy `[CHECK]`]; simulation with JuPedSim [JuPedSim `[CHECK]`].
- **Time-series foundation models.** Chronos [Ansari et al. 2024] and Chronos-2 [`[CHECK]` Amazon 2025] forecast without task-specific training.
- **Conformal prediction.** Split conformal calibration gives finite-sample coverage guarantees under exchangeability [Angelopoulos & Bates 2023 `[CHECK]`]. Here it calibrates the forecast's upper bound.
- **Optical flow.** RAFT [Teed & Deng 2020].

---

## 3. Method

### 3.1 Overview
Every second, the pipeline runs these steps (architecture in `architecture.md`):
1. **Density map:** CLIP-EBC on the current frame.
2. **Optical flow:** RAFT between the current frame and the frame 0.2 s later.
3. **Zone features:** density and flow are projected through the calibrated camera onto the floor plan and aggregated per zone into a feature row:
   - density (persons/m²);
   - mean speed (m/s);
   - speed variance and pressure;
   - direction entropy and counter-flow.
4. **Forecast (every 5 s):** a forecaster predicts each zone's density 10–60 s ahead (120 s in simulation).
5. **Risk and alerts:** the risk engine assigns a level, and an alert manager applies hysteresis.

The same feature schema is produced from Jülich trajectories (PedPy-validated, max difference 0.0000 persons/m², `results/julich_features_summary.md`) and from simulations, so forecasting and alerting are evaluated identically on all three sources.

### 3.2 Counting
- **Models compared:** YOLO11l (COCO persons), a CrowdHuman YOLO11 person+head detector, and CLIP-EBC ViT-B/16 with NWPU weights in 224-px sliding windows. All are used without fine-tuning.
- **Fusion candidates:**
  - a learned per-frame weight from detector reliability features (sigmoid of a linear model);
  - a constant weight;
  - a 50/50 average;
  - a hard switch (the detector below a count threshold, density above it).
- **Exclusions:** regions outside the room (a glass partition in the lab) are removed from labels and predictions alike (`configs/roi.yaml`).

### 3.3 Camera calibration
- **Camera model:** a pinhole model with radial distortion (k1, k2), focal length and pose, fitted by robust least squares.
- **Classrooms:**
  - Desk tops are rectangles of known length (1.00 m), all parallel and on one plane. Each desk has its own unknown floor position, so row and column spacing are not assumed and remain independent checks.
  - Straight scene edges (door frames, ceiling duct, stage edge) add plumb-line constraints that pin the lens distortion.
- **Computer lab:** long table edges at known lateral positions and vertical edges are used instead.
- **Floor level:** the floor is placed using the measured camera height or desk height.
- **Head positions:** heads are mapped to the floor through the camera ray at seated head height (1.2 m); 1.1–1.6 m is tested as a sensitivity check.
- **Zones:** 3×3 seating blocks plus the front aisle, 4–16 m² each. Density = count in the zone's image mask / visible zone area.

### 3.4 Motion
- RAFT-large flow at 960 px is averaged per density-map cell.
- Each cell centre and its displaced position are traced to the head-height plane, giving velocity in m/s.
- Cells are weighted by the density map, so empty floor contributes nothing.
- Definitions match the trajectory features: mean speed, velocity variance, pressure = density × variance, 8-bin direction entropy, and counter-flow share.

### 3.5 Forecasting and conformal calibration
- **Forecasters** (all given the last 30 s at 1 Hz):
  - **persistence:** last value;
  - **physics fill-rate:** density + (inflow − outflow)/area × h;
  - **Chronos-2 per zone;**
  - **Chronos-2 joint:** all zones of a camera together.
- **Conformal step:** on validation runs only, the 90 % upper bound is shifted by the finite-sample conformal quantile of (target − p90), per model and horizon.
- **Splits:** data are split by experiment, scenario or camera, never within a run.

### 3.6 Risk levels and alerts
- **Levels** come from the worst calibrated upper bound over the horizons:
  - Low < 2 persons/m²;
  - Medium 2–4;
  - High 4–5;
  - Critical ≥ 5.
- **Precursors** would raise the level by one step: pressure, direction entropy, counter-flow, flow collapse, slowing, and rate of rise. They are kept only if, on validation data, their precision for "reaches 4 persons/m² within 60 s" is at least twice the base rate.
- **Alerts:** raised when the level stays ≥ High for 10 s and cleared after 30 s below. Each alert carries zone, level, time-to-critical and reasons.

### 3.7 Runtime
- A background reader keeps only the newest frames, dropping frames when processing is slow.
- A 1-Hz loop runs steps 1–5 above and logs every stage's duration.
- A FastAPI/WebSocket server and a Next.js dashboard replay stored runs with zones coloured by level, the forecast band and the alert feed.

---

## 4. Data and evaluation protocol

| Source | Content | Use |
|---|---|---|
| Own CCTV | 4 cameras, 3 rooms (classrooms H705 ×2 views and H706, computer lab H103), 2.6 h; 160 frames with 6,715 hand-labelled heads | counting, density, real forecasting, live latency |
| Jülich archive | trajectories of corridor, bottleneck and entrance experiments; test = uni-directional corridor 2013 + entrance semicircle (10 runs) | forecasting, alerts on real dangerous crowds |
| JuPedSim | 16 scenarios, 4 layouts; test = 5 held-out scenarios | rare dangerous events (up to 8 persons/m²) |
| ShanghaiTech A/B | public benchmark | sanity check of counting weights |

**Metrics:**
- MAE per frame for counts.
- MAE in persons/m² per zone for density.
- Forecast MAE, and empirical coverage of the calibrated bound.
- **Alerts:**
  - an *event* is a zone first reaching 4 (or 5) persons/m² after monitoring starts;
  - a warning is "in time" if an alert is raised within (longest horizon + 10 s) before the event and not in the first 15 s of monitoring;
  - false alarms are alerts with no event within 120 s, reported per zone-hour.
- **Why the strict alert metric:** a looser version, which counted alerts raised at the start of a run and simply kept on, inflated the early-warning rate several-fold (Section 6).

---

## 5. Results

### 5.1 Counting
**Table 1.** Zero-shot counting, MAE in people per frame (`results/counting_zeroshot_cctv.md`, `counting_zeroshot_newcams.md`, `counting_zeroshot_shanghaitech.md`).

| Model | H705 (100 frames) | H706 (30) | H103 lab (30) | ShanghaiTech A / B |
|---|---|---|---|---|
| CLIP-EBC | **5.10** | **2.58** | **1.97** | **66.95 / 11.66** |
| YOLO11l | 6.36 | 6.53 | 17.63 | 418.3 / 92.8 |
| CrowdHuman heads | 8.04 | 6.23 | 28.23 | 374.2 / 70.2 |

- The detectors fail under occlusion: in the lab, monitors hide most bodies, and YOLO11l misses about 18 people per frame.
- **Fusion (Table 2, `results/fusion_ablation.md`, `fusion_loco.md`):**
  - **Within one room** (H705, time-split), a constant fusion weight reached 1.59 and the learned weight 2.09, versus 5.30 for CLIP-EBC alone. The two models happened to err in opposite directions there (+5.1 and −6.0 people).
  - **Leave-one-room-out:**

| Method | Mean MAE on the held-out room |
|---|---|
| Hard switch | 3.16 |
| CLIP-EBC alone | 3.22 |
| 50/50 average | 4.48 |
| Constant weight | 4.56 |
| Learned fusion | 5.22 |
| YOLO11l alone | 10.17 |

  - **Conclusion:** learned fusion does not transfer across rooms. We therefore use CLIP-EBC, with a hard switch to the detector in near-empty scenes.

### 5.2 Calibration and zone density
**Calibration** (`results/calibration.md`):

| Room | Reprojection error | Independent checks |
|---|---|---|
| H705 | 4.2 px | row spacing 0.99 m vs 0.96 m measured (+3 %); mean desk + gap 1.69 m vs 1.77 m (−5 %; single gaps −1 to −10 %) |
| H706 | 3.6 px | desk + gap 1.78 m vs 1.77 m; door width 1.23 m vs 1.12 m (+10 %; the second door, partly hidden, gave 0.95 m) |
| H103 lab | 4.0 px | table edges of the two columns not used in the fit found within 0.03–0.18 m; camera height not independent (soft prior from the measured value) |

**Zone density, CLIP-EBC vs labels** (`results/zone_density.md`):

| Room | MAE (persons/m²) | Highest true zone density |
|---|---|---|
| H705 | 0.101 | 1.67 |
| H706 | 0.081 | 1.40 |
| H103 | 0.078 | 1.29 |

- 94–98 % of labelled heads fall in a zone.
- Changing the assumed head height from 1.1 to 1.6 m changes the MAE by at most 0.03 persons/m².
- Over the full videos, the busiest zone reached 3.2 persons/m² (H705 lecture). No classroom zone reached 4.

### 5.3 Forecasting
**Table 3.** Forecast error and reliability of the calibrated 90 % bound (`results/forecast_cctv.md`, `forecast_julich_a100.md`, `forecast_sim_a100.md`).

| Data (horizons) | Model | MAE (persons/m²) | Calibrated bound held |
|---|---|---|---|
| Own CCTV, test cameras (10/30/60 s) | Chronos-2 joint | 0.064 / 0.080 / 0.097 | 0.90 / 0.91 / 0.89 |
| | Persistence | 0.074 / 0.087 / 0.105 | 0.92 / 0.93 / 0.91 |
| Jülich test, at critical targets (10/30/60 s) | Chronos-2 per zone | 0.38 / 0.56 / 0.73 (all points) | 0.89 / 0.88 / 0.93 |
| | Persistence | 0.39 / 0.57 / 0.71 | 0.85 / 0.66 / 0.68 |
| Simulation test (10/30/60/120 s) | Chronos-2 joint | 0.16 / 0.36 / 0.57 / 0.76 | 0.94 / 0.92 / 0.91 / 0.96 |
| | Persistence | 0.17 / 0.34 / 0.56 / 0.75 | 0.89 / 0.88 / 0.87 / 0.93 |
| | Physics | 0.26 / 0.66 / 1.25 / 2.32 | 0.90 / 0.87 / 0.86 / 0.91 |

- **Average error:** Chronos-2 matches persistence, and is slightly better on our CCTV.
- **Worst case:** its calibrated worst-case bound stays near the 90 % target exactly when density is critical, where persistence under-covers badly.

### 5.4 Early warning
**Table 4.** Share of events (zone reaches 4 persons/m²) warned in time, strict metric (`results/alerts_sim.md`, `alerts_julich.md`, `alerts_cctv.md`).

| Test data | Current density only | Persistence | Physics | Chronos-2 per zone / joint |
|---|---|---|---|---|
| Simulation (47 events) | 0 % | 0 % | 32 % (median 25 s; 2.1 false alarms/zone-h) | 34 % (7.5 s; 1.6) / 30 % (7.5 s; 1.5) |
| Jülich (9 events) | 0 % | 11 % | 11 % (11.9 false alarms/zone-h) | 11 % / 11 % (1.8 / 0.5) |
| Own CCTV (8.6 zone-hours) | no events | no events, 0 false alarms | — | no events, 0 false alarms |

**Precursors.**
- **Tuned on Jülich validation runs only:**
  - slowing (≤ 0.21 m/s) reached 3.2× the base rate (precision 0.47 vs 0.145);
  - direction entropy reached 2×;
  - pressure, counter-flow and flow collapse did not pass.
- **Pooled with simulation validation data,** no precursor passed, so none is used (`results/alerts_julich.md`, `configs/risk.yaml`).

**Ablation of alert timing** (`results/alert_experiment.md`; persistence and physics on the laptop; 29 simulated and 9 Jülich events):

| Change | Effect |
|---|---|
| Shorter forecast history (15 s) | Physics improved most in simulation: 59 → 62 % warned, median 25 → 38 s, false alarms 1.6 → 2.3 per zone-hour |
| Shorter hold (5 s) + rate-of-rise + shorter history together | Raised current-density alerts from 0 to 24 % and persistence from 0 to 31 % with 0.2 false alarms per zone-hour, but lowered physics to 24 % |
| Shorter hold alone | Mainly added false alarms |
| Rate-of-rise alone | Almost no effect |

No variant improved all methods, so the default (30 s history, 10 s hold) is kept.

### 5.5 Live latency
**Table 5.** One A100 (shared), cam02 played at real speed (`results/latency_live_chronos.md`, `latency_live_persistence.md`, `latency_throughput.md`).

| Run | Median / p95 latency | CLIP-EBC | RAFT | Forecast | Cycles on time | GPU memory |
|---|---|---|---|---|---|---|
| Live, Chronos-2 (10 min) | 0.49 s / 1.6 s | 159 ms | 103 ms | 54 ms per 5 s | 94 % | 2.0 GB |
| Live, persistence (5 min) | 0.39 s / 1.5 s | 103 ms | 61 ms | 6 ms | 94 % | 1.5 GB |
| Unpaced (15 min of video) | 0.36 s / 0.45 s | 105 ms | 58 ms | 40 ms | 99.8 % | 2.0 GB |

- **Throughput:** the unpaced run processed 908 s of video in 196 s (4.6× real time).
- **What latency includes:** the 0.2 s wait for the second frame of the motion pair.
- **Spikes:** rare, up to 7 s, mostly from database writes on the shared server.
- **CPU only:** the pipeline needs about 19 s per second of video on a laptop CPU.

---

## 6. Discussion and limitations

- **Why warnings are short.**
  - In both the Jülich experiments and the simulations, zones go from safe to 4 persons/m² in roughly 10–20 s.
  - The alert design adds a 5 s decision step and a 10 s hold, and the forecaster needs 30 s of history.
  - From a single camera view, the cause of a surge (arrivals from a corridor, an exit closing) is often outside the frame.
  - **Implication:** minute-level warning would need upstream information, such as entrance cameras, gate counters or schedules, and neighbouring-zone inflow.
- **Measurement matters.** Our first alert scoring counted alerts raised at the start of a run and simply kept on, which produced early-warning rates of 56 % (Jülich) and 86 % (simulation). The strict metric reduces these to 11 % and about one third. We recommend the strict form for future work.
- **Precursors.** Crowd pressure and turbulence are documented at disaster densities (above 6–7 persons/m²). At the densities available here, they did not precede filling-up better than chance. Slowing did on real data but not in simulation, which suggests the simulation model does not reproduce slowing before congestion.
- **Limitations:**
  - Calibration is approximate (about ±5 %).
  - Our CCTV contains no dangerous densities, so danger-level evaluation relies on 9 real Jülich events and on simulation, which follows model rules rather than real crowd behaviour.
  - Latency was measured on a shared GPU with one camera.
  - Head height is assumed (seated).
  - Inflow and outflow are not yet measured from CCTV, so the physics forecaster was evaluated only on trajectories and simulation.
- **Ethics and privacy.** The pipeline estimates densities and flow fields only. No identities, faces or tracks are stored, and labels are head points.

## 7. Conclusion

Zero-shot components are sufficient to turn existing CCTV into calibrated, zone-level crowd density with errors around 0.1 persons/m², and to run the full pipeline live on one GPU with sub-second median latency. A conformally calibrated zero-shot forecaster gives a trustworthy worst-case bound where a naive one does not. Turning that bound into early warnings works for about a third of dangerous events in simulation, and only seconds ahead. We see longer warning horizons as a problem of sensing upstream of the camera rather than of the forecaster.

---

## Figures to include
All figures are made by scripts (`python scripts/paper_figures.py`, `scripts/alert_timeline.py`, `scripts/calibrate_cameras.py`, `scripts/zone_density.py`, `scripts/sim_preview.py`).
1. **Pipeline diagram:** `results/figures/paper_pipeline.png`.
2. **Calibration overlays:** `results/figures/calibration_H705.jpg`, `_H706.jpg`, `_H103.jpg`.
3. **Zones with labelled heads:** `results/figures/zones_H705.jpg` (or H706, H103).
4. **Zone density on the camera image (busiest second of each recording):** `results/figures/paper_heatmap_cam02.jpg` (H705, t = 309 s), `paper_heatmap_cam03.jpg` (H706, t = 1057 s), `paper_heatmap_cam04.jpg` (H103, t = 2134 s). All zones stay below 2 persons/m² (highest per room: H705 1.87, H706 1.94, H103 1.18), i.e. the classrooms never become crowded.
5. **Forecast example:** `results/figures/paper_forecast_sim_corridor.png`, the corridor-surge simulation, zone c0_9, forecast made 30 s before the zone first reaches 4 persons/m². The calibrated Chronos-2 bound rises (2.33 / 2.72 / 2.82 / 3.09 persons/m² at 10 / 30 / 60 / 120 s) while persistence stays flat (1.80 / 1.80 / 1.65 / 1.65), but the bound still stays below 4: the forecaster sees the rise coming without predicting how far it goes. This illustrates the short lead times in Table [alerts].
6. **Alert timelines:** `results/figures/timeline_julich_corridor.png` (real dangerous crowd) and `timeline_cam02.png` (classroom exit, no false alarm).
7. **Simulation snapshots:** `results/figures/sim_hall_700_A_closes.png`.
8. **Dashboard screenshot** [TODO: user captures a CCTV run with the video playing, from http://localhost:3000].

## References (verify all fields before submission)
- Angelopoulos, A. N., & Bates, S. (2023). Conformal prediction: A gentle introduction. *Foundations and Trends in Machine Learning*. `[CHECK]`
- Ansari, A. F., et al. (2024). Chronos: Learning the language of time series. *Transactions on Machine Learning Research*. `[CHECK]`
- Chronos-2 (Amazon, 2025), model card `amazon/chronos-2` and technical report. `[CHECK: exact title and authors]`
- Forschungszentrum Jülich. Pedestrian Dynamics Data Archive. `[CHECK: per-experiment citations]`
- Fruin, J. J. (1971). *Pedestrian Planning and Design*. `[CHECK]`
- Helbing, D., Johansson, A., & Al-Abideen, H. Z. (2007). Dynamics of crowd disasters: An empirical study. *Physical Review E*, 75, 046109. `[CHECK]`
- JuPedSim, PedPy (software, Forschungszentrum Jülich). `[CHECK: citation format]`
- Ma, Y., Sanchez, V., & Guha, T. (2024). CLIP-EBC: CLIP can count accurately through enhanced blockwise classification. arXiv. `[CHECK]`
- Teed, Z., & Deng, J. (2020). RAFT: Recurrent all-pairs field transforms for optical flow. *ECCV*.
- Ultralytics (2024). YOLO11. `[CHECK]`
- Weidmann, U. (1993). Transporttechnik der Fussgänger. ETH Zürich (walking-speed distribution used in the simulations). `[CHECK]`
- Zhang, Y., Zhou, D., Chen, S., Gao, S., & Ma, Y. (2016). Single-image crowd counting via multi-column CNN. *CVPR*.
- CrowdHuman (Shao et al., 2018) and the `Sharath33/Person` model. `[CHECK]`
- ByteTrack is not used in the reported results; remove it from the related work if it is not mentioned elsewhere.

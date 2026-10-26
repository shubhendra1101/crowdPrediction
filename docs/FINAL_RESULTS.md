# CrowdSafe: final results (2026-10-26)

One page to show the results. Every number comes from a file in `results/`, linked in each section. Nothing is estimated by hand.

**What the system does:** from ordinary CCTV, it measures how crowded each zone of a room is (people per m²), forecasts the next 10–120 s, and raises an alarm before a zone becomes dangerous (above 4 people/m²).

**Data:**
- **Own CCTV:** 4 cameras in 3 rooms, 2.6 h of video, with 6,715 hand-labelled heads on 160 frames.
- **Jülich:** real crowd experiments with measured trajectories.
- **Simulations:** 16 JuPedSim scenarios (synthetic, reported separately).

**All models are used zero-shot (no training).**

---

## 1. Counting people (own CCTV)
Mean absolute error in people per frame; lower is better.

| Room | CLIP-EBC (chosen) | YOLO11l | Head detector |
|---|---|---|---|
| H705 classroom | **5.1** | 6.4 | 8.0 |
| H706 classroom | **2.6** | 6.5 | 6.2 |
| H103 computer lab | **2.0** | 17.6 | 28.2 |

**Files:**
- `results/counting_zeroshot_cctv.md` and `counting_zeroshot_newcams.md`
- `results/counting_zeroshot_shanghaitech.md` (public benchmark: CLIP-EBC 67 / 11.7 vs YOLO 418 / 93)

**Learned YOLO + CLIP-EBC fusion: negative result.** Each room was held out in turn and the fusion fitted on the others:

| Method | Mean error on the held-out room |
|---|---|
| Hard switch | 3.16 |
| CLIP-EBC alone | 3.22 |
| Learned fusion | 5.22 |
| YOLO alone | 10.17 |

The learned fusion does not transfer to unseen rooms. Files: `results/fusion_loco.md`, `fusion_ablation_heldout.md`.

## 2. Camera calibration and people per m²
**Method:** a full camera model (lens distortion, position, tilt) fitted from desk tops and straight edges. It was checked against measurements the fit did not use:
- row spacing within 3 %;
- desk + gap spacing within 1–5 %;
- door width within 6 %;
- lab table positions within 0.03–0.18 m.

Files: `results/calibration.md`, `results/figures/calibration_*.jpg`.

**Density per zone:** error of CLIP-EBC against the labels, in people per m².

| Room | Error | Highest true density |
|---|---|---|
| H705 | 0.10 | 1.7 |
| H706 | 0.08 | 1.4 |
| H103 | 0.08 | 1.3 |

94–98 % of labelled heads fall inside a zone. Files: `results/zone_density.md`, `results/figures/zones_*.jpg`.

## 3. Forecasting (zero-shot Chronos-2 vs simple baselines)
"Upper limit held" = share of times the true density stayed below the calibrated 90 % worst case (target 90 %).

| Data | Chronos-2 error (people/m²) | Chronos-2 upper limit held | Persistence upper limit held |
|---|---|---|---|
| Own CCTV, test cameras (10–60 s ahead) | **0.064–0.097** (lowest) | 89–91 % | 91–93 % |
| Jülich, real crowds (10–60 s) | 0.37–0.74 (≈ persistence) | 84–93 % at dangerous moments | 66–85 % |
| Simulations (10–120 s) | 0.16–0.77 (≈ persistence) | 91–96 % | 87–93 % |

**Files:**
- `results/forecast_cctv.md`
- `results/forecast_julich_a100.md`
- `results/forecast_sim_a100.md`

## 4. Early-warning alarms
**Alarm rule:** the zone's forecast worst case must reach High (≥ 4 people/m²) for 10 s; the alarm clears after 30 s below.

**Scoring rule (strict):**
- the alarm must come within the forecast window before the zone reaches 4 people/m²;
- alarms raised in the first 15 s of monitoring do not count.

| Test data | Alarm on current density only | Persistence | Physics (in − out) | Chronos-2 |
|---|---|---|---|---|
| Simulations (47 events) | 0 % in time | 0 % | 32 % (median 25 s early) | **34 %** (median 7.5 s) |
| Jülich, real (9 events) | 0 % | 11 % | 11 % | 11 % |
| Own CCTV | no dangerous moments → **0 alarms, 0 false alarms** in 8.6 zone-hours | | | |

**Files:**
- `results/alerts_sim.md`, `alerts_julich.md`, `alerts_cctv.md`
- timelines: `results/figures/timeline_cam02.png`, `timeline_julich_corridor.png`, `timeline_julich_entrance.png`

**Main finding:** an alarm based only on current density never warns in time. Forecasting turns about a third of dangerous events into warnings, typically seconds ahead (not minutes), because zones fill in 10–20 s.

**Extra experiment (ablation):** shorter wait, rate-of-rise warning, shorter history. No change improved every method, so the original setting is kept. File: `results/alert_experiment.md`.

**Crowd-motion warning signs tested and rejected:** crowd pressure, counter-flow, flow collapse, and slowing down (the last passed on Jülich only). File: `results/alerts_julich.md`.

## 5. Simulations (synthetic)
16 scenarios in 4 layouts (classroom, corridor + bottleneck, hall with 2 exits, two-way corridor), with surges, exit closures and opposing flows:
- 10 scenarios pass 5 people/m², up to 8.0;
- classrooms stay below 4.

Files: `results/simulation_summary.md`, `results/figures/sim_*.png`; data in `data/sim/csv/` (Excel).

## Limitations (say these openly)
- Calibration is approximate (about ±5 %).
- Real dangerous events are few (9 in the Jülich test set), so those percentages are rough.
- The simulations follow model rules, not real panic.
- Our classrooms never become dangerous, so danger-level testing relies on Jülich and the simulations.
- Warnings come seconds ahead. Minute-level warnings would need upstream information: entrance cameras, gate counters, schedules.

## Earlier figures that are superseded (do not quote)
- "56 % / 86 % warned early": a loose scoring rule; replaced by section 4.
- "Fusion is the main contribution": replaced by the negative result in section 1 (decision D7).

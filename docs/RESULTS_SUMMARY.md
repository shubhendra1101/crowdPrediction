# CrowdSafe: results so far (2026-10-22, about week 4 of 7)

A plain-language summary to share and discuss. Every number comes from a script in this repo, with the full tables in `results/`.

## The project in one line
From ordinary CCTV, estimate how crowded each part of a room is (people per m²), forecast it a few minutes ahead, and warn early when it is getting dangerous.

The pipeline has four steps: **count people → convert to people per m² → forecast → alert**.

## 1. Counting people: works, with one clear winner

We tested three ready-made models without retraining them (zero-shot) on our own CCTV. The ground truth is 2,352+ heads we marked by hand in CVAT, across 3 rooms and 4 cameras.

Average error, in people per frame (lower is better):

| Model | What it does | H705 classroom | H706 classroom | H103 lab |
|---|---|---|---|---|
| **CLIP-EBC** | estimates a "crowd density map" | **5.1** | **2.6** | **2.0** |
| YOLO11l | draws a box around each visible person | 6.4 | 6.5 | 17.6 |
| CrowdHuman head detector | draws a box around each visible head | — | 6.2 | 28.2 |

**Conclusion:** the counting works. CLIP-EBC is accurate in every room. The box detectors fail when people are partly hidden: in the lab, monitors cover the bodies, and YOLO misses most people.

On the public ShanghaiTech benchmark, CLIP-EBC's error is 67 (Part A, very dense crowds) and 11.7 (Part B). YOLO's is 418 and 93. Same picture.

**Known issues:**
- **H705:** CLIP-EBC counts about 5 people too many on every frame, and YOLO about 6 too few. The cause is a consistent offset, not random noise.
- **H103:** a person seen through the glass window into the next room was being counted. We exclude that window area for the labels and for every model (`configs/roi.yaml`), so the scoring is fair.

## 2. The planned "smart fusion": does NOT generalise (an honest negative result)

**The idea:** a small learned rule decides, frame by frame, how much to trust YOLO versus CLIP-EBC, using cues like detector confidence and box overlap.

**What happened:**
- **Within one room (H705)**, it looked great: error 1.6–2.1, against 5.3 for CLIP-EBC alone. That was only because the two models erred in opposite directions in that room (+5 and −6), so averaging cancelled the errors out.
- **The fair test** fits the rule on two rooms and tests it on the third, unseen room, repeated for each room:

| Method | Mean error on the unseen room |
|---|---|
| Hard switch (use YOLO only when the scene is nearly empty) | **3.16** |
| CLIP-EBC alone | **3.22** |
| 50/50 average | 4.48 |
| One fixed weight | 4.56 |
| Learned per-frame fusion (the planned contribution) | 5.22 |
| YOLO alone | 10.17 |

**Conclusion:** the learned fusion is *worse* than CLIP-EBC alone on rooms it hasn't seen. It overfits each room's quirks. This is still a valid and useful finding for the paper: "a density model is robust across rooms; detector fusion does not transfer."

**What can be done:**
- **(A) Recommended:** use CLIP-EBC, with the simple hard switch for sparse scenes. Report the fusion as a tested-and-rejected baseline, and put the paper's weight on the later stages (calibration, motion, forecasting with reliable warnings).
- **(B)** Keep pushing the fusion with more rooms and labels. This costs much more labelling time, and the chances look low.
- **(C) Not planned:** fine-tune CLIP-EBC on our frames to remove the H705 offset. The project rules say to ask first, and errors of 2–3 people are already small.

## 3. People per m² (camera calibration): done, approximate

Counting gives people *per frame*. Safety thresholds are in people per **m²** (below 2 safe, 2–4 crowded, 4–5 high risk, above 5 critical). Converting needs the camera's exact view of the floor.

**Method** (
esults/calibration.md):
- A full camera model is fitted per room: zoom, lens bending (the CCTV lenses are wide-angle and bend straight lines) and the camera's position and tilt.
- **Classrooms:** every desk top is a 1.00 m rectangle; straight edges (doors, ceiling duct, stage edge) pin the lens bending.
- **Lab:** the table edges' known positions and the vertical edges are used.
- People are placed on the floor plan through their heads, at seated head height (1.2 m).

**Checks against measurements the fit never used:**

| Room | Check | Calibration | Measured |
|---|---|---|---|
| H705 | row spacing | 0.99 m | 0.96 m (+3 %) |
| H705 | desk + gap across | 1.69 m | 1.77 m (−5 %) |
| H706 | desk + gap across | 1.78 m | 1.77 m (+1 %) |
| H706 | door width | 1.19 m | 1.12 m (+6 %) |
| H103 | hidden columns 3–4 position | within 0.03–0.18 m | — |

**Two things that disagree with the notes:**
- The **H706 camera** is mounted near the ceiling (~2.6–2.7 m), not at door height.
- H706 rows look about **0.87 m** apart in the image, not 0.96 m. The H706 notes were copied from H705.

**Zones** are 3×3 blocks of seats plus the front aisle, each 4–16 m².

**Density per zone, CLIP-EBC vs hand labels, 160 frames:**

| Room | Error (persons/m²) | Highest true density |
|---|---|---|
| H705 | 0.10 | 1.7 |
| H706 | 0.08 | 1.4 |
| H103 | 0.08 | 1.3 |

- 94–98 % of labelled heads fall inside a zone.
- Changing the assumed head height (1.1–1.6 m) changes the error by at most 0.03.

## 4. Forecasting a few minutes ahead: works where it matters

Tested on the public Jülich pedestrian experiments, which have real measured crowd densities. **Chronos-2** is a pretrained time-series model used without training. It was compared with "persistence" (the simplest baseline: assume nothing changes).

- **Typical error:** about the same for both (about 0.37–0.73 persons/m² at 10–60 s ahead). Short-term crowd density is hard to beat on the average value.
- **Safe upper limit, during dangerous (critical) moments:** both methods give an upper bound that is meant to be exceeded only 10% of the time (target 90% coverage). Chronos-2's upper bound held **88–93%** of the time. Persistence's held only **66–85%**, so it often under-warns exactly when the crowd is dangerous.

**Conclusion:** the forecaster's value is a trustworthy "worst case in the next minute", which is exactly what an early-warning system needs.

## 5. Still to do (weeks 4–7)
1. Crowd-motion warning signs from video (RAFT optical flow: flow collapse, turbulence, "crowd pressure").
2. Risk engine and tiered alerts, with hysteresis so alerts don't flicker. Measure how many seconds of warning they give.
3. Simulated dangerous scenarios (JuPedSim), since we can't film a real stampede.
4. Live dashboard demo (FastAPI + Next.js).
5. Final evaluation and paper figures.

## Bottom line
- **Is the model working?** Yes. Counting is accurate (2–5 people error per frame) on rooms it was never trained on, density per zone is within about 0.1 persons/m², and the forecaster gives reliable upper-bound warnings.
- **What didn't work:** the learned YOLO+CLIP-EBC fusion. It is reported honestly as a negative result.
- **What's blocking:** nothing. Optional: re-measure the H706 row spacing and confirm the H706 camera height.

# CrowdSafe — project overview and status

*Plain-language summary for anyone joining or reviewing the project. Last updated: 15 October 2026.*
*Code: https://github.com/shubhendra1101/crowdPrediction*

---

## 1. What CrowdSafe is

Crowd crushes (stampedes) happen when too many people end up packed into one space — gates, platforms,
exits. The warning signs appear minutes earlier: the crowd gets denser, people slow down, movement turns
chaotic. Staff watching many CCTV screens miss them.

**CrowdSafe watches ordinary CCTV cameras and warns, minutes in advance, that an area is about to become
dangerously crowded.** Think of it as a weather forecast for crowds.

How it works, in five steps:

1. **Calibrate** each camera: turn pixels into metres so we can measure **people per square metre** (what actually decides danger).
2. **Count** people per zone with two AI models and combine them (one detects individuals, one estimates crowd density).
3. **Measure movement** (speed, chaos, people entering/leaving) with an optical-flow model.
4. **Forecast** the next 30 s – 10 min with a pretrained time-series model, including a trustworthy "upper limit".
5. **Alert**: Low (< 2 people/m²) · Medium (2–4) · High (4–5) · Critical (> 5), raised one level when warning signs appear (pressure, slowing flow, counter-flow). Every alert says *why*.

Privacy: no face recognition, no identities — the system only counts and measures movement. Videos never leave the laptop / college server.

---

## 2. What changed from the original plan

The original guide (`CrowdSafe Implementation Guide.md`, 29 Sep) is kept for history. These changes were made since, each recorded with its reason in `docs/decisions.md`:

| Topic | Original plan | Now | Why |
| --- | --- | --- | --- |
| Starting point | Extend an old CrowdSafe repo (YOLO11m + CSRNet + existing dashboard) | **New project from scratch**; old code not reused; dashboard will be a new build | The old idea was replaced; the old code was not available/needed (D2) |
| Old-system baseline (T0.5) | Measure the old pipeline first | **Dropped**; its "hard switch" idea is re-implemented as a comparison baseline | No old code to run (D2) |
| Where GPU work runs | Directly on an A100 | **Jupyter notebooks** the student runs on the college A100 (Kubeflow); models come back as **ONNX** files; laptop is CPU-only | Only way to reach the A100 (D3) |
| Density model | CLIP-EBC weights from GitHub | **CLIP-EBC ViT-B/16 (NWPU) from Hugging Face** | Prefer Hugging Face sources (D4) |
| Optional point counter | APGCC | **PET-Finetuned** (Hugging Face), still optional | Hugging Face availability (D4) |
| Head detector | Any CrowdHuman YOLO | **Sharath33/Person** (YOLO11s person + head, ONNX) | Fixed choice (D4) |
| Forecasting model | Graph neural network (later GRU) | **Chronos-2 zero-shot** + simple baselines; GRU optional; GNN = future work | No training needed, gives uncertainty bands |
| Lab network | — | Workarounds: RAFT weights from a checksum-verified mirror, ShanghaiTech from a Hugging Face copy if Kaggle fails, installs that never replace the server's PyTorch | Blocked downloads on the college network (D5) |
| Code layout | Folders at top level | Python package `crowdsafe/` | Avoids a name clash with Hugging Face `datasets` (D1) |
| Real-world video | Venue CCTV (3–5 cameras) | **College classrooms + a computer lab** (4 recordings, 3 rooms), with permission | Accessible footage |

---

## 3. What has been done (timeline)

**Week 1 (30 Sep – 6 Oct) — setup, models, data, first results**
- Repo, task list, decision log, automated tests (~60 tests).
- On the A100: 4 pretrained models exported to ONNX and checked against the originals (YOLO11l, RAFT optical flow, CLIP-EBC, CrowdHuman head detector). Speed: YOLO 15 ms/frame, RAFT 41 ms per frame pair.
- Downloaded ~23 GB of public data: Jülich crowd experiments (real people, exact trajectories), ShanghaiTech, JHU-Crowd++, Mall, UCSD.
- Built the "one row per zone per second" data table from 144 Jülich experiment runs (speed, density, pressure, flow…); cross-checked against the official PedPy library (exact match).

**Week 2 (7 – 15 Oct) — own camera data**
- Classroom CCTV H705 audited (Full HD, 25 fps, steady); 100 frames pre-labelled by a model, corrected by hand in CVAT (4,379 heads).
- Counting tested on the classroom; the counting comparison (fusion ablation) run.
- Two new recordings audited: H706 classroom (students leave at break) and H103 computer lab (empty → students enter).
- 60 new frames from these cameras prepared for labelling; calibration code written; room dimensions recorded.

---

## 4. Key results so far (real measured numbers)

**Counting on a public benchmark (ShanghaiTech), no training — average error per image**

| Model | Dense crowds (Part A) | Medium crowds (Part B) |
| --- | --- | --- |
| **CLIP-EBC** (density model) | **67** | **12** |
| CrowdHuman head detector | 374 | 70 |
| YOLO11l (person detector) | 418 | 93 |

Detectors miss most people in dense crowds; the density model is far better there.

**Counting on our classroom camera (100 hand-labelled frames, up to 57 people)**

| Method | Average error (people per frame) |
| --- | --- |
| YOLO11l alone | 6.25 (undercounts) |
| CLIP-EBC alone | 5.30 (overcounts) |
| Hard switch (old idea) | 5.30 |
| Plain 50/50 average | 1.66 |
| **One fixed weight (0.43)** | **1.59** |
| Learned per-frame weight (planned main contribution) | 2.09 |

Combining the two models cuts the error by ~70 % because their mistakes go in opposite directions. **Honest
finding:** the learned per-frame weighting is *not* yet better than a single fixed weight — with one camera it
overfits. This is being tested next on two new cameras.

**Forecasting crowd density on Jülich lab data (held-out experiments) — error in people/m²**

| Model | 10 s ahead | 60 s ahead | "Upper limit" reliability (target 90 %) |
| --- | --- | --- | --- |
| **Chronos-2** (pretrained, zero-shot) | 0.38 | 0.73 | **89–93 %** |
| "Nothing changes" baseline | 0.39 | 0.71 | 66–85 % |
| Physics fill-rate formula | 1.05 | 4.45 | 70–80 % |

Chronos-2's typical guess is about as good as "nothing changes", but **only Chronos-2 gives an upper limit you
can trust** — which is exactly what the alerts use.

Other findings: the lab experiments reach up to 9 people/m² (above the 5 people/m² danger level), but runs are
only 1–5 minutes long, so minute-ahead warnings will be tested on simulations.

---

## 5. What is planned next

| Step | What | Needs |
| --- | --- | --- |
| Test the counter on new cameras | Fit on H705, test on H706 + H103 (cameras never seen in tuning) | 60 corrected labels (in progress) |
| Calibration | Pixels → metres per camera; draw zones; counts become people/m² | Desk/bench measurements in H705, H103 |
| Alert engine (T7) | Levels, warning signs, "hold 10 s to raise, 30 s to clear", alert reasons; measure **warning lead time** | Nothing — can start now |
| Simulations (T5.4) | JuPedSim crowd scenarios (exit bottleneck, merging corridor, two-way corridor) incl. surges and exit closures | Approval of 3 layouts |
| Motion on real video (T4) | RAFT optical flow on the classroom videos → speed, pressure, flow | A100 notebook |
| Dashboard (T8) | FastAPI + Next.js: heatmap, forecast chart, alert feed, replay mode | — |
| Evaluation & paper (T9) | All tables and figures; methodology from measured results only | — |

Headline number for the paper: **how many minutes before an area reached 5 people/m² the system warned, and how few false alarms it raised.**

---

## 6. Open questions / risks

- **Main contribution:** if the learned weighting still doesn't beat a fixed weight on new cameras, the paper will report a calibrated two-model combination instead (with the learned version as an honest comparison).
- **Classrooms are not crowd crushes:** they test counting and calibration well, but dangerous densities come from the Jülich experiments and simulations. A corridor/door camera at class change would help.
- **Calibration** currently depends on a few tape measurements (room corners are hidden by desks).

---

## 7. Where to find things

| File | What |
| --- | --- |
| `task.md` | Full task checklist with every measured result |
| `docs/decisions.md` | Every change to the plan, with reasons |
| `results/` | All result tables (Markdown + CSV) |
| `architecture.md`, `plan.md` | System design and 7-week plan |
| `CrowdSafe Implementation Guide.md` | Original plan (historical; see section 2 for changes) |
| `docs/HANDOFF.md` | Detailed technical context for continuing the work |

## 8. Glossary

- **Homography:** a 3×3 table that converts a pixel on the floor into a real position in metres.
- **Zero-shot:** using a pretrained model as-is, without training it on our data.
- **ONNX:** a standard file format for trained models, so they run outside the original framework.
- **Conformal calibration:** adjusting a forecast's upper limit on held-out data so it is exceeded only ~10 % of the time.
- **Crowd pressure:** density × variation in people's speeds; rises before crowd disasters.
- **MAE:** mean absolute error — the average size of the mistake.

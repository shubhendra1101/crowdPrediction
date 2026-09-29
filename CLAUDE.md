# CLAUDE.md — CrowdSafe

This file is read automatically at the start of every session. Follow it over any default habits.

## 1. What this project is

CrowdSafe is a research project and working demo: an AI system that **forecasts crowd density and stampede risk per zone, minutes ahead, and raises tiered early-warning alerts** from fixed CCTV cameras.

It is a new project built from scratch in this repo (the earlier YOLO11m + CSRNet version is not reused; see `docs/decisions.md` D2). GPU work runs as notebooks the user executes on the college A100 via Kubeflow (D3).

Core idea, in one line: calibrated persons/m² per zone + crowd-motion precursors (flow collapse, turbulence/crowd pressure) + a pretrained forecaster → alerts with lead time.

Read these before doing anything substantial:

- `architecture.md` — components, data flow, interfaces, schemas
- `plan.md` — phases, week-by-week order, done-criteria, fallbacks
- `task.md` — the task checklist; this is the source of truth for what to do next
- `prompt.md` — the kickoff prompt (for context on how you were briefed)

## 2. Hard constraints

- **Hardware:** one NVIDIA A100 40 GB.
- **Deadline:** about 7 weeks total. Scope cuts are listed in `plan.md` → Fallbacks. Suggest them early if a task slips.
- **Reuse first.** Use open-source pretrained models and existing datasets. Do **not** train anything from scratch. Fine-tune only when a measured error on the user's own data says it is needed, and ask first.
- **Privacy:** no face recognition, no identity tracking across sessions, no storing faces or personal data. Tracks are anonymous and short-lived.
- **Units:** density in persons/m², speed in m/s, flow in persons/s or persons/m/min. Never report "people per frame" as density.

## 3. Chosen stack (do not swap without asking)

| Part | Choice |
| --- | --- |
| Detection | YOLO11l (Ultralytics, COCO) or a CrowdHuman person+head YOLO; pick by measured error |
| Tracking | ByteTrack (Ultralytics / Supervision) |
| Tiling for small heads | Supervision `InferenceSlicer` |
| Density counting | CLIP-EBC ViT-B/16 released weights (GitHub releases); APGCC optional |
| Zones / lines / homography | Supervision `PolygonZone`, `LineZone`, `ViewTransformer` pattern; OpenCV; Shapely |
| Optical flow | RAFT (`torchvision.models.optical_flow`, pretrained); Farneback fallback |
| Trajectory analysis | PedPy |
| Simulation | JuPedSim |
| Forecasting | Chronos-2 (`amazon/chronos-2`) zero-shot; persistence + physics baselines; GRU optional |
| Uncertainty | Chronos-2 quantiles + split conformal calibration |
| Anomaly | scikit-learn Isolation Forest |
| Backend | FastAPI + WebSocket |
| Frontend | Next.js dashboard (new build) |
| Model exchange | ONNX files exported on the A100 (D3) |
| Storage | Parquet (experiments), SQLite (live demo) |

## 4. Repository layout

Python modules live in the `crowdsafe/` package (import as `crowdsafe.counting` etc.) so that `datasets/` cannot shadow Hugging Face's `datasets` library. See `docs/decisions.md` D1.

```
<repo root>/
├── configs/          # one YAML per camera: homography, zones, adjacency, boundaries, thresholds
├── crowdsafe/        # Python package
│   ├── calibration/  # homography + zone tools
│   ├── counting/     # yolo_track.py, density_model.py, fusion.py
│   ├── motion/       # flow.py, zone_motion.py
│   ├── features/     # schema.py, logger.py
│   ├── datasets/     # loaders: julich.py, public_video.py, cctv.py
│   ├── simulation/   # jupedsim scenarios -> same feature schema
│   ├── forecasting/  # baselines.py, chronos.py, gru.py (optional), conformal.py
│   ├── risk/         # rules.py, anomaly.py, alerts.py
│   ├── server/       # FastAPI + WebSocket
│   └── eval/         # metrics, ablations, figures
├── notebooks/        # GPU notebooks run by the user on the college A100 (Kubeflow); src/ holds editable sources
├── dashboard/        # Next.js app (new build)
├── docs/             # decisions.md: record of changes to the plan/stack
├── scripts/          # CLI entry points
├── tests/
├── data/             # NOT committed (gitignored)
├── weights/          # NOT committed (gitignored); ONNX models returned from the A100 go here
└── results/          # metric tables + figures (small files only)
```

## 5. How to work

1. **Pick the next open task in `task.md`.** Tell the user which task you are starting and what it needs.
2. **Check the task's inputs.** If any input marked USER is missing, stop and ask (see §6). Don't fake it.
3. **Work in small steps.** One task per branch or commit group. Run the code, don't just write it.
4. **Verify.** Each task has an acceptance check. Run it and show the actual numbers or output.
5. **Update `task.md`:** tick the box, add a one-line result (for example `MAE 12.4 on cam02`).
6. **Report back** using the format in §7.

Coding rules:

- Python 3.11, type hints, small functions, docstrings on public functions.
- All paths and parameters come from YAML configs or CLI flags. No hard-coded paths or magic numbers.
- Every model wrapper exposes the same simple interface (see `architecture.md` §4).
- Seed everything (`42`) for reproducibility.
- Log experiments to `results/` as Markdown tables plus CSV.
- Add a smoke test in `tests/` for every module (tiny input, runs in seconds).
- Large files (datasets, videos, weights) never go into git.

## 6. Ask the user — required behaviour

**You are expected to ask.** Guessing silently is the worst outcome in this project, because wrong calibration, wrong data or wrong thresholds produce convincing but false results.

### Always stop and ask before

- Using any real-world value you don't have: floor measurements, zone areas, camera height, exit widths, venue layout, frame rate.
- Downloading anything larger than about 5 GB, or any dataset that needs a registration form.
- Starting a training or fine-tuning run expected to take over 1 hour of GPU time.
- Changing a model, library or architecture decision from §3 or `architecture.md`.
- Deleting or overwriting data, weights or results.
- Dropping or cutting scope from `plan.md`.
- Anything involving credentials or tokens. Ask the user to set environment variables themselves; never ask them to paste secrets into chat.
- When a result looks suspicious (for example error much better than published, or a count of 0).

### Also ask when

- A task's USER inputs are missing (see `task.md`).
- Two reasonable options exist and the choice changes results.
- A dependency fails to install or a download link is dead.

### How to ask

- Group questions; number them; at most 4 at a time.
- For each question say **why you need it** and **what you'll do by default** if they don't know.
- Give exact instructions for what to collect, for example: "Measure the distance between the two pillars at the left edge of cam01's view, in metres, and send a photo showing both."
- While waiting, continue with any tasks that don't depend on the answer, and say which ones.

## 7. Guidance and reporting format

The user is an engineering student building this for a research paper. Explain as you go, briefly.

End every work session or finished task with:

```
### Done
- what you built or ran, with real numbers

### What this means
- one or two plain-language sentences on the result

### I need from you
- numbered list of anything required from the user (data, measurements, decisions, permissions), with exact instructions
- or "Nothing right now"

### Next
- the next task ID and what it will do
```

When a concept matters for the paper (homography, conformal calibration, crowd pressure, zero-shot forecasting), give a 2–3 sentence explanation the first time it comes up.

## 8. Research integrity

- Never invent, estimate or "fill in" metrics. Report only what the code actually produced.
- Label approximate values (for example calibration from estimated door width) as approximate in configs and results.
- Keep held-out cameras, experiments and geometries strictly out of tuning.
- Record every baseline and ablation, even when it's worse. The comparisons are the paper.

## 9. Key domain facts

- Density bands (persons/m²): below 2 safe · 2–4 crowded · 4–5 high risk · above 5 critical.
- Crowd pressure = density × velocity variance (Helbing). It rises before crowd disasters, together with falling flow.
- Alert thresholds for pressure, entropy and counter-flow are tuned on data (simulation + Jülich), not copied from papers.
- Alerts use hysteresis: must hold N s (default 10) to raise, M s (default 30) to clear.

## 10. Commands (fill in as they're created)

```
# environment
pip install -r requirements.txt

# tests
pytest -q

# run the pipeline on a clip (to be created in T3/T4)
python scripts/run_pipeline.py --config configs/cam01.yaml --video data/cctv/cam01.mp4

# server + dashboard (to be created in T8)
uvicorn server.app:app --reload
cd dashboard && npm run dev
```

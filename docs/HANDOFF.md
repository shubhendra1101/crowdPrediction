# CrowdSafe — Handoff / session context (for any AI assistant)

Paste this whole file into a new chat (ChatGPT, Gemini, Claude, …) together with: *"You are my engineering
partner on this project. Read this handoff, then CLAUDE.md, architecture.md, plan.md and task.md from the repo,
and continue from 'Next steps'."* Repo: https://github.com/shubhendra1101/crowdPrediction (public, branch `main`).
Local folder: `C:\Users\hp\OneDrive\Desktop\CrownPrediction`. Last updated: 2026-09-30 (end of week 1 of 7).

---

## 1. Project in one paragraph
CrowdSafe is a final-year research project + demo (≈7 weeks, one A100 40 GB). It forecasts **crowd density
(persons/m²) and stampede risk per zone, minutes ahead, from fixed CCTV**, and raises tiered alerts
(Low <2, Medium 2–4, High 4–5, Critical >5 persons/m²; escalate on precursors: crowd pressure = density × velocity
variance, flow collapse, counter-flow, direction entropy; hysteresis: hold 10 s to raise, 30 s to clear).
Pipeline: **video → calibration (homography, zones in m²) → counting (YOLO + CLIP-EBC, confidence-weighted fusion =
main contribution) → motion (RAFT optical flow → speed, variance, pressure, entropy, counter-flow, in/outflow) →
feature table (1 row / zone / second) → forecasting (Chronos-2 zero-shot + persistence + physics baselines, split
conformal upper bound) → risk rules → FastAPI WebSocket → Next.js dashboard.** The prediction layer only reads the
feature table, so it is developed on Jülich lab trajectories and JuPedSim simulations, then run on real CCTV.

## 2. Rules the user set (follow them)
- Read `CLAUDE.md` (working rules), `architecture.md`, `plan.md`, `task.md` (source of truth; tick tasks with real results).
- Reuse pretrained models; **never train from scratch**; ask before fine-tuning > 1 h GPU, downloads > 5 GB, changing the stack.
- **Never invent** measurements, frame rates, areas or metrics; label approximations; flag suspicious results.
- Privacy: no face recognition / identity tracking; CCTV frames stay local (`data/` is git-ignored).
- Units: persons/m², m/s, persons/s. Seed 42. Paths/params from YAML or CLI. A smoke test per module.
- Model sourcing: prefer Hugging Face / pip weights; ask before GitHub-only weights.
- Report format after each task: **Done / What this means / I need from you / Next**. The user is a student:
  explain simply, give exact step-by-step instructions, short answers when they ask for "quick".
- Record every change of plan/stack in `docs/decisions.md`. Commit messages end with a Co-Authored-By line.
- **Secrets:** never commit keys. `kaggle.json` sits in the project root (git-ignored). `notebooks/local/` (git-ignored)
  holds notebook builds with the Kaggle key embedded. Before every commit, grep staged files for the key.
  The user pasted the key in chat once → they must rotate it.

## 3. Environment
- **Laptop (dev):** Windows 11, Python 3.11.9, torch 2.0.1 **CPU only**, 7.8 GB RAM (~1 GB free → keep memory low,
  never hold many video frames in RAM), ~34 GB free disk. Git Bash can crash (Win32 error 1455) → use PowerShell.
  Installed: supervision 0.30.6, pyarrow, shapely, pedpy 1.5.1, onnxruntime, kaggle, certifi, pytest.
- **A100 (college Kubeflow JupyterLab):** image `pytorch:25.03-py3_kubeflow` (NVIDIA torch `2.7.0a0+7c8ec84dab.nv25.03`,
  CUDA 12.8, driver 470), A100-SXM4-40GB. Quirks:
  - Sees 256 cores but quota ~8 CPUs / 16 GiB → always cap threads (`crowdsafe.nbenv.limit_threads`) and run heavy
    steps in subprocesses (a kernel died before this fix).
  - pip could replace NVIDIA torch → always install with `crowdsafe.nbenv.safe_pip` (pins torch/numpy).
  - `download.pytorch.org` blocked → RAFT weights from HF mirror `Dominoc/raft_large` (SHA-256 verified).
  - **No persistent volume:** `/root/crowdsafe_work` (data ~23 GB, caches) is lost if the server stops → re-run notebook 02.
  - File browser root is a "do not delete" folder; `/root/crowdsafe_work` is not visible and right-click shows Chrome's
    menu. To download: in a notebook cell `!cp /root/crowdsafe_work/<file>.zip .` then
    `from IPython.display import FileLink; FileLink("<file>.zip")` and click the link.
  - Workflow: the user uploads notebooks from `notebooks/local/`, runs *Kernel → Restart & Run All*, downloads the
    report zips into the laptop's `weights/` folder, says "done"; the assistant unpacks, verifies and records results.
  - Notebook sources are `notebooks/src/*.py` (percent format); build with `python scripts/build_notebooks.py`
    (+ `--embed-kaggle kaggle.json` for the local builds). Each notebook clones/pulls the GitHub repo itself.

## 4. Decisions (docs/decisions.md)
- **D1** Python code lives in package `crowdsafe/` (so `datasets/` cannot shadow HF `datasets`).
- **D2** New project from scratch; old YOLO11m+CSRNet not reused; T0.5 dropped; "hard switch" ablation re-implemented.
- **D3** GPU work = notebooks run by the user on the A100; models returned as ONNX.
- **D4** (user) CLIP-EBC ViT-B/16 **NWPU** weights from HF `Yiming-M/CLIP-EBC` (loading as in its `app.py`); PET-Finetuned
  (HF `Awiros/crowd-counting-and-localization`) replaces APGCC (optional); CrowdHuman detector = HF `Sharath33/Person`.
- **D5** Lab workarounds: RAFT HF mirror; ShanghaiTech HF fallback `KTAEHWA/shanghaitech-crowd-counting` (counts only);
  pip pins NVIDIA torch/numpy.

## 5. What is built (all tested; `pytest -q` → ~50 passing)
| Area | Files |
| --- | --- |
| Env helpers (A100) | `crowdsafe/nbenv.py` (safe_pip, fix_cv2, gpu_summary, cpu_limit, limit_threads, exit_reason, RAFT weights) |
| Datasets | `configs/datasets.yaml`, `crowdsafe/datasets/download.py` (HTTP/Kaggle/HF, SHA-256 manifests, split checks), `scripts/download_data.py`, `crowdsafe/datasets/shanghaitech.py`, `crowdsafe/datasets/julich.py`, `crowdsafe/datasets/cctv.py` |
| Counting | `crowdsafe/counting/yolo_track.py` (YoloDetector + tiling + ByteTrack; PersonHeadOnnx with 3-output decoder; reliability features), `crowdsafe/counting/density_model.py` (CLIP-EBC torch/ONNX, window/whole) |
| Features | `crowdsafe/features/schema.py` (14 columns), `crowdsafe/features/logger.py` (Parquet, SQLite) |
| Forecasting | `crowdsafe/forecasting/baselines.py`, `evaluate.py` (rolling origin, no look-ahead), `conformal.py`, `chronos.py` |
| Eval | `crowdsafe/eval/metrics.py` |
| Scripts | `export_onnx.py`, `bench_counting.py`, `julich_features.py`, `eval_forecast.py`, `camera_audit.py`, `prelabel_cvat.py`, `build_notebooks.py` |
| Notebooks (A100) | 01 env + ONNX export · 02 download datasets · 03 ShanghaiTech counting benchmark · 04 Chronos-2 on Jülich |
| Configs | `configs/cam_template.yaml`, `datasets.yaml`, `julich.yaml`, `forecast_julich.yaml`, `bench_shanghaitech.yaml` |
| Docs | `docs/decisions.md`, `docs/cvat_guide.md`, this file |

## 6. Results so far (real numbers; files in `results/`)
- **ONNX models** (`weights/crowdsafe_nb01_onnx/onnx/`, checksums verified): yolo11l 102 MB (ONNX = PyTorch, IoU 0.9999),
  raft_large 21 MB (recovers 6×3 px shift, diff 1.4e-4 px), clipebc_vitb16_nwpu 389 MB (927.54 vs 927.54),
  yolo11s CrowdHuman person+head 38 MB (bus.jpg: 4 persons, 3 heads). A100 speed: YOLO11l 1080p 15.5 ms, RAFT 544×960 40.8 ms.
- **Datasets on A100** (~23 GB): 8 Jülich experiments (trajectories + videos), ShanghaiTech (Kaggle copy has every file twice;
  one copy used), JHU-Crowd++ 2272/500/1600, Mall 2000, UCSD, ShanghaiTech-HF 300/182/400/316.
- **Zero-shot counting, ShanghaiTech test, MAE Part A / Part B** (`results/counting_zeroshot_shanghaitech.md`):
  CLIP-EBC window **66.95 / 11.66**, whole 66.0 / 11.98 · CrowdHuman heads 374.2 / 70.2, persons 377.1 / 58.8 ·
  YOLO11l 418.3 / 92.8 · YOLO tiled 421.7 / 95.4. Detectors undercount dense crowds; CLIP-EBC best in every band
  (even <50 on B: 1.95 vs 8.95) → fusion benefit must be shown on the user's CCTV (larger heads).
- **Jülich features** (`results/julich_features_summary.md`): 144 runs; 2 m × 2 m zones kept if ≥90% walked on;
  units mixed m/cm (header wins over config), fps 16/25/50; tracking glitches > 4 m/s ignored (≤0.3%); PedPy
  cross-check diff 0.0000. Max density 9.0 (entrance_semicircle), 5.65 (entrance_corridor), 5.3 (corridor_uni_2013);
  643 zone-seconds ≥ 5 p/m². bottleneck1 too narrow for 2 m zones. Runs are short (median 70–177 s).
- **Forecasting on Jülich test** (`results/forecast_julich_a100.md`; test = corridor_uni_2013 + entrance_semicircle, val =
  bottleneck2 + entrance_corridor + corridor_bi_2009; context 30 s; MAE p/m² at 10/30/60 s):
  Chronos-2 per-zone 0.376/0.558/0.727, joint 0.372/0.569/0.741, persistence 0.387/0.568/0.711, physics 1.049/2.507/4.449.
  **Calibrated p90 coverage (target 0.90):** Chronos-2 per-zone 0.892/0.875/0.925, joint 0.861/0.835/0.880,
  persistence 0.850/0.660/0.676, physics 0.798/0.700/0.751 → Chronos-2 ≈ persistence on MAE but only it gives a
  trustworthy upper bound (what alerts use).
- **User CCTV** (classroom "H705 Front", one camera, corner, ~2–2.5 m high, permission granted): `data/cctv/cam01.asf`
  (72 min lecture), `cam02.asf` (15 min: lecture → mass exit at ~min 7 → empty). 1920×1080, 25 fps, h264 ASF
  (**cannot seek → read sequentially**). Audit (`results/camera_audit_cam0*.md`): steady (≤0.64 px/s), well lit,
  heads median 40–53 px (p10 31–41 px) → detectors usable; max 51 persons (cam01), 45 (cam02). No calibration yet.
- **Labelling pack:** 100 frames (cam01 86, cam02 14; 0–55 heads) with 3,562 pre-placed head points →
  `data/cctv/labels/images.zip` + `cvat_prelabels.zip` (CVAT 1.1, label `head`, points); guide `docs/cvat_guide.md`.

## 7. task.md status
Done: T0.1–T0.4, T0.6, T1.1, T1.4, T1.5, T5.1, T5.2, T6.1, T6.2, T6.3 (T0.5 dropped). Partly: T1.2 (1 camera), T3.1–T3.3
(wrappers built/tested), T3.5 (ShanghaiTech done; Jülich-video + CCTV parts pending), T5.5 (Jülich split proposed),
T6.4 (Jülich done; sim + CCTV pending). Open: T1.3, T1.6–T1.7, T2.x, T3.4, T3.6–T3.8, T4.x, T5.3–T5.4, T6.5–T6.6, T7.x, T8.x, T9.x.

## 8. Waiting on the user
1. **CVAT correction** of the 100 frames (T1.6), export "CVAT for images 1.1" into `data/cctv/labels/`, say "labels done".
2. **Floor calibration for H705** (T2.1): 4 tape marks forming a large rectangle on the front aisle, all 4 sides + 1 diagonal
   in metres, a few seconds of CCTV with tape visible (camera unmoved); plus one desk length and the door width (validation).
3. **Venue layout sketch** with exit/door widths (T2.2).
4. **A second camera** (corridor/door/stairs preferred) — needed for a held-out camera test.
5. Approve defaults: 2 m zones for Jülich; the Jülich val/test split above.
6. Rotate the Kaggle key; delete notebook 02 from the A100 server.

## 9. Next steps (in order)
1. **T7 risk engine** (laptop): density bands on the conformal p90, precursor rules (thresholds tuned on val only),
   hysteresis 10 s / 30 s, alert reasons, time-to-critical; replay Jülich test runs → alert lead time before 5 p/m²,
   precision/recall, false alarms per hour.
2. **T2.3 calibration tool** (click points → homography, `ViewTransformer` pattern, zone areas via Shapely) — ready for measurements.
3. **T5.4 JuPedSim**: propose 3 geometries (bottleneck exit, corridor merge, two-way corridor) for user approval; 15–20 runs
   incl. surges/exit closures > 5 p/m²; long runs give the 2–10 min horizons Jülich cannot.
4. After labels: import CVAT export (T1.7), counting error per model on the classroom, **fusion** (T3.7) + ablation
   YOLO / CLIP-EBC / hard switch / fusion (T3.8). Split is by time/video until a second camera exists — say so in results.
5. New A100 notebook when needed: RAFT motion + fused counts on the classroom videos (T4, T5.3); check first whether
   `/root/crowdsafe_work/data/raw` still exists, else re-run notebook 02.
6. T8 backend + dashboard; T9 evaluation and write-up.

## 10. Lessons / pitfalls already hit
- ASF CCTV files silently ignore seeking (returned frame 0 every time) → sequential reads + identical-frame guard.
- Holding 1,000 decoded 1080p frames exhausted laptop RAM → two-pass sampling.
- `Sharath33/Person` ONNX has 3 outputs (boxes cx,cy,w,h; scores; classes), not the usual combined tensor.
- `.gitignore` does not allow inline comments after a pattern.
- Jülich: `readme.txt` inside an archive; one corridor file declares 25 fps vs 16 on the page; PedPy renumbers frames from 0.
- Physics fill-rate baseline is poor on 2 m zones (noisy boundary crossings); conformal coverage drops when val/test geometries differ.
- Red text in Jupyter is usually warnings (CLIP-EBC FutureWarnings, `Unexpected keys: ['proj']` is normal).

## 11. Useful commands
```
pytest -q
python scripts/build_notebooks.py                      # public notebooks
python scripts/build_notebooks.py --embed-kaggle kaggle.json   # notebooks/local/ (git-ignored)
python scripts/julich_features.py                      # Jülich → data/features/julich
python scripts/eval_forecast.py                        # baselines + conformal (laptop)
python scripts/camera_audit.py --videos data/cctv/cam01.asf --phd-onnx weights/crowdsafe_nb01_onnx/onnx/yolo11s_crowdhuman_person_head.onnx
python scripts/prelabel_cvat.py --videos data/cctv/cam01.asf data/cctv/cam02.asf --phd-onnx <same onnx> --n 100
```

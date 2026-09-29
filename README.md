# CrowdSafe

Forecasts crowd density (persons/m²) and stampede risk per zone, minutes ahead, from fixed CCTV cameras, and raises tiered early-warning alerts.

- `CLAUDE.md` — working rules and chosen stack
- `architecture.md` — components, data flow, schemas
- `plan.md` — 7-week plan and fallbacks
- `task.md` — task checklist (source of truth)
- `CrowdSafe Implementation Guide.md` — full step-by-step guide

## Setup

```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pytest
```

`data/`, `weights/` and videos are gitignored; keep them out of git.

## GPU notebooks (college A100, Kubeflow)

GPU work runs as notebooks in `notebooks/`. Edit the sources in `notebooks/src/`, then rebuild:

```
python scripts/build_notebooks.py
```

Upload the `.ipynb` to Kubeflow, run all cells, and return the zips it prints. Returned ONNX models go in `weights/`. Changes to the plan are recorded in `docs/decisions.md`.

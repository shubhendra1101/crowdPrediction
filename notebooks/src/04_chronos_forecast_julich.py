# %% [markdown]
# # CrowdSafe — Notebook 04: Chronos-2 zero-shot forecasting on Jülich (T6.2–T6.4, Jülich part)
#
# **Independent of the other notebooks** — downloads the Jülich trajectories itself if notebook 02 has not.
#
# **What it does**
# 1. Builds the zone feature table (1 row per zone per second) from all Jülich trajectories — same code as the laptop.
# 2. Forecasts zone density 10 / 30 / 60 s ahead, *no training*: persistence, physics fill-rate,
#    **Chronos-2 per zone** (with motion covariates) and **Chronos-2 joint** (all zones together).
# 3. Calibrates each model's upper bound on validation experiments (split conformal) and scores held-out test experiments.
#
# **How to run:** `Kernel → Restart & Run All` (~10–20 min). **Return:** `crowdsafe_nb04_report.zip`.

# %% [markdown]
# ## 1. Settings and project code

# %%
import os
import subprocess
import sys
from pathlib import Path

WORK = Path(os.environ.get("CROWDSAFE_WORK", Path.home() / "crowdsafe_work"))
RAW = Path(os.environ.get("CROWDSAFE_DATA", WORK / "data" / "raw"))
REPO_DIR = WORK / "crowdPrediction"
REPO_URL = "https://github.com/shubhendra1101/crowdPrediction.git"
FEATURES = WORK / "data" / "features" / "julich"
OUT = WORK / "outputs_nb04"
OUT.mkdir(parents=True, exist_ok=True)
if (REPO_DIR / ".git").exists():
    subprocess.run(["git", "-C", str(REPO_DIR), "pull", "--ff-only"], check=True)
else:
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(REPO_DIR)], check=True)
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))
COMMIT = subprocess.run(["git", "-C", str(REPO_DIR), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
os.chdir(REPO_DIR)
print("code commit:", COMMIT)

# %% [markdown]
# ## 2. Dependencies (PyTorch pinned) and GPU check

# %%
from crowdsafe import nbenv

print("CPU threads capped to", nbenv.limit_threads())   # container quota, not the 256 host cores

PKGS = ["chronos-forecasting>=2.0", "accelerate", "pedpy", "shapely", "pyarrow", "pandas", "pyyaml", "scipy",
        "certifi", "huggingface_hub>=0.24"]
print(nbenv.safe_pip(PKGS, log=OUT / "pip_install.log"))
print(nbenv.gpu_summary(require=True))


def run(cmd: list[str], log: str) -> int:
    """Run a project script, save its output, print the tail; return the exit code."""
    r = subprocess.run([sys.executable, *cmd], capture_output=True, text=True)
    (OUT / log).write_text(r.stdout + "\n" + r.stderr)
    print(r.stdout[-3000:], r.stderr[-2000:])
    return r.returncode

# %% [markdown]
# ## 3. Jülich trajectories (downloaded here if notebook 02 has not) and the feature table

# %%
if not any(RAW.glob("julich_*/trajectories")):
    run(["scripts/download_data.py", "--root", str(RAW), "--groups", "must", "--only-source", "julich",
         "--roles", "trajectories", "metadata", "--report", str(OUT / "julich_download.json")], "download_log.txt")
assert any(RAW.glob("julich_*/trajectories")), f"No Jülich trajectories under {RAW}"
code = run(["scripts/julich_features.py", "--raw-root", str(RAW), "--out-root", str(FEATURES),
            "--results", str(OUT / "julich_features_summary.md")], "features_log.txt")
assert code == 0 and any(FEATURES.rglob("*.parquet")), "Feature build failed — see features_log.txt"

# %% [markdown]
# ## 4. Chronos-2 check (one forecast per mode) — decides which modes go into the evaluation

# %%
import traceback

from crowdsafe.features.logger import read_features
from crowdsafe.forecasting.chronos import ChronosForecaster

one = read_features(sorted(FEATURES.rglob("*.parquet"))[0])
hist = one[one["timestamp"] < 30]
MODES_OK, CHECK = [], {}
for mode in ("per_zone", "joint"):
    try:
        fc = ChronosForecaster(mode=mode).predict(hist, [10, 30])
        assert len(fc) and fc[["p10", "p50", "p90"]].notna().all().all()
        MODES_OK.append(f"chronos2_{mode}")
        CHECK[mode] = "ok"
        print(mode, "OK\n", fc.head(4).to_string(index=False))
    except Exception as e:
        CHECK[mode] = traceback.format_exc()
        print(mode, "FAILED:", repr(e)[:500])
(OUT / "chronos_check.txt").write_text("\n\n".join(f"{k}: {v}" for k, v in CHECK.items()))

# %% [markdown]
# ## 5. Full evaluation: models × horizons 10/30/60 s, conformal on val, scored on test

# %%
MODELS = ["persistence", "physics"] + MODES_OK
print("evaluating:", MODELS)
run(["scripts/eval_forecast.py", "--features-root", str(FEATURES), "--models", *MODELS, "--device", "cuda",
     "--out", str(OUT / "forecast_julich.md")], "eval_log.txt")
res = OUT / "forecast_julich.md"
print(res.read_text() if res.exists() else "no results — see eval_log.txt")

# %% [markdown]
# ## 6. Package the report

# %%
import zipfile

report_zip = WORK / "crowdsafe_nb04_report.zip"
with zipfile.ZipFile(report_zip, "w", zipfile.ZIP_DEFLATED) as z:
    for p in OUT.rglob("*"):
        if p.suffix != ".parquet" or p.stat().st_size < 50e6:
            z.write(p, p.relative_to(OUT))
print("Download this file:", report_zip, f"({report_zip.stat().st_size / 1e6:.1f} MB)")

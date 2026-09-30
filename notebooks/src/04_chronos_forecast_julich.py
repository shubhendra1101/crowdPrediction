# %% [markdown]
# # CrowdSafe — Notebook 04: Chronos-2 zero-shot forecasting on Jülich (T6.2–T6.4, Jülich part)
#
# **Run after notebook 02** (needs the Jülich trajectories it downloaded). Independent of notebooks 01 and 03.
#
# **What it does**
# 1. Builds the zone feature table (1 row per zone per second) from all Jülich trajectories — same code as on the laptop.
# 2. Forecasts zone density 10 / 30 / 60 s ahead with four models, *no training*:
#    persistence, physics fill-rate, **Chronos-2 per zone** (with motion covariates), **Chronos-2 joint** (all zones together).
# 3. Calibrates each model's upper bound on the validation experiments (split conformal) and scores the held-out test experiments.
#
# **Zero-shot** means Chronos-2 was pretrained on a large collection of other time series and is used as-is on crowd data.
#
# **How to run:** `Kernel → Restart & Run All` (~10–20 min). **Return:** `crowdsafe_nb04_report.zip`.

# %% [markdown]
# ## 1. Settings

# %%
import os
from pathlib import Path

WORK = Path(os.environ.get("CROWDSAFE_WORK", Path.home() / "crowdsafe_work"))
REPO_DIR = WORK / "crowdPrediction"
REPO_URL = "https://github.com/shubhendra1101/crowdPrediction.git"
RAW = Path(os.environ.get("CROWDSAFE_DATA", WORK / "data" / "raw"))
FEATURES = WORK / "data" / "features" / "julich"
OUT = WORK / "outputs_nb04"
MODELS = ["persistence", "physics", "chronos2_per_zone", "chronos2_joint"]
OUT.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## 2. Latest code + dependencies

# %%
import subprocess
import sys

if (REPO_DIR / ".git").exists():
    subprocess.run(["git", "-C", str(REPO_DIR), "pull", "--ff-only"], check=True)
else:
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(REPO_DIR)], check=True)
COMMIT = subprocess.run(["git", "-C", str(REPO_DIR), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
PKGS = ["chronos-forecasting>=2.0", "pedpy", "shapely", "pyarrow", "pandas", "pyyaml", "scipy", "supervision==0.30.6", "pytest"]
res = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *PKGS], capture_output=True, text=True)
print("commit", COMMIT, "| pip exit", res.returncode, res.stderr[-800:])
os.chdir(REPO_DIR)


def run(cmd: list[str], log: str) -> None:
    """Run a project script, save its output to OUT/log, print the tail."""
    r = subprocess.run([sys.executable, *cmd], capture_output=True, text=True)
    (OUT / log).write_text(r.stdout + "\n" + r.stderr)
    print(r.stdout[-3000:], r.stderr[-2000:])

# %% [markdown]
# ## 3. Build the Jülich feature table

# %%
assert any(RAW.glob("julich_*/trajectories")), f"No Jülich trajectories under {RAW} — run notebook 02 first"
run(["scripts/julich_features.py", "--raw-root", str(RAW), "--out-root", str(FEATURES),
     "--results", str(OUT / "julich_features_summary.md")], "features_log.txt")

# %% [markdown]
# ## 4. Quick Chronos-2 check (one forecast) — fails fast if the package API differs

# %%
import pandas as pd

sys.path.insert(0, str(REPO_DIR))
from crowdsafe.features.logger import read_features
from crowdsafe.forecasting.chronos import ChronosForecaster

one = read_features(next(FEATURES.rglob("*.parquet")))
hist = one[one["timestamp"] < 30]
for mode in ("per_zone", "joint"):
    fc = ChronosForecaster(mode=mode).predict(hist, [10, 30])
    print(mode, fc.head(4).to_string(index=False))

# %% [markdown]
# ## 5. Full evaluation: 4 models × 3 horizons, conformal on val, scored on test

# %%
run(["scripts/eval_forecast.py", "--features-root", str(FEATURES), "--models", *MODELS, "--device", "cuda",
     "--out", str(OUT / "forecast_julich.md")], "eval_log.txt")
print((OUT / "forecast_julich.md").read_text() if (OUT / "forecast_julich.md").exists() else "no results")

# %% [markdown]
# ## 6. Package the report

# %%
import zipfile

report_zip = WORK / "crowdsafe_nb04_report.zip"
with zipfile.ZipFile(report_zip, "w", zipfile.ZIP_DEFLATED) as z:
    for p in OUT.rglob("*"):
        z.write(p, p.relative_to(OUT))
print("Return this file to the agent:", report_zip, f"({report_zip.stat().st_size / 1e6:.1f} MB)")

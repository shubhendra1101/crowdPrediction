# %% [markdown]
# # CrowdSafe — Notebook 05: CCTV zone features + Chronos-2 on simulations (T4, T5.3, T6.4)
#
# **What it does**
# 1. For each uploaded CCTV video: every second, CLIP-EBC density → **persons/m² per zone** (calibrated cameras,
#    `configs/cameras/`, `configs/zones/`) and RAFT optical flow → **walking speed (m/s) and motion features per zone**.
#    Saves progress every 5 min of video — if the kernel dies, just run the cells again; it continues.
# 2. Re-creates the 16 JuPedSim simulations (identical to the laptop: fixed seed) and forecasts them with
#    persistence, physics and **Chronos-2** (no training).
# 3. Forecasts the CCTV features (persistence + Chronos-2) for the alert replay.
#
# **Before running:** upload the CCTV videos `cam01.asf … cam04.asf` (from the laptop's `data/cctv/`) into
# `~/crowdsafe_work/data/cctv/`. Missing videos are skipped (the rest still runs).
#
# **How to run:** `Kernel → Restart & Run All`. **Return:** `crowdsafe_nb05_report.zip` (download with the link at the end).

# %% [markdown]
# ## 1. Settings and project code

# %%
import os
import subprocess
import sys
from pathlib import Path

WORK = Path(os.environ.get("CROWDSAFE_WORK", Path.home() / "crowdsafe_work"))
VIDEOS = WORK / "data" / "cctv"
REPO_DIR = WORK / "crowdPrediction"
REPO_URL = "https://github.com/shubhendra1101/crowdPrediction.git"
OUT = WORK / "outputs_nb05"
CCTV_FEATURES = OUT / "features_cctv"            # inside OUT so it is packaged and survives in the zip
SIM_FEATURES = REPO_DIR / "data" / "features" / "sim"   # where scripts/run_simulations.py writes (configs/simulation.yaml)
CLIPEBC_REPO = WORK / "cache" / "CLIP-EBC-hf"
for p in (VIDEOS, OUT, CCTV_FEATURES):
    p.mkdir(parents=True, exist_ok=True)
if (REPO_DIR / ".git").exists():
    subprocess.run(["git", "-C", str(REPO_DIR), "pull", "--ff-only"], check=True)
else:
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(REPO_DIR)], check=True)
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))
COMMIT = subprocess.run(["git", "-C", str(REPO_DIR), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
os.chdir(REPO_DIR)
(OUT / "code_commit.txt").write_text(COMMIT)
print("code commit:", COMMIT)

# %% [markdown]
# ## 2. Dependencies (PyTorch pinned), GPU check, model weights

# %%
from crowdsafe import nbenv

print("CPU threads capped to", nbenv.limit_threads())   # container quota, not the 256 host cores

PKGS = ["chronos-forecasting>=2.0", "accelerate", "jupedsim==1.4.2", "pedpy", "shapely>=2.0", "huggingface_hub>=0.24",
        "safetensors", "einops", "ftfy", "regex", "timm==0.9.16", "tensorboardX", "scipy", "pandas", "pyarrow", "pyyaml",
        "matplotlib", "supervision==0.30.6", "certifi"]
print(nbenv.safe_pip(PKGS, log=OUT / "pip_install.log"))
print(nbenv.fix_cv2())
print(nbenv.gpu_summary(require=True))
print("RAFT weights:", nbenv.ensure_raft_large_weights())   # HF mirror, SHA-256 checked (download.pytorch.org is blocked)

from huggingface_hub import snapshot_download

if not (CLIPEBC_REPO / "nwpu_weights" / "CLIP_EBC_ViT_B_16" / "model.safetensors").exists():
    snapshot_download("Yiming-M/CLIP-EBC", ignore_patterns=["nwpu_weights/CLIP_EBC_ViT_L_14/*"], local_dir=CLIPEBC_REPO)
print("CLIP-EBC ready:", (CLIPEBC_REPO / "nwpu_weights" / "CLIP_EBC_ViT_B_16" / "model.safetensors").exists())


def run(cmd: list[str], log: str) -> int:
    """Run a project script, save its output, print the tail; return the exit code."""
    r = subprocess.run([sys.executable, *cmd], capture_output=True, text=True)
    (OUT / log).write_text(r.stdout + "\n" + r.stderr)
    print(r.stdout[-3000:], r.stderr[-2000:])
    return r.returncode

# %% [markdown]
# ## 3. CCTV videos → zone features (≈ a few minutes per video on the A100; resumable)

# %%
CAMS = ["cam01", "cam02", "cam03", "cam04"]
found = {c: next((p for p in VIDEOS.glob(f"{c}.*") if p.suffix.lower() in (".asf", ".mp4", ".avi", ".mkv")), None) for c in CAMS}
for c, p in found.items():
    print(f"{c}: {p.name} ({p.stat().st_size / 1e9:.2f} GB)" if p else f"{c}: NOT UPLOADED — put it in {VIDEOS}")
specs = [f"{c}={p}" for c, p in found.items() if p]
if specs:
    code = run(["scripts/cctv_features.py", "--videos", *specs, "--out-dir", str(CCTV_FEATURES),
                "--clipebc-repo", str(CLIPEBC_REPO), "--device", "cuda", "--batch", "64"], "cctv_features_log.txt")
    print("exit code", code, "| finished:", sorted(p.name for p in CCTV_FEATURES.glob("cam0?.parquet")))

# %% [markdown]
# ## 4. Simulations (CPU, ~10 min) — same scenarios and seed as the laptop

# %%
code = run(["scripts/run_simulations.py", "--out", str(OUT / "simulation_summary.md")], "simulation_log.txt")
assert code == 0 and any(SIM_FEATURES.rglob("*.parquet")), "simulation failed — see simulation_log.txt"

# %% [markdown]
# ## 5. Chronos-2 check, then forecasts on the simulations (persistence, physics, Chronos-2)

# %%
import traceback

from crowdsafe.features.logger import read_features
from crowdsafe.forecasting.chronos import ChronosForecaster

one = read_features(sorted(SIM_FEATURES.rglob("corr_2p5.parquet")))
hist = one[one["timestamp"] < 60]
MODES_OK, CHECK = [], {}
for mode in ("per_zone", "joint"):
    try:
        fc = ChronosForecaster(mode=mode).predict(hist, [10, 30])
        assert len(fc) and fc[["p10", "p50", "p90"]].notna().all().all()
        MODES_OK.append(f"chronos2_{mode}")
        CHECK[mode] = "ok"
    except Exception:
        CHECK[mode] = traceback.format_exc()
        print(mode, "FAILED:", CHECK[mode][-800:])
(OUT / "chronos_check.txt").write_text("\n\n".join(f"{k}: {v}" for k, v in CHECK.items()))
print("Chronos-2 modes OK:", MODES_OK)
run(["scripts/eval_forecast.py", "--config", "configs/forecast_sim.yaml", "--features-root", str(SIM_FEATURES),
     "--models", "persistence", "physics", *MODES_OK, "--device", "cuda", "--out", str(OUT / "forecast_sim.md")], "forecast_sim_log.txt")
print((OUT / "forecast_sim.md").read_text() if (OUT / "forecast_sim.md").exists() else "no results — see forecast_sim_log.txt")

# %% [markdown]
# ## 6. Forecasts on the CCTV features (persistence + Chronos-2; needs at least one val and one test camera)

# %%
have = {p.stem for p in CCTV_FEATURES.glob("cam0?.parquet")}
if {"cam01", "cam03"} & have and {"cam02", "cam04"} & have:
    run(["scripts/eval_forecast.py", "--config", "configs/forecast_cctv.yaml", "--features-root", str(CCTV_FEATURES),
         "--models", "persistence", *MODES_OK, "--device", "cuda", "--out", str(OUT / "forecast_cctv.md")], "forecast_cctv_log.txt")
    print((OUT / "forecast_cctv.md").read_text() if (OUT / "forecast_cctv.md").exists() else "see forecast_cctv_log.txt")
else:
    print("skipped: CCTV features for", sorted(have), "— need one of cam01/cam03 and one of cam02/cam04")

# %% [markdown]
# ## 7. Package the report (features, forecasts, logs — no videos)

# %%
import zipfile

from IPython.display import FileLink, display

report_zip = WORK / "crowdsafe_nb05_report.zip"
with zipfile.ZipFile(report_zip, "w", zipfile.ZIP_DEFLATED) as z:
    for p in OUT.rglob("*"):
        if p.is_file() and (p.suffix != ".parquet" or p.stat().st_size < 200e6):
            z.write(p, p.relative_to(OUT))
print(f"{report_zip} ({report_zip.stat().st_size / 1e6:.1f} MB)")
# Jupyter can only link files below the notebook's folder: copy the zip next to the notebook, then click.
subprocess.run(["cp", str(report_zip), "."], check=False)
display(FileLink(report_zip.name))

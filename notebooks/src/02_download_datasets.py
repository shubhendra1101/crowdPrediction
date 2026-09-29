# %% [markdown]
# # CrowdSafe — Notebook 02: download datasets onto the A100 (T1.1)
#
# **What it does**
# 1. Gets the CrowdSafe code from GitHub (public repo) so the same downloader runs here and on the laptop.
# 2. Checks free disk and Kaggle credentials.
# 3. Downloads every dataset in `configs/datasets.yaml` for the groups below — Jülich trajectories **and videos**,
#    ShanghaiTech A/B, JHU-Crowd++, Mall, UCSD — into a persistent folder.
# 4. Checks file counts against the official split sizes and writes a manifest per dataset.
#
# Data **stays on the A100** (later notebooks read it from `DATA_ROOT`). Only a small report is returned.
#
# **Before running**
# - Kaggle key on this machine: upload your `kaggle.json` to `~/.kaggle/kaggle.json` using the Jupyter file browser
#   (enable hidden files, or run the helper cell below after uploading it to your home folder). Never paste the key into a cell.
#
# **How to run:** `Kernel → Restart & Run All`. Expected: ~12 GB download, ~25 GB disk with extracted copies,
# 20–60 min depending on the college network. Safe to re-run: finished files are skipped.
#
# **Return:** `crowdsafe_nb02_report.zip` (small) — printed by the last cell.

# %% [markdown]
# ## 1. Settings

# %%
import os
from pathlib import Path

WORK = Path(os.environ.get("CROWDSAFE_WORK", Path.home() / "crowdsafe_work"))
DATA_ROOT = Path(os.environ.get("CROWDSAFE_DATA", WORK / "data" / "raw"))   # must be on a persistent volume
REPO_URL = "https://github.com/shubhendra1101/crowdPrediction.git"
REPO_DIR = WORK / "crowdPrediction"
GROUPS = ["must", "should"]          # configs/datasets.yaml groups to fetch
ROLES = None                         # None = everything (trajectories, metadata, videos, images)
MIN_FREE_GB = 40                     # stop before downloading if less free space than this
OUT = WORK / "outputs_nb02"
for d in (DATA_ROOT, OUT):
    d.mkdir(parents=True, exist_ok=True)
print("Data root:", DATA_ROOT)

# %% [markdown]
# ## 2. Get the code and install the downloader's dependencies

# %%
import subprocess
import sys

if (REPO_DIR / ".git").exists():
    subprocess.run(["git", "-C", str(REPO_DIR), "pull", "--ff-only"], check=True)
else:
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(REPO_DIR)], check=True)
commit = subprocess.run(["git", "-C", str(REPO_DIR), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
res = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pyyaml", "kaggle", "certifi"],
                     capture_output=True, text=True)
print("repo commit:", commit, "| pip exit:", res.returncode, res.stderr[-1000:])
sys.path.insert(0, str(REPO_DIR))

# %% [markdown]
# ## 3. Pre-flight: disk and Kaggle key
# If `kaggle.json` is in your home folder but not in `~/.kaggle`, this moves it there and sets permissions.

# %%
import shutil

home_key, kaggle_key = Path.home() / "kaggle.json", Path.home() / ".kaggle" / "kaggle.json"
if home_key.exists() and not kaggle_key.exists():
    kaggle_key.parent.mkdir(exist_ok=True)
    shutil.move(str(home_key), kaggle_key)
if kaggle_key.exists():
    kaggle_key.chmod(0o600)
free_gb = shutil.disk_usage(DATA_ROOT).free / 1e9
preflight = {"free_gb": round(free_gb, 1), "kaggle_key_present": kaggle_key.exists() or bool(os.environ.get("KAGGLE_KEY")),
             "repo_commit": commit}
print(preflight)
assert free_gb >= MIN_FREE_GB, f"Only {free_gb:.0f} GB free at {DATA_ROOT}; need {MIN_FREE_GB} GB. Set CROWDSAFE_DATA to a bigger volume."
if not preflight["kaggle_key_present"]:
    print("WARNING: no Kaggle key — ShanghaiTech, JHU and UCSD will fail; Jülich and Mall still download.")

# %% [markdown]
# ## 4. Download
# One dataset at a time; a failure is recorded and the next dataset continues.

# %%
import json
from dataclasses import asdict

from crowdsafe.datasets.download import fetch_dataset, load_registry, select

reg = load_registry(REPO_DIR / "configs" / "datasets.yaml")
results = []
for name, entry in select(reg, GROUPS).items():
    print(f"--> {name}", flush=True)
    r = fetch_dataset(name, entry, DATA_ROOT, ROLES, reg["max_single_download_gb"])
    print("   ", r.status, r.error or "", {p: f"{v['found']}/{v['expected']}" for p, v in r.expect.items()})
    results.append(asdict(r))

# %% [markdown]
# ## 5. Summary and report zip

# %%
import time
import zipfile

def du_gb(p: Path) -> float:
    """Disk use of a folder in GB."""
    return round(sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e9, 2)

summary = {"finished": time.strftime("%Y-%m-%d %H:%M:%S"), "data_root": str(DATA_ROOT), "preflight": preflight,
           "datasets": {r["name"]: {"status": r["status"], "error": r["error"],
                                    "expect": r["expect"], "disk_gb": du_gb(DATA_ROOT / r["name"]) if (DATA_ROOT / r["name"]).exists() else 0}
                        for r in results},
           "free_gb_after": round(shutil.disk_usage(DATA_ROOT).free / 1e9, 1)}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
(OUT / "results.json").write_text(json.dumps(results, indent=2, default=str))

lines = ["| Dataset | Status | GB | Count checks |", "| --- | --- | --- | --- |"]
for n, d in summary["datasets"].items():
    checks = "; ".join(f"{v['found']}/{v['expected']}" for v in d["expect"].values()) or "-"
    lines.append(f"| {n} | {d['status']} | {d['disk_gb']} | {checks} |")
(OUT / "summary.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))

report_zip = WORK / "crowdsafe_nb02_report.zip"
with zipfile.ZipFile(report_zip, "w", zipfile.ZIP_DEFLATED) as z:
    for p in OUT.rglob("*"):
        z.write(p, p.relative_to(OUT))
    for m in DATA_ROOT.glob("*/manifest.json"):
        z.write(m, Path("manifests") / f"{m.parent.name}.json")
print("\nReturn this file to the agent:", report_zip, f"({report_zip.stat().st_size / 1e3:.0f} KB)")

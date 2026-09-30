# %% [markdown]
# # CrowdSafe — Notebook 02: download datasets onto the A100 (T1.1)
#
# **What it does**
# 1. Gets the project code and installs the downloader's dependencies (PyTorch untouched).
# 2. Downloads every dataset in `configs/datasets.yaml` (groups *must* + *should*): Jülich trajectories **and
#    videos**, ShanghaiTech A/B, JHU-Crowd++, Mall, UCSD — into `DATA_ROOT`.
# 3. If Kaggle is blocked or has no key, gets **ShanghaiTech from Hugging Face** instead (counts only, same splits).
# 4. Checks file counts against the official split sizes; writes a manifest per dataset.
#
# **Storage:** this server has no persistent volume — data lives until the server is deleted. Re-running is safe:
# finished files are skipped.
#
# **Kaggle key (optional):** upload `kaggle.json` to your home folder with the file browser; it is moved into
# place automatically. Never paste the key into a cell. Without it, JHU and UCSD are skipped and
# ShanghaiTech comes from Hugging Face.
#
# **How to run:** `Kernel → Restart & Run All` (20–60 min, ~12 GB). **Return:** `crowdsafe_nb02_report.zip`.

# %% [markdown]
# ## 1. Settings and project code

# %%
import os
import subprocess
import sys
from pathlib import Path

WORK = Path(os.environ.get("CROWDSAFE_WORK", Path.home() / "crowdsafe_work"))
DATA_ROOT = Path(os.environ.get("CROWDSAFE_DATA", WORK / "data" / "raw"))
REPO_DIR = WORK / "crowdPrediction"
REPO_URL = "https://github.com/shubhendra1101/crowdPrediction.git"
GROUPS = ["must", "should"]
ROLES = None                        # None = everything
MIN_FREE_GB = 40
OUT = WORK / "outputs_nb02"
for d in (DATA_ROOT, OUT):
    d.mkdir(parents=True, exist_ok=True)
if (REPO_DIR / ".git").exists():
    subprocess.run(["git", "-C", str(REPO_DIR), "pull", "--ff-only"], check=True)
else:
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(REPO_DIR)], check=True)
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))
COMMIT = subprocess.run(["git", "-C", str(REPO_DIR), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
print("Data root:", DATA_ROOT, "| code commit:", COMMIT)

# %% [markdown]
# ## 2. Dependencies (PyTorch pinned)

# %%
from crowdsafe import nbenv

PIP = nbenv.safe_pip(["pyyaml", "kaggle", "certifi", "huggingface_hub>=0.24", "pyarrow", "pandas", "scipy"],
                     log=OUT / "pip_install.log")
print(PIP)

# %% [markdown]
# ## 3. Pre-flight: disk and Kaggle key

# %%
import shutil

home_key, kaggle_key = Path.home() / "kaggle.json", Path.home() / ".kaggle" / "kaggle.json"
if home_key.exists() and not kaggle_key.exists():
    kaggle_key.parent.mkdir(exist_ok=True)
    shutil.move(str(home_key), kaggle_key)
if kaggle_key.exists():
    kaggle_key.chmod(0o600)
free_gb = shutil.disk_usage(DATA_ROOT).free / 1e9
HAVE_KAGGLE = kaggle_key.exists() or bool(os.environ.get("KAGGLE_KEY")) or bool(os.environ.get("KAGGLE_API_TOKEN"))
preflight = {"free_gb": round(free_gb, 1), "kaggle_key_present": HAVE_KAGGLE, "repo_commit": COMMIT}
print(preflight)
assert free_gb >= MIN_FREE_GB, f"Only {free_gb:.0f} GB free at {DATA_ROOT}; need {MIN_FREE_GB} GB."

# %% [markdown]
# ## 4. Download (one dataset at a time; failures are recorded, the rest continue)

# %%
import json
from dataclasses import asdict

from crowdsafe.datasets.download import fetch_dataset, load_registry, select

reg = load_registry(REPO_DIR / "configs" / "datasets.yaml")
results = []
for name, entry in select(reg, GROUPS).items():
    if entry["source"] == "kaggle" and not HAVE_KAGGLE:
        results.append({"name": name, "status": "skipped", "error": "no Kaggle key", "files": [], "expect": {}})
        print(f"--> {name}: skipped (no Kaggle key)")
        continue
    print(f"--> {name}", flush=True)
    r = fetch_dataset(name, entry, DATA_ROOT, ROLES, reg["max_single_download_gb"])
    print("   ", r.status, r.error or "", {p: f"{v['found']}/{v['expected']}" for p, v in r.expect.items()})
    results.append(asdict(r))

# %% [markdown]
# ## 5. ShanghaiTech fallback from Hugging Face (only if the Kaggle copy is missing or incomplete)

# %%
sht = next((r for r in results if r["name"] == "shanghaitech"), None)
sht_ok = bool(sht) and sht["status"] == "ok" and sht["expect"] and all(v["match"] for v in sht["expect"].values())
if sht_ok:
    print("ShanghaiTech from Kaggle is complete — fallback not needed.")
else:
    print("Kaggle ShanghaiTech missing/incomplete -> downloading the Hugging Face copy")
    r = fetch_dataset("shanghaitech_hf", reg["datasets"]["shanghaitech_hf"], DATA_ROOT, None, reg["max_single_download_gb"])
    print("   ", r.status, r.error or "", {p: f"{v['found']}/{v['expected']}" for p, v in r.expect.items()})
    results.append(asdict(r))

# %% [markdown]
# ## 6. Summary and report zip

# %%
import time
import zipfile


def du_gb(p: Path) -> float:
    return round(sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e9, 2) if p.exists() else 0.0


summary = {"finished": time.strftime("%Y-%m-%d %H:%M:%S"), "data_root": str(DATA_ROOT), "preflight": preflight,
           "datasets": {r["name"]: {"status": r["status"], "error": r["error"], "expect": r["expect"],
                                    "disk_gb": du_gb(DATA_ROOT / r["name"])} for r in results},
           "free_gb_after": round(shutil.disk_usage(DATA_ROOT).free / 1e9, 1)}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
(OUT / "results.json").write_text(json.dumps(results, indent=2, default=str))
lines = ["| Dataset | Status | GB | Count checks | Error |", "| --- | --- | --- | --- | --- |"]
for n, d in summary["datasets"].items():
    checks = "; ".join(f"{v['found']}/{v['expected']}" for v in d["expect"].values()) or "-"
    lines.append(f"| {n} | {d['status']} | {d['disk_gb']} | {checks} | {(d['error'] or '')[:120]} |")
(OUT / "summary.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))

report_zip = WORK / "crowdsafe_nb02_report.zip"
with zipfile.ZipFile(report_zip, "w", zipfile.ZIP_DEFLATED) as z:
    for p in OUT.rglob("*"):
        z.write(p, p.relative_to(OUT))
    for m in DATA_ROOT.glob("*/manifest.json"):
        z.write(m, Path("manifests") / f"{m.parent.name}.json")
print("\nDownload this file:", report_zip, f"({report_zip.stat().st_size / 1e3:.0f} KB)")

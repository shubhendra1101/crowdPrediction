# %% [markdown]
# # CrowdSafe — Notebook 06: live-pipeline latency on the A100 (T9.2)
#
# **What it does:** plays `cam02.asf` **at real speed**, as if it were a live camera, through the full pipeline
# (CLIP-EBC density → RAFT motion → zones in persons/m² → Chronos-2 forecast every 5 s → risk → alerts) and times
# every step. Same models and settings as the results so far; only the real-time loop is new.
#
# Three runs:
# 1. **Live, Chronos-2** — 10 min of video at real speed (main result).
# 2. **Live, persistence forecast** — 5 min, shows what the forecaster costs.
# 3. **Maximum throughput** — 5 min read as fast as possible, shows how many cameras one A100 could serve.
#
# **Before running:** `cam02.asf` must be in `crowdsafe_work1/data/cctv/` (it is still there from notebook 05 unless the
# server was reset; otherwise upload it again, 0.5 GB). **How to run:** `Kernel → Restart Kernel and Run All Cells`
# (~25–30 min). **Return:** `crowdsafe_nb06_report.zip` — the link at the end, or select it in the file browser → Download.

# %% [markdown]
# ## 1. Settings and project code

# %%
import os
import subprocess
import sys
from pathlib import Path

NB_DIR = Path.cwd()                                  # the folder shown in the Jupyter file browser
_c = [Path(os.environ["CROWDSAFE_WORK"])] if "CROWDSAFE_WORK" in os.environ else []
_c += [NB_DIR / "crowdsafe_work1", Path.home() / "crowdsafe_work1"]
WORK = next((c.resolve() for c in _c if (c / "data" / "cctv").exists()), _c[0].resolve())
VIDEO = WORK / "data" / "cctv" / "cam02.asf"
REPO_DIR = WORK / "crowdPrediction"
REPO_URL = "https://github.com/shubhendra1101/crowdPrediction.git"
OUT = WORK / "outputs_nb06"
CLIPEBC_REPO = WORK / "cache" / "CLIP-EBC-hf"
OUT.mkdir(parents=True, exist_ok=True)
if (REPO_DIR / ".git").exists():
    subprocess.run(["git", "-C", str(REPO_DIR), "pull", "--ff-only"], check=True)
else:
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(REPO_DIR)], check=True)
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))
COMMIT = subprocess.run(["git", "-C", str(REPO_DIR), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
os.chdir(REPO_DIR)
(OUT / "code_commit.txt").write_text(COMMIT)
print("work folder:", WORK, "| code commit:", COMMIT)
print("video:", VIDEO, "OK" if VIDEO.exists() else "MISSING — upload cam02.asf into crowdsafe_work1/data/cctv/")

# %% [markdown]
# ## 2. Dependencies (order matters on this server: install, then fix huggingface_hub, then make Chronos-2 importable)

# %%
from crowdsafe import nbenv

print("CPU threads capped to", nbenv.limit_threads())
PKGS = ["chronos-forecasting>=2.0", "accelerate", "av", "shapely>=2.0", "huggingface_hub>=0.24", "safetensors", "einops",
        "ftfy", "regex", "timm==0.9.16", "tensorboardX", "scipy", "pandas", "pyarrow", "pyyaml", "supervision==0.30.6", "certifi"]
print(nbenv.safe_pip(PKGS, log=OUT / "pip_install.log"))
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--force-reinstall", "--no-deps", "huggingface_hub>=0.34,<1.0"])
print(nbenv.ensure_chronos_importable())
print(nbenv.fix_cv2())
print(nbenv.gpu_summary(require=True))
print("RAFT weights:", nbenv.ensure_raft_large_weights())
from huggingface_hub import snapshot_download

if not (CLIPEBC_REPO / "nwpu_weights" / "CLIP_EBC_ViT_B_16" / "model.safetensors").exists():
    snapshot_download("Yiming-M/CLIP-EBC", ignore_patterns=["nwpu_weights/CLIP_EBC_ViT_L_14/*"], local_dir=CLIPEBC_REPO)
print("CLIP-EBC ready:", (CLIPEBC_REPO / "nwpu_weights" / "CLIP_EBC_ViT_B_16" / "model.safetensors").exists())


def run(args: list[str], log: str) -> int:
    """Run the live pipeline, save its output, print the tail; return the exit code."""
    r = subprocess.run([sys.executable, "scripts/live_pipeline.py", "--source", str(VIDEO), "--camera", "cam02",
                        "--clipebc-repo", str(CLIPEBC_REPO), "--device", "cuda", *args], capture_output=True, text=True)
    (OUT / log).write_text(r.stdout + "\n" + r.stderr)
    print(r.stdout[-3500:], r.stderr[-1500:])
    return r.returncode

# %% [markdown]
# ## 3. Run 1 — live at real speed, Chronos-2 forecaster (10 min)

# %%
run(["--forecaster", "chronos2_joint", "--duration-s", "600", "--db", str(OUT / "live_chronos.sqlite"),
     "--out", str(OUT / "latency_live_chronos.md")], "run1_log.txt")

# %% [markdown]
# ## 4. Run 2 — live at real speed, persistence forecaster (5 min)

# %%
run(["--forecaster", "persistence", "--duration-s", "300", "--db", str(OUT / "live_persistence.sqlite"),
     "--out", str(OUT / "latency_live_persistence.md")], "run2_log.txt")

# %% [markdown]
# ## 5. Run 3 — maximum throughput (not paced; how many video-seconds per real second)

# %%
run(["--forecaster", "chronos2_joint", "--no-realtime", "--duration-s", "300", "--db", str(OUT / "live_throughput.sqlite"),
     "--out", str(OUT / "latency_throughput.md")], "run3_log.txt")

# %% [markdown]
# ## 6. Package the report and make the download link

# %%
import shutil
import zipfile

from IPython.display import FileLink, display

report_zip = WORK / "crowdsafe_nb06_report.zip"
with zipfile.ZipFile(report_zip, "w", zipfile.ZIP_DEFLATED) as z:
    for p in OUT.rglob("*"):
        if p.is_file():
            z.write(p, p.relative_to(OUT))
shutil.copy(report_zip, NB_DIR / report_zip.name)      # links only work for files in the notebook's own folder
os.chdir(NB_DIR)
print(f"{report_zip.name}: {report_zip.stat().st_size / 1e6:.1f} MB")
display(FileLink(report_zip.name))

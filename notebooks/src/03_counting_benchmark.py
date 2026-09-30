# %% [markdown]
# # CrowdSafe — Notebook 03: zero-shot counting benchmark on ShanghaiTech (T3.5, part 1)
#
# **Run after notebooks 01 and 02** (it uses the CLIP-EBC download and CrowdHuman ONNX from 01, and ShanghaiTech from 02).
#
# **What it does:** counts people in the 182 + 316 ShanghaiTech test images with each model, *without any training*,
# and compares with the hand-annotated head counts:
#
# | Column | Model |
# | --- | --- |
# | `yolo11l_1280` | YOLO11l (COCO), full image at 1280 px |
# | `yolo11l_tiled640` | YOLO11l on 640 px tiles (`sv.InferenceSlicer`) — helps small heads |
# | `phd_heads` / `phd_persons` | CrowdHuman YOLO11s person+head (ONNX), head and person counts |
# | `clipebc_window` | CLIP-EBC ViT-B/16 NWPU, 224 px windows (repo evaluation style) |
# | `clipebc_whole` | CLIP-EBC, whole image in one pass (`app.py` style) |
#
# Output: MAE / RMSE per part and per count band, per-image predictions (also reused for the fusion, T3.7), speed.
#
# **How to run:** `Kernel → Restart & Run All` (~10–25 min). **Return:** `crowdsafe_nb03_report.zip`.

# %% [markdown]
# ## 1. Settings

# %%
import os
from pathlib import Path

WORK = Path(os.environ.get("CROWDSAFE_WORK", Path.home() / "crowdsafe_work"))
REPO_DIR = WORK / "crowdPrediction"
REPO_URL = "https://github.com/shubhendra1101/crowdPrediction.git"
DATA_SHT = Path(os.environ.get("CROWDSAFE_DATA", WORK / "data" / "raw")) / "shanghaitech"
CLIPEBC_REPO = WORK / "cache" / "CLIP-EBC-hf"          # downloaded by notebook 01
ONNX_DIR = WORK / "outputs" / "onnx"                   # written by notebook 01
OUT = WORK / "outputs_nb03"
DEVICE = "0"                                            # GPU index
SMOKE_IMAGES = 3                                        # quick trial before the full run
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
PKGS = ["ultralytics>=8.3", "supervision==0.30.6", "onnxruntime>=1.18", "huggingface_hub>=0.24", "safetensors",
        "einops", "ftfy", "regex", "timm==0.9.16", "scipy", "pandas", "pyyaml", "pytest"]
res = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *PKGS], capture_output=True, text=True)
print("commit", COMMIT, "| pip exit", res.returncode, res.stderr[-800:])
os.chdir(REPO_DIR)

# %% [markdown]
# ## 3. Check inputs and run the project's tests on this machine

# %%
from huggingface_hub import snapshot_download

if not (CLIPEBC_REPO / "nwpu_weights" / "CLIP_EBC_ViT_B_16" / "model.safetensors").exists():
    snapshot_download("Yiming-M/CLIP-EBC", ignore_patterns=["nwpu_weights/CLIP_EBC_ViT_L_14/*"], local_dir=CLIPEBC_REPO)
checks = {"shanghaitech": DATA_SHT.exists(), "phd_onnx": (ONNX_DIR / "yolo11s_crowdhuman_person_head.onnx").exists(),
          "clipebc": (CLIPEBC_REPO / "models").exists()}
print(checks)
assert checks["shanghaitech"], f"ShanghaiTech not found at {DATA_SHT} — run notebook 02 first"
t = subprocess.run([sys.executable, "-m", "pytest", "-q"], capture_output=True, text=True)
(OUT / "pytest.txt").write_text(t.stdout + t.stderr)
print(t.stdout[-600:])

# %% [markdown]
# ## 4. Smoke run (3 images per part) — catches errors in a minute

# %%
base = [sys.executable, "scripts/bench_counting.py", "--data-root", str(DATA_SHT), "--clipebc-repo", str(CLIPEBC_REPO),
        "--weights-dir", str(ONNX_DIR), "--device", DEVICE, "--commit", COMMIT]
if not checks["phd_onnx"]:
    print("CrowdHuman ONNX missing — skipping 'phd' (run notebook 01 to include it).")
models = ["yolo11l_1280", "yolo11l_tiled640", "clipebc_window", "clipebc_whole"] + (["phd"] if checks["phd_onnx"] else [])
smoke = subprocess.run(base + ["--models", *models, "--limit", str(SMOKE_IMAGES), "--out", str(OUT / "smoke.md")],
                       capture_output=True, text=True)
(OUT / "smoke_log.txt").write_text(smoke.stdout + smoke.stderr)
print(smoke.stdout[-3000:], smoke.stderr[-3000:])

# %% [markdown]
# ## 5. Full run on all test images

# %%
full = subprocess.run(base + ["--models", *models, "--out", str(OUT / "counting_zeroshot_shanghaitech.md")],
                      capture_output=True, text=True)
(OUT / "full_log.txt").write_text(full.stdout + full.stderr)
print(full.stdout[-4000:], full.stderr[-2000:])
print((OUT / "counting_zeroshot_shanghaitech.md").read_text() if (OUT / "counting_zeroshot_shanghaitech.md").exists() else "no results file")

# %% [markdown]
# ## 6. Package the report

# %%
import zipfile

report_zip = WORK / "crowdsafe_nb03_report.zip"
with zipfile.ZipFile(report_zip, "w", zipfile.ZIP_DEFLATED) as z:
    for p in OUT.rglob("*"):
        z.write(p, p.relative_to(OUT))
print("Return this file to the agent:", report_zip, f"({report_zip.stat().st_size / 1e3:.0f} KB)")

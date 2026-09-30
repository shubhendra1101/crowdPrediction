# %% [markdown]
# # CrowdSafe — Notebook 03: zero-shot counting benchmark on ShanghaiTech (T3.5, part 1)
#
# **Run after notebook 02** (ShanghaiTech data). Uses notebook 01's downloads when present, otherwise fetches
# what it needs itself.
#
# Counts people in the 182 + 316 ShanghaiTech test images with each model, *without training*, and compares
# with the annotated counts:
#
# | Column | Model |
# | --- | --- |
# | `yolo11l_1280` | YOLO11l (COCO), full image at 1280 px |
# | `yolo11l_tiled640` | YOLO11l on 640 px tiles (`sv.InferenceSlicer`) |
# | `phd_heads` / `phd_persons` | CrowdHuman YOLO11s person+head (ONNX) |
# | `clipebc_window` | CLIP-EBC ViT-B/16 NWPU, 224 px windows |
# | `clipebc_whole` | CLIP-EBC, whole image in one pass (`app.py` style) |
#
# **How to run:** `Kernel → Restart & Run All` (~10–25 min). **Return:** `crowdsafe_nb03_report.zip`.

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
CLIPEBC_REPO = WORK / "cache" / "CLIP-EBC-hf"
ONNX_DIR = WORK / "outputs" / "onnx"          # notebook 01 output (optional)
OUT = WORK / "outputs_nb03"
DEVICE = "0"
SMOKE_IMAGES = 3
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
# ## 2. Dependencies (PyTorch pinned), OpenCV and GPU check

# %%
os.environ["YOLO_AUTOINSTALL"] = "false"
from crowdsafe import nbenv

print("CPU threads capped to", nbenv.limit_threads())   # container quota, not the 256 host cores

PKGS = ["ultralytics>=8.3", "supervision==0.30.6", "onnxruntime>=1.18", "huggingface_hub>=0.24", "safetensors",
        "einops", "ftfy", "regex", "timm==0.9.16", "tensorboardX", "scipy", "pandas", "pyarrow", "pyyaml", "pytest"]
print(nbenv.safe_pip(PKGS, log=OUT / "pip_install.log"))
print(nbenv.fix_cv2())
print(nbenv.gpu_summary(require=True))

# %% [markdown]
# ## 3. Inputs: ShanghaiTech copy, CLIP-EBC code + weights, CrowdHuman ONNX

# %%
import shutil

from huggingface_hub import hf_hub_download, snapshot_download


def has_test_images(root: Path) -> bool:
    return len(list(root.rglob("part_A*/test_data/images/*.jpg"))) >= 182 and \
        len(list(root.rglob("part_B*/test_data/images/*.jpg"))) >= 316


DATA_SHT = next((RAW / n for n in ("shanghaitech", "shanghaitech_hf") if has_test_images(RAW / n)), None)
assert DATA_SHT, f"No complete ShanghaiTech copy under {RAW} — run notebook 02 first"
print("ShanghaiTech from:", DATA_SHT)

if not (CLIPEBC_REPO / "nwpu_weights" / "CLIP_EBC_ViT_B_16" / "model.safetensors").exists():
    snapshot_download("Yiming-M/CLIP-EBC", ignore_patterns=["nwpu_weights/CLIP_EBC_ViT_L_14/*"], local_dir=CLIPEBC_REPO)
WEIGHTS = WORK / "weights_nb03"
WEIGHTS.mkdir(exist_ok=True)
phd = WEIGHTS / "yolo11s_crowdhuman_person_head.onnx"
if not phd.exists():
    src = ONNX_DIR / phd.name
    shutil.copy(src if src.exists() else hf_hub_download("Sharath33/Person", "yolov11_phd_s.onnx",
                                                        local_dir=WORK / "cache" / "phd"), phd)
t = subprocess.run([sys.executable, "-m", "pytest", "-q"], capture_output=True, text=True)
(OUT / "pytest.txt").write_text(t.stdout + t.stderr)
print("project tests on this machine:", t.stdout.strip().splitlines()[-1] if t.stdout.strip() else t.stderr[-300:])

# %% [markdown]
# ## 4. Smoke run (3 images per part)

# %%
MODELS = ["yolo11l_1280", "yolo11l_tiled640", "phd", "clipebc_window", "clipebc_whole"]
base = [sys.executable, "scripts/bench_counting.py", "--data-root", str(DATA_SHT), "--clipebc-repo", str(CLIPEBC_REPO),
        "--weights-dir", str(WEIGHTS), "--device", DEVICE, "--commit", COMMIT, "--models", *MODELS]
smoke = subprocess.run(base + ["--limit", str(SMOKE_IMAGES), "--out", str(OUT / "smoke.md")], capture_output=True, text=True)
(OUT / "smoke_log.txt").write_text(smoke.stdout + smoke.stderr)
print(smoke.stdout[-3000:], smoke.stderr[-3000:])

# %% [markdown]
# ## 5. Full run on all test images

# %%
full = subprocess.run(base + ["--out", str(OUT / "counting_zeroshot_shanghaitech.md")], capture_output=True, text=True)
(OUT / "full_log.txt").write_text(full.stdout + full.stderr)
print(full.stdout[-4000:], full.stderr[-2000:])
res = OUT / "counting_zeroshot_shanghaitech.md"
print(res.read_text() if res.exists() else "no results file — see full_log.txt")

# %% [markdown]
# ## 6. Package the report

# %%
import zipfile

(OUT / "source.txt").write_text(f"shanghaitech copy: {DATA_SHT}\ncommit: {COMMIT}\n")
report_zip = WORK / "crowdsafe_nb03_report.zip"
with zipfile.ZipFile(report_zip, "w", zipfile.ZIP_DEFLATED) as z:
    for p in OUT.rglob("*"):
        z.write(p, p.relative_to(OUT))
print("Download this file:", report_zip, f"({report_zip.stat().st_size / 1e3:.0f} KB)")

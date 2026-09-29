# %% [markdown]
# # CrowdSafe — Notebook 01: A100 environment check + ONNX export
#
# **Tasks:** T0.2 (A100 environment), T0.6 (ONNX models for T3.1 / T3.2 / T3.3 / T4.1).
#
# **What it does**
# 1. Records the machine: GPU, CUDA, RAM, disk, internet access, whether tokens are set (never their values).
# 2. Exports pretrained models to ONNX (no training):
#    - **YOLO11l** (Ultralytics, COCO) — person detector, dynamic input size.
#    - **RAFT-large** (torchvision) — optical flow, fixed 544×960 input.
#    - **CLIP-EBC ViT-B/16, NWPU weights** from the official Hugging Face repo `Yiming-M/CLIP-EBC` — density model, 224×224 windows.
# 3. Downloads the **CrowdHuman YOLO11s person+head** model (`Sharath33/Person`, already ONNX) and smoke-tests it.
# 4. Checks every ONNX file against the original PyTorch model on the same input and writes the differences to a report.
#
# **How to run**
# 1. Start a Kubeflow notebook server with the A100 attached. Upload this notebook.
# 2. `Kernel → Restart & Run All`. Expected time: ~10–20 min, mostly downloads (~1.5 GB) and export.
# 3. When it finishes, the last cell prints two file paths. Download both from the Jupyter file browser
#    (right-click → Download) and give them back to the agent:
#    - `crowdsafe_nb01_report.zip` (small: reports, logs) — **always return this, even if something failed**
#    - `crowdsafe_nb01_onnx.zip` (~0.6–1 GB: the ONNX models)
#
# **If a cell fails:** don't fix it by hand. Run the remaining cells (each export is independent),
# then return the report zip — the error text is saved in it.
#
# Nothing here uses faces or identities. Seed: 42.

# %% [markdown]
# ## 1. Settings
# All paths and parameters are here. Change `WORK` only if your home folder is small or not persistent.

# %%
import os
from pathlib import Path

WORK = Path(os.environ.get("CROWDSAFE_WORK", Path.home() / "crowdsafe_work"))
OUT = WORK / "outputs"            # everything returned to the agent
ONNX_DIR = OUT / "onnx"
LOG_DIR = OUT / "logs"
CACHE = WORK / "cache"            # downloads, cloned repos (not returned)
SEED = 42
OPSET = 17

# YOLO (T3.1)
YOLO_WEIGHTS = "yolo11l.pt"       # auto-downloaded by Ultralytics
YOLO_EXPORT_IMGSZ = 1280          # export trace size; model is exported with dynamic H/W
YOLO_CHECK_IMGSZ = [640, 1280]    # sizes used for the PyTorch-vs-ONNX check
PERSON_CLASS = 0                  # COCO "person"

# RAFT (T4.1) — architecture.md §2.3: ~960 px wide. H must be a multiple of 8.
RAFT_H, RAFT_W = 544, 960
RAFT_ITERS = 12                   # torchvision default number of flow updates
RAFT_TEST_SHIFT = (6, 3)          # (dx, dy) px synthetic shift for the sanity check

# CLIP-EBC (T3.3) — official Hugging Face repo (model code + NWPU weights + config.json)
CLIPEBC_HF_REPO = "Yiming-M/CLIP-EBC"
CLIPEBC_HF_WEIGHTS = "nwpu_weights/CLIP_EBC_ViT_B_16"      # model.safetensors + config.json
CLIPEBC_HF_IGNORE = ["nwpu_weights/CLIP_EBC_ViT_L_14/*"]  # ViT-L not used
CLIPEBC_TEST_IMAGES = ["example1.jpg", "example2.jpg"]     # crowd images shipped in the HF repo
IMAGENET_MEAN = [0.485, 0.456, 0.406]   # CLIP-EBC app.py / datasets/crowd.py
IMAGENET_STD = [0.229, 0.224, 0.225]
# Optional ShanghaiTech-A checkpoint exists only on GitHub releases. CLAUDE.md sourcing rule:
# keep False unless the user has approved it.
USE_GITHUB_SHA = False
CLIPEBC_GITHUB_SHA = "https://github.com/Yiming-M/CLIP-EBC/releases/download/v1.0.0/ShanghaiTech_A_CLIP_EBC_ViT_B_16_Word.tgz"

# CrowdHuman person+head YOLO11s (T3.2) — already ONNX on Hugging Face; downloaded and smoke-tested only
PHD_HF_REPO = "Sharath33/Person"
PHD_HF_FILE = "yolov11_phd_s.onnx"
PHD_CLASSES = {0: "person", 1: "head"}   # from the model card
PHD_CONF, PHD_IOU = 0.2, 0.6             # model card inference.py defaults

TEST_IMAGE_URL = "https://ultralytics.com/images/bus.jpg"   # small public test image

for d in (OUT, ONNX_DIR, LOG_DIR, CACHE):
    d.mkdir(parents=True, exist_ok=True)
print("Work dir:", WORK)

# %% [markdown]
# ## 2. Install packages
# Installs only what this notebook needs. PyTorch is **not** reinstalled (the Kubeflow image's CUDA build is kept).

# %%
import subprocess
import sys

PKGS = ["ultralytics>=8.3", "onnx>=1.16", "onnxruntime>=1.18", "onnxslim", "huggingface_hub>=0.24",
        "safetensors", "einops", "ftfy", "regex", "timm==0.9.16", "tensorboardX", "scipy", "psutil",
        "requests", "matplotlib"]
res = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *PKGS], capture_output=True, text=True)
(LOG_DIR / "pip_install.log").write_text(res.stdout + "\n" + res.stderr)
print("pip exit code:", res.returncode)
if res.returncode != 0:
    print(res.stderr[-3000:])

# %% [markdown]
# ## 3. Environment report (T0.2)

# %%
import json
import platform
import random
import shutil
import time

import numpy as np
import psutil
import requests
import torch

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

REPORT: dict = {"notebook": "01_env_and_onnx_export", "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                "seed": SEED, "exports": {}}


def reachable(url: str) -> bool:
    """Return True if a HEAD request to url succeeds within 10 s."""
    try:
        return requests.head(url, timeout=10, allow_redirects=True).status_code < 400
    except Exception:
        return False


env = {
    "python": platform.python_version(),
    "platform": platform.platform(),
    "torch": torch.__version__,
    "torch_cuda_build": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
    "gpus": [
        {"name": torch.cuda.get_device_name(i),
         "total_mem_gb": round(torch.cuda.get_device_properties(i).total_memory / 1e9, 1)}
        for i in range(torch.cuda.device_count())
    ],
    "cpu_count": os.cpu_count(),
    "ram_total_gb": round(psutil.virtual_memory().total / 1e9, 1),
    "disk_work_total_gb": round(shutil.disk_usage(WORK).total / 1e9, 1),
    "disk_work_free_gb": round(shutil.disk_usage(WORK).free / 1e9, 1),
    "work_dir": str(WORK),
    "internet": {u: reachable(u) for u in ["https://huggingface.co", "https://github.com",
                                           "https://download.pytorch.org", "https://www.kaggle.com",
                                           "https://ped.fz-juelich.de"]},
    "HF_TOKEN_set": bool(os.environ.get("HF_TOKEN")),
    "kaggle_json_present": (Path.home() / ".kaggle" / "kaggle.json").exists(),
}
try:
    smi = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=30).stdout
except Exception as e:  # nvidia-smi missing is not fatal
    smi = f"nvidia-smi unavailable: {e}"
(LOG_DIR / "nvidia_smi.txt").write_text(smi)
REPORT["environment"] = env
print(json.dumps(env, indent=2))
print(smi)

# %% [markdown]
# ## 4. Shared helpers

# %%
import hashlib
import inspect
import traceback

import cv2


def sha256(path: Path) -> str:
    """SHA-256 of a file, so the agent can confirm the returned file is intact."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def onnx_export(model: torch.nn.Module, args: tuple, path: Path, **kw) -> None:
    """torch.onnx.export with the classic (TorchScript) exporter on every torch version."""
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        kw["dynamo"] = False
    torch.onnx.export(model, args, str(path), opset_version=OPSET, do_constant_folding=True, **kw)


def save_meta(name: str, meta: dict) -> None:
    """Write a JSON sidecar describing how to feed the ONNX model."""
    (ONNX_DIR / f"{name}.json").write_text(json.dumps(meta, indent=2))


def record(name: str, fn) -> None:
    """Run one export step; store its result or its error in REPORT without stopping the notebook."""
    t0 = time.time()
    try:
        out = fn()
        out["status"] = "ok"
    except Exception as e:
        out = {"status": "failed", "error": repr(e), "traceback": traceback.format_exc()}
        print(out["traceback"])
    out["seconds"] = round(time.time() - t0, 1)
    REPORT["exports"][name] = out
    print(name, "->", {k: v for k, v in out.items() if k != "traceback"})


test_img_path = CACHE / "bus.jpg"
if not test_img_path.exists():
    test_img_path.write_bytes(requests.get(TEST_IMAGE_URL, timeout=60).content)
TEST_BGR = cv2.imread(str(test_img_path))
TEST_RGB = cv2.cvtColor(TEST_BGR, cv2.COLOR_BGR2RGB)
print("Test image:", TEST_RGB.shape)

# %% [markdown]
# ## 5. YOLO11l → ONNX (T3.1)
# Exported with dynamic height/width so tiled inference (`InferenceSlicer`) and full frames both work.
# Check: person detections from PyTorch and ONNX on the same image must match (count and box IoU).

# %%
from ultralytics import YOLO


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two sets of xyxy boxes."""
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), axis=2)
    area = lambda x: np.prod(x[:, 2:] - x[:, :2], axis=1)
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def export_yolo() -> dict:
    """Export YOLO11l to ONNX and compare person detections with PyTorch."""
    pt = YOLO(YOLO_WEIGHTS)
    src = Path(pt.export(format="onnx", imgsz=YOLO_EXPORT_IMGSZ, dynamic=True, simplify=True, opset=OPSET))
    dst = ONNX_DIR / "yolo11l.onnx"
    shutil.copy(src, dst)
    ox = YOLO(str(dst), task="detect")
    checks = {}
    for sz in YOLO_CHECK_IMGSZ:
        kw = dict(imgsz=sz, classes=[PERSON_CLASS], conf=0.25, verbose=False)
        a = pt.predict(TEST_BGR, device=0 if torch.cuda.is_available() else "cpu", **kw)[0].boxes
        b = ox.predict(TEST_BGR, device="cpu", **kw)[0].boxes
        ab, bb = a.xyxy.cpu().numpy(), b.xyxy.cpu().numpy()
        miou = float(box_iou(ab, bb).max(axis=1).mean()) if len(ab) and len(bb) else None
        checks[f"imgsz_{sz}"] = {"persons_pytorch": len(ab), "persons_onnx": len(bb),
                                 "mean_best_iou": miou,
                                 "max_conf_diff": float(abs(np.sort(a.conf.cpu().numpy()) - np.sort(b.conf.cpu().numpy())).max())
                                 if len(ab) == len(bb) and len(ab) else None}
    save_meta("yolo11l", {"source": f"Ultralytics {YOLO_WEIGHTS} (COCO)", "opset": OPSET,
                          "input": "use via ultralytics YOLO('yolo11l.onnx', task='detect'); dynamic HxW (multiple of 32)",
                          "person_class": PERSON_CLASS, "sha256": sha256(dst)})
    return {"file": dst.name, "size_mb": round(dst.stat().st_size / 1e6, 1), "checks": checks}


record("yolo11l", export_yolo)

# %% [markdown]
# ## 6. RAFT-large → ONNX (T4.1)
# The ONNX model takes two RGB frames scaled to **[0, 1]**, shape `1×3×544×960`, and returns the final
# flow `1×2×544×960` in pixels (x, y). The [-1, 1] normalisation RAFT expects is built in.
# Check: (a) PyTorch vs ONNX difference; (b) on a frame shifted by a known (dx, dy), both must recover that shift.

# %%
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large


class RaftONNX(torch.nn.Module):
    """RAFT wrapper: [0,1] RGB inputs, returns only the final flow field."""

    def __init__(self, iters: int) -> None:
        super().__init__()
        self.net = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).eval()
        self.iters = iters

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.net(a * 2 - 1, b * 2 - 1, num_flow_updates=self.iters)[-1]


def to_tensor01(rgb: np.ndarray) -> torch.Tensor:
    """HxWx3 uint8 RGB -> 1x3xHxW float in [0, 1]."""
    return torch.from_numpy(rgb).permute(2, 0, 1)[None].float() / 255.0


def export_raft() -> dict:
    """Export RAFT-large to ONNX and check it on a synthetic known shift."""
    import onnxruntime as ort

    model = RaftONNX(RAFT_ITERS).eval()
    f1 = cv2.resize(TEST_RGB, (RAFT_W, RAFT_H))
    dx, dy = RAFT_TEST_SHIFT
    f2 = np.roll(f1, shift=(dy, dx), axis=(0, 1))          # content moves by +dx, +dy
    a, b = to_tensor01(f1), to_tensor01(f2)
    dst = ONNX_DIR / "raft_large.onnx"
    with torch.no_grad():
        onnx_export(model, (a, b), dst, input_names=["frame_a", "frame_b"], output_names=["flow"])
        ref = model(a, b).numpy()
    sess = ort.InferenceSession(str(dst), providers=["CPUExecutionProvider"])
    got = sess.run(None, {"frame_a": a.numpy(), "frame_b": b.numpy()})[0]
    m = 32                                                   # ignore wrapped borders
    centre = lambda f: [float(np.median(f[0, c, m:-m, m:-m])) for c in (0, 1)]
    save_meta("raft_large", {"source": "torchvision raft_large, Raft_Large_Weights.DEFAULT", "opset": OPSET,
                             "inputs": {"frame_a": [1, 3, RAFT_H, RAFT_W], "frame_b": [1, 3, RAFT_H, RAFT_W]},
                             "input_format": "RGB float32 in [0,1], NCHW; resize frames to W=960, H=544",
                             "output": "flow 1x2xHxW, pixels, channel 0 = x, 1 = y",
                             "flow_updates": RAFT_ITERS, "sha256": sha256(dst)})
    return {"file": dst.name, "size_mb": round(dst.stat().st_size / 1e6, 1),
            "checks": {"true_shift_px": [dx, dy],
                       "median_flow_pytorch": centre(ref), "median_flow_onnx": centre(got),
                       "max_abs_diff_px": float(np.abs(ref - got).max()),
                       "mean_abs_diff_px": float(np.abs(ref - got).mean())}}


record("raft_large", export_raft)

# %% [markdown]
# ## 7. CLIP-EBC ViT-B/16 (NWPU) → ONNX (T3.3)
# Downloads the author's official Hugging Face repo `Yiming-M/CLIP-EBC` (model code + `nwpu_weights/CLIP_EBC_ViT_B_16`),
# rebuilds the model exactly as its `app.py` does (settings from the shipped `config.json`), loads the weights
# **strictly** (any mismatch = error, not a silent bad model), and exports the 224×224 window model.
# Sliding-window stitching happens outside ONNX, in `crowdsafe/counting/density_model.py`.
#
# ONNX input: RGB float in **[0, 1]**, `N×3×224×224` (ImageNet normalisation built in).
# ONNX output: density `N×1×28×28`; the sum is the count in that window.
#
# Checks on the repo's own crowd example images:
# (a) PyTorch vs ONNX on the same windows; (b) our sliding window vs the repo's `sliding_window_predict`;
# (c) the whole-image count the way `app.py` computes it — so we can see how much windowing changes the count.

# %%
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

CLIPEBC_DIR = Path(snapshot_download(CLIPEBC_HF_REPO, repo_type="model", ignore_patterns=CLIPEBC_HF_IGNORE,
                                     local_dir=CACHE / "CLIP-EBC-hf"))
if str(CLIPEBC_DIR) not in sys.path:
    sys.path.insert(0, str(CLIPEBC_DIR))
os.chdir(CLIPEBC_DIR)   # the repo resolves some paths relative to itself
REPORT["clipebc_source"] = {"hf_repo": CLIPEBC_HF_REPO, "weights": CLIPEBC_HF_WEIGHTS,
                            "files": sorted(str(p.relative_to(CLIPEBC_DIR)) for p in (CLIPEBC_DIR / CLIPEBC_HF_WEIGHTS).iterdir())}


class ClipEbcONNX(torch.nn.Module):
    """CLIP-EBC window model: [0,1] RGB in, density map out (ImageNet normalisation built in)."""

    def __init__(self, net: torch.nn.Module) -> None:
        super().__init__()
        self.net = net.eval()
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net((x - self.mean) / self.std)


def build_clipebc_hf(weights_dir: Path) -> tuple[torch.nn.Module, dict, str]:
    """Rebuild CLIP-EBC from the HF config.json and load model.safetensors strictly (as in app.py)."""
    from models import get_model

    cfg = json.loads((weights_dir / "config.json").read_text())
    net = get_model(backbone=cfg["backbone"], input_size=cfg["input_size"], reduction=cfg["reduction"],
                    bins=[(float(lo), float(hi)) for lo, hi in cfg["bins"]],
                    anchor_points=[float(p) for p in cfg["anchor_points"]],
                    prompt_type=cfg["prompt_type"], num_vpt=cfg["num_vpt"],
                    vpt_drop=cfg["vpt_drop"], deep_vpt=cfg["deep_vpt"])
    sd = load_file(str(weights_dir / "model.safetensors"))
    # app.py strips "model." from keys; try a prefix-only strip first, then app.py's exact rule.
    attempts = {
        "prefix_strip": {(k[len("model."):] if k.startswith("model.") else k): v for k, v in sd.items()},
        "app_py_replace": {k.replace("model.", ""): v for k, v in sd.items()},
        "as_is": sd,
    }
    errors = {}
    for how, cand in attempts.items():
        try:
            net.load_state_dict(cand, strict=True)
            return net.eval(), cfg, how
        except RuntimeError as e:
            errors[how] = str(e)[:500]
    raise RuntimeError(f"No key mapping loaded strictly: {errors}")


def sliding_window_np(run, rgb01: np.ndarray, win: int, reduction: int) -> np.ndarray:
    """Non-overlapping sliding window over a zero-padded image; run(batch NCHW) -> density N1hw."""
    h, w = rgb01.shape[:2]
    H, W = -(-h // win) * win, -(-w // win) * win
    pad = np.zeros((H, W, 3), np.float32)
    pad[:h, :w] = rgb01
    tiles = [pad[y:y + win, x:x + win].transpose(2, 0, 1) for y in range(0, H, win) for x in range(0, W, win)]
    dens = run(np.stack(tiles).astype(np.float32))
    r = win // reduction
    out = np.zeros((H // reduction, W // reduction), np.float32)
    i = 0
    for y in range(0, H // reduction, r):
        for x in range(0, W // reduction, r):
            out[y:y + r, x:x + r] = dens[i, 0]
            i += 1
    return out


def export_clipebc_hf(name: str, weights_dir: Path, source: str) -> dict:
    """Build, strictly load, export and check the HF CLIP-EBC checkpoint."""
    import onnxruntime as ort

    net, cfg, key_mapping = build_clipebc_hf(weights_dir)
    model = ClipEbcONNX(net).eval()
    win, red = cfg["input_size"], cfg["reduction"]
    dst = ONNX_DIR / f"{name}.onnx"
    with torch.no_grad():
        onnx_export(model, (torch.rand(2, 3, win, win),), dst, input_names=["image"], output_names=["density"],
                    dynamic_axes={"image": {0: "batch"}, "density": {0: "batch"}})
    sess = ort.InferenceSession(str(dst), providers=["CPUExecutionProvider"])

    def run_torch(batch: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return model(torch.from_numpy(batch)).numpy()

    def run_onnx(batch: np.ndarray) -> np.ndarray:
        return sess.run(None, {"image": batch})[0]

    checks = {}
    for img_name in CLIPEBC_TEST_IMAGES:
        path = CLIPEBC_DIR / img_name
        if not path.exists():
            checks[img_name] = "image not in repo snapshot"
            continue
        rgb01 = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        d_t = sliding_window_np(run_torch, rgb01, win, red)
        d_o = sliding_window_np(run_onnx, rgb01, win, red)
        c = {"size_hw": list(rgb01.shape[:2]), "count_window_pytorch": float(d_t.sum()),
             "count_window_onnx": float(d_o.sum()), "max_abs_diff_density": float(np.abs(d_t - d_o).max())}
        x = torch.from_numpy(rgb01).permute(2, 0, 1)[None]
        with torch.no_grad():
            try:   # whole image in one pass, as app.py does
                c["count_whole_image_app_py"] = float(model(x).sum())
            except Exception as e:
                c["count_whole_image_app_py"] = f"not run: {e!r}"
            try:   # repo's reference sliding window on the same zero-padded image
                from utils.eval_utils import sliding_window_predict
                h, w = rgb01.shape[:2]
                H, W = -(-h // win) * win, -(-w // win) * win
                img = torch.zeros(1, 3, H, W)
                img[0, :, :h, :w] = x[0]
                c["count_repo_sliding_window"] = float(sliding_window_predict(net, (img - model.mean) / model.std, win, win).sum())
            except Exception as e:
                c["count_repo_sliding_window"] = f"not run: {e!r}"
        checks[img_name] = c
    save_meta(name, {"source": source, "config": cfg, "key_mapping": key_mapping, "opset": OPSET,
                     "input": f"image Nx3x{win}x{win} RGB float32 in [0,1] (ImageNet normalisation built in)",
                     "output": f"density Nx1x{win // red}x{win // red}; sum = count",
                     "sliding_window": f"non-overlapping {win}px windows over zero-padded frame", "sha256": sha256(dst)})
    return {"file": dst.name, "size_mb": round(dst.stat().st_size / 1e6, 1), "key_mapping": key_mapping,
            "config": cfg, "checks": checks}


record("clipebc_vitb16_nwpu", lambda: export_clipebc_hf(
    "clipebc_vitb16_nwpu", CLIPEBC_DIR / CLIPEBC_HF_WEIGHTS, f"Hugging Face {CLIPEBC_HF_REPO}/{CLIPEBC_HF_WEIGHTS}"))
os.chdir(WORK)

if USE_GITHUB_SHA:
    print("USE_GITHUB_SHA=True: the optional ShanghaiTech-A checkpoint is handled in a later notebook once approved.")

# %% [markdown]
# ## 7b. CrowdHuman person + head YOLO11s (T3.2) — download and smoke test
# `Sharath33/Person` is already ONNX, so nothing is exported. This cell downloads it, records its input/output
# shapes, and counts persons and heads on the test images with the model card's own settings
# (BGR, ÷255, letterbox with 114 padding, conf 0.2, NMS IoU 0.6). The accuracy comparison is T3.5.

# %%
from huggingface_hub import hf_hub_download


def letterbox(bgr: np.ndarray, h_in: int, w_in: int) -> tuple[np.ndarray, float, int, int]:
    """Resize keeping aspect ratio and pad with 114 (model card preprocessing)."""
    h, w = bgr.shape[:2]
    s = min(w_in / w, h_in / h)
    nh, nw = int(round(h * s)), int(round(w * s))
    canvas = np.full((h_in, w_in, 3), 114, np.uint8)
    top, left = (h_in - nh) // 2, (w_in - nw) // 2
    canvas[top:top + nh, left:left + nw] = cv2.resize(bgr, (nw, nh))
    return canvas, s, left, top


def phd_counts(sess, bgr: np.ndarray) -> dict:
    """Run the person+head ONNX model; return per-class counts after NMS."""
    inp = sess.get_inputs()[0]
    h_in, w_in = (int(inp.shape[2]), int(inp.shape[3])) if isinstance(inp.shape[2], int) else (640, 640)
    canvas, _, _, _ = letterbox(bgr, h_in, w_in)
    x = canvas.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
    preds = sess.run(None, {inp.name: x})[0][0].T            # (anchors, 4 + n_classes)
    scores = preds[:, 4:]
    cls = scores.argmax(1)
    conf = scores.max(1)
    keep = conf >= PHD_CONF
    boxes = preds[keep, :4].copy()
    boxes[:, :2] -= boxes[:, 2:] / 2                           # cx,cy,w,h -> x,y,w,h
    idx = cv2.dnn.NMSBoxes(boxes.tolist(), conf[keep].tolist(), PHD_CONF, PHD_IOU)
    idx = np.array(idx).reshape(-1)
    kept_cls = cls[keep][idx] if len(idx) else np.array([], int)
    return {PHD_CLASSES.get(int(k), str(k)): int((kept_cls == k).sum()) for k in PHD_CLASSES}


def check_phd() -> dict:
    """Download the CrowdHuman person+head ONNX model and smoke-test it."""
    import onnxruntime as ort

    src = Path(hf_hub_download(PHD_HF_REPO, PHD_HF_FILE, local_dir=CACHE / "phd"))
    dst = ONNX_DIR / "yolo11s_crowdhuman_person_head.onnx"
    shutil.copy(src, dst)
    sess = ort.InferenceSession(str(dst), providers=["CPUExecutionProvider"])
    io = {"inputs": [(i.name, i.shape) for i in sess.get_inputs()],
          "outputs": [(o.name, o.shape) for o in sess.get_outputs()]}
    imgs = {"bus.jpg": TEST_BGR}
    for n in CLIPEBC_TEST_IMAGES:
        p = CLIPEBC_DIR / n
        if p.exists():
            imgs[n] = cv2.imread(str(p))
    checks = {n: phd_counts(sess, im) for n, im in imgs.items()}
    save_meta("yolo11s_crowdhuman_person_head", {"source": f"Hugging Face {PHD_HF_REPO}/{PHD_HF_FILE}",
                                                 "classes": PHD_CLASSES, "io": io,
                                                 "input_format": "BGR, /255, letterbox pad 114, NCHW",
                                                 "defaults": {"conf": PHD_CONF, "nms_iou": PHD_IOU},
                                                 "license": "openrail++", "sha256": sha256(dst)})
    return {"file": dst.name, "size_mb": round(dst.stat().st_size / 1e6, 1), "io": io, "checks": checks}


record("yolo11s_crowdhuman_person_head", check_phd)

# %% [markdown]
# ## 8. GPU smoke timing
# One quick number per model on the A100 with PyTorch (not ONNX), so later runtime estimates have a reference.
# This is a smoke check, not the T9.2 runtime evaluation.

# %%
def time_it(fn, n: int = 20) -> float:
    """Median milliseconds of fn() over n runs after 3 warm-ups."""
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    return float(np.median(ts))


timing = {}
if torch.cuda.is_available():
    try:
        raft = RaftONNX(RAFT_ITERS).cuda().eval()
        x = torch.rand(1, 3, RAFT_H, RAFT_W, device="cuda")
        with torch.no_grad():
            timing["raft_large_544x960_ms"] = time_it(lambda: raft(x, x))
        del raft
    except Exception as e:
        timing["raft_error"] = repr(e)
    try:
        yolo = YOLO(YOLO_WEIGHTS)
        frame = cv2.resize(TEST_BGR, (1920, 1080))
        timing["yolo11l_1080p_imgsz1280_ms"] = time_it(
            lambda: yolo.predict(frame, imgsz=1280, device=0, classes=[PERSON_CLASS], verbose=False), n=10)
    except Exception as e:
        timing["yolo_error"] = repr(e)
    timing["peak_gpu_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
REPORT["gpu_timing"] = timing
print(timing)

# %% [markdown]
# ## 9. Write report and package the outputs
# Produces the two zips to return. The report zip is small; the ONNX zip holds the models.

# %%
import zipfile

REPORT["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
(OUT / "report.json").write_text(json.dumps(REPORT, indent=2, default=str))

lines = ["# Notebook 01 report", "", f"Finished: {REPORT['finished']}", "",
         "| Export | Status | Size MB | Seconds |", "| --- | --- | --- | --- |"]
for k, v in REPORT["exports"].items():
    lines.append(f"| {k} | {v['status']} | {v.get('size_mb', '')} | {v['seconds']} |")
(OUT / "report.md").write_text("\n".join(lines) + "\n")

report_zip = WORK / "crowdsafe_nb01_report.zip"
onnx_zip = WORK / "crowdsafe_nb01_onnx.zip"
with zipfile.ZipFile(report_zip, "w", zipfile.ZIP_DEFLATED) as z:
    for p in OUT.rglob("*"):
        if p.is_file() and p.suffix != ".onnx":
            z.write(p, p.relative_to(OUT))
with zipfile.ZipFile(onnx_zip, "w", zipfile.ZIP_STORED) as z:
    for p in ONNX_DIR.iterdir():
        z.write(p, p.relative_to(OUT))

print("\n".join(lines))
print("\nReturn these files to the agent:")
print(" ", report_zip, f"({report_zip.stat().st_size / 1e6:.1f} MB)")
print(" ", onnx_zip, f"({onnx_zip.stat().st_size / 1e6:.1f} MB)")

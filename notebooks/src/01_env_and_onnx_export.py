# %% [markdown]
# # CrowdSafe — Notebook 01: A100 environment check + ONNX export
#
# **Tasks:** T0.2 (A100 environment), T0.6 (ONNX models for T3.1 / T3.2 / T3.3 / T4.1).
#
# **What it does**
# 1. Gets the project code from GitHub and installs packages **without replacing the container's PyTorch**.
# 2. Records the machine: GPU, CUDA, RAM, disk, internet access, whether tokens are set (never their values).
# 3. Exports pretrained models to ONNX (no training): **YOLO11l** (COCO), **RAFT-large** (optical flow),
#    **CLIP-EBC ViT-B/16 NWPU** (density, from Hugging Face `Yiming-M/CLIP-EBC`); downloads the
#    **CrowdHuman YOLO11s person+head** ONNX (`Sharath33/Person`). Every ONNX file is checked against PyTorch.
#
# **How to run:** `Kernel → Restart & Run All` (~10–20 min). Each export is independent: if one fails the
# others still run, and the error is saved. The last cell prints a status table and two zips to download:
# - `crowdsafe_nb01_report.zip` (small) — **always return this**
# - `crowdsafe_nb01_onnx.zip` (~0.6–1 GB) — the ONNX models
#
# Lab-network notes handled here: `download.pytorch.org` is blocked (RAFT weights come from a verified
# Hugging Face mirror); the container's NVIDIA PyTorch build is pinned during installs.

# %% [markdown]
# ## 1. Settings and project code

# %%
import os
import subprocess
import sys
from pathlib import Path

WORK = Path(os.environ.get("CROWDSAFE_WORK", Path.home() / "crowdsafe_work"))
REPO_DIR = WORK / "crowdPrediction"
REPO_URL = "https://github.com/shubhendra1101/crowdPrediction.git"
OUT = WORK / "outputs"            # everything returned to the agent
ONNX_DIR = OUT / "onnx"
LOG_DIR = OUT / "logs"
CACHE = WORK / "cache"            # downloads (not returned)
SEED = 42
OPSET = 17

# YOLO (T3.1)
YOLO_WEIGHTS = "yolo11l.pt"       # auto-downloaded by Ultralytics (GitHub assets)
YOLO_EXPORT_IMGSZ = 1280          # trace size; exported with dynamic H/W
YOLO_CHECK_IMGSZ = [640, 1280]
PERSON_CLASS = 0

# RAFT (T4.1) — architecture.md §2.3: ~960 px wide; H multiple of 8
RAFT_H, RAFT_W = 544, 960
RAFT_ITERS = 12
RAFT_TEST_SHIFT = (6, 3)

# CLIP-EBC (T3.3)
CLIPEBC_HF_REPO = "Yiming-M/CLIP-EBC"
CLIPEBC_HF_WEIGHTS = "nwpu_weights/CLIP_EBC_ViT_B_16"
CLIPEBC_HF_IGNORE = ["nwpu_weights/CLIP_EBC_ViT_L_14/*"]
CLIPEBC_TEST_IMAGES = ["example1.jpg", "example2.jpg"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# CrowdHuman person+head (T3.2)
PHD_HF_REPO, PHD_HF_FILE = "Sharath33/Person", "yolov11_phd_s.onnx"
PHD_CLASSES = {0: "person", 1: "head"}
PHD_CONF, PHD_IOU = 0.2, 0.6

TEST_IMAGE_URL = "https://ultralytics.com/images/bus.jpg"

for d in (OUT, ONNX_DIR, LOG_DIR, CACHE):
    d.mkdir(parents=True, exist_ok=True)
if (REPO_DIR / ".git").exists():
    subprocess.run(["git", "-C", str(REPO_DIR), "pull", "--ff-only"], check=True)
else:
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(REPO_DIR)], check=True)
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))
COMMIT = subprocess.run(["git", "-C", str(REPO_DIR), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
print("Work dir:", WORK, "| code commit:", COMMIT)

# %% [markdown]
# ## 2. Install packages (PyTorch pinned) and check OpenCV / GPU

# %%
os.environ["YOLO_AUTOINSTALL"] = "false"    # stop Ultralytics from pip-installing things mid-run
from crowdsafe import nbenv

PKGS = ["ultralytics>=8.3", "onnx>=1.16", "onnxruntime>=1.18", "onnxslim", "onnxscript", "huggingface_hub>=0.24",
        "safetensors", "einops", "ftfy", "regex", "timm==0.9.16", "tensorboardX", "scipy", "psutil",
        "requests", "matplotlib", "pyyaml", "supervision==0.30.6"]
PIP =nbenv.safe_pip(PKGS, log=LOG_DIR / "pip_install.log")
print("pip:", PIP)
print(nbenv.fix_cv2())
GPU = nbenv.gpu_summary(require=False)
print("GPU:", GPU)
if not GPU["cuda_available"]:
    print("WARNING: no GPU visible — exports still run on CPU, but the timing section is skipped.")

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

REPORT: dict = {"notebook": "01_env_and_onnx_export", "commit": COMMIT, "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                "seed": SEED, "pip": PIP, "exports": {}}


def reachable(url: str) -> bool:
    """True if a GET to url returns any HTTP response within 10 s (some sites reject HEAD)."""
    try:
        requests.get(url, timeout=10, stream=True)
        return True
    except Exception:
        return False


def cgroup(path_v2: str, path_v1: str) -> str:
    for p in (path_v2, path_v1):
        try:
            return Path(p).read_text().strip()
        except OSError:
            continue
    return "unknown"


env = {
    "python": platform.python_version(), "platform": platform.platform(),
    "torch": torch.__version__, "torch_cuda_build": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
    "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
    "gpus": [{"name": torch.cuda.get_device_name(i),
              "total_mem_gb": round(torch.cuda.get_device_properties(i).total_memory / 1e9, 1)}
             for i in range(torch.cuda.device_count())],
    "cpu_limit_cgroup": cgroup("/sys/fs/cgroup/cpu.max", "/sys/fs/cgroup/cpu/cpu.cfs_quota_us"),
    "ram_limit_bytes_cgroup": cgroup("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"),
    "disk_work_free_gb": round(shutil.disk_usage(WORK).free / 1e9, 1), "work_dir": str(WORK),
    "internet": {u: reachable(u) for u in ["https://huggingface.co", "https://github.com", "https://download.pytorch.org",
                                           "https://www.kaggle.com", "https://ped.fz-juelich.de"]},
    "HF_TOKEN_set": bool(os.environ.get("HF_TOKEN")),
    "kaggle_json_present": (Path.home() / ".kaggle" / "kaggle.json").exists(),
}
smi = subprocess.run("nvidia-smi 2>&1", shell=True, capture_output=True, text=True).stdout
(LOG_DIR / "nvidia_smi.txt").write_text(smi)
REPORT["environment"] = env
print(json.dumps(env, indent=2))

# %% [markdown]
# ## 4. Shared helpers
# ONNX export tries, in order: TorchScript exporter at opset 17, at opset 18, then the dynamo exporter.
# PyTorch's fused attention fast path (used in eval mode) cannot be exported, so it is switched off.

# %%
import hashlib
import inspect
import traceback

import cv2

if hasattr(torch.backends, "mha") and hasattr(torch.backends.mha, "set_fastpath_enabled"):
    torch.backends.mha.set_fastpath_enabled(False)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def onnx_export(model: torch.nn.Module, args: tuple, path: Path, **kw) -> str:
    """Export with fallbacks; return which exporter/opset worked. Raises with all errors if none did."""
    has_dynamo = "dynamo" in inspect.signature(torch.onnx.export).parameters
    attempts = [("torchscript", OPSET), ("torchscript", 18)] + ([("dynamo", 18)] if has_dynamo else [])
    errors = {}
    for how, opset in attempts:
        try:
            extra = {"dynamo": how == "dynamo"} if has_dynamo else {}
            torch.onnx.export(model, args, str(path), opset_version=opset, do_constant_folding=True, **extra, **kw)
            return f"{how}/opset{opset}"
        except Exception as e:
            errors[f"{how}/opset{opset}"] = f"{type(e).__name__}: {str(e)[:400]}"
            print(f"   export {how}/opset{opset} failed: {errors[f'{how}/opset{opset}'][:200]}")
    raise RuntimeError(f"All ONNX export attempts failed: {errors}")


def save_meta(name: str, meta: dict) -> None:
    (ONNX_DIR / f"{name}.json").write_text(json.dumps(meta, indent=2, default=str))


def record(name: str, fn) -> None:
    """Run one step; store its result or its error in REPORT without stopping the notebook."""
    t0 = time.time()
    try:
        out = fn()
        out["status"] = "ok"
    except Exception as e:
        out = {"status": "failed", "error": f"{type(e).__name__}: {e}"[:2000], "traceback": traceback.format_exc()}
        print(out["traceback"])
    out["seconds"] = round(time.time() - t0, 1)
    REPORT["exports"][name] = out
    print(name, "->", {k: v for k, v in out.items() if k != "traceback"})


test_img_path = CACHE / "bus.jpg"
if not test_img_path.exists():
    try:
        test_img_path.write_bytes(requests.get(TEST_IMAGE_URL, timeout=60).content)
    except Exception:   # fall back to an image shipped with ultralytics
        import ultralytics
        shutil.copy(Path(ultralytics.__file__).parent / "assets" / "bus.jpg", test_img_path)
TEST_BGR = cv2.imread(str(test_img_path))
TEST_RGB = cv2.cvtColor(TEST_BGR, cv2.COLOR_BGR2RGB)
print("Test image:", TEST_RGB.shape)

# %% [markdown]
# ## 5. YOLO11l → ONNX (T3.1)

# %%
from ultralytics import YOLO


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), axis=2)
    area = lambda x: np.prod(x[:, 2:] - x[:, :2], axis=1)
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def export_yolo() -> dict:
    pt = YOLO(YOLO_WEIGHTS)
    src = Path(pt.export(format="onnx", imgsz=YOLO_EXPORT_IMGSZ, dynamic=True, simplify=True, opset=OPSET, device="cpu"))
    dst = ONNX_DIR / "yolo11l.onnx"
    shutil.copy(src, dst)
    pt = YOLO(YOLO_WEIGHTS)                       # fresh model (export can modify the in-memory one)
    ox = YOLO(str(dst), task="detect")
    checks = {}
    for sz in YOLO_CHECK_IMGSZ:
        kw = dict(imgsz=sz, classes=[PERSON_CLASS], conf=0.25, verbose=False)
        a = pt.predict(TEST_BGR, device=0 if torch.cuda.is_available() else "cpu", **kw)[0].boxes
        b = ox.predict(TEST_BGR, device="cpu", **kw)[0].boxes
        ab, bb = a.xyxy.cpu().numpy(), b.xyxy.cpu().numpy()
        checks[f"imgsz_{sz}"] = {"persons_pytorch": len(ab), "persons_onnx": len(bb),
                                 "mean_best_iou": float(box_iou(ab, bb).max(axis=1).mean()) if len(ab) and len(bb) else None}
    save_meta("yolo11l", {"source": f"Ultralytics {YOLO_WEIGHTS} (COCO)", "opset": OPSET,
                          "input": "use via ultralytics YOLO('yolo11l.onnx', task='detect'); dynamic HxW (multiple of 32)",
                          "person_class": PERSON_CLASS, "sha256": sha256(dst)})
    return {"file": dst.name, "size_mb": round(dst.stat().st_size / 1e6, 1), "checks": checks}


record("yolo11l", export_yolo)

# %% [markdown]
# ## 6. RAFT-large → ONNX (T4.1)
# Inputs: two RGB frames in **[0, 1]**, `1×3×544×960`; output: final flow `1×2×544×960` in pixels.
# Check: PyTorch vs ONNX difference, and both must recover a known synthetic shift.

# %%
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

print(nbenv.ensure_raft_large_weights())


class RaftONNX(torch.nn.Module):
    def __init__(self, iters: int) -> None:
        super().__init__()
        self.net = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).eval()
        self.iters = iters

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.net(a * 2 - 1, b * 2 - 1, num_flow_updates=self.iters)[-1]


def to_tensor01(rgb: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(rgb).permute(2, 0, 1)[None].float() / 255.0


def export_raft() -> dict:
    import onnxruntime as ort

    model = RaftONNX(RAFT_ITERS).eval()
    f1 = cv2.resize(TEST_RGB, (RAFT_W, RAFT_H))
    dx, dy = RAFT_TEST_SHIFT
    f2 = np.roll(f1, shift=(dy, dx), axis=(0, 1))
    a, b = to_tensor01(f1), to_tensor01(f2)
    dst = ONNX_DIR / "raft_large.onnx"
    with torch.no_grad():
        how = onnx_export(model, (a, b), dst, input_names=["frame_a", "frame_b"], output_names=["flow"])
        ref = model(a, b).numpy()
    got = ort.InferenceSession(str(dst), providers=["CPUExecutionProvider"]).run(
        None, {"frame_a": a.numpy(), "frame_b": b.numpy()})[0]
    m = 32
    centre = lambda f: [float(np.median(f[0, c, m:-m, m:-m])) for c in (0, 1)]
    save_meta("raft_large", {"source": "torchvision raft_large, Raft_Large_Weights.DEFAULT (C_T_SKHT_V2)", "exporter": how,
                             "inputs": {"frame_a": [1, 3, RAFT_H, RAFT_W], "frame_b": [1, 3, RAFT_H, RAFT_W]},
                             "input_format": "RGB float32 in [0,1], NCHW; resize frames to W=960, H=544",
                             "output": "flow 1x2xHxW, pixels, channel 0 = x, 1 = y", "flow_updates": RAFT_ITERS,
                             "sha256": sha256(dst)})
    return {"file": dst.name, "exporter": how, "size_mb": round(dst.stat().st_size / 1e6, 1),
            "checks": {"true_shift_px": [dx, dy], "median_flow_pytorch": centre(ref), "median_flow_onnx": centre(got),
                       "max_abs_diff_px": float(np.abs(ref - got).max()), "mean_abs_diff_px": float(np.abs(ref - got).mean())}}


record("raft_large", export_raft)

# %% [markdown]
# ## 7. CLIP-EBC ViT-B/16 (NWPU) → ONNX (T3.3)
# Built with the repo's own code from its `config.json`, weights loaded strictly (see
# `crowdsafe/counting/density_model.py`). ONNX input: RGB in **[0, 1]**, `N×3×224×224`; output density `N×1×28×28`.
# Checks on the repo's crowd example images: PyTorch vs ONNX (windows), our stitching vs the repo's
# `sliding_window_predict`, and the whole-image count as `app.py` computes it.

# %%
from huggingface_hub import snapshot_download

from crowdsafe.counting.density_model import load_clipebc_torch, sliding_window

CLIPEBC_DIR = Path(snapshot_download(CLIPEBC_HF_REPO, repo_type="model", ignore_patterns=CLIPEBC_HF_IGNORE,
                                     local_dir=CACHE / "CLIP-EBC-hf"))


class ClipEbcONNX(torch.nn.Module):
    def __init__(self, net: torch.nn.Module) -> None:
        super().__init__()
        self.net = net.eval()
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net((x - self.mean) / self.std)


def export_clipebc(name: str) -> dict:
    import onnxruntime as ort

    cwd = os.getcwd()
    os.chdir(CLIPEBC_DIR)                      # the repo resolves some paths relative to itself
    try:
        net, cfg = load_clipebc_torch(CLIPEBC_DIR, CLIPEBC_DIR / CLIPEBC_HF_WEIGHTS)
    finally:
        os.chdir(cwd)
    net = net.cpu().eval()
    model = ClipEbcONNX(net).eval()
    win, red = cfg["input_size"], cfg["reduction"]
    dst = ONNX_DIR / f"{name}.onnx"
    with torch.no_grad():
        how = onnx_export(model, (torch.rand(2, 3, win, win),), dst, input_names=["image"], output_names=["density"],
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
        c = {"size_hw": list(rgb01.shape[:2])}
        try:
            d_t = sliding_window(run_torch, rgb01, win, red)
            d_o = sliding_window(run_onnx, rgb01, win, red)
            c.update(count_window_pytorch=float(d_t.sum()), count_window_onnx=float(d_o.sum()),
                     max_abs_diff_density=float(np.abs(d_t - d_o).max()))
        except Exception as e:
            c["window_check"] = f"failed: {e!r}"[:300]
        x = torch.from_numpy(rgb01).permute(2, 0, 1)[None]
        with torch.no_grad():
            try:
                c["count_whole_image_app_py"] = float(model(x).sum())
            except Exception as e:
                c["count_whole_image_app_py"] = f"not run: {e!r}"[:300]
            try:
                sys.path.insert(0, str(CLIPEBC_DIR))
                from utils.eval_utils import sliding_window_predict
                h, w = rgb01.shape[:2]
                H, W = -(-h // win) * win, -(-w // win) * win
                img = torch.zeros(1, 3, H, W)
                img[0, :, :h, :w] = x[0]
                c["count_repo_sliding_window"] = float(sliding_window_predict(net, (img - model.mean) / model.std, win, win).sum())
            except Exception as e:
                c["count_repo_sliding_window"] = f"not run: {e!r}"[:300]
        checks[img_name] = c
    save_meta(name, {"source": f"Hugging Face {CLIPEBC_HF_REPO}/{CLIPEBC_HF_WEIGHTS}", "config": cfg, "exporter": how,
                     "input": f"image Nx3x{win}x{win} RGB float32 in [0,1] (ImageNet normalisation built in)",
                     "output": f"density Nx1x{win // red}x{win // red}; sum = count",
                     "sliding_window": f"non-overlapping {win}px windows over zero-padded frame", "sha256": sha256(dst)})
    return {"file": dst.name, "exporter": how, "size_mb": round(dst.stat().st_size / 1e6, 1), "config": cfg, "checks": checks}


record("clipebc_vitb16_nwpu", lambda: export_clipebc("clipebc_vitb16_nwpu"))

# %% [markdown]
# ## 7b. CrowdHuman person + head YOLO11s (T3.2) — download and smoke test (already ONNX)

# %%
from huggingface_hub import hf_hub_download

from crowdsafe.counting.yolo_track import PersonHeadOnnx


def check_phd() -> dict:
    src = Path(hf_hub_download(PHD_HF_REPO, PHD_HF_FILE, local_dir=CACHE / "phd"))
    dst = ONNX_DIR / "yolo11s_crowdhuman_person_head.onnx"
    shutil.copy(src, dst)
    det = PersonHeadOnnx(dst, conf=PHD_CONF, iou=PHD_IOU)
    io = {"inputs": [(i.name, i.shape) for i in det.sess.get_inputs()],
          "outputs": [(o.name, o.shape) for o in det.sess.get_outputs()]}
    imgs = {"bus.jpg": TEST_BGR}
    for n in CLIPEBC_TEST_IMAGES:
        if (CLIPEBC_DIR / n).exists():
            imgs[n] = cv2.imread(str(CLIPEBC_DIR / n))
    checks = {}
    for n, im in imgs.items():
        d = det(im)
        checks[n] = {PHD_CLASSES[k]: int((d.class_id == k).sum()) for k in PHD_CLASSES}
    save_meta("yolo11s_crowdhuman_person_head", {"source": f"Hugging Face {PHD_HF_REPO}/{PHD_HF_FILE}", "classes": PHD_CLASSES,
                                                 "io": io, "input_format": "BGR, /255, letterbox pad 114, NCHW",
                                                 "defaults": {"conf": PHD_CONF, "nms_iou": PHD_IOU},
                                                 "license": "openrail++", "sha256": sha256(dst)})
    return {"file": dst.name, "size_mb": round(dst.stat().st_size / 1e6, 1), "io": io, "checks": checks}


record("yolo11s_crowdhuman_person_head", check_phd)

# %% [markdown]
# ## 8. GPU smoke timing (PyTorch, not ONNX) — a reference number, not the T9.2 runtime evaluation

# %%
def time_it(fn, n: int = 20) -> float:
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
        timing["raft_error"] = repr(e)[:300]
    try:
        yolo = YOLO(YOLO_WEIGHTS)
        frame = cv2.resize(TEST_BGR, (1920, 1080))
        timing["yolo11l_1080p_imgsz1280_ms"] = time_it(
            lambda: yolo.predict(frame, imgsz=1280, device=0, classes=[PERSON_CLASS], verbose=False), n=10)
    except Exception as e:
        timing["yolo_error"] = repr(e)[:300]
    timing["peak_gpu_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
else:
    timing["skipped"] = "no GPU visible"
REPORT["gpu_timing"] = timing
print(timing)

# %% [markdown]
# ## 9. Status table, report and zips — download both zips

# %%
import zipfile

REPORT["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
(OUT / "report.json").write_text(json.dumps(REPORT, indent=2, default=str))
lines = ["# Notebook 01 report", "", f"Finished: {REPORT['finished']} · commit {COMMIT}", "",
         "| Step | Status | Exporter | Size MB | Seconds | Error |", "| --- | --- | --- | --- | --- | --- |"]
for k, v in REPORT["exports"].items():
    err = (v.get("error") or "").splitlines()[0][:150] if v.get("error") else ""
    lines.append(f"| {k} | {v['status']} | {v.get('exporter', '')} | {v.get('size_mb', '')} | {v['seconds']} | {err} |")
(OUT / "report.md").write_text("\n".join(lines) + "\n")

report_zip, onnx_zip = WORK / "crowdsafe_nb01_report.zip", WORK / "crowdsafe_nb01_onnx.zip"
with zipfile.ZipFile(report_zip, "w", zipfile.ZIP_DEFLATED) as z:
    for p in OUT.rglob("*"):
        if p.is_file() and p.suffix != ".onnx":
            z.write(p, p.relative_to(OUT))
with zipfile.ZipFile(onnx_zip, "w", zipfile.ZIP_STORED) as z:
    for p in ONNX_DIR.iterdir():
        z.write(p, p.relative_to(OUT))
print("\n".join(lines))
print("\nDownload these files (right-click → Download in the file browser):")
print(" ", report_zip, f"({report_zip.stat().st_size / 1e6:.1f} MB)")
print(" ", onnx_zip, f"({onnx_zip.stat().st_size / 1e6:.1f} MB)")

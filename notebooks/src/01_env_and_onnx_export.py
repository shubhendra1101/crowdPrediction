# %% [markdown]
# # CrowdSafe — Notebook 01: A100 environment check + ONNX export
#
# **Tasks:** T0.2 (A100 environment), T0.6 (ONNX models for T3.1 / T3.2 / T3.3 / T4.1).
#
# **What it does**
# 1. Gets the project code from GitHub; installs packages **without replacing the container's PyTorch**.
# 2. Records the machine: GPU, CUDA, CPU/RAM limits, disk, internet access (never token values).
# 3. Exports **YOLO11l**, **RAFT-large** and **CLIP-EBC ViT-B/16 NWPU** to ONNX and downloads the
#    **CrowdHuman YOLO11s person+head** ONNX — each checked against PyTorch.
#
# **Crash-proofing:** the container sees 256 host cores but may use only a few (and ~16 GiB RAM). Threads are
# capped to the real quota, and **each export runs in its own process**: if one is killed (e.g. out of memory)
# the notebook keeps going and records why. YOLO retries once without ONNX simplification.
#
# **How to run:** `Kernel → Restart & Run All` (~15–25 min). The last cell prints a status table and two zips:
# - `crowdsafe_nb01_report.zip` (small) — **always download**
# - `crowdsafe_nb01_onnx.zip` (~0.6–1 GB) — the ONNX models

# %% [markdown]
# ## 1. Settings, project code, thread limit

# %%
import os
import subprocess
import sys
from pathlib import Path

WORK = Path(os.environ.get("CROWDSAFE_WORK", Path.home() / "crowdsafe_work"))
REPO_DIR = WORK / "crowdPrediction"
REPO_URL = "https://github.com/shubhendra1101/crowdPrediction.git"
OUT = WORK / "outputs"
ONNX_DIR = OUT / "onnx"
STEP_DIR = OUT / "steps"
LOG_DIR = OUT / "logs"
CACHE = WORK / "cache"
TEST_IMAGE_URL = "https://ultralytics.com/images/bus.jpg"
STEP_TIMEOUT_S = 3600
for d in (OUT, ONNX_DIR, STEP_DIR, LOG_DIR, CACHE):
    d.mkdir(parents=True, exist_ok=True)

if (REPO_DIR / ".git").exists():
    subprocess.run(["git", "-C", str(REPO_DIR), "pull", "--ff-only"], check=True)
else:
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(REPO_DIR)], check=True)
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))
COMMIT = subprocess.run(["git", "-C", str(REPO_DIR), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()

from crowdsafe import nbenv   # stdlib only — safe before numpy/torch

THREADS = nbenv.limit_threads()          # must happen before numpy / torch / cv2 are imported
os.environ["YOLO_AUTOINSTALL"] = "false"
print("Work dir:", WORK, "| code commit:", COMMIT, "| CPU threads capped to", THREADS)

# %% [markdown]
# ## 2. Install packages (PyTorch pinned), check OpenCV and GPU

# %%
PKGS = ["ultralytics>=8.3", "onnx>=1.16", "onnxruntime>=1.18", "onnxslim", "onnxscript", "huggingface_hub>=0.24",
        "safetensors", "einops", "ftfy", "regex", "timm==0.9.16", "tensorboardX", "scipy", "psutil",
        "requests", "matplotlib", "pyyaml", "supervision==0.30.6"]
PIP = nbenv.safe_pip(PKGS, log=LOG_DIR / "pip_install.log")
print("pip:", PIP)
print(nbenv.fix_cv2())
GPU = nbenv.gpu_summary(require=False)
print("GPU:", GPU)

# %% [markdown]
# ## 3. Environment report (T0.2)

# %%
import json
import shutil
import time

import psutil
import requests
import torch

torch.set_num_threads(THREADS)
REPORT: dict = {"notebook": "01_env_and_onnx_export", "commit": COMMIT, "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                "pip": PIP, "threads": THREADS, "exports": {}}


def reachable(url: str) -> bool:
    try:
        requests.get(url, timeout=10, stream=True)
        return True
    except Exception:
        return False


def read(path: str) -> str:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return "n/a"


env = {
    "python": sys.version.split()[0], "torch": torch.__version__, "torch_cuda_build": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "gpus": [{"name": torch.cuda.get_device_name(i),
              "total_mem_gb": round(torch.cuda.get_device_properties(i).total_memory / 1e9, 1)}
             for i in range(torch.cuda.device_count())],
    "cpu_seen": os.cpu_count(), "cpu_quota": THREADS, "cgroup_cpu_max": read("/sys/fs/cgroup/cpu.max"),
    "ram_limit_gb": round(int(read("/sys/fs/cgroup/memory.max")) / 1e9, 1) if read("/sys/fs/cgroup/memory.max").isdigit() else read("/sys/fs/cgroup/memory.max"),
    "ram_host_gb": round(psutil.virtual_memory().total / 1e9, 1),
    "disk_work_free_gb": round(shutil.disk_usage(WORK).free / 1e9, 1), "work_dir": str(WORK),
    "internet": {u: reachable(u) for u in ["https://huggingface.co", "https://github.com", "https://download.pytorch.org",
                                           "https://www.kaggle.com", "https://ped.fz-juelich.de"]},
    "HF_TOKEN_set": bool(os.environ.get("HF_TOKEN")),
}
(LOG_DIR / "nvidia_smi.txt").write_text(subprocess.run("nvidia-smi 2>&1", shell=True, capture_output=True, text=True).stdout)
REPORT["environment"] = env
print(json.dumps(env, indent=2))

# %% [markdown]
# ## 4. Test image and the step runner
# Each export runs as `python scripts/export_onnx.py <step>` in its own process (code in the repo, unit-tested).

# %%
test_img = CACHE / "bus.jpg"
if not test_img.exists():
    try:
        test_img.write_bytes(requests.get(TEST_IMAGE_URL, timeout=60).content)
    except Exception:
        import ultralytics
        shutil.copy(Path(ultralytics.__file__).parent / "assets" / "bus.jpg", test_img)


def run_step(step: str, *extra: str) -> dict:
    """Run one export step in a subprocess; record its JSON result or the reason it died."""
    result = STEP_DIR / f"{step}.json"
    result.unlink(missing_ok=True)
    cmd = [sys.executable, str(REPO_DIR / "scripts" / "export_onnx.py"), step, "--onnx-dir", str(ONNX_DIR),
           "--cache", str(CACHE), "--test-image", str(test_img), "--result", str(result), *extra]
    t0 = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=STEP_TIMEOUT_S, cwd=str(REPO_DIR))
        code, log = p.returncode, p.stdout + "\n" + p.stderr
    except subprocess.TimeoutExpired as e:
        code, log = "timeout", f"timed out after {STEP_TIMEOUT_S} s\n{e.stdout or ''}\n{e.stderr or ''}"
    (LOG_DIR / f"{step}{'_' + '_'.join(extra) if extra else ''}.log").write_text(str(log))
    if code == 0 and result.exists():
        out = json.loads(result.read_text())
    else:
        tail = [ln for ln in str(log).strip().splitlines() if ln.strip()][-12:]
        out = {"status": "failed", "exit": code,
               "reason": nbenv.exit_reason(code) if isinstance(code, int) else "timeout",
               "error": "\n".join(tail)[-1500:], "seconds": round(time.time() - t0, 1)}
    print(step, "->", {k: v for k, v in out.items() if k != "error"})
    if out.get("status") == "failed":
        print("   last lines:\n   " + out["error"].replace("\n", "\n   "))
    return out

# %% [markdown]
# ## 5. YOLO11l → ONNX (T3.1) — retries without simplification if the first try fails

# %%
res = run_step("yolo")
if res.get("status") != "ok":
    print("retrying YOLO export without onnxslim simplification")
    res = run_step("yolo", "--no-simplify")
REPORT["exports"]["yolo11l"] = res

# %% [markdown]
# ## 6. RAFT-large → ONNX (T4.1) — weights from a SHA-256-verified HF mirror (download.pytorch.org is blocked)

# %%
REPORT["exports"]["raft_large"] = run_step("raft")

# %% [markdown]
# ## 7. CLIP-EBC ViT-B/16 (NWPU) → ONNX (T3.3)

# %%
REPORT["exports"]["clipebc_vitb16_nwpu"] = run_step("clipebc")

# %% [markdown]
# ## 7b. CrowdHuman person + head YOLO11s (T3.2) — download + smoke test (already ONNX)

# %%
REPORT["exports"]["yolo11s_crowdhuman_person_head"] = run_step("phd")

# %% [markdown]
# ## 8. GPU smoke timing (reference only)

# %%
REPORT["gpu_timing"] = run_step("timing")

# %% [markdown]
# ## 9. Status table, report and zips — download both

# %%
import zipfile

REPORT["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
(OUT / "report.json").write_text(json.dumps(REPORT, indent=2, default=str))
lines = ["# Notebook 01 report", "", f"Finished: {REPORT['finished']} · commit {COMMIT} · threads {THREADS}", "",
         "| Step | Status | Exporter | Size MB | Seconds | Problem |", "| --- | --- | --- | --- | --- | --- |"]
for k, v in REPORT["exports"].items():
    problem = v.get("reason", "") + (": " + v["error"].splitlines()[-1][:140] if v.get("error") else "")
    lines.append(f"| {k} | {v.get('status')} | {v.get('exporter', '')} | {v.get('size_mb', '')} | {v.get('seconds', '')} | {problem} |")
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
print("\nDownload these files (right-click → Download):")
print(" ", report_zip, f"({report_zip.stat().st_size / 1e6:.1f} MB)")
print(" ", onnx_zip, f"({onnx_zip.stat().st_size / 1e6:.1f} MB)")

"""Environment helpers for the A100 notebooks (NVIDIA PyTorch container on Kubeflow).

The lab image ships NVIDIA's own PyTorch build (e.g. ``2.7.0a0+...nv25.03``) matched to the
host driver. A plain ``pip install`` of packages that depend on torch can replace it with a
PyPI wheel that does not work with that driver. :func:`safe_pip` pins the installed torch
family (and numpy, whose ABI other container packages rely on) and verifies nothing changed.
"""
from __future__ import annotations

import hashlib
import importlib.metadata as md
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PINNED = ["torch", "torchvision", "torchaudio", "triton", "pytorch-triton", "numpy"]
TORCH_FAMILY = {"torch", "torchvision", "torchaudio", "triton", "pytorch-triton"}


def installed(pkg: str) -> str | None:
    """Installed version of a distribution, or None."""
    try:
        return md.version(pkg)
    except md.PackageNotFoundError:
        return None


def constraints_text() -> str:
    """pip constraints pinning the currently installed torch family and numpy."""
    return "\n".join(f"{p}=={v}" for p in PINNED if (v := installed(p))) + "\n"


def _pip(args: list[str], constraints: Path) -> subprocess.CompletedProcess:
    cmd = [sys.executable, "-m", "pip", "install", "-q", "--disable-pip-version-check", "-c", str(constraints), *args]
    return subprocess.run(cmd, capture_output=True, text=True)


def _base_name(req: str) -> str:
    return re.split(r"[\s\[<>=!~;]", req.strip(), maxsplit=1)[0].lower().replace("_", "-")


def safe_pip(pkgs: list[str], log: Path | None = None) -> dict:
    """Install packages without replacing the container's torch/numpy.

    Tries a normal constrained install; if the resolver cannot satisfy it, installs each
    package with ``--no-deps`` and then its non-torch dependencies (constrained).
    Raises RuntimeError if a pinned package changed version anyway.
    """
    before = {p: installed(p) for p in PINNED}
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(constraints_text())
        cons = Path(f.name)
    logs, mode = [], "constrained"
    r = _pip(pkgs, cons)
    logs.append(r.stdout + r.stderr)
    if r.returncode != 0:
        mode = "no-deps fallback"
        for pkg in pkgs:
            r1 = _pip(["--no-deps", pkg], cons)
            logs.append(r1.stdout + r1.stderr)
            name = _base_name(pkg)
            deps = [d.split(";")[0].strip() for d in (md.requires(name) or []) if "extra ==" not in d]
            deps = [d for d in deps if _base_name(d) not in TORCH_FAMILY]
            if deps:
                r2 = _pip(deps, cons)
                logs.append(r2.stdout + r2.stderr)
    after = {p: installed(p) for p in PINNED}
    if log:
        Path(log).write_text("\n".join(logs))
    changed = {p: (before[p], after[p]) for p in PINNED if before[p] and before[p] != after[p]}
    if changed:
        raise RuntimeError(f"pip changed pinned packages {changed} — restart the kernel and tell the agent.")
    missing = [p for p in pkgs if not installed(_base_name(p))]
    return {"mode": mode, "missing": missing, "torch": after["torch"], "numpy": after["numpy"]}


def cv2_ok() -> tuple[bool, str]:
    """Whether ``import cv2`` works in a fresh interpreter (and the error if not)."""
    r = subprocess.run([sys.executable, "-c", "import cv2; print(cv2.__version__)"], capture_output=True, text=True)
    return r.returncode == 0, (r.stdout + r.stderr).strip()[-300:]


def fix_cv2() -> str:
    """Containers often lack libGL: replace GUI OpenCV with the headless build if import fails."""
    ok, msg = cv2_ok()
    if ok:
        return f"cv2 ok {msg}"
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "opencv-python", "opencv-contrib-python"],
                   capture_output=True, text=True)
    safe_pip(["opencv-python-headless"])
    ok, msg = cv2_ok()
    if not ok:
        raise RuntimeError(f"OpenCV still fails to import: {msg}")
    return f"cv2 fixed (headless) {msg}"


def gpu_summary(require: bool = True) -> dict:
    """Torch/CUDA/GPU facts; raise if ``require`` and no GPU is visible."""
    import torch

    info = {"torch": torch.__version__, "cuda_build": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    if require and not info["cuda_available"]:
        raise RuntimeError("No GPU visible to PyTorch. Check the Kubeflow server has a GPU; "
                           "if it is days old, recreate it (see nvidia-smi).")
    return info


def sha256(path: Path) -> str:
    """SHA-256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


RAFT_LARGE_FILE = "raft_large_C_T_SKHT_V2-ff5fadd5.pth"
RAFT_LARGE_SHA256 = "ff5fadd56d26b40647388883af1547351ea17868b765c05b27231e72dd16a322"
RAFT_HF_MIRROR = ("Dominoc/raft_large", RAFT_LARGE_FILE)


def ensure_raft_large_weights() -> str:
    """Make torchvision's RAFT-large weights available offline.

    download.pytorch.org is blocked on the lab network, so if the file is not cached, fetch the
    identical file from a Hugging Face mirror, verify its SHA-256 against the hash torchvision
    encodes in the file name, and place it in torch hub's checkpoint cache.
    """
    import torch

    dst = Path(torch.hub.get_dir()) / "checkpoints" / RAFT_LARGE_FILE
    if dst.exists() and sha256(dst) == RAFT_LARGE_SHA256:
        return f"cached {dst}"
    from huggingface_hub import hf_hub_download

    src = Path(hf_hub_download(*RAFT_HF_MIRROR))
    digest = sha256(src)
    if digest != RAFT_LARGE_SHA256:
        raise RuntimeError(f"RAFT mirror hash mismatch: {digest}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dst)
    return f"from HF mirror {RAFT_HF_MIRROR[0]}, sha256 verified -> {dst}"

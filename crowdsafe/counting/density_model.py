"""CLIP-EBC density model (architecture.md §4 DensityModel): RGB frame in, density map out.

Two backends with identical output:
- ``torch``: rebuilds the model from the official Hugging Face repo ``Yiming-M/CLIP-EBC``
  (code + ``nwpu_weights/CLIP_EBC_ViT_B_16``), loading weights as its ``app.py`` does.
- ``onnx``: the 224×224 window model exported by notebook 01 (normalisation built in).

Two inference modes:
- ``window`` (default): non-overlapping windows of the training input size over a zero-padded
  frame, stitched back together — what the repo's evaluation uses for ViT models.
- ``whole``: one forward pass on the whole frame, as ``app.py`` does (torch backend only).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable

import numpy as np

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def sliding_window(run: Callable[[np.ndarray], np.ndarray], rgb01: np.ndarray, win: int, reduction: int,
                   max_batch: int = 64) -> np.ndarray:
    """Run ``run`` (N×3×win×win -> N×1×(win/r)×(win/r)) over non-overlapping windows of a zero-padded
    image and stitch the outputs. Returns the density map cropped to ceil(h/r)×ceil(w/r)."""
    h, w = rgb01.shape[:2]
    H, W = -(-h // win) * win, -(-w // win) * win
    pad = np.zeros((H, W, 3), np.float32)
    pad[:h, :w] = rgb01
    coords = [(y, x) for y in range(0, H, win) for x in range(0, W, win)]
    r = win // reduction
    out = np.zeros((H // reduction, W // reduction), np.float32)
    for i in range(0, len(coords), max_batch):
        chunk = coords[i:i + max_batch]
        batch = np.stack([pad[y:y + win, x:x + win].transpose(2, 0, 1) for y, x in chunk])
        dens = run(batch)
        for (y, x), d in zip(chunk, dens):
            out[y // reduction:y // reduction + r, x // reduction:x // reduction + r] = d[0]
    return out[:-(-h // reduction), :-(-w // reduction)]


def load_clipebc_torch(repo_dir: Path, weights_dir: Path):
    """Build CLIP-EBC with the repo's get_model from config.json and load model.safetensors strictly."""
    from safetensors.torch import load_file

    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))
    from models import get_model  # from the CLIP-EBC repo

    cfg = json.loads((Path(weights_dir) / "config.json").read_text())
    net = get_model(backbone=cfg["backbone"], input_size=cfg["input_size"], reduction=cfg["reduction"],
                    bins=[(float(a), float(b)) for a, b in cfg["bins"]],
                    anchor_points=[float(p) for p in cfg["anchor_points"]], prompt_type=cfg["prompt_type"],
                    num_vpt=cfg["num_vpt"], vpt_drop=cfg["vpt_drop"], deep_vpt=cfg["deep_vpt"])
    sd = load_file(str(Path(weights_dir) / "model.safetensors"))
    for mapping in (lambda k: k[6:] if k.startswith("model.") else k, lambda k: k.replace("model.", ""), lambda k: k):
        try:
            net.load_state_dict({mapping(k): v for k, v in sd.items()}, strict=True)
            return net.eval(), cfg
        except RuntimeError as err:
            last = err
    raise last


class ClipEbcDensity:
    """CLIP-EBC density model. ``__call__(rgb uint8 H×W×3) -> density map``; ``.count(rgb)`` sums it."""

    def __init__(self, backend: str = "torch", mode: str = "window", repo_dir: Path | None = None,
                 weights_dir: Path | None = None, onnx_path: Path | None = None, device: str = "cuda",
                 input_size: int = 224, reduction: int = 8, max_batch: int = 64) -> None:
        self.backend, self.mode, self.max_batch = backend, mode, max_batch
        if backend == "torch":
            import torch

            self.torch = torch
            self.net, cfg = load_clipebc_torch(Path(repo_dir), Path(weights_dir))
            self.device = device if torch.cuda.is_available() or device == "cpu" else "cpu"
            self.net.to(self.device)
            self.win, self.red = cfg["input_size"], cfg["reduction"]
        elif backend == "onnx":
            if mode != "window":
                raise ValueError("The ONNX model takes fixed windows; use mode='window'.")
            import onnxruntime as ort

            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device != "cpu" else ["CPUExecutionProvider"]
            self.sess = ort.InferenceSession(str(onnx_path), providers=providers)
            self.win, self.red = input_size, reduction
        else:
            raise ValueError(f"Unknown backend {backend}")

    def _run_window(self, batch01: np.ndarray) -> np.ndarray:
        if self.backend == "onnx":
            return self.sess.run(None, {"image": batch01.astype(np.float32)})[0]
        x = (batch01 - IMAGENET_MEAN[:, None, None]) / IMAGENET_STD[:, None, None]
        with self.torch.no_grad():
            return self.net(self.torch.from_numpy(x.astype(np.float32)).to(self.device)).float().cpu().numpy()

    def _run_whole(self, rgb01: np.ndarray) -> np.ndarray:
        """app.py style: upscale if a side is below the input size, normalise, one forward pass."""
        import torch.nn.functional as F

        x = self.torch.from_numpy(((rgb01 - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1).copy())[None]
        h, w = x.shape[-2:]
        if h < self.win or w < self.win:
            ratio = max(self.win / h, self.win / w)
            x = F.interpolate(x, size=(int(h * ratio) + 1, int(w * ratio) + 1), mode="bicubic", antialias=True)
        with self.torch.no_grad():
            return self.net(x.to(self.device)).float().cpu().numpy()[0, 0]

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        """Density map for an RGB uint8 frame; its sum is the estimated person count."""
        rgb01 = rgb.astype(np.float32) / 255.0
        if self.mode == "whole":
            return self._run_whole(rgb01)
        return sliding_window(self._run_window, rgb01, self.win, self.red, self.max_batch)

    def count(self, rgb: np.ndarray) -> float:
        """Estimated number of people in the frame."""
        return float(self(rgb).sum())

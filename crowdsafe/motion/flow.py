"""Optical flow between two frames (T4.1; architecture.md §2.3).

`FlowModel(frame_a, frame_b) -> H×W×2` pixel displacement (x, y) at the input resolution.
Backends: RAFT-large (torchvision, pretrained) on GPU — frames resized to ~`width` px (multiple of 8)
— and OpenCV Farneback as the CPU fallback / laptop test.
"""
from __future__ import annotations

import cv2
import numpy as np


class FlowModel:
    def __init__(self, backend: str = "raft", width: int = 960, device: str = "cuda", iters: int = 12):
        self.backend, self.width, self.iters = backend, width, iters
        if backend == "raft":
            import torch
            from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

            self.torch = torch
            self.device = device if torch.cuda.is_available() else "cpu"
            self.net = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).eval().to(self.device)
        elif backend != "farneback":
            raise ValueError(f"Unknown flow backend {backend}")

    def _size(self, h: int, w: int) -> tuple[int, int]:
        s = self.width / w
        return int(round(h * s / 8)) * 8, int(round(w * s / 8)) * 8

    def __call__(self, frame_a: np.ndarray, frame_b: np.ndarray) -> np.ndarray:
        """BGR uint8 frames -> flow (H, W, 2) in pixels of the input frames."""
        h, w = frame_a.shape[:2]
        sh, sw = self._size(h, w)
        a, b = cv2.resize(frame_a, (sw, sh)), cv2.resize(frame_b, (sw, sh))
        if self.backend == "farneback":
            ga, gb = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
            f = cv2.calcOpticalFlowFarneback(ga, gb, None, 0.5, 4, 21, 3, 5, 1.1, 0)
        else:
            t = self.torch
            x = t.from_numpy(np.stack([cv2.cvtColor(a, cv2.COLOR_BGR2RGB), cv2.cvtColor(b, cv2.COLOR_BGR2RGB)]))
            x = x.permute(0, 3, 1, 2).float().div(255).mul(2).sub(1).to(self.device)   # RAFT expects [-1, 1]
            with t.no_grad():
                f = self.net(x[:1], x[1:], num_flow_updates=self.iters)[-1][0].permute(1, 2, 0).cpu().numpy()
        f = cv2.resize(f, (w, h))
        f[..., 0] *= w / sw
        f[..., 1] *= h / sh
        return f

"""Edge helpers for calibration: snap rough clicks to the strongest nearby image edge (T2.3).

Used to turn a few rough points along a physically straight edge (wall corner, door frame, ceiling
duct) into dense sub-pixel samples for the plumb-line constraint, and to tighten desk-top outlines.
"""
from __future__ import annotations

import cv2
import numpy as np


def gradient(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sobel gradients of a lightly blurred grayscale image."""
    g = cv2.GaussianBlur(gray.astype(np.float32), (3, 3), 0)
    return cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)


def trace_edge(gray: np.ndarray, rough: np.ndarray, spacing_px: float = 8.0, search_px: float = 6.0,
               min_grad: float = 20.0) -> np.ndarray:
    """Dense edge samples along a rough polyline.

    The polyline (≥ 2 points, in order) is resampled every `spacing_px`; each sample moves along the
    local normal (± `search_px`) to the pixel with the largest gradient across the polyline. Samples
    with a weak edge (< `min_grad`) are dropped. Returns (N, 2) float points.
    """
    gx, gy = gradient(gray)
    h, w = gray.shape[:2]
    rough = np.asarray(rough, float)
    out = []
    for a, b in zip(rough[:-1], rough[1:]):
        seg = b - a
        n_s = max(2, int(np.linalg.norm(seg) / spacing_px))
        d = seg / np.linalg.norm(seg)
        nrm = np.array([-d[1], d[0]])
        for t in np.linspace(0, 1, n_s, endpoint=False):
            c = a + t * seg
            best, bv = None, min_grad
            for s in np.arange(-search_px, search_px + 0.25, 0.5):
                x, y = c + s * nrm
                if 1 <= x < w - 1 and 1 <= y < h - 1:
                    v = abs(gx[int(y), int(x)] * nrm[0] + gy[int(y), int(x)] * nrm[1])
                    if v > bv:
                        best, bv = (x, y), v
            if best is not None:
                out.append(best)
    return np.asarray(out, float).reshape(-1, 2)


def snap_quad(gray: np.ndarray, quad: np.ndarray, search_px: float = 7.0, min_grad: float = 20.0,
              n_samples: int = 25) -> np.ndarray | None:
    """Tighten a rough bright quadrilateral (e.g. a desk top) to its edges.

    Each side is snapped to the strongest bright-inside/dark-outside edge near it and fitted with a
    robust line; corners are the intersections of neighbouring side lines. Returns (4, 2) or None if a
    side has too few edge samples.
    """
    gx, gy = gradient(gray)
    h, w = gray.shape[:2]
    Q = np.asarray(quad, float)
    cen = Q.mean(0)
    lines = []
    for i in range(4):
        p, q = Q[i], Q[(i + 1) % 4]
        d = (q - p) / np.linalg.norm(q - p)
        nrm = np.array([-d[1], d[0]])
        if np.dot(cen - (p + q) / 2, nrm) > 0:
            nrm = -nrm                                    # outward normal
        pts = []
        for t in np.linspace(0.15, 0.85, n_samples):
            c = p + t * (q - p)
            best, bv = None, min_grad
            for s in np.arange(-search_px, search_px + 0.25, 0.5):
                x, y = c + s * nrm
                if 1 <= x < w - 1 and 1 <= y < h - 1:
                    v = -(gx[int(y), int(x)] * nrm[0] + gy[int(y), int(x)] * nrm[1])
                    if v > bv:
                        best, bv = c + s * nrm, v
            if best is not None:
                pts.append(best)
        if len(pts) < 6:
            return None
        vx, vy, x0, y0 = cv2.fitLine(np.asarray(pts, np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        lines.append((np.array([x0, y0]), np.array([vx, vy])))
    corners = []
    for i in range(4):
        (p, d), (q, e) = lines[i - 1], lines[i]
        t = np.linalg.solve(np.array([d, -e]).T, q - p)
        corners.append(p + t[0] * d)
    return np.asarray(corners)

"""Image ↔ floor calibration with a planar homography (architecture.md §2.1, T2.3).

A homography H (3×3) maps image pixels of points *on the floor* to floor coordinates in metres.
It is fitted from ≥ 4 image points whose floor positions are known (room corners, desk-leg feet,
tape marks). With more than 4 points it uses RANSAC and reports the reprojection error, and an
extra known distance that was not used for fitting validates it (target < 10 % error, T2.5).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from shapely.geometry import Polygon


@dataclass
class FloorCalibration:
    """Pixel → metre mapping for one camera."""

    H: np.ndarray                                  # 3×3, image px -> floor m
    rms_error_m: float                             # reprojection error of the fitting points, metres
    n_points: int
    note: str = ""
    inliers: list[bool] = field(default_factory=list)

    def to_floor(self, px: np.ndarray) -> np.ndarray:
        """(N, 2) image pixels -> (N, 2) floor metres."""
        pts = np.asarray(px, np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.H).reshape(-1, 2)

    def to_image(self, xy_m: np.ndarray) -> np.ndarray:
        """(N, 2) floor metres -> (N, 2) image pixels."""
        pts = np.asarray(xy_m, np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, np.linalg.inv(self.H)).reshape(-1, 2)

    def area_m2(self, polygon_px: list[tuple[float, float]]) -> float:
        """Floor area (m²) of a zone polygon drawn on the image floor."""
        return float(Polygon(self.to_floor(np.asarray(polygon_px))).area)

    def distance_m(self, a_px: tuple[float, float], b_px: tuple[float, float]) -> float:
        """Floor distance (m) between two image points on the floor."""
        a, b = self.to_floor(np.array([a_px, b_px]))
        return float(np.hypot(*(a - b)))

    def validate(self, a_px, b_px, true_m: float) -> dict:
        """Compare a known distance (not used for fitting) with the calibrated one."""
        est = self.distance_m(a_px, b_px)
        return {"true_m": true_m, "estimated_m": round(est, 3), "error_pct": round(100 * abs(est - true_m) / true_m, 1)}


def fit(image_px: list[tuple[float, float]], floor_m: list[tuple[float, float]], note: str = "",
        ransac_px: float = 8.0) -> FloorCalibration:
    """Fit H from matching image/floor points (≥ 4; RANSAC when > 4)."""
    src = np.asarray(image_px, np.float64)
    dst = np.asarray(floor_m, np.float64)
    if len(src) < 4 or len(src) != len(dst):
        raise ValueError("Need at least 4 matching image/floor points")
    if len(src) == 4:
        H, mask = cv2.findHomography(src, dst, 0)
    else:
        # RANSAC threshold is in floor units (m); convert the pixel tolerance roughly via point spread
        scale = np.ptp(dst, axis=0).max() / max(np.ptp(src, axis=0).max(), 1)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransac_px * scale)
    if H is None:
        raise ValueError("Homography could not be estimated (points collinear or inconsistent)")
    cal = FloorCalibration(H, 0.0, len(src), note, [bool(m) for m in mask.ravel()] if mask is not None else [])
    err = np.linalg.norm(cal.to_floor(src) - dst, axis=1)
    cal.rms_error_m = float(np.sqrt(np.mean(err ** 2)))
    return cal


def line_intersection(p1, p2, q1, q2) -> tuple[float, float]:
    """Intersection of line p1–p2 with line q1–q2 (image px) — e.g. a hidden room corner from two wall–floor lines."""
    (x1, y1), (x2, y2), (x3, y3), (x4, y4) = p1, p2, q1, q2
    d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(d) < 1e-9:
        raise ValueError("Lines are parallel")
    a, b = x1 * y2 - y1 * x2, x3 * y4 - y3 * x4
    return ((a * (x3 - x4) - (x1 - x2) * b) / d, (a * (y3 - y4) - (y1 - y2) * b) / d)

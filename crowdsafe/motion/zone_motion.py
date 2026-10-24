"""Pixel flow -> per-zone crowd motion in m/s (T4.2; architecture.md §2.3).

Work on the density-map grid (one cell = `reduction` px, 8 for CLIP-EBC): each cell's mean flow
moves its centre from p0 to p1 = p0 + flow. Both points are traced to the head-height plane through
the calibrated camera, so velocity = (XY(p1) − XY(p0)) / dt in m/s. Cells are weighted by the density
map (people per cell), so empty floor and static furniture contribute nothing.

Per zone (same definitions as the trajectory features in crowdsafe.datasets.julich):
mean_speed (weighted), speed_var (weighted variance of vx + vy), dir_entropy (8-bin, normalised,
moving cells only), counterflow (weighted share moving against the zone's mean direction).
"""
from __future__ import annotations

import cv2
import numpy as np

from crowdsafe.calibration.camera_model import CameraModel


class ZoneMotion:
    def __init__(self, cam: CameraModel, masks: dict[str, np.ndarray], z: float, min_cell_density: float = 0.02,
                 min_speed: float = 0.1, max_speed: float = 4.0, bins: int = 8):
        self.cam, self.masks, self.z = cam, masks, z
        self.h, self.w = next(iter(masks.values())).shape
        self.sx, self.sy = cam.image_size[0] / self.w, cam.image_size[1] / self.h
        jj, ii = np.meshgrid(np.arange(self.w), np.arange(self.h))
        self.p0 = np.c_[(jj.ravel() + 0.5) * self.sx, (ii.ravel() + 0.5) * self.sy]
        self.xy0 = cam.to_plane(self.p0, z)                    # fixed per camera: computed once
        self.min_d, self.min_speed, self.max_speed, self.bins = min_cell_density, min_speed, max_speed, bins

    def __call__(self, flow: np.ndarray, density: np.ndarray, dt: float) -> dict[str, dict]:
        """flow: H×W×2 px (full frame); density: h×w map; returns zone -> motion features (NaN if no people)."""
        f = cv2.resize(flow, (self.w, self.h), interpolation=cv2.INTER_AREA).reshape(-1, 2)
        d = density.reshape(-1).astype(float)
        use = (d >= self.min_d) & np.isfinite(self.xy0).all(1)
        v = np.full((len(d), 2), np.nan)
        if use.any():
            xy1 = self.cam.to_plane(self.p0[use] + f[use], self.z)
            v[use] = (xy1 - self.xy0[use]) / dt
        sp = np.hypot(v[:, 0], v[:, 1])
        ok = use & np.isfinite(sp) & (sp <= self.max_speed)
        out = {}
        for zone, m in self.masks.items():
            sel = ok & (m.reshape(-1) > 0)
            if d[sel].sum() <= 0:
                out[zone] = {"mean_speed": np.nan, "speed_var": np.nan, "dir_entropy": np.nan, "counterflow": np.nan}
                continue
            wgt, vz, sz = d[sel], v[sel], sp[sel]
            W = wgt.sum()
            mv = (vz * wgt[:, None]).sum(0) / W
            var = float((wgt * ((vz - mv) ** 2).sum(1)).sum() / W)
            mov = sz >= self.min_speed
            ent = cf = np.nan
            if mov.any():
                ang = np.arctan2(vz[mov, 1], vz[mov, 0])
                hist = np.histogram(ang, bins=self.bins, range=(-np.pi, np.pi), weights=wgt[mov])[0]
                p = hist[hist > 0] / hist.sum()
                ent = float(-(p * np.log(p)).sum() / np.log(self.bins))
                mm = (vz[mov] * wgt[mov, None]).sum(0)
                if np.hypot(*mm) > 1e-9:
                    cf = float(wgt[mov][(vz[mov] @ mm) < 0].sum() / wgt[mov].sum())
            out[zone] = {"mean_speed": float((wgt * sz).sum() / W), "speed_var": var, "dir_entropy": ent, "counterflow": cf}
        return out

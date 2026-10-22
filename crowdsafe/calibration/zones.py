"""Zones on the floor plan and their image footprints (T2.4): counts -> persons/m².

A zone is a floor polygon (metres). People are localised by their *head*: the image point of a head is
traced back along the camera ray to a horizontal plane at head height (seated ≈ 1.2 m), which gives
its floor position. The same plane is used to build each zone's image mask, so a density map summed
inside the mask and labelled head points assigned by ray give the same zone membership.

Density = people in zone / *visible* zone area (the part of the zone whose head-height plane falls
inside the image), in persons/m².
"""
from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.path as mpath
import numpy as np
import yaml

from crowdsafe.calibration.camera_model import CameraModel


def load_room(room: str, cameras_dir: Path = Path("configs/cameras"),
              zones_dir: Path = Path("configs/zones")) -> tuple[CameraModel, dict[str, list], float]:
    """Camera model, zone polygons and head height for one room."""
    cam_cfg = yaml.safe_load((Path(cameras_dir) / f"{room}.yaml").read_text(encoding="utf-8"))
    zcfg = yaml.safe_load((Path(zones_dir) / f"{room}.yaml").read_text(encoding="utf-8"))
    return CameraModel.from_dict(cam_cfg["camera"]), zcfg["zones"], float(zcfg.get("head_height_m", 1.2))


def camera_rooms(zones_dir: Path = Path("configs/zones")) -> dict[str, str]:
    """camera id -> room, from the zone files."""
    out = {}
    for p in sorted(Path(zones_dir).glob("*.yaml")):
        z = yaml.safe_load(p.read_text(encoding="utf-8"))
        out.update({c: z["room"] for c in z["cameras"]})
    return out


def zone_masks(cam: CameraModel, zones: dict[str, list], z: float, shape_hw: tuple[int, int],
               step_m: float = 0.05) -> dict[str, np.ndarray]:
    """name -> uint8 mask (shape_hw) of pixels whose ray meets plane Z = z inside the zone.

    Built by rasterising the zone at `step_m` on the plane and projecting each cell, which stays correct
    when parts of the zone are outside the image or bent by lens distortion.
    """
    h, w = shape_hw
    sx, sy = w / cam.image_size[0], h / cam.image_size[1]
    out = {}
    for name, poly in zones.items():
        P = np.asarray(poly, float)
        xs = np.arange(P[:, 0].min(), P[:, 0].max() + step_m, step_m)
        ys = np.arange(P[:, 1].min(), P[:, 1].max() + step_m, step_m)
        gx, gy = np.meshgrid(xs, ys)
        pts = np.c_[gx.ravel(), gy.ravel()]
        pts = pts[_inside_fast(P, pts)]
        xyz = np.c_[pts, np.full(len(pts), z)]
        ok = cam.in_view(xyz)
        m = np.zeros((h, w), np.uint8)
        if ok.any():
            uv = cam.project(xyz[ok]) * [sx, sy]
            ij = np.floor(uv).astype(int)
            good = (ij[:, 0] >= 0) & (ij[:, 0] < w) & (ij[:, 1] >= 0) & (ij[:, 1] < h)
            m[ij[good, 1], ij[good, 0]] = 1
            # close the gaps between projected cells (cells near the camera spread over several px)
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        out[name] = m
    return out


def _inside_fast(poly: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return mpath.Path(poly).contains_points(pts)


def visible_area_m2(cam: CameraModel, poly: list, z: float, step_m: float = 0.05) -> float:
    """Area (m²) of the zone part whose head-height plane is inside the image."""
    P = np.asarray(poly, float)
    xs = np.arange(P[:, 0].min() + step_m / 2, P[:, 0].max(), step_m)
    ys = np.arange(P[:, 1].min() + step_m / 2, P[:, 1].max(), step_m)
    gx, gy = np.meshgrid(xs, ys)
    pts = np.c_[gx.ravel(), gy.ravel()]
    pts = pts[_inside_fast(P, pts)]
    ok = cam.in_view(np.c_[pts, np.full(len(pts), z)])
    return float(ok.sum() * step_m ** 2)


def assign_points(cam: CameraModel, zones: dict[str, list], px: np.ndarray, z: float) -> list[str | None]:
    """Zone name (or None) for each image point, via the ray to plane Z = z."""
    if len(px) == 0:
        return []
    xy = cam.to_plane(np.asarray(px, float).reshape(-1, 2), z)
    polys = {n: np.asarray(p, float) for n, p in zones.items()}
    out = []
    for p in xy:
        hit = None
        if not np.isnan(p).any():
            for n, P in polys.items():
                if _inside_fast(P, p[None])[0]:
                    hit = n
                    break
        out.append(hit)
    return out

"""Smoke tests for the camera model, the desk-rectangle fit, the edge fit and zones (synthetic camera)."""
import cv2
import numpy as np
import pytest

from crowdsafe.calibration.camera_model import (CameraModel, _look_rotation, fit_from_edges, fit_from_rectangles,
                                                rect_corners)
from crowdsafe.calibration.zones import assign_points, visible_area_m2, zone_masks

W, H = 1920, 1080


def _camera(f=1300.0, k1=-0.25, k2=0.05, center=(2.5, -2.2, 1.45), yaw=-35, pitch=28) -> CameraModel:
    R = _look_rotation(np.radians(yaw), np.radians(pitch))
    C = np.array(center, float)
    return CameraModel(np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]]), np.array([k1, k2, 0, 0, 0]),
                       cv2.Rodrigues(R)[0].ravel(), -R @ C, (W, H))


def _desks(cam, L=1.0, d=0.38, cols=(-3.4, -1.7, 0.0, 1.7), rows=range(0, 6), pitch=0.96):
    obs, truth = {}, {}
    for c, x in enumerate(cols):
        for r in rows:
            P = rect_corners(x, r * pitch, L, d)
            if cam.in_view(P, margin_px=-20).all():
                obs[f"c{c}r{r}"] = cam.project(P)
                truth[f"c{c}r{r}"] = (x, r * pitch)
    return obs, truth


def test_rectangle_fit_recovers_camera_and_desk_layout() -> None:
    cam = _camera()
    obs, truth = _desks(cam)
    ref = "c2r0"                                   # at (0, 0)
    rng = np.random.default_rng(42)
    noisy = {k: v + rng.normal(0, 0.7, v.shape) for k, v in obs.items()}
    fit = fit_from_rectangles(noisy, 1.0, (W, H), ref=ref)
    assert fit.rms_px < 2.0
    assert fit.params["f_px"] == pytest.approx(1300, rel=0.05)
    assert fit.camera.center[2] == pytest.approx(1.45, abs=0.08)
    for k, (x, y) in truth.items():
        assert np.allclose(fit.positions[k], (x, y), atol=0.08), k


def test_to_plane_inverts_project_and_floor_shift() -> None:
    cam = _camera()
    pts = np.array([[0.5, 1.0, 0.0], [-2.0, 4.0, 0.0], [1.0, 6.0, 0.0]])
    assert np.allclose(cam.to_plane(cam.project(pts), 0.0), pts[:, :2], atol=1e-4)
    shifted = cam.with_floor_at(-0.8)                # desk plane (Z = 0) 0.8 m above the floor
    assert shifted.center[2] == pytest.approx(cam.center[2] + 0.8)
    assert np.allclose(shifted.project(pts + [0, 0, 0.8]), cam.project(pts), atol=1e-6)


def test_edge_fit_recovers_camera_height() -> None:
    cam = _camera(f=1200, k1=-0.28, k2=0.04, center=(0.3, 0.0, 1.78), yaw=35, pitch=25)
    ys = np.linspace(2.5, 11, 40)
    edges = []
    for X in (2.74, 3.86):
        P = np.c_[np.full_like(ys, X), ys, np.zeros_like(ys)]
        ok = cam.in_view(P)
        edges.append({"pts": cam.project(P[ok]), "axis": "X", "value": X, "z": 0.0})
    verts = []
    for x, y in [(6.0, 13.0), (9.0, 8.0), (9.0, 12.0), (4.0, 13.0)]:
        P = np.c_[np.full(20, x), np.full(20, y), np.linspace(-0.7, 2.0, 20)]
        ok = cam.in_view(P)
        if ok.sum() > 5:
            verts.append(cam.project(P[ok]))
    fit = fit_from_edges(edges, (W, H), verticals=verts, cam_xy_init=(0.5, 0.0), cam_z_init=1.8, fixed=("cy",))
    assert fit.rms_px["edges"] < 1.0
    assert fit.camera.center[2] == pytest.approx(1.78, abs=0.1)


def test_zones_masks_areas_and_assignment() -> None:
    cam = _camera().with_floor_at(-0.8)
    zones = {"a": [[-1, 0], [1, 0], [1, 2], [-1, 2]], "b": [[1, 0], [3, 0], [3, 2], [1, 2]]}
    assert visible_area_m2(cam, zones["a"], 1.2) == pytest.approx(4.0, rel=0.05)
    heads = cam.project(np.array([[0.0, 1.0, 1.2], [2.0, 1.0, 1.2], [10.0, 10.0, 1.2]]))
    assert assign_points(cam, zones, heads, 1.2) == ["a", "b", None]
    m = zone_masks(cam, zones, 1.2, (H // 8, W // 8))
    u, v = (heads[0] / 8).astype(int)
    assert m["a"][v, u] == 1 and m["b"][v, u] == 0

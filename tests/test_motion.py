"""Smoke tests for optical flow (Farneback fallback) and zone motion in m/s."""
import cv2
import numpy as np
import pytest

from crowdsafe.calibration.camera_model import CameraModel, _look_rotation
from crowdsafe.calibration.zones import zone_masks
from crowdsafe.motion.flow import FlowModel
from crowdsafe.motion.zone_motion import ZoneMotion

W, H = 640, 360


def _cam() -> CameraModel:
    R = _look_rotation(0.0, np.radians(35))
    C = np.array([0.0, -3.0, 2.5])
    return CameraModel(np.array([[450.0, 0, W / 2], [0, 450.0, H / 2], [0, 0, 1]]), np.zeros(5),
                       cv2.Rodrigues(R)[0].ravel(), -R @ C, (W, H))


def test_farneback_recovers_a_shift() -> None:
    rng = np.random.default_rng(42)
    a = cv2.GaussianBlur((rng.random((H, W)) * 255).astype(np.uint8), (0, 0), 3)
    b = np.roll(a, 4, axis=1)
    f = FlowModel("farneback", width=W)(cv2.cvtColor(a, cv2.COLOR_GRAY2BGR), cv2.cvtColor(b, cv2.COLOR_GRAY2BGR))
    assert f.shape == (H, W, 2)
    # Farneback is the CPU fallback and underestimates on fine random texture (~3.4 px); RAFT is used on the A100
    assert np.median(f[40:-40, 40:-40, 0]) == pytest.approx(4.0, abs=1.0)


def test_zone_motion_converts_pixel_flow_to_metres_per_second() -> None:
    cam, z = _cam(), 1.2
    zones = {"Z": [[-1.5, 1.0], [1.5, 1.0], [1.5, 4.0], [-1.5, 4.0]]}
    masks = zone_masks(cam, zones, z, (H // 8, W // 8))
    zm = ZoneMotion(cam, masks, z)
    # everyone walks +1 m/s along X: build the true pixel flow for a 0.2 s step
    dt, v = 0.2, np.array([1.0, 0.0])
    flow = np.zeros((H, W, 2), np.float32)
    ys, xs = np.mgrid[0:H, 0:W]
    px = np.c_[xs.ravel() + 0.5, ys.ravel() + 0.5]
    xy = cam.to_plane(px, z)
    ok = np.isfinite(xy).all(1)
    uv1 = cam.project(np.c_[xy[ok] + v * dt, np.full(ok.sum(), z)])
    flow.reshape(-1, 2)[ok] = uv1 - px[ok]
    density = np.full((H // 8, W // 8), 0.1, np.float32)
    out = zm(flow, density, dt)["Z"]
    assert out["mean_speed"] == pytest.approx(1.0, rel=0.1)
    assert out["speed_var"] < 0.05 and out["counterflow"] == pytest.approx(0.0, abs=0.05)

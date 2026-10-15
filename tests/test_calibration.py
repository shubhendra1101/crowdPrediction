"""Smoke tests for the floor homography."""
import numpy as np
import pytest

from crowdsafe.calibration.homography import fit, line_intersection

# A synthetic camera: known homography floor(m) -> image(px)
H_TRUE = np.array([[120.0, 35.0, 300.0], [-10.0, 60.0, 200.0], [0.01, 0.06, 1.0]])


def _project(xy):
    p = np.c_[xy, np.ones(len(xy))] @ H_TRUE.T
    return p[:, :2] / p[:, 2:]


def test_fit_recovers_distances_and_areas() -> None:
    floor = np.array([[0, 0], [10.363, 0], [10.363, 8.026], [0, 8.026]])
    cal = fit(_project(floor), floor)
    assert cal.rms_error_m < 1e-6
    a, b = _project(np.array([[1.0, 1.0], [4.0, 5.0]]))
    assert cal.validate(a, b, 5.0)["error_pct"] < 0.01
    sq = _project(np.array([[2, 2], [4, 2], [4, 4], [2, 4]]))
    assert cal.area_m2(sq) == pytest.approx(4.0, rel=1e-6)


def test_ransac_ignores_a_wrong_point() -> None:
    floor = np.array([[0, 0], [5, 0], [5, 4], [0, 4], [2.5, 2], [1, 3], [4, 1]], float)
    img = _project(floor)
    img[6] += [80, -60]                     # one badly clicked point
    cal = fit(img, floor)
    assert cal.inliers.count(False) >= 1
    assert cal.validate(*_project(np.array([[0.5, 0.5], [3.5, 3.5]])), np.hypot(3, 3))["error_pct"] < 2


def test_hidden_corner_from_two_lines() -> None:
    assert line_intersection((0, 0), (2, 2), (0, 2), (2, 0)) == pytest.approx((1.0, 1.0))

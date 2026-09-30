"""Smoke test for CCTV metrics and the audit on a tiny synthetic video (no detector)."""
from pathlib import Path

import cv2
import numpy as np

from crowdsafe.datasets.cctv import blockiness, brightness, global_shift, sample_frames, video_info


def _video(path: Path, n: int = 40, fps: int = 10) -> Path:
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (160, 120))
    rng = np.random.default_rng(42)
    base = (rng.random((120, 160, 3)) * 255).astype(np.uint8)
    for i in range(n):
        w.write(np.clip(base.astype(int) + i, 0, 255).astype(np.uint8))   # frames differ
    w.release()
    return path


def test_video_info_and_sampling(tmp_path: Path) -> None:
    v = _video(tmp_path / "v.avi")
    info = video_info(v)
    assert (info.width, info.height) == (160, 120) and abs(info.fps_claimed - 10) < 0.5
    frames = list(sample_frames(v, every_s=1.0, fps=info.fps_claimed))
    assert len(frames) == 4 and frames[1][0] == 1.0 and frames[0][2] is None
    paired = list(sample_frames(v, every_s=1.0, fps=info.fps_claimed, pair_gap_s=0.5))
    assert len(paired) == 4 and all(p[2] is not None for p in paired)


def test_metrics_on_known_images() -> None:
    img = np.full((64, 64, 3), 100, np.uint8)
    assert brightness(img) == 100
    rng = np.random.default_rng(1)
    tex = cv2.GaussianBlur((rng.random((240, 320, 3)) * 255).astype(np.uint8), (5, 5), 0)
    shifted = np.roll(tex, 6, axis=1)
    assert 5 < global_shift(tex, shifted, width=320) < 7
    assert 0.5 < blockiness(tex) < 1.5


def test_sequential_sampling_returns_different_frames(tmp_path: Path) -> None:
    v = _video(tmp_path / "v.avi")
    fr = [f for _, f, _ in sample_frames(v, every_s=1.0, fps=10)]
    assert len({round(float(f.mean()), 1) for f in fr}) == len(fr)


def test_prelabel_helpers() -> None:
    import xml.etree.ElementTree as ET

    import pandas as pd

    from scripts.prelabel_cvat import cvat_xml, stratified_pick

    counts = pd.Series([0, 0, 1, 2, 5, 8, 20, 30, 31, 40])
    picked = stratified_pick(counts, 5, bins=5, seed=42)
    assert len(picked) == 5 and len(set(picked)) == 5
    assert counts[picked].max() >= 30 and counts[picked].min() <= 1       # spans empty to busy
    root = ET.fromstring(cvat_xml([{"name": "a.jpg", "width": 10, "height": 8, "points": [(1, 2), (3.5, 4)]}]))
    pts = root.findall("image/points")
    assert len(pts) == 2 and pts[1].get("points") == "3.5,4.0" and root.find("meta/task/labels/label/name").text == "head"

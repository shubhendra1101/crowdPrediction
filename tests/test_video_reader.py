"""The PyAV fallback reader must give the same frames as OpenCV (used when OpenCV lacks FFmpeg)."""
import cv2
import numpy as np
import pytest

from crowdsafe.datasets import cctv

pytest.importorskip("av")


def test_pyav_fallback_matches_opencv(tmp_path, monkeypatch) -> None:
    path = tmp_path / "clip.avi"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 48))
    for i in range(30):
        w.write(np.full((48, 64, 3), i * 8, np.uint8))
    w.release()
    a = [(t, f.mean(), later is not None) for t, f, later in cctv.sample_frames(path, 1.0, 10.0, pair_gap_s=0.2)]
    monkeypatch.setenv("CROWDSAFE_VIDEO_BACKEND", "av")
    assert type(cctv.open_video(path)).__name__ == "_AvReader"
    info = cctv.video_info(path)
    b = [(t, f.mean(), later is not None) for t, f, later in cctv.sample_frames(path, 1.0, 10.0, pair_gap_s=0.2)]
    assert (info.width, info.height) == (64, 48) and info.fps_claimed == pytest.approx(10.0)
    assert [x[0] for x in a] == [x[0] for x in b] == [0.0, 1.0, 2.0]
    assert np.allclose([x[1] for x in a], [x[1] for x in b], atol=3)

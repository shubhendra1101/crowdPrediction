"""Smoke tests for the live frame source: real-time pacing and offline back-pressure."""
import time

import cv2
import numpy as np

from crowdsafe.datasets.live import LiveSource


def _clip(path, n=40, fps=20.0):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (32, 24))
    for i in range(n):
        w.write(np.full((24, 32, 3), i * 5, np.uint8))
    w.release()
    return path


def test_realtime_pacing_and_pairs(tmp_path) -> None:
    src = LiveSource(_clip(tmp_path / "a.avi"), fps=20.0, realtime=True).start()
    time.sleep(0.5)
    f = src.latest()
    assert f is not None and 0.2 <= f.video_t <= 0.7          # about 0.5 s of video after 0.5 s of wall time
    g = src.after(f.video_t + 0.2)
    assert g is not None and g.video_t >= f.video_t + 0.2 and g.wall_t >= f.wall_t
    src.stop()


def test_offline_backpressure_skips_nothing(tmp_path) -> None:
    src = LiveSource(_clip(tmp_path / "b.avi"), fps=20.0, realtime=False, buffer_s=0.5).start()
    seen, t = [], 0.0
    while True:
        f = src.after(t, timeout_s=2.0)
        if f is None:
            break
        seen.append(f.video_t)
        src.consumed_t = f.video_t
        t = f.video_t + 0.25
        time.sleep(0.02)                                        # slow consumer
    src.stop()
    assert len(seen) >= 7 and np.allclose(np.diff(seen), 0.25)


def test_regular_grid_fills_skipped_seconds() -> None:
    import pandas as pd

    from scripts.live_pipeline import regular_grid

    h = pd.DataFrame({"timestamp": [0, 1, 3, 0, 1, 3], "zone_id": ["a"] * 3 + ["b"] * 3, "density": [1, 2, 4, 0, 0, 1.0]})
    r = regular_grid(h, 1.0).sort_values(["zone_id", "timestamp"])
    assert r["timestamp"].tolist() == [0, 1, 2, 3] * 2
    assert r.loc[(r["zone_id"] == "a") & (r["timestamp"] == 2), "density"].item() == 2.0

"""CCTV video access and quality metrics for the camera audit (T1.4).

All metrics work on single frames or frame pairs, so they run on a CPU laptop by sampling.
Nothing here stores or analyses identities; frames are only measured.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import os

import cv2
import numpy as np


class _AvReader:
    """PyAV reader with the cv2.VideoCapture grab/retrieve interface — for builds of OpenCV without FFmpeg
    (e.g. NVIDIA containers), which cannot open ASF/H.264 CCTV files."""

    def __init__(self, path: Path):
        import av

        self.c = av.open(str(path))
        self.s = self.c.streams.video[0]
        self.s.thread_type = "AUTO"
        self.it = self.c.decode(self.s)
        self.cur = None

    def isOpened(self) -> bool:
        return True

    def grab(self) -> bool:
        for _ in range(50):                      # skip a few undecodable packets instead of stopping
            try:
                self.cur = next(self.it)
                return True
            except StopIteration:
                return False
            except Exception:                    # av.error.InvalidDataError and friends
                continue
        return False

    def retrieve(self) -> tuple[bool, np.ndarray | None]:
        return (True, self.cur.to_ndarray(format="bgr24")) if self.cur is not None else (False, None)

    def pos_msec(self) -> float:
        return float(self.cur.time * 1000) if self.cur is not None and self.cur.time is not None else float("nan")

    def release(self) -> None:
        self.c.close()


def open_video(path: Path):
    """cv2.VideoCapture if this OpenCV build can read the file, else a PyAV reader (same interface).
    CROWDSAFE_VIDEO_BACKEND=av forces PyAV (tests)."""
    if os.environ.get("CROWDSAFE_VIDEO_BACKEND", "").lower() != "av":
        cap = cv2.VideoCapture(str(path))
        if cap.isOpened():
            return cap
        cap.release()
    try:
        return _AvReader(path)
    except ImportError as e:
        raise IOError(f"Cannot open {path}: OpenCV has no FFmpeg support and PyAV is not installed (pip install av)") from e


@dataclass
class VideoInfo:
    """Basic properties of a video file."""

    path: Path
    width: int
    height: int
    fps_claimed: float
    frames_claimed: int
    codec: str
    fps_measured: float | None = None

    @property
    def duration_s(self) -> float:
        fps = self.fps_measured or self.fps_claimed
        return self.frames_claimed / fps if fps else 0.0


def video_info(path: Path, measure_frames: int = 250) -> VideoInfo:
    """Read container metadata and measure the real frame rate from frame timestamps."""
    cap = open_video(path)
    if isinstance(cap, _AvReader):
        s = cap.s
        fps = float(s.average_rate or s.base_rate or 0)
        frames = s.frames or int((cap.c.duration or 0) / 1e6 * fps)
        info = VideoInfo(Path(path), s.codec_context.width, s.codec_context.height, fps, int(frames), s.codec_context.name)
        stamps = []
        for _ in range(measure_frames):
            if not cap.grab():
                break
            stamps.append(cap.pos_msec())
        cap.release()
        d = np.diff(np.asarray(stamps))
        d = d[d > 0]
        if len(d) > 10:
            info.fps_measured = float(1000.0 / np.median(d))
        return info
    if not cap.isOpened():
        raise IOError(f"Cannot open {path}")
    info = VideoInfo(Path(path), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                     float(cap.get(cv2.CAP_PROP_FPS)), int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                     int(cap.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, "little").decode(errors="replace").strip("\x00"))
    stamps = []
    for _ in range(measure_frames):
        if not cap.grab():
            break
        stamps.append(cap.get(cv2.CAP_PROP_POS_MSEC))
    cap.release()
    d = np.diff(np.asarray(stamps))
    d = d[d > 0]
    if len(d) > 10:
        info.fps_measured = float(1000.0 / np.median(d))
    return info


def sample_frames(path: Path, every_s: float, fps: float, max_frames: int | None = None,
                  pair_gap_s: float | None = None) -> Iterator[tuple[float, np.ndarray, np.ndarray | None]]:
    """Yield (time_s, frame, frame ``pair_gap_s`` later or None) every ``every_s`` seconds.

    Reads the file sequentially (grab every frame, decode only the needed ones) because
    CCTV containers such as ASF often do not support seeking — ``CAP_PROP_POS_FRAMES`` is
    silently ignored and returns the first frame again.
    """
    cap = open_video(path)
    step = max(1, int(round(every_s * fps)))
    gap = int(round(pair_gap_s * fps)) if pair_gap_s else None
    idx, n, pending = 0, 0, None           # pending = (t, frame) waiting for its pair
    while cap.grab():
        if pending is not None and gap and idx == pending[2] + gap:
            ok, later = cap.retrieve()
            yield pending[0], pending[1], later if ok else None
            pending = None
            n += 1
        if idx % step == 0:
            if pending is not None:          # gap longer than step: emit without pair
                yield pending[0], pending[1], None
                pending = None
                n += 1
            ok, frame = cap.retrieve()
            if ok:
                if gap:
                    pending = (idx / fps, frame, idx)
                else:
                    yield idx / fps, frame, None
                    n += 1
        if max_frames and n >= max_frames:
            break
        idx += 1
    if pending is not None:
        yield pending[0], pending[1], None
    cap.release()


def brightness(bgr: np.ndarray) -> float:
    """Mean luminance, 0–255."""
    return float(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).mean())


def contrast(bgr: np.ndarray) -> float:
    """Standard deviation of luminance."""
    return float(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).std())


def sharpness(bgr: np.ndarray) -> float:
    """Variance of the Laplacian (higher = sharper)."""
    return float(cv2.Laplacian(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def blockiness(bgr: np.ndarray, block: int = 8) -> float:
    """Compression blocking: mean gradient across 8-px block borders ÷ mean gradient elsewhere (≈1 = none)."""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    dx = np.abs(np.diff(g, axis=1))
    cols = np.arange(dx.shape[1])
    border = (cols % block) == block - 1
    return float(dx[:, border].mean() / (dx[:, ~border].mean() + 1e-6))


def global_shift(a: np.ndarray, b: np.ndarray, width: int = 640) -> float:
    """Whole-frame translation (px at full resolution) between two frames via phase correlation."""
    s = width / a.shape[1]
    ga = cv2.cvtColor(cv2.resize(a, None, fx=s, fy=s), cv2.COLOR_BGR2GRAY).astype(np.float32)
    gb = cv2.cvtColor(cv2.resize(b, None, fx=s, fy=s), cv2.COLOR_BGR2GRAY).astype(np.float32)
    (dx, dy), _ = cv2.phaseCorrelate(ga, gb)
    return float(np.hypot(dx, dy) / s)

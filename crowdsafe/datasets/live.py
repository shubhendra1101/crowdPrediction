"""Live frame source for the real-time pipeline (T9.2): an RTSP/HTTP stream or a video file played at real speed.

A background thread decodes continuously and keeps only the most recent frames (a short ring buffer), so a slow
consumer drops frames instead of falling further and further behind — this is how a live system must behave, and
it makes the measured latency honest. Every frame carries the wall-clock time it arrived (`wall_t`).
"""
from __future__ import annotations

import collections
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from crowdsafe.datasets.cctv import open_video


@dataclass
class Frame:
    video_t: float      # seconds since the start of the stream/file
    wall_t: float       # time.perf_counter() when the frame was available
    image: np.ndarray   # BGR


class LiveSource:
    """Background reader. `realtime=True` paces a file to its frame rate (simulated live camera, frames dropped when the
    consumer is slow); `realtime=False` reads a file without pacing but waits for the consumer (`consumed_t`), so no
    video time is skipped (offline tests)."""

    def __init__(self, source: str | Path, fps: float = 25.0, realtime: bool = True, buffer_s: float = 2.0):
        self.source, self.fps, self.realtime = str(source), fps, realtime
        self.buf: collections.deque[Frame] = collections.deque(maxlen=max(2, int(buffer_s * fps)))
        self.lock = threading.Lock()
        self.done = threading.Event()
        self.n_read = 0
        self.consumed_t = 0.0                              # set by the consumer; used for back-pressure on files
        self.buffer_s = buffer_s
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> "LiveSource":
        self.t0 = time.perf_counter()
        self.thread.start()
        return self

    def _run(self) -> None:
        cap = open_video(self.source)
        try:
            while not self.done.is_set():
                if not cap.grab():
                    break
                ok, img = cap.retrieve()
                if not ok:
                    continue
                video_t = self.n_read / self.fps
                if self.realtime:                          # a file: wait until this frame would exist on a live camera
                    delay = self.t0 + video_t - time.perf_counter()
                    if delay > 0:
                        time.sleep(delay)
                else:                                      # offline test: never run more than buffer_s ahead of the consumer
                    while video_t > self.consumed_t + 0.8 * self.buffer_s and not self.done.is_set():
                        time.sleep(0.005)
                with self.lock:
                    self.buf.append(Frame(video_t, time.perf_counter(), img))
                self.n_read += 1
        finally:
            cap.release()
            self.done.set()

    def latest(self) -> Frame | None:
        with self.lock:
            return self.buf[-1] if self.buf else None

    def after(self, video_t: float, timeout_s: float = 2.0) -> Frame | None:
        """First frame at or after `video_t` (waits for it to arrive)."""
        end = time.perf_counter() + timeout_s
        while time.perf_counter() < end:
            with self.lock:
                for f in self.buf:
                    if f.video_t >= video_t:
                        return f
            if self.done.is_set():
                return None
            time.sleep(0.005)
        return None

    def stop(self) -> None:
        self.done.set()

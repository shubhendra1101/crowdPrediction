"""Convert CCTV videos (.asf) to browser-playable MP4 (H.264) for the dashboard (T8.3).

Frame n of the source gets presentation time n / fps — the same clock the feature tables use (frame index / fps),
so dashboard time t and video time t are the same moment. Downscaled (default 960 px wide) to keep files small.

    python scripts/make_web_videos.py --videos cam02=data/cctv/cam02.asf cam03=data/cctv/cam03.asf cam04=data/cctv/cam04.asf
"""
from __future__ import annotations

import argparse
import sys
import time
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import av  # noqa: E402
import cv2  # noqa: E402

from crowdsafe.datasets.cctv import open_video, video_info  # noqa: E402


def convert(src: Path, dst: Path, width: int, crf: int, preset: str) -> dict:
    info = video_info(src)
    fps = round(info.fps_measured or info.fps_claimed)
    h = int(round(info.height * width / info.width / 2)) * 2
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".part.mp4")
    out = av.open(str(tmp), "w", options={"movflags": "faststart"})
    st = out.add_stream("libx264", rate=fps)
    st.width, st.height, st.pix_fmt = width, h, "yuv420p"
    st.time_base = Fraction(1, fps)
    st.options = {"crf": str(crf), "preset": preset}
    cap = open_video(src)
    n, t0 = 0, time.time()
    while cap.grab():
        ok, img = cap.retrieve()
        if not ok:
            continue
        frame = av.VideoFrame.from_ndarray(cv2.resize(img, (width, h), interpolation=cv2.INTER_AREA), format="bgr24")
        frame.pts, frame.time_base = n, Fraction(1, fps)
        for pkt in st.encode(frame):
            out.mux(pkt)
        n += 1
        if n % (fps * 300) == 0:
            print(f"  {src.name}: {n / fps:.0f} s of video, {n / (time.time() - t0):.0f} fps", flush=True)
    cap.release()
    for pkt in st.encode():
        out.mux(pkt)
    out.close()
    tmp.replace(dst)
    return {"frames": n, "seconds": round(n / fps, 1), "fps": fps, "size_mb": round(dst.stat().st_size / 1e6, 1),
            "minutes": round((time.time() - t0) / 60, 1)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--videos", nargs="+", required=True, help="camera=path")
    ap.add_argument("--out-dir", type=Path, default=Path("data/cctv/web"))
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--crf", type=int, default=28)
    ap.add_argument("--preset", default="veryfast")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    for spec in a.videos:
        cam, path = spec.split("=", 1)
        dst = a.out_dir / f"{cam}.mp4"
        if dst.exists() and not a.overwrite:
            print(f"{cam}: exists ({dst})")
            continue
        print(cam, convert(Path(path), dst, a.width, a.crf, a.preset), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

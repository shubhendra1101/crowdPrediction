"""Camera audit (T1.4): quality, head size and a people-count timeline per CCTV video.

Writes results/camera_audit.md + .csv (numbers only, no images) and saves a few reference frames
(including the emptiest one, for calibration and zone drawing) to data/cctv/audit/ (git-ignored).

    python scripts/camera_audit.py --videos data/cctv/cam01.asf data/cctv/cam02.asf \
        --phd-onnx weights/crowdsafe_nb01_onnx/onnx/yolo11s_crowdhuman_person_head.onnx
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crowdsafe.datasets.cctv import (blockiness, brightness, contrast, global_shift, sample_frames,  # noqa: E402
                                     sharpness, video_info)

# Rules of thumb from the implementation guide (Step 1b); used only to phrase recommendations.
SMALL_HEAD_PX = 15
LOW_FPS = 5
DARK = 60
BLOCKY = 1.3
SHAKE_PX = 2.0


def audit_video(path: Path, every_s: float, detector, frames_dir: Path) -> tuple[dict, pd.DataFrame]:
    """Metrics for one video; returns a summary row and the per-sample timeline."""
    info = video_info(path)
    fps = info.fps_measured or info.fps_claimed
    rows, shifts, keep = [], [], {}
    for t, frame, later in sample_frames(path, every_s, fps, pair_gap_s=1.0):
        r = {"t_s": round(t, 1), "brightness": brightness(frame), "contrast": contrast(frame),
             "sharpness": sharpness(frame), "blockiness": blockiness(frame)}
        if later is not None:
            shifts.append(global_shift(frame, later))      # frames 1 s apart
        if detector is not None:
            d = detector(frame)
            heads = d[d.class_id == detector.HEAD]
            h = heads.xyxy[:, 3] - heads.xyxy[:, 1] if len(heads) else np.array([])
            r.update(persons=int((d.class_id == detector.PERSON).sum()), heads=len(heads),
                     head_h_median=float(np.median(h)) if len(h) else np.nan,
                     head_h_p10=float(np.percentile(h, 10)) if len(h) else np.nan)
            n = r["persons"]
            if "emptiest" not in keep or n < keep["emptiest"][0]:
                keep["emptiest"] = (n, t, frame)
            if "busiest" not in keep or n > keep["busiest"][0]:
                keep["busiest"] = (n, t, frame)
        rows.append(r)
    tl = pd.DataFrame(rows)
    if len(tl) > 3 and tl["brightness"].round(2).nunique() == 1 and tl["sharpness"].round(1).nunique() == 1:
        raise RuntimeError(f"{path.name}: all sampled frames are identical — the video could not be read past the start")
    frames_dir.mkdir(parents=True, exist_ok=True)
    saved = {}
    for name, (n, t, img) in keep.items():
        out = frames_dir / f"{path.stem}_{name}_t{int(t)}s.jpg"
        cv2.imwrite(str(out), img)
        saved[name] = out.name
    summary = {"camera": path.stem, "file": path.name, "resolution": f"{info.width}x{info.height}",
               "fps_claimed": round(info.fps_claimed, 2),
               "fps_measured": round(info.fps_measured, 2) if info.fps_measured else None,
               "duration_min": round(info.duration_s / 60, 1), "codec": info.codec, "samples": len(tl),
               "brightness_median": round(float(tl["brightness"].median()), 1),
               "contrast_median": round(float(tl["contrast"].median()), 1),
               "sharpness_median": round(float(tl["sharpness"].median()), 1),
               "blockiness_median": round(float(tl["blockiness"].median()), 3),
               "frames_distinct": int(tl["brightness"].round(2).nunique()),
               "shake_px_median": round(float(np.median(shifts)), 2) if shifts else None,
               "shake_px_max": round(float(np.max(shifts)), 2) if shifts else None, "saved_frames": saved}
    if "heads" in tl:
        summary.update(persons_max=int(tl["persons"].max()), persons_median=float(tl["persons"].median()),
                       heads_max=int(tl["heads"].max()),
                       head_px_median=round(float(np.nanmedian(tl["head_h_median"])), 1) if tl["head_h_median"].notna().any() else None,
                       head_px_p10=round(float(np.nanmedian(tl["head_h_p10"])), 1) if tl["head_h_p10"].notna().any() else None)
    return summary, tl


def recommendations(s: dict) -> list[str]:
    """Plain-language recommendations from the guide's rules of thumb."""
    rec = []
    fps = s.get("fps_measured") or s["fps_claimed"]
    rec.append(f"Frame rate {fps:g} fps: fine for flow (≥ {LOW_FPS})." if fps >= LOW_FPS else
               f"Frame rate {fps:g} fps is low: compute flow between consecutive frames only.")
    hp = s.get("head_px_p10")
    if hp is not None:
        rec.append(f"Smallest heads ≈ {hp:g} px (10th pct): " + ("detectors should work; fusion can use YOLO." if hp >= SMALL_HEAD_PX
                   else "small — rely more on CLIP-EBC; try tiled detection."))
    rec.append("Lighting OK." if s["brightness_median"] >= DARK else "Dark: apply CLAHE on luminance before counting.")
    rec.append("Compression blocking low." if s["blockiness_median"] < BLOCKY else "Visible compression blocking: light denoise before optical flow.")
    if s.get("shake_px_max") is not None:
        rec.append("Camera steady (shift < 2 px over 1 s)." if s["shake_px_max"] < SHAKE_PX else
                   f"Frame shift up to {s['shake_px_max']} px over 1 s — check for shake/pan before motion features (people moving can also cause this).")
    rec.append("Needs floor calibration (4+ measured floor points) before any persons/m² result.")
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--videos", nargs="+", type=Path, required=True)
    ap.add_argument("--phd-onnx", type=Path, default=None, help="CrowdHuman person+head ONNX for head size / counts")
    ap.add_argument("--every-s", type=float, default=10.0)
    ap.add_argument("--frames-dir", type=Path, default=Path("data/cctv/audit"))
    ap.add_argument("--out", type=Path, default=Path("results/camera_audit.md"))
    a = ap.parse_args()
    det = None
    if a.phd_onnx and a.phd_onnx.exists():
        from crowdsafe.counting.yolo_track import PersonHeadOnnx
        det = PersonHeadOnnx(a.phd_onnx, conf=0.2, iou=0.6)
    summaries = []
    for v in a.videos:
        t0 = time.time()
        s, tl = audit_video(v, a.every_s, det, a.frames_dir)
        s["audit_seconds"] = round(time.time() - t0, 1)
        tl.to_csv(a.out.with_name(f"camera_audit_{v.stem}_timeline.csv"), index=False)
        summaries.append(s)
        print(json.dumps(s, default=str))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).drop(columns="saved_frames").to_csv(a.out.with_suffix(".csv"), index=False)
    lines = ["# Camera audit (T1.4)", "", f"Generated {time.strftime('%Y-%m-%d %H:%M')}. Frames sampled every {a.every_s:g} s. "
             "Counts/head sizes from the CrowdHuman YOLO11s person+head model (zero-shot, conf 0.2) — **indicative only**, "
             "not ground truth. No calibration yet, so nothing here is in persons/m².", ""]
    for s in summaries:
        lines += [f"## {s['camera']} ({s['file']})", "", "| Metric | Value |", "| --- | --- |"]
        lines += [f"| {k} | {v} |" for k, v in s.items() if k not in ("camera", "file", "saved_frames")]
        lines += ["", "**Recommendations**", ""] + [f"- {r}" for r in recommendations(s)] + [""]
    a.out.write_text("\n".join(lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Sample CCTV frames and pre-label head points for correction in CVAT (T1.5).

Frames are at least ``--min-gap-s`` apart and spread across crowd levels (stratified by the
detector's person count). Pre-labels are head centres from the CrowdHuman person+head detector
(heads here are large enough for it); the user corrects every frame in CVAT, so the final labels
are human ground truth. Output (all under data/, git-ignored):

    <out>/images/<camera>_t<sec>.jpg
    <out>/cvat_prelabels.zip      (annotations.xml, CVAT for images 1.1, label "head", points)
    <out>/frames.csv              (image, camera, t_s, prelabel heads/persons)

    python scripts/prelabel_cvat.py --videos data/cctv/cam01.asf data/cctv/cam02.asf \
        --phd-onnx weights/crowdsafe_nb01_onnx/onnx/yolo11s_crowdhuman_person_head.onnx --n 100
"""
from __future__ import annotations

import argparse
import random
import sys
import zipfile
from pathlib import Path
from xml.sax.saxutils import quoteattr

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crowdsafe.datasets.cctv import sample_frames, video_info  # noqa: E402


def stratified_pick(counts: pd.Series, n: int, bins: int, seed: int) -> list[int]:
    """Indices spread across count quantile bins, as evenly as the data allows."""
    rng = random.Random(seed)
    ranks = counts.rank(method="first")
    b = pd.qcut(ranks, q=min(bins, counts.nunique() or 1), labels=False, duplicates="drop")
    groups = [list(counts.index[b == k]) for k in sorted(set(b))]
    for g in groups:
        rng.shuffle(g)
    picked = []
    while len(picked) < n and any(groups):
        for g in groups:
            if g and len(picked) < n:
                picked.append(g.pop())
    return sorted(picked)


def cvat_xml(images: list[dict], label: str = "head") -> str:
    """CVAT for images 1.1 annotation XML with one point per head."""
    out = ['<?xml version="1.0" encoding="utf-8"?>', "<annotations>", "  <version>1.1</version>",
           f"  <meta><task><labels><label><name>{label}</name><type>points</type><attributes/></label></labels></task></meta>"]
    for i, im in enumerate(images):
        out.append(f'  <image id="{i}" name={quoteattr(im["name"])} width="{im["width"]}" height="{im["height"]}">')
        for x, y in im["points"]:
            out.append(f'    <points label="{label}" source="auto" occluded="0" points="{x:.1f},{y:.1f}" z_order="0"/>')
        out.append("  </image>")
    out.append("</annotations>")
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--videos", nargs="+", type=Path, required=True)
    ap.add_argument("--phd-onnx", type=Path, required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--min-gap-s", type=float, default=5.0)
    ap.add_argument("--bins", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("data/cctv/labels"))
    a = ap.parse_args()
    from crowdsafe.counting.yolo_track import PersonHeadOnnx

    det = PersonHeadOnnx(a.phd_onnx, conf=0.2, iou=0.6)
    cands = []
    for v in a.videos:
        info = video_info(v)
        fps = info.fps_measured or info.fps_claimed
        for t, frame, _ in sample_frames(v, a.min_gap_s, fps):
            d = det(frame)
            heads = d[d.class_id == det.HEAD]
            centres = [((x0 + x1) / 2, (y0 + y1) / 2) for x0, y0, x1, y1 in heads.xyxy]
            cands.append({"camera": v.stem, "video": v, "fps": fps, "t_s": round(t, 1), "points": centres,
                          "persons": int((d.class_id == det.PERSON).sum()), "heads": len(heads),
                          "width": frame.shape[1], "height": frame.shape[0]})   # no pixels kept: low memory
        print(f"{v.name}: {sum(c['camera'] == v.stem for c in cands)} candidate frames", flush=True)
    counts = pd.Series([c["heads"] for c in cands])
    picked = stratified_pick(counts, a.n, a.bins, a.seed)
    (a.out / "images").mkdir(parents=True, exist_ok=True)
    rows, images = [], []
    # second pass: re-read each video sequentially and write only the picked frames
    wanted = {}
    for i in picked:
        wanted.setdefault(cands[i]["video"], {})[cands[i]["t_s"]] = cands[i]
    for v, by_t in wanted.items():
        fps = next(iter(by_t.values()))["fps"]
        for t, frame, _ in sample_frames(v, a.min_gap_s, fps):
            c = by_t.get(round(t, 1))
            if c is not None:
                cv2.imwrite(str(a.out / "images" / f"{c['camera']}_t{int(c['t_s']):05d}.jpg"), frame)
    for i in picked:
        c = cands[i]
        name = f"{c['camera']}_t{int(c['t_s']):05d}.jpg"
        images.append({"name": name, "width": c["width"], "height": c["height"], "points": c["points"]})
        rows.append({"image": name, "camera": c["camera"], "t_s": c["t_s"], "prelabel_heads": c["heads"],
                     "prelabel_persons": c["persons"]})
    pd.DataFrame(rows).to_csv(a.out / "frames.csv", index=False)
    with zipfile.ZipFile(a.out / "cvat_prelabels.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("annotations.xml", cvat_xml(images))
    with zipfile.ZipFile(a.out / "images.zip", "w", zipfile.ZIP_STORED) as z:
        for r in rows:
            z.write(a.out / "images" / r["image"], r["image"])
    df = pd.DataFrame(rows)
    print(f"picked {len(df)} frames; per camera {df['camera'].value_counts().to_dict()}; "
          f"prelabel heads min/median/max {df['prelabel_heads'].min()}/{df['prelabel_heads'].median()}/{df['prelabel_heads'].max()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

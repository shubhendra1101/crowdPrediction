"""Zero-shot counting on the user's labelled CCTV frames (T3.5, CCTV part) — CPU, ONNX models.

Ground truth = the user's corrected CVAT head points. For each frame and model it stores the count
plus the detector reliability features used later by the fusion (T3.7). Models run one at a time
to keep memory low on the laptop.

    python scripts/eval_cctv_counting.py --labels data/cctv/labels/heads_corrected.zip \
        --images data/cctv/labels/images --onnx-dir weights/crowdsafe_nb01_onnx/onnx
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crowdsafe import nbenv  # noqa: E402

nbenv.limit_threads(nbenv.cpu_limit(default=4))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from crowdsafe.datasets.cvat import read_cvat_points  # noqa: E402
from crowdsafe.eval.metrics import band_labels, count_metrics  # noqa: E402


def run_model(name: str, args, frames: list[Path]) -> pd.DataFrame:
    """Per-frame counts (and features) for one model."""
    from crowdsafe.counting.yolo_track import PersonHeadOnnx, YoloOnnx, reliability_features

    rows, times = [], []
    if name == "yolo11l":
        det = YoloOnnx(args.onnx_dir / "yolo11l.onnx", imgsz=args.yolo_imgsz, conf=0.25, iou=0.7)
    elif name == "phd":
        det = PersonHeadOnnx(args.onnx_dir / "yolo11s_crowdhuman_person_head.onnx", conf=0.2, iou=0.6)
    elif name == "clipebc":
        from crowdsafe.counting.density_model import ClipEbcDensity
        det = ClipEbcDensity(backend="onnx", onnx_path=args.onnx_dir / "clipebc_vitb16_nwpu.onnx", device="cpu",
                             max_batch=args.clipebc_batch)
    else:
        raise ValueError(name)
    for f in frames:
        bgr = cv2.imread(str(f))
        t0 = time.perf_counter()
        if name == "clipebc":
            r = {"clipebc": det.count(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))}
        elif name == "yolo11l":
            d = det(bgr)
            r = {"yolo11l": len(d), **{f"yolo11l__{k}": v for k, v in reliability_features(d).items()}}
        else:
            d = det(bgr)
            heads = d[d.class_id == det.HEAD]
            r = {"phd_heads": len(heads), "phd_persons": int((d.class_id == det.PERSON).sum()),
                 **{f"phd_heads__{k}": v for k, v in reliability_features(heads).items()}}
        times.append(time.perf_counter() - t0)
        rows.append({"image": f.name, **r})
    print(f"{name}: {len(rows)} frames, median {1000 * np.median(times):.0f} ms/frame (CPU)", flush=True)
    del det
    gc.collect()
    return pd.DataFrame(rows).set_index("image")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--onnx-dir", type=Path, required=True)
    ap.add_argument("--frames-csv", type=Path, default=Path("data/cctv/labels/frames.csv"))
    ap.add_argument("--models", nargs="*", default=["phd", "yolo11l", "clipebc"])
    ap.add_argument("--yolo-imgsz", type=int, default=1280)
    ap.add_argument("--clipebc-batch", type=int, default=4)
    ap.add_argument("--bands", type=float, nargs="*", default=[20, 40])
    ap.add_argument("--out", type=Path, default=Path("results/counting_zeroshot_cctv.md"))
    a = ap.parse_args()

    gt = read_cvat_points(a.labels)
    frames = sorted(a.images / n for n in gt if (a.images / n).exists())
    base = pd.DataFrame({"image": [f.name for f in frames], "gt": [len(gt[f.name]) for f in frames]}).set_index("image")
    if a.frames_csv.exists():
        base = base.join(pd.read_csv(a.frames_csv).set_index("image")[["camera", "t_s"]])
    per = base.join([run_model(m, a, frames) for m in a.models])
    per["band"] = band_labels(per["gt"].to_numpy(), a.bands)
    preds = [c for c in ("yolo11l", "phd_heads", "phd_persons", "clipebc") if c in per]
    table = count_metrics(per.reset_index(), preds, "gt", "band")
    table["mape_%"] = [round(100 * float(((per[m] - per["gt"]).abs() / per["gt"].clip(lower=1)).mean()), 1)
                       for m in table["model"]] if len(table) else []
    a.out.parent.mkdir(parents=True, exist_ok=True)
    per.reset_index().to_csv(a.out.with_name(a.out.stem + "_per_frame.csv"), index=False)
    table.to_csv(a.out.with_suffix(".csv"), index=False)
    md = ["# Zero-shot counting on the classroom CCTV (T3.5, CCTV part)", "",
          f"Generated {time.strftime('%Y-%m-%d %H:%M')}. Ground truth: user-corrected CVAT head points, {len(per)} frames "
          f"({per['camera'].value_counts().to_dict() if 'camera' in per else ''}), {int(per['gt'].sum())} heads. "
          "Same camera for all frames (H705 Front) — no held-out camera. ONNX on CPU; YOLO11l at "
          f"{a.yolo_imgsz} px, conf 0.25; CrowdHuman conf 0.2; CLIP-EBC 224 px windows. Bands = heads per frame "
          "(no floor calibration yet, so not persons/m²). `mape_%` = mean absolute error as % of the true count.", "",
          "| Model | Band | n | MAE | RMSE | Bias | MAPE % |", "| --- | --- | --- | --- | --- | --- | --- |"]
    md += [f"| {r.model} | {r.band} | {r.n} | {r.mae} | {r.rmse} | {r.bias} | {r._7 if r.band == 'all' else ''} |"
           for r in table.itertuples()]
    a.out.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(table.to_string(index=False))
    print(json.dumps({"frames": len(per), "heads": int(per["gt"].sum())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

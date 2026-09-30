"""Zero-shot counting benchmark on ShanghaiTech A/B test sets (T3.5, part 1).

Writes per-image predictions (+ detector reliability features for the fusion, T3.7) and a
MAE/RMSE table per part and per count band.

Example (A100):
    python scripts/bench_counting.py --data-root ~/crowdsafe_work/data/raw/shanghaitech \
        --clipebc-repo ~/crowdsafe_work/cache/CLIP-EBC-hf --weights-dir ~/crowdsafe_work/outputs/onnx
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crowdsafe.datasets.shanghaitech import list_split  # noqa: E402
from crowdsafe.eval.metrics import band_labels, count_metrics  # noqa: E402

Predictor = Callable[[np.ndarray], dict[str, float]]   # BGR image -> {column: value}


def build_model(name: str, spec: dict, args: argparse.Namespace) -> Predictor:
    """Create a predictor for one model spec from the benchmark config."""
    import cv2

    if spec["type"] == "yolo":
        from crowdsafe.counting.yolo_track import YoloDetector, reliability_features

        det = YoloDetector(spec["weights"], imgsz=spec["imgsz"], conf=spec["conf"], device=args.device,
                           slice_wh=spec.get("slice_wh"), overlap_px=spec.get("overlap_px", 100))

        def run(bgr: np.ndarray) -> dict[str, float]:
            d = det(bgr)
            return {name: len(d), **{f"{name}__{k}": v for k, v in reliability_features(d).items()}}
        return run

    if spec["type"] == "person_head":
        from crowdsafe.counting.yolo_track import PersonHeadOnnx, reliability_features

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if args.device != "cpu" else None
        det = PersonHeadOnnx(Path(args.weights_dir) / spec["onnx"], conf=spec["conf"], iou=spec["iou"],
                             providers=providers)

        def run(bgr: np.ndarray) -> dict[str, float]:
            d = det(bgr)
            heads = d[d.class_id == det.HEAD]
            return {f"{name}_heads": len(heads), f"{name}_persons": int((d.class_id == det.PERSON).sum()),
                    **{f"{name}_heads__{k}": v for k, v in reliability_features(heads).items()}}
        return run

    if spec["type"] == "clipebc":
        from crowdsafe.counting.density_model import ClipEbcDensity

        repo = Path(args.clipebc_repo)
        model = ClipEbcDensity(backend="torch", mode=spec["mode"], repo_dir=repo,
                               weights_dir=repo / "nwpu_weights" / "CLIP_EBC_ViT_B_16",
                               device="cpu" if args.device == "cpu" else "cuda")

        def run(bgr: np.ndarray) -> dict[str, float]:
            return {name: model.count(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))}
        return run

    raise ValueError(f"Unknown model type {spec['type']}")


def seed_all(seed: int) -> None:
    """Seed Python, NumPy and (if present) torch."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def benchmark(cfg: dict, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Run every selected model over every test image; return per-image rows, metrics, timing."""
    import cv2

    samples = [(p, s) for p in cfg["dataset"]["parts"]
               for s in list_split(Path(args.data_root), p, cfg["dataset"]["split"])[: args.limit]]
    base = pd.DataFrame([{"part": p, "image": s.image.name, "gt": s.count} for p, s in samples])
    names = args.models or list(cfg["models"])
    timing, cols = {}, []
    for name in names:
        print(f"--> {name}", flush=True)
        try:
            model = build_model(name, cfg["models"][name], args)
        except Exception as e:  # a model that can't load is reported, not fatal
            timing[name] = {"error": repr(e)}
            print("    failed to load:", repr(e))
            continue
        rows, t = [], []
        for _, s in samples:
            bgr = cv2.imread(str(s.image))
            t0 = time.perf_counter()
            rows.append(model(bgr))
            t.append(time.perf_counter() - t0)
        cols.append(pd.DataFrame(rows))
        timing[name] = {"ms_per_image_median": round(1000 * float(np.median(t)), 1), "images": len(t)}
        del model
    per_image = pd.concat([base] + cols, axis=1)
    per_image["band"] = band_labels(per_image["gt"].to_numpy(), cfg["count_bands"])
    pred_cols = [c for c in per_image.columns if c in names or c.endswith(("_heads", "_persons"))]
    metrics = pd.concat([count_metrics(g, pred_cols, "gt", "band").assign(part=p)
                         for p, g in per_image.groupby("part")], ignore_index=True)
    return per_image, metrics, timing


def write_results(per_image: pd.DataFrame, metrics: pd.DataFrame, timing: dict, out: Path, meta: dict) -> None:
    """Write per-image CSV, metrics CSV/Markdown and a JSON run record."""
    out.parent.mkdir(parents=True, exist_ok=True)
    per_image.to_csv(out.with_name(out.stem + "_per_image.csv"), index=False)
    metrics.to_csv(out.with_suffix(".csv"), index=False)
    lines = [f"# Zero-shot counting — ShanghaiTech {meta['split']} (T3.5)", "",
             f"Run: {meta['finished']} · commit {meta.get('commit', '?')} · device {meta['device']} · seed {meta['seed']}",
             "", "Bands are **people per image** (ShanghaiTech has no floor calibration, so no persons/m²).", ""]
    for part, g in metrics.groupby("part"):
        lines += [f"## Part {part}", "", "| Model | Band | n | MAE | RMSE | Bias |", "| --- | --- | --- | --- | --- | --- |"]
        lines += [f"| {r.model} | {r.band} | {r.n} | {r.mae} | {r.rmse} | {r.bias} |" for r in g.itertuples()]
        lines.append("")
    lines += ["## Speed", "", "| Model | ms / image (median) |", "| --- | --- |"]
    lines += [f"| {k} | {v.get('ms_per_image_median', v.get('error'))} |" for k, v in timing.items()]
    out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    out.with_name(out.stem + "_run.json").write_text(json.dumps({**meta, "timing": timing}, indent=2), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=Path("configs/bench_shanghaitech.yaml"))
    ap.add_argument("--data-root", type=Path, required=True, help="folder containing part_A / part_B")
    ap.add_argument("--clipebc-repo", type=Path, default=None, help="snapshot of HF Yiming-M/CLIP-EBC")
    ap.add_argument("--weights-dir", type=Path, default=Path("weights"), help="folder with the ONNX files")
    ap.add_argument("--models", nargs="*", default=None, help="subset of config model names")
    ap.add_argument("--limit", type=int, default=None, help="first N images per part (smoke runs)")
    ap.add_argument("--device", default="0", help="GPU index or 'cpu'")
    ap.add_argument("--out", type=Path, default=Path("results/counting_zeroshot_shanghaitech.md"))
    ap.add_argument("--commit", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seed_all(cfg["seed"])
    per_image, metrics, timing = benchmark(cfg, args)
    meta = {"finished": time.strftime("%Y-%m-%d %H:%M:%S"), "split": cfg["dataset"]["split"], "device": args.device,
            "seed": cfg["seed"], "commit": args.commit, "limit": args.limit, "config": cfg}
    write_results(per_image, metrics, timing, args.out, meta)
    print(metrics.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

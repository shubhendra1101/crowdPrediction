"""Fusion fit + ablation on the labelled CCTV frames (T3.7 / T3.8).

Variants: detector only · CLIP-EBC only · hard switch (re-implemented, D2) · confidence-weighted fusion.
θ and τ are fitted on the validation split only; metrics are reported on the test split.

Split (single camera, so no held-out camera): time blocks, never interleaved frames —
validation = the first ``--val-share`` of cam01 by time; test = the rest of cam01 + all of cam02.

    python scripts/fusion_ablation.py --per-frame results/counting_zeroshot_cctv_per_frame.csv
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crowdsafe.counting.fusion import Fusion, fit_constant_weight, fit_hard_switch, hard_switch  # noqa: E402
from crowdsafe.eval.metrics import band_labels, count_metrics  # noqa: E402

FEATURES = ["__mean_conf", "__n_det", "__overlap_share", "__mean_box_h"]


def split_time_blocks(df: pd.DataFrame, val_camera: str, val_share: float) -> pd.Series:
    """'val' for the first val_share of val_camera's frames by time, 'test' for everything else."""
    s = pd.Series("test", index=df.index)
    cam = df[df["camera"] == val_camera].sort_values("t_s")
    s[cam.index[: int(round(len(cam) * val_share))]] = "val"
    return s


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--per-frame", type=Path, default=Path("results/counting_zeroshot_cctv_per_frame.csv"))
    ap.add_argument("--detector", default="yolo11l", help="yolo11l or phd_heads")
    ap.add_argument("--density", default="clipebc")
    ap.add_argument("--val-camera", default="cam01")
    ap.add_argument("--val-share", type=float, default=0.5)
    ap.add_argument("--bands", type=float, nargs="*", default=[20, 40])
    ap.add_argument("--test-per-frame", type=Path, default=None,
                    help="held-out cameras: fit on ALL rows of --per-frame, test on these rows (reported per camera)")
    ap.add_argument("--out", type=Path, default=Path("results/fusion_ablation.md"))
    a = ap.parse_args()

    df = pd.read_csv(a.per_frame)
    if a.test_per_frame:
        df = pd.concat([df.assign(split="val"), pd.read_csv(a.test_per_frame).assign(split="test")], ignore_index=True)
    else:
        df["split"] = split_time_blocks(df, a.val_camera, a.val_share)
    feat_cols = [a.detector + f for f in FEATURES] + [a.density]
    val, test = df[df.split == "val"], df[df.split == "test"]
    fus = Fusion().fit(val[feat_cols].to_numpy(), val[a.detector], val[a.density], val["gt"])
    tau = fit_hard_switch(val[a.detector], val[a.density], val["gt"])
    w_const = fit_constant_weight(val[a.detector], val[a.density], val["gt"])
    out = df.copy()
    out["detector_only"] = df[a.detector].astype(float)
    out["density_only"] = df[a.density].astype(float)
    out["hard_switch"] = hard_switch(df[a.detector], df[a.density], tau)
    out["average_50_50"] = 0.5 * df[a.detector] + 0.5 * df[a.density]
    out["constant_weight"] = w_const * df[a.detector] + (1 - w_const) * df[a.density]
    out["fusion"] = fus(df[feat_cols].to_numpy(), df[a.detector], df[a.density])
    out["fusion_weight"] = fus.weights(df[feat_cols].to_numpy())
    out["band"] = band_labels(out["gt"].to_numpy(), a.bands)
    variants = ["detector_only", "density_only", "hard_switch", "average_50_50", "constant_weight", "fusion"]
    tables = {s: count_metrics(out[out.split == s], variants, "gt", "band") for s in ("val", "test")}
    per_cam = pd.concat([count_metrics(g, variants, "gt").assign(camera=c)
                         for c, g in out[out.split == "test"].groupby("camera")], ignore_index=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out.with_name(a.out.stem + "_per_frame.csv"), index=False)
    pd.concat([t.assign(split=s) for s, t in tables.items()]).to_csv(a.out.with_suffix(".csv"), index=False)
    per_cam.to_csv(a.out.with_name(a.out.stem + "_per_camera.csv"), index=False)
    t = tables["test"]
    split_text = (f"Split: **fit** = all frames of `{a.per_frame.name}` ({len(val)} frames, cameras "
                  f"{sorted(val['camera'].unique())}) — θ, τ and w fitted here only; **test** = `{a.test_per_frame.name}` "
                  f"({len(test)} frames, cameras {sorted(test['camera'].unique())}) — **cameras never used for fitting**."
                  if a.test_per_frame else
                  f"Split: **val** = first {a.val_share:.0%} of {a.val_camera} by time ({len(val)} frames) — θ and τ fitted here only; "
                  f"**test** = rest of {a.val_camera} + other recordings ({len(test)} frames). Single camera: this is a *time* hold-out, "
                  "not a held-out camera.")
    md = ["# Fusion ablation on the classroom CCTV (T3.7 / T3.8)", "",
          f"Generated {time.strftime('%Y-%m-%d %H:%M')}. Detector = `{a.detector}`, density = `{a.density}`. "
          f"{split_text} Frame-level counts (zones need calibration). "
          f"Hard switch = detector if density count < τ = {tau:.1f}, else density (re-implemented, D2). "
          f"`average_50_50` = plain mean of both counts; `constant_weight` = one detector weight w = {w_const:.2f} fitted on val; "
          "`fusion` = learned per-frame weight from reliability features (architecture.md §2.2).", "",
          "## Test split", "", "| Variant | Band | n | MAE | RMSE | Bias |", "| --- | --- | --- | --- | --- | --- |"]
    md += [f"| {r.model} | {r.band} | {r.n} | {r.mae} | {r.rmse} | {r.bias} |" for r in t.itertuples()]
    md += ["", "## Test split, per camera (all bands)", "", "| Camera | Variant | n | MAE | RMSE | Bias |",
           "| --- | --- | --- | --- | --- | --- |"]
    md += [f"| {r.camera} | {r.model} | {r.n} | {r.mae} | {r.rmse} | {r.bias} |" for r in per_cam.itertuples()]
    md += ["", f"Mean fusion weight on test (share given to the detector): {out[out.split == 'test']['fusion_weight'].mean():.2f}",
           "", "Validation-split numbers are in the CSV (in-sample for θ/τ, so not a result)."]
    a.out.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(t[t.band == "all"].to_string(index=False))
    print(per_cam.to_string(index=False))
    print(json.dumps({"w_const": round(w_const, 3), "tau": tau, "theta": fus.theta.round(3).tolist(), "val": len(val), "test": len(test)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Leave-one-room-out evaluation of the counting variants (T3.8, generalisation).

For each room: fit θ (learned fusion), τ (hard switch) and w (constant weight) on the frames of the
OTHER rooms, then score every variant on the held-out room. Rooms: H705 (cam01, cam02), H706 (cam03),
H103 (cam04).

    python scripts/fusion_loco.py --per-frame results/counting_zeroshot_cctv_per_frame.csv \
        results/counting_zeroshot_newcams_per_frame.csv
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crowdsafe.counting.fusion import Fusion, fit_constant_weight, fit_hard_switch, hard_switch  # noqa: E402
from crowdsafe.eval.metrics import count_metrics  # noqa: E402

FEATURES = ["__mean_conf", "__n_det", "__overlap_share", "__mean_box_h"]
ROOMS = {"cam01": "H705", "cam02": "H705", "cam03": "H706", "cam04": "H103"}
VARIANTS = ["detector_only", "density_only", "hard_switch", "average_50_50", "constant_weight", "fusion"]


def predict_variants(fit: pd.DataFrame, test: pd.DataFrame, det: str, dens: str) -> tuple[pd.DataFrame, dict]:
    """Fit all fitted variants on `fit`, return `test` with one column per variant."""
    feats = [det + f for f in FEATURES] + [dens]
    fus = Fusion().fit(fit[feats].to_numpy(), fit[det], fit[dens], fit["gt"])
    tau = fit_hard_switch(fit[det], fit[dens], fit["gt"])
    w = fit_constant_weight(fit[det], fit[dens], fit["gt"])
    out = test.copy()
    out["detector_only"] = test[det].astype(float)
    out["density_only"] = test[dens].astype(float)
    out["hard_switch"] = hard_switch(test[det], test[dens], tau)
    out["average_50_50"] = 0.5 * test[det] + 0.5 * test[dens]
    out["constant_weight"] = w * test[det] + (1 - w) * test[dens]
    out["fusion"] = fus(test[feats].to_numpy(), test[det], test[dens])
    return out, {"tau": round(tau, 1), "w_const": round(w, 3)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--per-frame", type=Path, nargs="+", required=True)
    ap.add_argument("--detector", default="yolo11l")
    ap.add_argument("--density", default="clipebc")
    ap.add_argument("--out", type=Path, default=Path("results/fusion_loco.md"))
    a = ap.parse_args()
    df = pd.concat([pd.read_csv(p) for p in a.per_frame], ignore_index=True)
    df["room"] = df["camera"].map(ROOMS)
    tables, params = [], {}
    for room in sorted(df["room"].unique()):
        pred, params[room] = predict_variants(df[df.room != room], df[df.room == room], a.detector, a.density)
        tables.append(count_metrics(pred, VARIANTS, "gt").assign(held_out=room))
    t = pd.concat(tables, ignore_index=True)
    mean = t.groupby("model", sort=False)[["mae", "rmse"]].mean().round(2).reset_index()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(a.out.with_suffix(".csv"), index=False)
    md = ["# Leave-one-room-out counting evaluation (T3.8)", "",
          f"Generated {time.strftime('%Y-%m-%d %H:%M')}. Each room is held out in turn; θ / τ / w are fitted on the other two "
          f"rooms only. Frames: {df.groupby('room').size().to_dict()}. Ground truth = user-corrected CVAT heads "
          "(H103 glass-window area excluded for labels and models alike, `configs/roi.yaml`). Error in people per frame.", "",
          "## Per held-out room", "", "| Held-out room | Variant | n | MAE | RMSE | Bias |", "| --- | --- | --- | --- | --- | --- |"]
    md += [f"| {r.held_out} | {r.model} | {r.n} | {r.mae} | {r.rmse} | {r.bias} |" for r in t.itertuples()]
    md += ["", "## Mean over the three held-out rooms", "", "| Variant | mean MAE | mean RMSE |", "| --- | --- | --- |"]
    md += [f"| {r.model} | {r.mae} | {r.rmse} |" for r in mean.itertuples()]
    md += ["", "Fitted parameters per held-out room: " + "; ".join(f"{k}: τ={v['tau']}, w={v['w_const']}" for k, v in params.items())]
    a.out.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(t.pivot(index="model", columns="held_out", values="mae").reindex(VARIANTS).to_string())
    print(mean.to_string(index=False))
    print(params)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

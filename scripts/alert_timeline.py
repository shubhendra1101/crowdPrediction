"""Alert timeline figures: observed zone density, the forecaster's calibrated worst case, risk bands and alerts (T7.4, T9.3).

    python scripts/alert_timeline.py --features data/features/cctv --points weights/crowdsafe_nb05_report/forecast_cctv_points.parquet \
        --timelines results/alerts_cctv_timelines.csv --run cam02 --out results/figures/timeline_cam02.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crowdsafe.features.logger import read_features  # noqa: E402

BANDS = [(0, 2, "#d9f2d9", "safe"), (2, 4, "#fff3c4", "crowded"), (4, 5, "#ffd9b3", "high risk"), (5, 9, "#ffc2c2", "critical")]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--features", type=Path, required=True, help="feature parquet file or folder")
    ap.add_argument("--points", type=Path, required=True)
    ap.add_argument("--timelines", type=Path, required=True)
    ap.add_argument("--run", required=True, help="camera_id / run id")
    ap.add_argument("--model", default="chronos2_joint")
    ap.add_argument("--zones", nargs="*", help="zones to plot (default: the 4 busiest)")
    ap.add_argument("--title", default=None)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    f = read_features(a.features)
    f = f[f["camera_id"] == a.run]
    p = pd.read_parquet(a.points)
    p = p[(p["camera_id"] == a.run) & (p["model"] == a.model)]
    p = p.assign(ub=p["p90_cal"].fillna(p["p90"]))
    worst = p.groupby(["zone_id", "origin"])["ub"].max().reset_index()
    tl = pd.read_csv(a.timelines)
    tl = tl[(tl["camera_id"] == a.run) & tl["variant"].str.startswith(a.model) & tl["zone_id"].notna()]
    zones = a.zones or f.groupby("zone_id")["density"].max().sort_values(ascending=False).index[:4].tolist()
    ymax = max(2.5, float(np.nanmax([f["density"].max(), worst["ub"].max() if len(worst) else 0])) * 1.1)
    fig, axes = plt.subplots(len(zones), 1, figsize=(11, 2.2 * len(zones)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, z in zip(axes, zones):
        for lo, hi, col, _ in BANDS:
            ax.axhspan(lo, min(hi, ymax), color=col, zorder=0)
        g = f[f["zone_id"] == z].sort_values("timestamp")
        ax.plot(g["timestamp"], g["density"], color="black", lw=1, label="observed density")
        w = worst[worst["zone_id"] == z].sort_values("origin")
        ax.plot(w["origin"], w["ub"], color="tab:blue", lw=1.2, label=f"{a.model}: worst case next 60 s (90 % bound)")
        for r in tl[tl["zone_id"] == z].itertuples():
            ax.axvline(r.since, color="red", lw=2, zorder=5)
            ax.text(r.since, ymax * 0.92, f" alert: {r.level}", color="red", fontsize=8, va="top")
        ax.set_ylim(0, ymax)
        ax.set_ylabel(f"{z}\npersons/m²", fontsize=8)
    axes[0].legend(loc="upper left", fontsize=8)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(a.title or f"{a.run}: zone density, forecast worst case and alerts")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=90, bbox_inches="tight")
    print("wrote", a.out, "zones", zones, "alerts", len(tl))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

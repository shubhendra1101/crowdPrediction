"""Paper figures (T9.3): pipeline diagram, zone-density heatmap on a camera frame, forecast fan chart.

    python scripts/paper_figures.py --out results/figures

Inputs come from configs/dashboard.yaml (same runs as the replay dashboard), so the figures show exactly what the
demo shows. Every number drawn is read from stored features / forecasts; nothing is computed fresh here.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crowdsafe.server import replay  # noqa: E402

# density bands (persons/m²) -> colour, as in the dashboard
BANDS = [(2.0, (46, 158, 68)), (4.0, (224, 177, 0)), (5.0, (240, 125, 26)), (np.inf, (215, 38, 61))]
BAND_FILL = [(0, 2, "#d9f2d9"), (2, 4, "#fff3c4"), (4, 5, "#ffd9b3"), (5, 99, "#ffc2c2")]


def band_rgb(d: float) -> tuple[int, int, int]:
    """Dashboard colour of a density value."""
    for hi, rgb in BANDS:
        if d < hi:
            return rgb
    return BANDS[-1][1]


def pipeline_diagram(out: Path) -> None:
    """Block diagram of the system (architecture.md §2)."""
    boxes = [
        ("CCTV camera\n(fixed, 1 frame/s)", 0.2, 0.75, "#e8eef9"),
        ("Camera model\n(pinhole + lens distortion,\nfitted to desks / edges)", 3.3, 0.3, "#f2f2f2"),
        ("CLIP-EBC\ndensity map", 1.75, 0.95, "#e8f5e9"),
        ("RAFT\noptical flow", 1.75, 0.55, "#e8f5e9"),
        ("Zones on the floor (m²)\ndensity p/m², speed m/s,\npressure, entropy", 3.3, 0.75, "#fff8e1"),
        ("Chronos-2 zero-shot\nforecast 10/30/60 s\n+ split conformal bound", 4.85, 0.75, "#fff3e0"),
        ("Risk engine\nbands 2/4/5 p/m²\non calibrated worst case", 6.4, 0.75, "#fdecea"),
        ("Alert manager\nraise 10 s / clear 30 s", 6.4, 0.3, "#fdecea"),
        ("FastAPI + WebSocket\n→ dashboard", 4.85, 0.3, "#ede7f6"),
    ]
    w, h = 1.35, 0.32
    fig, ax = plt.subplots(figsize=(12, 3.6))
    pos = {}
    for text, x, y, col in boxes:
        ax.add_patch(FancyBboxPatch((x, y - h / 2), w, h, boxstyle="round,pad=0.02", fc=col, ec="#555", lw=1))
        ax.text(x + w / 2, y, text, ha="center", va="center", fontsize=8.5)
        pos[text.split("\n")[0]] = (x, y)

    def arrow(a: str, b: str, side_a: str = "r", side_b: str = "l") -> None:
        (xa, ya), (xb, yb) = pos[a], pos[b]
        pa = {"r": (xa + w, ya), "b": (xa + w / 2, ya - h / 2), "t": (xa + w / 2, ya + h / 2), "l": (xa, ya)}[side_a]
        pb = {"l": (xb, yb), "t": (xb + w / 2, yb + h / 2), "b": (xb + w / 2, yb - h / 2), "r": (xb + w, yb)}[side_b]
        ax.add_patch(FancyArrowPatch(pa, pb, arrowstyle="-|>", mutation_scale=12, color="#333", lw=1))

    arrow("CCTV camera", "CLIP-EBC")
    arrow("CCTV camera", "RAFT")
    arrow("CLIP-EBC", "Zones on the floor (m²)")
    arrow("RAFT", "Zones on the floor (m²)")
    arrow("Camera model", "Zones on the floor (m²)", "t", "b")
    arrow("Zones on the floor (m²)", "Chronos-2 zero-shot")
    arrow("Chronos-2 zero-shot", "Risk engine")
    arrow("Risk engine", "Alert manager", "b", "t")
    arrow("Alert manager", "FastAPI + WebSocket", "l", "r")
    ax.set_xlim(0, 7.95)
    ax.set_ylim(0.05, 1.2)
    ax.axis("off")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _video_frame(path: Path, t: float) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * cap.get(cv2.CAP_PROP_FPS))))
    ok, img = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"cannot read {path} at {t} s")
    return img


def heatmap_frame(run_id: str, out: Path, t: float | None = None) -> float:
    """Camera frame with each floor zone filled in its density-band colour and labelled with persons/m²."""
    tl = replay.timeline(run_id)
    if t is None:                                                  # busiest second of the recording
        t = max(tl, key=lambda m: sum((z["density"] or 0) for z in m["zones"]))["timestamp"]
    msg = min(tl, key=lambda m: abs(m["timestamp"] - t))
    lay = replay.layout(run_id)
    vp = replay.video_path(run_id)
    img = _video_frame(vp, msg["timestamp"]) if vp else cv2.imread(str(replay.ROOT / replay._run_cfg(run_id)["frame"]))
    s = img.shape[1] / lay["width"]
    over = img.copy()
    dens = {z["zone_id"]: z["density"] for z in msg["zones"]}
    for z in lay["zones"]:
        d = dens.get(z["zone_id"])
        if d is None or len(z["points"]) < 3:
            continue
        P = (np.asarray(z["points"]) * s).astype(np.int32)
        r, g, b = band_rgb(d)
        cv2.fillPoly(over, [P], (b, g, r))
        cv2.polylines(img, [P], True, (255, 255, 255), 1, cv2.LINE_AA)
    img = cv2.addWeighted(over, 0.35, img, 0.65, 0)
    for z in lay["zones"]:
        d = dens.get(z["zone_id"])
        if d is None or len(z["points"]) < 3:
            continue
        c = (np.asarray(z["points"]) * s).mean(0).astype(int)
        label = f"{d:.2f}"
        cv2.putText(img, label, tuple(c), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, label, tuple(c), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(out), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return float(msg["timestamp"])


def forecast_fan(run_id: str, out: Path, model: str = "chronos2_joint", baseline: str = "persistence",
                 event_density: float = 4.0, lead_s: float = 30.0, history_s: float = 120.0) -> dict:
    """History + forecast quantiles at one origin `lead_s` before the busiest zone first reaches `event_density`."""
    r = replay._run_cfg(run_id)
    _, feat, _ = replay._load(run_id)
    import pandas as pd
    pts = pd.read_parquet(replay.ROOT / r["points"])
    pts = pts[pts["camera_id"] == r["camera"]]
    zone = feat.groupby("zone_id")["density"].max().idxmax()
    g = feat[feat["zone_id"] == zone].sort_values("timestamp")
    hit = g[g["density"] >= event_density]["timestamp"]
    t_event = float(hit.iloc[0]) if len(hit) else float(g.loc[g["density"].idxmax(), "timestamp"])
    origins = np.sort(pts["origin"].unique())
    origin = float(origins[np.argmin(np.abs(origins - (t_event - lead_s)))])
    fig, ax = plt.subplots(figsize=(8, 3.4))
    for lo, hi, col in BAND_FILL:
        ax.axhspan(lo, hi, color=col, zorder=0)
    win = g[(g["timestamp"] >= origin - history_s) & (g["timestamp"] <= origin + max(70.0, float(pts["horizon_s"].max()) + 10))]
    past, fut = win[win["timestamp"] <= origin], win[win["timestamp"] >= origin]
    ax.plot(past["timestamp"] - origin, past["density"], color="black", lw=1.2, label="observed (input)")
    ax.plot(fut["timestamp"] - origin, fut["density"], color="black", lw=1.2, ls=":", label="observed (future, not seen)")
    info = {"zone": zone, "origin": origin, "t_event": t_event}
    for m, col in ((model, "tab:blue"), (baseline, "tab:gray")):
        f = pts[(pts["model"] == m) & (pts["zone_id"] == zone) & (pts["origin"] == origin)].sort_values("horizon_s")
        if f.empty:
            continue
        h = np.r_[0, f["horizon_s"].to_numpy(float)]
        last = float(f["last"].iloc[0])
        ub = f["p90_cal"].fillna(f["p90"]).to_numpy(float)
        if m == model:
            ax.fill_between(h, np.r_[last, f["p10"]], np.r_[last, f["p90"]], color=col, alpha=0.25, label=f"{m} 10–90 %")
            ax.plot(h, np.r_[last, f["p50"]], color=col, ls="--", lw=1.2, label=f"{m} median")
        ax.plot(h, np.r_[last, ub], color=col, lw=1.6, label=f"{m} calibrated 90 % bound")
        info[m] = {int(k): round(float(v), 2) for k, v in zip(f["horizon_s"], ub)}
    ax.axvline(0, color="#444", lw=0.8)
    ymax = max(6.0, float(win["density"].max()) * 1.15)
    ax.set_ylim(0, ymax)
    ax.set_xlabel("time relative to forecast origin (s)")
    ax.set_ylabel(f"zone {zone} density (persons/m²)")
    ax.legend(fontsize=7, loc="upper left")
    ax.set_title(r["title"], fontsize=9)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return info


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=Path("results/figures"))
    ap.add_argument("--heatmap-runs", nargs="*", default=["cam02", "cam03", "cam04"])
    ap.add_argument("--forecast-run", default="sim_corridor")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    pipeline_diagram(a.out / "paper_pipeline.png")
    print("wrote", a.out / "paper_pipeline.png")
    for run in a.heatmap_runs:
        t = heatmap_frame(run, a.out / f"paper_heatmap_{run}.jpg")
        print("wrote", a.out / f"paper_heatmap_{run}.jpg", f"t={t:.0f}s")
    info = forecast_fan(a.forecast_run, a.out / f"paper_forecast_{a.forecast_run}.png")
    print("wrote", a.out / f"paper_forecast_{a.forecast_run}.png", info)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Paper figures (T9.3): pipeline diagram, zone-density heatmap on a camera frame, forecast fan chart, alert timeline.

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
    """End-to-end pipeline (paper Fig. 1): blocks, what passes between them, and which parts are evaluation-only.

    Vector output when `out` ends in .pdf/.svg. Dashed block/arrow = evaluated but not in the deployed zone path.
    """
    fig, ax = plt.subplots(figsize=(7.16, 4.9))
    ax.set_xlim(0, 14.4)
    ax.set_ylim(-0.55, 9.45)
    ax.axis("off")
    fs, fs_t, fs_a = 5.25, 5.9, 5.0
    col = {"in": "#e8eef9", "vis": "#e8f5e9", "geo": "#f2f2f2", "feat": "#fff8e1", "fc": "#fff3e0",
           "risk": "#fdecea", "out": "#ede7f6", "eval": "#ffffff"}
    lab = dict(fontsize=fs_a, style="italic", color="#1f3a68")
    box = {}

    def b(key, x, y, w, h, title, body, c, dashed=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.1", fc=col[c],
                                    ec="#444", lw=0.7, ls="--" if dashed else "-"))
        ax.text(x + w / 2, y + h - 0.12, title, ha="center", va="top", fontsize=fs_t, weight="bold")
        ax.text(x + w / 2, y + h - 0.44, body, ha="center", va="top", fontsize=fs, linespacing=1.2)
        box[key] = (x, y, w, h)

    def pt(key, side, f=0.5):
        x, y, w, h = box[key]
        return {"l": (x, y + f * h), "r": (x + w, y + f * h), "t": (x + f * w, y + h), "b": (x + f * w, y)}[side]

    def arr(p0, p1, dashed=False, rad=0.0):
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=7, color="#222", lw=0.7,
                                     ls="--" if dashed else "-", connectionstyle=f"arc3,rad={rad}", shrinkA=0.5, shrinkB=0.5))

    def note(x, y, text, ha="center", va="center"):
        ax.text(x, y, text, ha=ha, va=va, **lab)

    # inputs
    b("cctv", 0.3, 7.95, 2.6, 1.05, "CCTV video", "fixed camera\n1080p, 25 fps", "in")
    b("admin", 6.4, 7.95, 7.7, 1.05, "Administrator configuration",
      "measured desk/table sizes, camera height · floor zone polygons (m) · exclusion masks\n"
      "risk bands 2 / 4 / 5 persons/m² · alert hold 10 s, clear 30 s", "in")
    # per-frame vision: sampling (top-left), RAFT under it, CLIP-EBC right of sampling
    b("samp", 0.3, 6.15, 2.6, 0.95, "Frame sampling", "1 frame/s + partner frame\n0.2 s later; exclusion mask", "vis")
    b("raft", 0.3, 5.0, 2.6, 0.85, "RAFT optical flow", "pretrained, 960 px", "vis")
    b("clip", 3.7, 5.0, 2.8, 2.1, "CLIP-EBC ViT-B/16", "zero-shot, NWPU weights;\n224-px sliding windows\n→ density map D_t\n(persons per cell)", "vis")
    b("yolo", 7.1, 5.75, 2.6, 1.35, "YOLO11l + switch", "detector count used\nwhen density count < τ;\nframe-level evaluation only", "eval", dashed=True)
    b("cam", 10.0, 5.0, 4.1, 2.1, "Camera model (fitted offline)",
      "pinhole + radial distortion k1, k2,\nfitted to desk tops and straight edges;\n→ zone masks M_z, visible area A_z (m²),\npixel → floor at head height 1.2 m", "geo")
    # zone features
    b("agg", 0.3, 2.75, 6.0, 1.6, "Zone aggregation (zone z, second t)",
      "count c = sum of D_t inside M_z;   density ρ = c / A_z\nflow traced to the floor → speed (m/s), speed variance,\npressure ρ·var, direction entropy, counter-flow", "feat")
    b("table", 6.9, 2.75, 3.4, 1.6, "Zone feature table",
      "one row per (camera, z, t):\ncount, density, speed, variance,\npressure, entropy, counter-flow,\ninflow*, outflow*, source", "feat")
    b("trajf", 10.9, 2.75, 3.2, 1.6, "Trajectory data",
      "Jülich (measured) and\nJuPedSim (simulated) runs\n→ 2-m cells, same columns;\ninflow/outflow from crossings", "in")
    # forecast -> alert
    w6, gap = 2.08, 0.32
    xs = [0.3 + i * (w6 + gap) for i in range(6)]
    for key, title, body, c, x in (
            ("win", "Forecast window", "last 30 s per zone\nat 1 Hz; live: 1-s\ngrid, gaps\nforward-filled", "fc", xs[0]),
            ("fc", "Forecasters", "Chronos-2, zero-shot:\nper zone (+ motion\ncovariates) or joint;\npersistence, physics*", "fc", xs[1]),
            ("conf", "Conformal bound", "U_h = p90_h + q_h;\nq_h fitted on the\nvalidation split", "fc", xs[2]),
            ("risk", "Risk level", "band of max(U_h, ρ):\nLow / Medium /\nHigh / Critical", "risk", xs[3]),
            ("alert", "Alert manager", "raise: ≥ High for 10 s\nclear: < High for 30 s;\nreason text,\ntime-to-critical", "risk", xs[4]),
            ("outp", "Outputs", "alert events;\nstage log (live run);\nFastAPI/WebSocket\n→ replay dashboard", "out", xs[5])):
        b(key, x, 0.4, w6, 1.7, title, body, c)

    # arrows and data labels
    arr(pt("cctv", "b"), pt("samp", "t")); note(1.75, 7.55, "video frames", ha="left")
    arr(pt("admin", "b", 0.72), pt("cam", "t", 0.55)); note(12.4, 7.55, "measurements, polygons", ha="left")
    arr(pt("samp", "r", 0.5), pt("clip", "l", 0.77)); note(3.3, 6.86, "frame I_t")
    arr(pt("samp", "b", 0.5), pt("raft", "t", 0.5)); note(1.75, 5.99, "pair (I_t, I_t+0.2 s)", ha="left")
    arr(pt("clip", "r", 0.8), pt("yolo", "l", 0.6), dashed=True); note(6.8, 6.25, "frame\ncount")
    arr(pt("raft", "b", 0.5), pt("agg", "t", 0.25)); note(1.75, 4.65, "flow F_t (px/frame pair)", ha="left")
    arr(pt("clip", "b", 0.5), pt("agg", "t", 0.77)); note(4.95, 4.65, "D_t", ha="left")
    arr(pt("cam", "b", 0.1), pt("agg", "t", 0.98)); note(8.6, 4.8, "M_z, A_z, pixel → floor map", ha="left")
    arr(pt("agg", "r", 0.4), pt("table", "l", 0.4)); note(6.6, 3.82, "rows")
    arr(pt("trajf", "l", 0.4), pt("table", "r", 0.4)); note(10.6, 3.82, "rows")
    arr(pt("table", "b", 0.5), pt("win", "t", 0.6), rad=-0.1); note(5.0, 2.55, "time-indexed zone series", ha="left")
    chain = ["win", "fc", "conf", "risk", "alert", "outp"]
    labels = ["context", "p10/p50/p90 at h", "bound U_h", "level", "alert event"]
    for i in range(5):
        arr(pt(chain[i], "r", 0.55), pt(chain[i + 1], "l", 0.55))
        note(xs[i] + w6 + gap / 2, 0.18, labels[i], va="top")
    ax.text(0.3, -0.5, "* inflow/outflow exist only for trajectory data (missing for CCTV); the physics forecaster is "
            "evaluated only there.\nDashed: evaluated, not part of the zone path.   Horizons h = 10, 30, 60 s "
            "(also 120 s in simulation).", fontsize=4.9, ha="left", va="bottom", color="#555", linespacing=1.3)
    fig.savefig(out, bbox_inches="tight", dpi=300)
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


def alert_timeline(features: Path, points: Path, timelines: Path, run: str, zones: list[str], out: Path,
                   variant: str = "chronos2_per_zone upper bound", model: str = "chronos2_per_zone",
                   event_density: float = 4.0, window_s: float = 70.0, warmup_s: float = 15.0) -> list[dict]:
    """Alert timeline from stored evaluation outputs (paper Fig. 3): observed density, the calibrated worst-case bound,
    the event threshold and event time, alert raise times and whether each event counts as warned in time.

    Uses the same rule as scripts/eval_alerts.py `score`: an event is the first second at or above `event_density`
    after monitoring starts; it is warned in time if an alert was raised within `window_s` before it and later than
    `warmup_s` after monitoring started."""
    import pandas as pd
    from crowdsafe.features.logger import read_features

    f = read_features(features)
    f = f[f["camera_id"] == run]
    p = pd.read_parquet(points)
    p = p[(p["camera_id"] == run) & (p["model"] == model)]
    ub = p.assign(u=p["p90_cal"].fillna(p["p90"])).groupby(["zone_id", "origin"])["u"].max().reset_index()
    tl = pd.read_csv(timelines)
    tl = tl[(tl["camera_id"] == run) & (tl["variant"] == variant)]
    start = float(tl[tl["zone_id"].isna()]["monitor_start"].iloc[0])
    al = tl[tl["zone_id"].notna()]
    fig, axes = plt.subplots(len(zones), 1, figsize=(3.5, 1.15 * len(zones) + 0.35), sharex=True)
    axes = np.atleast_1d(axes)
    info = []
    ymax = 9.0
    for ax, z in zip(axes, zones):
        for lo, hi, c in BAND_FILL:
            ax.axhspan(lo, min(hi, ymax), color=c, zorder=0, lw=0)
        ax.axvspan(0, start + warmup_s, color="#bbbbbb", alpha=0.35, lw=0, zorder=1)
        g = f[f["zone_id"] == z].sort_values("timestamp")
        ax.plot(g["timestamp"], g["density"], color="black", lw=0.8, label="observed density", zorder=3)
        u = ub[ub["zone_id"] == z].sort_values("origin")
        ax.plot(u["origin"], u["u"], color="tab:blue", lw=0.9, label="calibrated upper bound (max over h)", zorder=3)
        ax.axhline(event_density, color="#7a3e00", lw=0.6, ls="--", zorder=2)
        hi = g.loc[(g["density"] >= event_density) & (g["timestamp"] >= start), "timestamp"]
        ev = float(hi.min()) if len(hi) else None
        za = np.sort(al[al["zone_id"] == z]["since"].to_numpy(float))
        for s in za:
            ax.axvline(s, color="red", lw=1.0, zorder=4)
        if ev is not None:
            ax.axvline(ev, color="green", lw=1.0, ls=":", zorder=4)
            before = za[za <= ev]
            strict = before[(before >= ev - window_s) & (before > start + warmup_s)]
            early = before[(before < ev - window_s) & (before > start + warmup_s)]
            if len(strict):
                verdict = f"warned in time ({ev - strict.min():.0f} s ahead)"
            elif len(early):
                verdict = f"not in time: alert {ev - early.max():.0f} s before (> {window_s:.0f} s)"
            elif len(before):
                verdict = "alert only in warm-up: not counted"
            elif len(za[(za > ev) & (za <= ev + 30)]):
                verdict = f"late: alert {za[za > ev].min() - ev:.0f} s after event"
            else:
                verdict = "missed"
        else:
            verdict = "no event"
        info.append({"zone": z, "event_s": ev, "alerts_s": za.tolist(), "verdict": verdict})
        ax.text(0.99, 0.93, f"zone {z}: {verdict}", transform=ax.transAxes, ha="right", va="top", fontsize=5.8,
                bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.85))
        ax.set_ylim(0, ymax)
        ax.set_ylabel("p/m²", fontsize=6)
        ax.tick_params(labelsize=5.5)
    axes[-1].set_xlabel("time in run (s)", fontsize=6)
    handles = [plt.Line2D([], [], color="black", lw=0.8), plt.Line2D([], [], color="tab:blue", lw=0.9),
               plt.Line2D([], [], color="#7a3e00", lw=0.6, ls="--"), plt.Line2D([], [], color="green", lw=1, ls=":"),
               plt.Line2D([], [], color="red", lw=1), plt.Rectangle((0, 0), 1, 1, color="#bbbbbb", alpha=0.35)]
    fig.legend(handles, ["observed density", "calibrated upper bound", "event threshold 4 p/m²", "event (first ≥ 4)",
                         "alert raised", "before forecasts + 15 s warm-up"], loc="upper center", ncol=3, fontsize=5.2,
               frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.9), h_pad=0.3)
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return info


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=Path("results/figures"))
    ap.add_argument("--heatmap-runs", nargs="*", default=["cam02", "cam03", "cam04"])
    ap.add_argument("--forecast-run", default="sim_corridor")
    ap.add_argument("--timeline-features", default="data/features/julich")
    ap.add_argument("--timeline-points", default="weights/crowdsafe_nb04_report/forecast_julich_points.parquet")
    ap.add_argument("--timeline-csv", default="results/alerts_julich_timelines.csv")
    ap.add_argument("--timeline-run", default="julich_corridor_uni_2013/uni_corr_500_09")
    ap.add_argument("--timeline-zones", nargs="*", default=["c3_2", "c2_3", "c4_3"])
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    pipeline_diagram(a.out / "paper_pipeline.png")
    print("wrote", a.out / "paper_pipeline.png")
    for run in a.heatmap_runs:
        t = heatmap_frame(run, a.out / f"paper_heatmap_{run}.jpg")
        print("wrote", a.out / f"paper_heatmap_{run}.jpg", f"t={t:.0f}s")
    info = forecast_fan(a.forecast_run, a.out / f"paper_forecast_{a.forecast_run}.png")
    print("wrote", a.out / f"paper_forecast_{a.forecast_run}.png", info)
    pipeline_diagram(a.out / "paper_pipeline.pdf")
    print("wrote", a.out / "paper_pipeline.pdf")
    for ext in ("pdf", "png"):
        info = alert_timeline(Path(a.timeline_features), Path(a.timeline_points), Path(a.timeline_csv), a.timeline_run,
                              a.timeline_zones, a.out / f"paper_alert_timeline.{ext}")
    print("wrote", a.out / "paper_alert_timeline.pdf", info)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

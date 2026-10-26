"""Real-time pipeline with per-stage latency logging (T9.2; architecture.md §2.7).

Every `--step-s` (1 s) of wall time: take the newest frame and the frame `--pair-gap-s` later, run CLIP-EBC density
and RAFT flow, aggregate to zones (persons/m², m/s), log to SQLite; every `--forecast-every-s` (5 s, once 30 s of
history exist) run the forecaster with the conformal offsets fitted on the validation cameras, then the risk engine
and the alert manager (configs/risk.yaml). Same models and settings as the offline results — only the loop is new.

Timings per cycle (ms): wait_pair, density, flow, zones, log, forecast, risk, total compute, and end-to-end latency =
time from the newest frame's arrival to the result being ready. Writes a CSV per cycle and a Markdown summary.

    python scripts/live_pipeline.py --source data/cctv/cam02.asf --camera cam02 --duration-s 600 --clipebc-repo <dir>   # A100
    python scripts/live_pipeline.py --source data/cctv/cam02.asf --camera cam02 --density-backend onnx \
        --flow-backend farneback --device cpu --forecaster persistence --duration-s 20 --no-realtime                    # laptop test
    python scripts/live_pipeline.py --source rtsp://<camera> --camera cam01 ...                                         # real camera
"""
from __future__ import annotations

import argparse
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
import yaml  # noqa: E402

from crowdsafe.calibration.zones import camera_rooms, load_room, visible_area_m2, zone_masks  # noqa: E402
from crowdsafe.datasets.live import LiveSource  # noqa: E402
from crowdsafe.features.logger import SqliteLogger  # noqa: E402
from crowdsafe.motion.flow import FlowModel  # noqa: E402
from crowdsafe.motion.zone_motion import ZoneMotion  # noqa: E402
from crowdsafe.risk.alerts import AlertManager  # noqa: E402
from crowdsafe.risk.rules import RiskEngine  # noqa: E402
from scripts.cctv_features import build_density  # noqa: E402
from scripts.eval_cctv_counting import density_mask, load_roi  # noqa: E402

HORIZONS = [10, 30, 60]


def make_forecaster(name: str, device: str):
    if name == "persistence":
        from crowdsafe.forecasting.baselines import Persistence
        return Persistence()
    from crowdsafe.forecasting.chronos import ChronosForecaster
    return ChronosForecaster(mode=name.split("_", 1)[1], device=device)


def regular_grid(hist: pd.DataFrame, step: float) -> pd.DataFrame:
    """Every zone on every grid second from the first to the last timestamp; a second skipped by a slow cycle is
    filled with the previous value (live timing is irregular, Chronos-2 needs a fixed frequency)."""
    grid = np.round(np.arange(hist["timestamp"].min(), hist["timestamp"].max() + step / 2, step), 6)
    parts = []
    for z, g in hist.groupby("zone_id"):
        g = g.set_index("timestamp").reindex(grid).ffill()
        g["zone_id"] = z
        parts.append(g.rename_axis("timestamp").reset_index())
    return pd.concat(parts, ignore_index=True)


def gpu_sync(device: str) -> None:
    if device.startswith("cuda"):
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()


def gpu_peak_mb(device: str) -> float | None:
    if device.startswith("cuda"):
        import torch
        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / 2 ** 20, 1)
    return None


def summarise(df: pd.DataFrame, a, info: dict) -> list[str]:
    stages = ["wait_pair_ms", "density_ms", "flow_ms", "zones_ms", "log_ms", "forecast_ms", "risk_ms", "compute_ms", "latency_ms"]
    rows = []
    for s in stages:
        v = df[s].dropna()
        if len(v):
            rows.append(f"| {s.replace('_ms', '')} | {len(v)} | {v.mean():.0f} | {v.median():.0f} | {v.quantile(0.95):.0f} | {v.max():.0f} |")
    lag = df["behind_s"]
    return [f"# Live pipeline latency (T9.2) — {a.camera}", "",
            f"Generated {time.strftime('%Y-%m-%d %H:%M')} by `scripts/live_pipeline.py`. Source `{a.source}` "
            f"({'played at real speed (simulated live camera)' if a.realtime else 'read as fast as possible (not real time)'}), "
            f"{len(df)} cycles of {a.step_s} s, device `{a.device}`, density `{a.density_backend}`, flow `{a.flow_backend}` "
            f"({a.flow_width} px), forecaster `{a.forecaster}` every {a.forecast_every_s} s.", "",
            "- `wait_pair`: waiting for the second frame of the motion pair (= pair gap; part of latency, not compute).",
            "- `latency`: newest frame arrived → zone features, risk level and alerts ready (excludes the network/dashboard).",
            "- `forecast` only in the cycles that run it.", "",
            "| Stage | cycles | mean ms | median ms | p95 ms | max ms |", "| --- | --- | --- | --- | --- | --- |", *rows, "",
            f"- Cycles that finished within the {a.step_s} s budget: {100 * (df['compute_ms'] + df['wait_pair_ms'] <= a.step_s * 1000).mean():.1f} %",
            f"- How far processing fell behind the live stream at the end: {lag.iloc[-1]:.1f} s (max {lag.max():.1f} s)",
            f"- Frames decoded {info['frames_read']} in {info['wall_s']:.0f} s ({info['frames_read'] / max(info['wall_s'], 1e-9):.1f} fps)",
            f"- GPU peak memory: {info['gpu_peak_mb']} MiB" if info["gpu_peak_mb"] is not None else "- GPU: not used",
            f"- Alerts raised: {info['alerts']}",
            "", "Design delays on top of compute (not measured here): forecasts every "
            f"{a.forecast_every_s} s, an alert must hold {info['raise_after_s']} s before it is raised."]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", required=True, help="video file or rtsp:// / http:// stream")
    ap.add_argument("--camera", required=True, help="camera id with a calibration (configs/zones/)")
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--no-realtime", dest="realtime", action="store_false", help="read a file as fast as possible")
    ap.add_argument("--duration-s", type=float, default=600.0)
    ap.add_argument("--step-s", type=float, default=1.0)
    ap.add_argument("--pair-gap-s", type=float, default=0.2)
    ap.add_argument("--forecast-every-s", type=float, default=5.0)
    ap.add_argument("--context-s", type=float, default=30.0)
    ap.add_argument("--forecaster", default="chronos2_joint", choices=["chronos2_joint", "chronos2_per_zone", "persistence"])
    ap.add_argument("--offsets", type=Path, default=Path("results/forecast_cctv.csv"), help="conformal offsets q (val cameras)")
    ap.add_argument("--density-backend", choices=["torch", "onnx"], default="torch")
    ap.add_argument("--clipebc-repo", type=Path)
    ap.add_argument("--onnx", type=Path, default=Path("weights/crowdsafe_nb01_onnx/onnx/clipebc_vitb16_nwpu.onnx"))
    ap.add_argument("--flow-backend", choices=["raft", "farneback"], default="raft")
    ap.add_argument("--flow-width", type=int, default=960)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--roi", type=Path, default=Path("configs/roi.yaml"))
    ap.add_argument("--risk", type=Path, default=Path("configs/risk.yaml"))
    ap.add_argument("--min-area", type=float, default=2.0)
    ap.add_argument("--warmup-cycles", type=int, default=3, help="first cycles excluded from the summary (model warm-up)")
    ap.add_argument("--db", type=Path, default=Path("data/live/live.sqlite"))
    ap.add_argument("--out", type=Path, default=Path("results/latency_live.md"))
    a = ap.parse_args()

    cam, zones, hz = load_room(camera_rooms()[a.camera])
    areas = {z: visible_area_m2(cam, p, hz) for z, p in zones.items()}
    polys = load_roi(a.roi).get(a.camera, [])
    density, flow = build_density(a), FlowModel(a.flow_backend, a.flow_width, a.device)
    fcaster = make_forecaster(a.forecaster, a.device)
    off = pd.read_csv(a.offsets)
    off = off[off["model"] == a.forecaster].set_index("horizon_s")["q"].to_dict()
    risk = yaml.safe_load(a.risk.read_text(encoding="utf-8"))
    eng = RiskEngine(bands=tuple(risk["bands"]), thresholds=risk["thresholds"], collapse_window_s=risk["collapse_window_s"],
                     collapse_min_density=risk["collapse_min_density"], use_precursors=False)
    mgr = AlertManager(risk["alerts"]["raise_after_s"], risk["alerts"]["clear_after_s"], risk["alerts"]["alert_level"])
    a.db.parent.mkdir(parents=True, exist_ok=True)
    log = SqliteLogger(a.db)
    masks = motion = roi = None
    hist, rows, last_fc_t, fc = [], [], -1e9, None
    src = LiveSource(a.source, a.fps, a.realtime).start()
    t_start = time.perf_counter()
    next_tick = t_start
    try:
        while time.perf_counter() - t_start < a.duration_s and not (src.done.is_set() and src.latest() is None):
            if a.realtime:                                   # one cycle per step of wall time
                time.sleep(max(0.0, next_tick - time.perf_counter()))
                next_tick += a.step_s
            if a.realtime:                                   # live: always the newest frame (older ones are dropped)
                f0 = src.latest()
                if f0 is None:
                    time.sleep(0.01)
                    continue
            else:                                            # offline test: exactly one step of video per cycle
                f0 = src.after(rows[-1]["video_t"] + a.step_s if rows else 0.0, timeout_s=30.0)
                if f0 is None:
                    break
            src.consumed_t = f0.video_t
            c0 = time.perf_counter()
            f1 = src.after(f0.video_t + a.pair_gap_s)
            c1 = time.perf_counter()
            D = density(cv2.cvtColor(f0.image, cv2.COLOR_BGR2RGB)).astype(np.float32)
            gpu_sync(a.device)
            c2 = time.perf_counter()
            if masks is None:
                masks = zone_masks(cam, zones, hz, D.shape)
                motion = ZoneMotion(cam, masks, hz)
                roi = density_mask(D.shape, (cam.image_size[1], cam.image_size[0]), polys) if polys else None
            if roi is not None:
                D = D * roi
            fl = flow(f0.image, f1.image) if f1 is not None else None
            gpu_sync(a.device)
            c3 = time.perf_counter()
            mot = motion(fl, D, f1.video_t - f0.video_t) if fl is not None else {}
            t = float(round(f0.video_t / a.step_s) * a.step_s)   # regular grid (Chronos-2 needs a fixed frequency)
            frame_rows = []
            for z in zones:
                cnt = float((D * masks[z]).sum())
                dens = cnt / areas[z] if areas[z] >= a.min_area else np.nan
                m = mot.get(z, {})
                sv = m.get("speed_var", np.nan)
                frame_rows.append({"timestamp": t, "camera_id": a.camera, "zone_id": z, "count": cnt, "density": dens,
                                   "fusion_weight": np.nan, "mean_speed": m.get("mean_speed", np.nan), "speed_var": sv,
                                   "pressure": dens * sv if np.isfinite(sv) else np.nan, "dir_entropy": m.get("dir_entropy", np.nan),
                                   "counterflow": m.get("counterflow", np.nan), "inflow": np.nan, "outflow": np.nan, "source": "real"})
            hist.extend(frame_rows)
            c4 = time.perf_counter()
            log.append(pd.DataFrame(frame_rows))
            c5 = time.perf_counter()
            hdf = pd.DataFrame([r for r in hist if r["timestamp"] > t - a.context_s])
            hdf = hdf.drop_duplicates(["timestamp", "zone_id"], keep="last")      # two cycles in one grid second
            hist = hdf.to_dict("records")
            fc_ms = None
            if t - last_fc_t >= a.forecast_every_s and t - hdf["timestamp"].min() >= a.context_s - a.step_s - 1e-6:
                g = regular_grid(hdf.dropna(subset=["density"]), a.step_s)
                fc = fcaster.predict(g, HORIZONS)
                fc["p90"] = np.clip(fc["p90"] + fc["horizon_s"].map(off).fillna(0.0), 0, None)
                gpu_sync(a.device)
                fc_ms = (time.perf_counter() - c5) * 1000
                last_fc_t = t
            c6 = time.perf_counter()
            risks = eng(hdf.dropna(subset=["density"]), fc[["zone_id", "horizon_s", "p90"]] if fc is not None else None)
            alerts = mgr.update(t, risks)
            c7 = time.perf_counter()
            rows.append({"video_t": t, "wait_pair_ms": (c1 - c0) * 1000, "density_ms": (c2 - c1) * 1000,
                         "flow_ms": (c3 - c2) * 1000, "zones_ms": (c4 - c3) * 1000, "log_ms": (c5 - c4) * 1000,
                         "forecast_ms": fc_ms, "risk_ms": (c7 - c6) * 1000,
                         "compute_ms": (c7 - c1) * 1000,              # density .. risk, incl. forecast when it ran
                         "latency_ms": (c7 - f0.wall_t) * 1000,
                         "behind_s": (src.latest().video_t - t) if src.latest() is not None else np.nan,
                         "peak_density": max((r["density"] for r in frame_rows if np.isfinite(r["density"])), default=np.nan),
                         "max_level": max((r.level_idx for r in risks), default=0), "active_alerts": len(alerts)})
            if len(rows) % 30 == 0 or len(rows) <= 3:
                r = rows[-1]
                print(f"  t={t:7.1f}s  latency {r['latency_ms']:6.0f} ms  (density {r['density_ms']:.0f}, flow {r['flow_ms']:.0f}"
                      f"{', forecast %.0f' % fc_ms if fc_ms else ''})  peak {r['peak_density']:.2f} p/m²  behind {r['behind_s']:.1f} s", flush=True)
    finally:
        src.stop()
        log.close()
    wall = time.perf_counter() - t_start
    df = pd.DataFrame(rows)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out.with_suffix(".csv"), index=False)
    keep = df.iloc[a.warmup_cycles:] if len(df) > a.warmup_cycles else df
    info = {"frames_read": src.n_read, "wall_s": wall, "gpu_peak_mb": gpu_peak_mb(a.device), "alerts": len(mgr.history),
            "raise_after_s": risk["alerts"]["raise_after_s"]}
    a.out.write_text("\n".join(summarise(keep, a, info)) + "\n", encoding="utf-8")
    print("\n".join(summarise(keep, a, info)))
    print(json.dumps({"cycles": len(df), **info}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

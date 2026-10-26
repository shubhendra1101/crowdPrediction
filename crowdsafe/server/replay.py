"""Replay timelines for the dashboard (T8.1/T8.2): stored features + forecasts -> per-second messages.

The message format follows architecture.md §2.7. The risk engine (crowdsafe.risk.rules) and the alert manager
(crowdsafe.risk.alerts) run at every forecast origin with configs/risk.yaml; between origins the last levels and
forecasts are held, while the observed density updates every second.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

from crowdsafe.features.logger import read_features
from crowdsafe.risk.alerts import AlertManager
from crowdsafe.risk.rules import RiskEngine

ROOT = Path(__file__).resolve().parents[2]


def _cfg() -> dict:
    return yaml.safe_load((ROOT / "configs" / "dashboard.yaml").read_text(encoding="utf-8"))


def list_runs() -> list[dict]:
    """Runs whose input files exist."""
    out = []
    for r in _cfg()["runs"]:
        ok = (ROOT / r["points"]).exists() and (r["features"] is None or (ROOT / r["features"]).exists())
        if ok:
            out.append({"id": r["id"], "title": r["title"], "kind": r["kind"]})
    return out


def _run_cfg(run_id: str) -> dict:
    for r in _cfg()["runs"]:
        if r["id"] == run_id:
            return r
    raise KeyError(run_id)


@lru_cache(maxsize=8)
def _points(path: str) -> pd.DataFrame:
    return pd.read_parquet(ROOT / path)


def _load(run_id: str) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    r = _run_cfg(run_id)
    model = _cfg()["forecast_model"]
    pts = _points(r["points"])
    pts = pts[(pts["camera_id"] == r["camera"]) & (pts["model"] == model)].copy()
    pts["p90"] = pts["p90_cal"].fillna(pts["p90"]) if "p90_cal" in pts else pts["p90"]
    if r["features"] is None:
        ref = pts.drop_duplicates(["zone_id", "origin"])
        feat = pd.DataFrame({"timestamp": ref["origin"].astype(float), "zone_id": ref["zone_id"],
                             "density": ref["last"].astype(float)})
        for c in ("mean_speed", "speed_var", "pressure", "dir_entropy", "counterflow"):
            feat[c] = np.nan
    else:
        feat = read_features(ROOT / r["features"])
        feat = feat[feat["camera_id"] == r["camera"]].copy()
    return r, feat.sort_values(["timestamp", "zone_id"]).reset_index(drop=True), pts


def _f(x) -> float | None:
    return None if x is None or not np.isfinite(x) else round(float(x), 3)


@lru_cache(maxsize=8)
def timeline(run_id: str) -> list[dict]:
    """Per-second messages for one run."""
    r, feat, pts = _load(run_id)
    risk = yaml.safe_load((ROOT / "configs" / "risk.yaml").read_text(encoding="utf-8"))
    eng = RiskEngine(bands=tuple(risk["bands"]), thresholds=risk["thresholds"], collapse_window_s=risk["collapse_window_s"],
                     collapse_min_density=risk["collapse_min_density"], use_precursors=False)
    mgr = AlertManager(risk["alerts"]["raise_after_s"], risk["alerts"]["clear_after_s"], risk["alerts"]["alert_level"])
    by_origin = dict(tuple(pts.groupby("origin")))
    origins = np.sort(np.array(list(by_origin)))
    ts = feat["timestamp"].to_numpy(float)
    zones = sorted(feat["zone_id"].unique())
    state = {z: {"level": "Low", "ttc": None, "forecast": None} for z in zones}
    alerts: list = []
    out = []
    oi = 0
    for t in np.unique(ts):
        now = feat[feat["timestamp"] == t].set_index("zone_id")
        while oi < len(origins) and origins[oi] <= t:          # a decision is due
            o = origins[oi]
            fc = by_origin[o]
            hist = feat.iloc[np.searchsorted(ts, o - risk["collapse_window_s"]):np.searchsorted(ts, o, side="right")]
            for zr in eng(hist, fc[["zone_id", "horizon_s", "p90"]]):
                f = fc[fc["zone_id"] == zr.zone_id].sort_values("horizon_s")
                state[zr.zone_id] = {"level": zr.level, "ttc": zr.time_to_critical_s,
                                     "forecast": {"h": f["horizon_s"].astype(int).tolist(), "p10": [_f(v) for v in f["p10"]],
                                                  "p50": [_f(v) for v in f["p50"]], "p90": [_f(v) for v in f["p90"]]},
                                     "reasons": zr.reasons}
            alerts = mgr.update(float(o), eng(hist, fc[["zone_id", "horizon_s", "p90"]]))
            oi += 1
        if oi == 0:                                        # no forecast yet (first 30 s): level from current density only
            for zr in eng(feat.iloc[:np.searchsorted(ts, t, side="right")], None):
                state[zr.zone_id] = {**state[zr.zone_id], "level": zr.level, "ttc": zr.time_to_critical_s}
        out.append({
            "timestamp": float(t), "camera_id": r["camera"],
            "zones": [{"zone_id": z, "density": _f(now["density"].get(z, np.nan)),
                       "mean_speed": _f(now["mean_speed"].get(z, np.nan)), "pressure": _f(now["pressure"].get(z, np.nan)),
                       "forecast": state[z]["forecast"], "level": state[z]["level"],
                       "time_to_critical_s": state[z]["ttc"]} for z in zones],
            "alerts": [a.to_message() for a in alerts],
        })
    return out


def _grid_cell(zone_id: str) -> tuple[int, int] | None:
    m = re.fullmatch(r"c(-?\d+)_(-?\d+)", zone_id)
    return (int(m[1]), int(m[2])) if m else None


@lru_cache(maxsize=8)
def layout(run_id: str) -> dict:
    """Zone shapes for drawing: CCTV -> polygons in image pixels over the camera frame; grid -> 2 m cells (metres)."""
    r, feat, _ = _load(run_id)
    zones = sorted(feat["zone_id"].unique())
    if r["kind"] == "cctv":
        from crowdsafe.calibration.zones import camera_rooms, load_room
        cam, polys, hz = load_room(camera_rooms()[r["camera"]])
        out = []
        for z in zones:
            P = np.asarray(polys[z], float)
            edge = np.vstack([a + np.linspace(0, 1, 40, endpoint=False)[:, None] * (b - a) for a, b in zip(P, np.roll(P, -1, axis=0))])
            xyz = np.c_[edge, np.full(len(edge), hz)]
            ok = cam.in_view(xyz, margin_px=200)
            uv = cam.project(xyz[ok]) if ok.any() else np.zeros((0, 2))
            uv = np.clip(uv, [-50, -50], [cam.image_size[0] + 50, cam.image_size[1] + 50])
            out.append({"zone_id": z, "points": np.round(uv, 1).tolist()})
        return {"kind": "cctv", "width": cam.image_size[0], "height": cam.image_size[1], "image": f"/api/runs/{run_id}/frame",
                "zones": out}
    cells = {z: _grid_cell(z) for z in zones}
    cells = {z: c for z, c in cells.items() if c is not None}
    xs, ys = [c[0] for c in cells.values()], [c[1] for c in cells.values()]
    return {"kind": "grid", "cell_m": 2.0, "x0": min(xs), "y0": min(ys), "nx": max(xs) - min(xs) + 1, "ny": max(ys) - min(ys) + 1,
            "zones": [{"zone_id": z, "ix": c[0] - min(xs), "iy": c[1] - min(ys)} for z, c in cells.items()]}


def frame_jpeg(run_id: str, width: int = 960) -> bytes:
    r = _run_cfg(run_id)
    img = cv2.imread(str(ROOT / r["frame"]))
    s = width / img.shape[1]
    ok, buf = cv2.imencode(".jpg", cv2.resize(img, None, fx=s, fy=s), [cv2.IMWRITE_JPEG_QUALITY, 80])
    return buf.tobytes()

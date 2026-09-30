"""Rolling-origin forecast evaluation on feature tables (T6.4).

For every run, a forecast is issued every ``stride`` seconds once ``context`` seconds of history
exist; each forecast is scored against the true density ``h`` seconds later. Only data up to the
origin is passed to the model, so there is no look-ahead.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd


class Forecaster(Protocol):
    name: str

    def predict(self, history: pd.DataFrame, horizons: list[int]) -> pd.DataFrame: ...


def forecast_points(features: pd.DataFrame, model: Forecaster, context_s: int, horizons: list[int],
                    stride_s: int) -> pd.DataFrame:
    """All (run, origin, zone, horizon) forecasts with the observed target density."""
    out = []
    for run, g in features.groupby("camera_id", sort=False):
        truth = g.set_index(["timestamp", "zone_id"])["density"]
        t_min, t_max = g["timestamp"].min(), g["timestamp"].max()
        for origin in np.arange(t_min + context_s - 1, t_max - min(horizons) + 1, stride_s):
            hist = g[(g["timestamp"] <= origin) & (g["timestamp"] > origin - context_s)]
            valid = [h for h in horizons if origin + h <= t_max]
            if not valid:
                continue
            fc = model.predict(hist, valid)
            fc["target"] = [truth.get((origin + h, z), np.nan) for z, h in zip(fc["zone_id"], fc["horizon_s"])]
            fc["last"] = fc["zone_id"].map(hist[hist["timestamp"] == origin].set_index("zone_id")["density"])
            fc["camera_id"], fc["origin"], fc["model"] = run, origin, model.name
            out.append(fc)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def score(points: pd.DataFrame, critical: float = 5.0) -> pd.DataFrame:
    """MAE / RMSE per model and horizon, overall and on points whose target is >= 2 persons/m²,
    plus empirical coverage of [p10, p90] and of p90 as an upper bound."""
    p = points.dropna(subset=["target"])
    rows = []
    for (model, h), g in p.groupby(["model", "horizon_s"]):
        err = g["p50"] - g["target"]
        dense = g[g["target"] >= 2]
        rows.append({"model": model, "horizon_s": h, "n": len(g),
                     "mae": round(float(err.abs().mean()), 3), "rmse": round(float(np.sqrt((err ** 2).mean())), 3),
                     "n_dense": len(dense),
                     "mae_dense": round(float((dense["p50"] - dense["target"]).abs().mean()), 3) if len(dense) else np.nan,
                     "cover_p10_p90": round(float(((g["target"] >= g["p10"]) & (g["target"] <= g["p90"])).mean()), 3),
                     "below_p90": round(float((g["target"] <= g["p90"]).mean()), 3),
                     "n_target_critical": int((g["target"] >= critical).sum())})
    return pd.DataFrame(rows)

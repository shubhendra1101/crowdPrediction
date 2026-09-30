"""Baseline forecasters (architecture.md §2.5, T6.1).

Both follow the Forecaster interface: ``predict(history, horizons) -> DataFrame`` with columns
zone_id, horizon_s, p10, p50, p90 (+ time_to_critical_s for the physics model). ``history`` is the
feature table of one camera/run up to the forecast origin (1 row per zone per second).
Baselines give a point forecast, so p10 = p50 = p90 until conformal calibration widens them (T6.3).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _last(history: pd.DataFrame) -> pd.DataFrame:
    """Latest row per zone."""
    return history.sort_values("timestamp").groupby("zone_id").tail(1).set_index("zone_id")


def zone_area(history: pd.DataFrame) -> pd.Series:
    """Zone area in m² recovered from count / density (density = count / area)."""
    h = history[history["density"] > 0]
    return (h["count"] / h["density"]).groupby(h["zone_id"]).median()


def _frame(zone_ids, horizons: list[int], p50: np.ndarray) -> pd.DataFrame:
    rows = [{"zone_id": z, "horizon_s": h, "p50": float(v)} for z, pz in zip(zone_ids, p50) for h, v in zip(horizons, pz)]
    df = pd.DataFrame(rows)
    df["p10"] = df["p90"] = df["p50"]
    return df[["zone_id", "horizon_s", "p10", "p50", "p90"]]


class Persistence:
    """Future density = latest density."""

    name = "persistence"

    def predict(self, history: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
        last = _last(history)
        return _frame(last.index, horizons, np.repeat(last["density"].to_numpy()[:, None], len(horizons), 1))


class PhysicsFillRate:
    """ρ(t+h) = ρ(t) + (inflow − outflow) / area × h, with flows averaged over the last window.

    Also reports time_to_critical_s: seconds until ρ reaches ``critical`` at the current net fill
    rate (inf if the zone is not filling).
    """

    name = "physics"

    def __init__(self, flux_window_s: int = 10, critical: float = 5.0) -> None:
        self.window, self.critical = flux_window_s, critical

    def predict(self, history: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
        last = _last(history)
        t_end = history["timestamp"].max()
        recent = history[history["timestamp"] > t_end - self.window]
        net = (recent["inflow"] - recent["outflow"]).groupby(recent["zone_id"]).mean().reindex(last.index).fillna(0)
        area = zone_area(history).reindex(last.index)
        rate = (net / area).fillna(0).to_numpy()                       # persons/m² per second
        rho = last["density"].to_numpy()
        p50 = np.clip(rho[:, None] + rate[:, None] * np.asarray(horizons)[None, :], 0, None)
        out = _frame(last.index, horizons, p50)
        with np.errstate(divide="ignore", invalid="ignore"):
            ttc = np.where(rho >= self.critical, 0.0, np.where(rate > 0, (self.critical - rho) / rate, np.inf))
        return out.merge(pd.DataFrame({"zone_id": last.index, "time_to_critical_s": ttc}), on="zone_id")

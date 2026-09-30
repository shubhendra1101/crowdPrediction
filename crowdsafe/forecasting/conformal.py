"""Split conformal calibration of the upper forecast bound (architecture.md §2.5, T6.3).

On validation forecasts, find per (model, horizon) the offset q such that
``target <= p90 + q`` holds for at least ``1 - alpha`` of the points (the finite-sample
conformal quantile). Apply the same offset to test forecasts as ``p90_cal``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def conformal_offset(residuals: np.ndarray, alpha: float = 0.1) -> float:
    """Finite-sample split-conformal quantile of residuals (target − p90)."""
    r = np.sort(np.asarray(residuals, float))
    n = len(r)
    if n == 0:
        return np.nan
    k = int(np.ceil((n + 1) * (1 - alpha))) - 1
    return float(r[min(k, n - 1)])


def fit(val_points: pd.DataFrame, alpha: float = 0.1) -> pd.DataFrame:
    """Offset q per (model, horizon_s) from validation points (needs columns target, p90)."""
    p = val_points.dropna(subset=["target"])
    rows = [{"model": m, "horizon_s": h, "q": conformal_offset(g["target"] - g["p90"], alpha), "n_val": len(g)}
            for (m, h), g in p.groupby(["model", "horizon_s"])]
    return pd.DataFrame(rows)


def apply(points: pd.DataFrame, offsets: pd.DataFrame) -> pd.DataFrame:
    """Add ``p90_cal = p90 + q`` (clipped at 0) using the fitted offsets."""
    out = points.merge(offsets[["model", "horizon_s", "q"]], on=["model", "horizon_s"], how="left")
    out["p90_cal"] = np.clip(out["p90"] + out["q"], 0, None)
    return out

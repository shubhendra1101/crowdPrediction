"""Counting metrics: MAE / RMSE overall and per band."""
from __future__ import annotations

import numpy as np
import pandas as pd


def mae_rmse(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    """Mean absolute error and root-mean-square error."""
    err = np.asarray(pred, float) - np.asarray(gt, float)
    return float(np.abs(err).mean()), float(np.sqrt((err ** 2).mean()))


def band_labels(values: np.ndarray, edges: list[float]) -> list[str]:
    """Label each value with its band, e.g. edges [50, 200] -> '<50', '50-200', '>=200'."""
    names = [f"<{edges[0]:g}"] + [f"{a:g}-{b:g}" for a, b in zip(edges[:-1], edges[1:])] + [f">={edges[-1]:g}"]
    return [names[i] for i in np.digitize(values, edges)]


def count_metrics(df: pd.DataFrame, pred_cols: list[str], gt_col: str, band_col: str | None = None) -> pd.DataFrame:
    """One row per (model, band) with n, MAE, RMSE and mean signed error (bias). Band 'all' is overall."""
    rows = []
    groups = [("all", df)] + (list(df.groupby(band_col, sort=False)) if band_col else [])
    for col in pred_cols:
        for band, g in groups:
            g = g[g[col].notna()]
            if g.empty:
                continue
            mae, rmse = mae_rmse(g[col], g[gt_col])
            rows.append({"model": col, "band": band, "n": len(g), "mae": round(mae, 2), "rmse": round(rmse, 2),
                         "bias": round(float((g[col] - g[gt_col]).mean()), 2)})
    return pd.DataFrame(rows)

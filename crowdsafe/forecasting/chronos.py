"""Chronos-2 zero-shot forecaster (T6.2) — Forecaster interface over ``Chronos2Pipeline.predict_df``.

Modes:
- ``per_zone``: one series per zone; target = density; past covariates = pressure, inflow,
  outflow, mean_speed (motion precursors). Zones of a run are sent in the same call.
- ``joint``: all zones of a run as one multivariate series (target = list of zone columns),
  so the model sees them together (spill-over between neighbouring zones).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

QUANTILES = [0.1, 0.5, 0.9]
EPOCH = pd.Timestamp("2000-01-01")


def _qcol(df: pd.DataFrame, q: float) -> str:
    """Find the output column holding quantile q (named '0.1' or 0.1 depending on version)."""
    for c in df.columns:
        try:
            if abs(float(c) - q) < 1e-9:
                return c
        except (TypeError, ValueError):
            continue
    raise KeyError(f"Quantile {q} not in Chronos output columns {list(df.columns)}")


class ChronosForecaster:
    """Zero-shot Chronos-2; returns zone_id, horizon_s, p10, p50, p90 like the baselines."""

    def __init__(self, mode: str = "per_zone", covariates: list[str] | None = None,
                 model_id: str = "amazon/chronos-2", device: str = "cuda", pipeline=None) -> None:
        self.mode = mode
        self.name = f"chronos2_{mode}"
        self.covariates = covariates if covariates is not None else ["pressure", "inflow", "outflow", "mean_speed"]
        if pipeline is None:
            from chronos import Chronos2Pipeline

            pipeline = Chronos2Pipeline.from_pretrained(model_id, device_map=device)
        self.pipeline = pipeline

    def _context(self, history: pd.DataFrame) -> tuple[pd.DataFrame, list[str] | str]:
        h = history.sort_values("timestamp")
        ts = EPOCH + pd.to_timedelta(h["timestamp"], unit="s")
        if self.mode == "per_zone":
            ctx = pd.DataFrame({"id": h["zone_id"].to_numpy(), "timestamp": ts.to_numpy(),
                                "target": h["density"].to_numpy()})
            for c in self.covariates:
                if c in h:   # motion covariates are absent for count-only sources
                    ctx[c] = h[c].fillna(0).to_numpy()
            return ctx, "target"
        wide = h.pivot_table(index="timestamp", columns="zone_id", values="density").sort_index()
        wide.columns = [f"z__{c}" for c in wide.columns]
        ctx = wide.reset_index()
        ctx["timestamp"] = EPOCH + pd.to_timedelta(ctx["timestamp"], unit="s")
        ctx.insert(0, "id", "run")
        return ctx, list(wide.columns)

    def predict(self, history: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
        """Quantile forecasts of density per zone at the given horizons (seconds = steps at 1 Hz)."""
        ctx, target = self._context(history)
        pred = self.pipeline.predict_df(ctx, prediction_length=int(max(horizons)), quantile_levels=QUANTILES,
                                        id_column="id", timestamp_column="timestamp", target=target)
        return self._parse(pred, history["timestamp"].max(), horizons)

    def _parse(self, pred: pd.DataFrame, origin: float, horizons: list[int]) -> pd.DataFrame:
        pred = pred.copy()
        step = ((pd.to_datetime(pred["timestamp"]) - EPOCH).dt.total_seconds() - origin).round().astype(int)
        pred["horizon_s"] = step
        if self.mode == "per_zone":
            pred["zone_id"] = pred["id"]
        else:   # multivariate output: one row per target column
            tcol = next((c for c in pred.columns if pred[c].astype(str).str.startswith("z__").all()), None)
            if tcol is None:
                raise KeyError(f"Cannot find the target-name column in Chronos output {list(pred.columns)}")
            pred["zone_id"] = pred[tcol].astype(str).str[3:]
        out = pred[pred["horizon_s"].isin(horizons)]
        res = pd.DataFrame({"zone_id": out["zone_id"].to_numpy(), "horizon_s": out["horizon_s"].to_numpy(),
                            "p10": out[_qcol(pred, 0.1)].to_numpy(float), "p50": out[_qcol(pred, 0.5)].to_numpy(float),
                            "p90": out[_qcol(pred, 0.9)].to_numpy(float)})
        res[["p10", "p50", "p90"]] = np.clip(np.sort(res[["p10", "p50", "p90"]].to_numpy(), axis=1), 0, None)
        return res

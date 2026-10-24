"""Risk levels per zone (T7.1, T7.2; architecture.md §2.6).

Base level = density band of the *calibrated upper bound* of the forecast (the worst case over the
forecast horizons), or of the current density if that is higher. A level is raised by one step when a
crowd-motion precursor fires:

- pressure above threshold (Helbing: density × velocity variance),
- crowd slowing: mean speed below threshold at the current density (jam forming),
- flow collapse: over the last `collapse_window_s` density rises while mean speed falls,
- direction entropy or counter-flow above threshold (disordered / opposing movement).

A precursor without a threshold (None in the config) is switched off; the tuning keeps only those that
beat chance clearly on validation data.

Thresholds come from configs/risk.yaml (tuned on Jülich validation runs by scripts/eval_alerts.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

LEVELS = ["Low", "Medium", "High", "Critical"]


@dataclass
class ZoneRisk:
    """Risk of one zone at one moment."""

    zone_id: str
    level: str
    base_level: str
    density: float                       # current, persons/m²
    upper_density: float                 # worst calibrated upper bound over the horizons, persons/m²
    time_to_critical_s: float | None     # first horizon whose upper bound reaches critical density
    reasons: list[str] = field(default_factory=list)

    @property
    def level_idx(self) -> int:
        return LEVELS.index(self.level)


def band(density: float, bands: tuple[float, float, float] = (2.0, 4.0, 5.0)) -> int:
    """0 Low (< 2), 1 Medium (2–4), 2 High (4–5), 3 Critical (≥ 5) for the default bands."""
    if density is None or not np.isfinite(density):
        return 0
    return int(np.searchsorted(np.asarray(bands), density, side="right"))


def _slope(t: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(t) & np.isfinite(y)
    if ok.sum() < 5 or np.ptp(t[ok]) == 0:
        return float("nan")
    return float(np.polyfit(t[ok], y[ok], 1)[0])


class RiskEngine:
    """features + forecast -> list[ZoneRisk] (interface of architecture.md §4)."""

    def __init__(self, bands=(2.0, 4.0, 5.0), thresholds: dict | None = None, collapse_window_s: float = 30.0,
                 collapse_min_density: float = 1.0, critical_density: float = 5.0, use_precursors: bool = True):
        self.bands = tuple(bands)
        self.th = {k: v for k, v in (thresholds or {}).items() if v is not None}
        self.win = collapse_window_s
        self.collapse_min = collapse_min_density
        self.critical = critical_density
        self.use_precursors = use_precursors

    def precursors(self, hist: pd.DataFrame) -> list[str]:
        """Reasons that fired for one zone; `hist` = its feature rows up to now (sorted by time)."""
        now = hist.iloc[-1]
        out = []
        for key, label in [("pressure", "crowd pressure"), ("dir_entropy", "disordered movement"),
                           ("counterflow", "counter-flow")]:
            v = now.get(key, np.nan)
            if key in self.th and np.isfinite(v) and v >= self.th[key]:
                out.append(f"{label} {v:.2f} ≥ {self.th[key]:.2f}")
        v = now.get("mean_speed", np.nan)
        if "slow_speed" in self.th and np.isfinite(v) and v <= self.th["slow_speed"] and now["density"] >= self.collapse_min:
            out.append(f"crowd slowing: {v:.2f} m/s ≤ {self.th['slow_speed']:.2f}")
        if not self.th.get("collapse_enabled", True):
            return out
        w = hist[hist["timestamp"] >= now["timestamp"] - self.win]
        if len(w) >= 5 and np.nanmean(w["density"]) >= self.collapse_min:
            sd = _slope(w["timestamp"].to_numpy(float), w["density"].to_numpy(float))
            sv = _slope(w["timestamp"].to_numpy(float), w["mean_speed"].to_numpy(float))
            if np.isfinite(sd) and np.isfinite(sv) and sd > self.th.get("collapse_density_slope", 0.0) \
                    and sv < -self.th.get("collapse_speed_slope", 0.0):
                out.append(f"flow collapse: density +{sd * 60:.2f}/min, speed {sv * 60:.2f} m/s per min")
        return out

    def __call__(self, features_now: pd.DataFrame, forecast: pd.DataFrame | None) -> list[ZoneRisk]:
        """features_now: recent feature rows (≥ collapse window) of one camera; forecast: zone_id, horizon_s,
        p90 (conformal-calibrated upper bound). Returns one ZoneRisk per zone present in features_now."""
        out = []
        fc = forecast.groupby("zone_id") if forecast is not None and len(forecast) else None
        for zone, hist in features_now.sort_values("timestamp").groupby("zone_id", sort=False):
            d_now = float(hist["density"].iloc[-1])
            upper, ttc = d_now, None
            if fc is not None and zone in fc.groups:
                f = fc.get_group(zone).sort_values("horizon_s")
                upper = max(d_now, float(np.nanmax(f["p90"]))) if f["p90"].notna().any() else d_now
                hit = f[f["p90"] >= self.critical]
                ttc = float(hit["horizon_s"].iloc[0]) if len(hit) else None
            if d_now >= self.critical:
                ttc = 0.0
            base = band(upper, self.bands)
            reasons = []
            if fc is not None and base > band(d_now, self.bands):
                reasons.append(f"forecast upper bound {upper:.1f} persons/m² (now {d_now:.1f})")
            elif base > 0:
                reasons.append(f"density {d_now:.1f} persons/m²")
            level = base
            if self.use_precursors:
                pre = self.precursors(hist)
                if pre:
                    level = min(base + 1, len(LEVELS) - 1)
                    reasons += pre
            out.append(ZoneRisk(str(zone), LEVELS[level], LEVELS[base], d_now, upper, ttc, reasons))
        return out

"""Zone feature table schema (architecture.md §2.4): one row per zone per second.

The same columns are produced from real video, Jülich trajectories and JuPedSim runs, so the
prediction layer never needs to know where a row came from (except via ``source``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# column -> (dtype, unit / meaning)
COLUMNS: dict[str, tuple[str, str]] = {
    "timestamp": ("float64", "seconds since the start of the clip / run"),
    "camera_id": ("string", "camera, experiment run or simulation scenario id"),
    "zone_id": ("string", "zone id within the camera / run"),
    "count": ("float64", "persons in the zone (mean over the second)"),
    "density": ("float64", "persons/m²"),
    "fusion_weight": ("float64", "YOLO weight w_z in the fused count; NaN for trajectory data"),
    "mean_speed": ("float64", "m/s"),
    "speed_var": ("float64", "variance of the velocity vector, m²/s² (var vx + var vy)"),
    "pressure": ("float64", "density × speed_var, 1/s² (Helbing crowd pressure)"),
    "dir_entropy": ("float64", "normalised entropy of 8-bin direction histogram, 0–1"),
    "counterflow": ("float64", "share of movement opposite the zone's mean direction, 0–1"),
    "inflow": ("float64", "persons/s entering the zone across its boundaries"),
    "outflow": ("float64", "persons/s leaving the zone across its boundaries"),
    "source": ("string", "real | julich | sim"),
}
SOURCES = {"real", "julich", "sim"}


def empty() -> pd.DataFrame:
    """An empty feature table with the right columns and dtypes."""
    return pd.DataFrame({c: pd.Series(dtype=t) for c, (t, _) in COLUMNS.items()})


def conform(df: pd.DataFrame) -> pd.DataFrame:
    """Return df with exactly the schema columns, in order, with schema dtypes (missing numeric -> NaN)."""
    out = df.copy()
    for col, (dtype, _) in COLUMNS.items():
        if col not in out:
            out[col] = np.nan if dtype == "float64" else pd.NA
    return out[list(COLUMNS)].astype({c: t for c, (t, _) in COLUMNS.items()})


def validate(df: pd.DataFrame) -> list[str]:
    """List of schema problems (empty list = valid)."""
    problems = [f"missing column {c}" for c in COLUMNS if c not in df]
    if problems:
        return problems
    if not set(df["source"].dropna().unique()) <= SOURCES:
        problems.append(f"unknown source values {set(df['source'].unique()) - SOURCES}")
    for col in ("count", "density", "inflow", "outflow"):
        if (df[col].dropna() < 0).any():
            problems.append(f"negative values in {col}")
    for col in ("dir_entropy", "counterflow"):
        v = df[col].dropna()
        if ((v < 0) | (v > 1)).any():
            problems.append(f"{col} outside [0, 1]")
    if df.duplicated(["camera_id", "zone_id", "timestamp"]).any():
        problems.append("duplicate (camera_id, zone_id, timestamp) rows")
    return problems

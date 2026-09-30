"""Jülich trajectories -> zone feature table (T5.2).

Pipeline for one run (one trajectory file):
1. :func:`read_trajectories` — parse the archive's text format; convert to metres and seconds.
2. :func:`resample` — put every person on a common time grid (``sample_hz``) by linear
   interpolation; velocity by central differences on that grid.
3. :func:`make_zones` — square cells of ``cell_m``; a cell becomes a zone only if people walked
   over at least ``min_walkable_share`` of it (walkable area inferred from the trajectories).
4. :func:`zone_features` — one row per zone per second in the shared schema
   (``crowdsafe.features.schema``): density, speed, speed variance, pressure, direction entropy,
   counter-flow, inflow and outflow. Directional features use only the current second (no look-ahead).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

UNIT_TO_M = {"m": 1.0, "cm": 0.01}


@dataclass
class Zones:
    """Square-cell zones of one run."""

    cell_m: float
    x0: float
    y0: float
    cells: dict[str, tuple[int, int]]           # zone_id -> (ix, iy)

    @property
    def area_m2(self) -> float:
        return self.cell_m ** 2

    def cell_index(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Grid cell (ix, iy) of each position."""
        return (np.floor((x - self.x0) / self.cell_m).astype(int), np.floor((y - self.y0) / self.cell_m).astype(int))

    def zone_of(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Zone id per position ('' when the position is not inside a kept zone)."""
        ix, iy = self.cell_index(x, y)
        lookup = {v: k for k, v in self.cells.items()}
        return np.array([lookup.get((a, b), "") for a, b in zip(ix, iy)], dtype=object)

    def polygon(self, zone_id: str) -> list[tuple[float, float]]:
        """Corner coordinates of a zone in metres."""
        ix, iy = self.cells[zone_id]
        x, y, c = self.x0 + ix * self.cell_m, self.y0 + iy * self.cell_m, self.cell_m
        return [(x, y), (x + c, y), (x + c, y + c), (x, y + c)]


def parse_header(path: Path) -> dict:
    """Frame rate and x unit declared in the file's '#' header, if any."""
    info = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.startswith("#"):
                break
            if m := re.search(r"framerate:\s*([\d.]+)\s*fps", line):
                info["fps"] = float(m[1])
            if m := re.search(r"\bx/(cm|m)\b", line):
                info["unit"] = m[1]
    return info


def read_trajectories(path: Path, unit: str, fps: float, time: str = "frame",
                      return_used: bool = False) -> pd.DataFrame | tuple[pd.DataFrame, dict]:
    """Read one Jülich trajectory file as columns id, t [s], x [m], y [m].

    ``unit``/``fps`` come from the config (archive page). If the file's own header declares a
    different value, the header wins and the difference is reported in ``used["note"]``.
    ``time`` is ``"frame"`` (frame number) or ``"sec:frame"`` (Jülich 2005 format).
    """
    hdr = parse_header(path)
    used = {"unit": hdr.get("unit", unit), "fps": hdr.get("fps", fps)}
    notes = [f"header {k}={hdr[k]} overrides config {k}={v}" for k, v in (("unit", unit), ("fps", fps))
             if k in hdr and hdr[k] != v]
    if notes:
        used["note"] = "; ".join(notes)
    raw = pd.read_csv(path, sep=r"\s+", comment="#", header=None, usecols=[0, 1, 2, 3],
                      names=["id", "time", "x", "y"], dtype={"id": "int64", "time": str, "x": "float64", "y": "float64"})
    if time == "sec:frame":
        parts = raw["time"].str.split(":", expand=True).astype(int)
        t = parts[0] + parts[1] / used["fps"]
    else:
        t = raw["time"].astype(float) / used["fps"]
    s = UNIT_TO_M[used["unit"]]
    df = pd.DataFrame({"id": raw["id"], "t": t.astype(float), "x": raw["x"] * s, "y": raw["y"] * s})
    return (df, used) if return_used else df


def resample(traj: pd.DataFrame, hz: float) -> pd.DataFrame:
    """Interpolate each person onto the grid t = k / hz; add velocity (m/s) by central differences."""
    out = []
    for pid, g in traj.sort_values(["id", "t"]).groupby("id", sort=False):
        t = g["t"].to_numpy()
        k = np.arange(np.ceil(t[0] * hz), np.floor(t[-1] * hz) + 1).astype(int)
        if len(k) == 0:
            continue
        tk = k / hz
        x, y = np.interp(tk, t, g["x"].to_numpy()), np.interp(tk, t, g["y"].to_numpy())
        if len(k) > 1:
            vx, vy = np.gradient(x, 1 / hz), np.gradient(y, 1 / hz)
        else:
            vx = vy = np.full(1, np.nan)
        out.append(pd.DataFrame({"id": pid, "k": k, "t": tk, "x": x, "y": y, "vx": vx, "vy": vy}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["id", "k", "t", "x", "y", "vx", "vy"])


def drop_implausible_velocity(samples: pd.DataFrame, max_speed: float) -> tuple[pd.DataFrame, int]:
    """Set velocity to NaN where speed > max_speed (tracking glitches). Positions are kept for density.
    Returns the cleaned samples and how many samples were affected."""
    bad = np.hypot(samples["vx"], samples["vy"]) > max_speed
    out = samples.copy()
    out.loc[bad, ["vx", "vy"]] = np.nan
    return out, int(bad.sum())


def make_zones(samples: pd.DataFrame, cell_m: float, res_m: float, min_share: float,
               body_radius_m: float = 0.25) -> Zones:
    """Square cells covering the run; keep cells whose floor was walked on for >= min_share of their area.

    Walked-on floor = fine grid cells (``res_m``) containing any position, grown by ``body_radius_m``
    because a person occupies roughly a 0.5 m wide disc, not a point.
    """
    from scipy.ndimage import binary_dilation

    x0 = np.floor(samples["x"].min() / cell_m) * cell_m
    y0 = np.floor(samples["y"].min() / cell_m) * cell_m
    per_cell = int(round(cell_m / res_m))
    fx = np.floor((samples["x"].to_numpy() - x0) / res_m).astype(int)
    fy = np.floor((samples["y"].to_numpy() - y0) / res_m).astype(int)
    nx, ny = (-(-(fx.max() + 1) // per_cell)) * per_cell, (-(-(fy.max() + 1) // per_cell)) * per_cell
    mask = np.zeros((nx, ny), bool)
    mask[fx, fy] = True
    r = int(round(body_radius_m / res_m))
    if r > 0:
        yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
        mask = binary_dilation(mask, structure=(xx ** 2 + yy ** 2) <= r * r)
    share = mask.reshape(nx // per_cell, per_cell, ny // per_cell, per_cell).mean(axis=(1, 3))
    kept = sorted(zip(*np.nonzero(share >= min_share)))
    return Zones(cell_m, float(x0), float(y0), {f"c{ix}_{iy}": (int(ix), int(iy)) for ix, iy in kept})


def _direction_stats(vx: np.ndarray, vy: np.ndarray, min_speed: float, bins: int) -> tuple[float, float]:
    """(normalised direction entropy, counter-flow share) of the moving samples; NaN if nobody moves."""
    moving = np.hypot(vx, vy) >= min_speed
    if moving.sum() == 0:
        return np.nan, np.nan
    vx, vy = vx[moving], vy[moving]
    hist = np.histogram(np.arctan2(vy, vx), bins=bins, range=(-np.pi, np.pi))[0]
    p = hist[hist > 0] / hist.sum()
    entropy = float(-(p * np.log(p)).sum() / np.log(bins))
    mx, my = vx.mean(), vy.mean()
    counter = float(((vx * mx + vy * my) < 0).mean()) if np.hypot(mx, my) > 1e-9 else np.nan
    return entropy, counter


def _crossings(samples: pd.DataFrame, zones: Zones, t_start: float) -> pd.DataFrame:
    """Per (zone, second): persons crossing into (inflow) and out of (outflow) the zone."""
    s = samples.sort_values(["id", "k"])
    ix, iy = zones.cell_index(s["x"].to_numpy(), s["y"].to_numpy())
    cell = pd.Series([f"c{a}_{b}" for a, b in zip(ix, iy)], index=s.index)
    same_id = s["id"].eq(s["id"].shift()) & s["k"].eq(s["k"].shift() + 1)
    prev = cell.shift()
    moved = same_id & cell.ne(prev)
    sec = np.floor(s["t"] - t_start).astype(int)
    rows = pd.concat([
        pd.DataFrame({"zone_id": cell[moved], "second": sec[moved], "inflow": 1, "outflow": 0}),
        pd.DataFrame({"zone_id": prev[moved], "second": sec[moved], "inflow": 0, "outflow": 1}),
    ])
    rows = rows[rows["zone_id"].isin(zones.cells.keys())]
    return rows.groupby(["zone_id", "second"])[["inflow", "outflow"]].sum()


def zone_features(samples: pd.DataFrame, zones: Zones, run_id: str, hz: float,
                  min_speed: float = 0.1, bins: int = 8) -> pd.DataFrame:
    """One row per zone per second (schema columns) for one resampled run."""
    t_start = samples["t"].min()
    s = samples.assign(zone_id=zones.zone_of(samples["x"].to_numpy(), samples["y"].to_numpy()),
                       second=np.floor(samples["t"] - t_start).astype(int))
    all_k = np.arange(samples["k"].min(), samples["k"].max() + 1)
    zone_ids = list(zones.cells)
    # counts per grid step (zeros included), then mean per second
    per_k = s[s["zone_id"] != ""].groupby(["k", "zone_id"]).size().unstack(fill_value=0)
    per_k = per_k.reindex(index=all_k, columns=zone_ids, fill_value=0)
    per_k["second"] = np.floor(per_k.index / hz - t_start).astype(int)
    count = per_k.groupby("second").mean().stack().rename("count")
    count.index.names = ["second", "zone_id"]

    moving = s[(s["zone_id"] != "") & s["vx"].notna()]
    motion = []
    for (zid, sec), g in moving.groupby(["zone_id", "second"]):
        vx, vy = g["vx"].to_numpy(), g["vy"].to_numpy()
        ent, cf = _direction_stats(vx, vy, min_speed, bins)
        motion.append({"zone_id": zid, "second": sec, "mean_speed": float(np.hypot(vx, vy).mean()),
                       "speed_var": float(vx.var() + vy.var()), "dir_entropy": ent, "counterflow": cf})
    motion_df = pd.DataFrame(motion, columns=["zone_id", "second", "mean_speed", "speed_var", "dir_entropy", "counterflow"])

    df = count.reset_index()
    df = df.merge(motion_df, on=["zone_id", "second"], how="left")
    df = df.merge(_crossings(samples, zones, t_start).reset_index(), on=["zone_id", "second"], how="left")
    df[["inflow", "outflow"]] = df[["inflow", "outflow"]].fillna(0).astype(float)
    df["density"] = df["count"] / zones.area_m2
    df["pressure"] = df["density"] * df["speed_var"]
    df["timestamp"] = df["second"].astype(float)
    df["camera_id"] = run_id
    df["fusion_weight"] = np.nan
    df["source"] = "julich"
    return df.drop(columns="second").sort_values(["timestamp", "zone_id"]).reset_index(drop=True)

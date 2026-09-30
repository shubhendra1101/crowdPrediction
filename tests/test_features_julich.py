"""Smoke tests for the feature schema/logger and the Jülich trajectory -> feature pipeline."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from crowdsafe.datasets.julich import make_zones, read_trajectories, resample, zone_features
from crowdsafe.features.logger import SqliteLogger, read_features, write_parquet
from crowdsafe.features.schema import COLUMNS, conform, validate


def _walkers(n: int = 20, speed: float = 1.0, hz_raw: float = 25, secs: float = 8, reverse_id: int | None = None):
    """n people spaced across a 4 m wide, 8 m long area, walking +x at `speed` (one optionally -x).

    Positions are offset by 0.013 m so nobody sits exactly on a zone edge (edge points are counted
    by our floor() binning but not by PedPy's strict `within`; real measured data never hits edges exactly).
    """
    rows = []
    t = np.arange(0, secs, 1 / hz_raw)
    for i in range(n):
        y = 0.113 + (i % 10) * 0.38
        x0 = 0.013 - (i // 10) * 0.5
        v = -speed if i == reverse_id else speed
        start = 8.0 if i == reverse_id else x0
        for tt in t:
            rows.append((i, tt, start + v * tt, y))
    return pd.DataFrame(rows, columns=["id", "t", "x", "y"])


def test_schema_conform_and_validate() -> None:
    df = conform(pd.DataFrame({"timestamp": [0.0], "camera_id": ["c"], "zone_id": ["z"], "count": [2.0],
                               "density": [0.5], "source": ["sim"]}))
    assert list(df.columns) == list(COLUMNS) and validate(df) == []
    bad = df.assign(source="video", counterflow=1.5)
    assert len(validate(bad)) == 2


def test_parquet_and_sqlite_roundtrip(tmp_path: Path) -> None:
    df = conform(pd.DataFrame({"timestamp": [0.0, 1.0], "camera_id": ["c", "c"], "zone_id": ["z", "z"],
                               "count": [1.0, 2.0], "density": [0.25, 0.5], "source": ["julich"] * 2}))
    p = write_parquet(df, tmp_path / "a" / "run.parquet")
    assert read_features(tmp_path).equals(read_features(p))
    log = SqliteLogger(tmp_path / "live.sqlite")
    log.append(df)
    assert len(log.latest("c", 0.5)) == 1
    log.close()
    with pytest.raises(ValueError):
        write_parquet(df.assign(density=-1.0), tmp_path / "bad.parquet")


def test_read_trajectories_formats(tmp_path: Path) -> None:
    f1 = tmp_path / "b1.txt"
    f1.write_text("1  0:24  0.45  1.25\n1  1:01  0.44  1.17\n")
    t = read_trajectories(f1, "m", 25, "sec:frame")
    np.testing.assert_allclose(t["t"], [0.96, 1.04])
    f2 = tmp_path / "e.txt"
    f2.write_text("# framerate: 25 fps\n# id frame x/cm y/cm z/cm\n1 50 100 200 176\n")
    t2 = read_trajectories(f2, "cm", 25)
    assert t2.loc[0, "t"] == 2.0 and t2.loc[0, "x"] == 1.0 and t2.loc[0, "y"] == 2.0
    t3, used = read_trajectories(f2, "m", 16, return_used=True)   # header wins over config
    assert used["unit"] == "cm" and used["fps"] == 25 and "overrides" in used["note"] and t3.loc[0, "x"] == 1.0


def test_zone_features_known_motion() -> None:
    hz = 5
    s = resample(_walkers(), hz)
    zones = make_zones(s, cell_m=2.0, res_m=0.25, min_share=0.9)
    assert zones.cells, "expected fully walked cells"
    f = zone_features(s, zones, "demo/run", hz)
    assert validate(conform(f)) == []
    busy = f[f["count"] > 0]
    assert busy["mean_speed"].between(0.99, 1.01).all()           # everyone walks exactly 1 m/s
    assert busy["counterflow"].fillna(0).eq(0).all()                # all in the same direction
    assert busy["speed_var"].max() < 1e-6
    assert (f["density"] == f["count"] / 4.0).all()
    assert f["inflow"].sum() > 0 and f["outflow"].sum() > 0


def test_counterflow_detected() -> None:
    hz = 5
    s = resample(_walkers(reverse_id=3), hz)
    zones = make_zones(s, 2.0, 0.25, 0.9)
    f = zone_features(s, zones, "demo/run", hz)
    assert f["counterflow"].max() > 0


def test_pedpy_cross_check_agrees() -> None:
    pytest.importorskip("pedpy")
    import scripts.julich_features as jf

    hz = 5
    s = resample(_walkers(), hz)
    zones = make_zones(s, 2.0, 0.25, 0.9)
    zid = next(iter(zones.cells))
    assert jf.pedpy_check(s, zones, zid, hz) < 1e-9


def test_drop_implausible_velocity() -> None:
    from crowdsafe.datasets.julich import drop_implausible_velocity

    s = pd.DataFrame({"vx": [1.0, 10.0, np.nan], "vy": [0.0, 0.0, 0.0], "x": [0, 1, 2]})
    out, n = drop_implausible_velocity(s, 4.0)
    assert n == 1 and np.isnan(out.loc[1, "vx"]) and out.loc[1, "x"] == 1

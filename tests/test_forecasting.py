"""Smoke tests for baseline forecasters and the rolling-origin evaluation."""
import numpy as np
import pandas as pd
import pytest

from crowdsafe.forecasting.baselines import Persistence, PhysicsFillRate, zone_area
from crowdsafe.forecasting.evaluate import forecast_points, score


def _filling_zone(seconds: int = 120, area: float = 4.0, net_in: float = 0.2) -> pd.DataFrame:
    """One zone gaining `net_in` persons/s: density rises by net_in/area per second."""
    t = np.arange(seconds, dtype=float)
    count = 2 + net_in * t
    return pd.DataFrame({"timestamp": t, "camera_id": "run1", "zone_id": "Z1", "count": count,
                         "density": count / area, "inflow": net_in + 0.1, "outflow": 0.1})


def test_persistence_and_physics_on_filling_zone() -> None:
    h = _filling_zone()
    hist = h[h["timestamp"] <= 59]
    assert zone_area(hist)["Z1"] == pytest.approx(4.0)
    p = Persistence().predict(hist, [10, 30])
    assert p["p50"].tolist() == pytest.approx([h.loc[59, "density"]] * 2)
    f = PhysicsFillRate(flux_window_s=10, critical=5.0).predict(hist, [10, 30])
    assert f.set_index("horizon_s").loc[30, "p50"] == pytest.approx(h.loc[89, "density"])
    rho, rate = h.loc[59, "density"], 0.2 / 4.0
    assert f["time_to_critical_s"].iloc[0] == pytest.approx((5.0 - rho) / rate)


def test_time_to_critical_not_filling_is_inf() -> None:
    h = _filling_zone(net_in=0.0)
    f = PhysicsFillRate().predict(h, [10])
    assert np.isinf(f["time_to_critical_s"].iloc[0])


def test_rolling_evaluation_scores() -> None:
    h = _filling_zone()
    pts = pd.concat([forecast_points(h, m, 30, [10, 30], 5) for m in (Persistence(), PhysicsFillRate())])
    s = score(pts).set_index(["model", "horizon_s"])
    assert s.loc[("physics", 30), "mae"] == pytest.approx(0.0, abs=1e-9)
    assert s.loc[("persistence", 30), "mae"] == pytest.approx(0.2 / 4.0 * 30)
    # no forecast may use data after its origin
    assert (pts["origin"] + pts["horizon_s"] <= h["timestamp"].max()).all()


def test_conformal_offset_reaches_coverage() -> None:
    from crowdsafe.forecasting import conformal

    rng = np.random.default_rng(42)
    val = pd.DataFrame({"model": "m", "horizon_s": 10, "p90": 0.0, "target": rng.normal(size=2000)})
    test = pd.DataFrame({"model": "m", "horizon_s": 10, "p90": 0.0, "target": rng.normal(size=2000)})
    off = conformal.fit(val, alpha=0.1)
    cal = conformal.apply(test, off)
    assert 0.87 <= (cal["target"] <= cal["p90"] + cal["q"]).mean() <= 0.93


def test_chronos_wrapper_parses_fake_pipeline() -> None:
    from crowdsafe.forecasting.chronos import EPOCH, ChronosForecaster

    class FakePipeline:
        """Mimics predict_df output: id, timestamp, predictions, '0.1', '0.5', '0.9'."""
        def predict_df(self, ctx, prediction_length, quantile_levels, id_column, timestamp_column, target):
            last = ctx.groupby("id")["timestamp"].max()
            rows = []
            for zid, t in last.items():
                for k in range(1, prediction_length + 1):
                    rows.append({"id": zid, "timestamp": t + pd.Timedelta(seconds=k), "predictions": 1.0,
                                 "0.1": 0.9, "0.5": 1.0, "0.9": float(k)})
            return pd.DataFrame(rows)

    h = _filling_zone(40)
    fc = ChronosForecaster(mode="per_zone", pipeline=FakePipeline()).predict(h, [10, 30])
    assert fc["horizon_s"].tolist() == [10, 30] and fc["p90"].tolist() == [10.0, 30.0]
    assert (fc["p10"] <= fc["p50"]).all()

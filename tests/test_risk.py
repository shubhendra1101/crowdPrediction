"""Smoke tests for risk levels, precursors and alert hysteresis."""
import numpy as np
import pandas as pd

from crowdsafe.risk.alerts import AlertManager
from crowdsafe.risk.rules import RiskEngine, band


def _hist(density, speed, zone="Z1", t0=0.0):
    n = len(density)
    return pd.DataFrame({"timestamp": t0 + np.arange(n, dtype=float), "zone_id": zone, "density": density,
                         "mean_speed": speed, "pressure": 0.1, "dir_entropy": 0.5, "counterflow": 0.0})


def test_bands() -> None:
    assert [band(x) for x in (0.5, 2.0, 3.9, 4.0, 5.0, 7.0)] == [0, 1, 1, 2, 3, 3]


def test_forecast_upper_bound_sets_level_and_time_to_critical() -> None:
    eng = RiskEngine(use_precursors=False)
    fc = pd.DataFrame({"zone_id": "Z1", "horizon_s": [10, 30, 60], "p90": [3.0, 4.5, 5.2]})
    r = eng(_hist([3.0] * 30, [0.8] * 30), fc)[0]
    assert r.level == "Critical" and r.base_level == "Critical" and r.time_to_critical_s == 60
    assert eng(_hist([3.0] * 30, [0.8] * 30), None)[0].level == "Medium"


def test_slow_crowd_precursor_escalates_one_level() -> None:
    eng = RiskEngine(thresholds={"slow_speed": 0.2, "collapse_enabled": False})
    r = eng(_hist([3.0] * 30, [0.1] * 30), None)[0]
    assert r.base_level == "Medium" and r.level == "High" and any("slowing" in s for s in r.reasons)


def test_hysteresis_raise_and_clear() -> None:
    eng, mgr = RiskEngine(use_precursors=False), AlertManager(raise_after_s=10, clear_after_s=30)
    dens = [4.5] * 5 + [1.0] * 2 + [4.5] * 20 + [1.0] * 40      # short spike, then a real one
    for t, d in enumerate(dens):
        mgr.update(float(t), eng(_hist([d], [0.5], t0=t), None))
    assert len(mgr.history) == 1                                # the 5 s spike did not alert
    a = mgr.history[0]
    assert a.since == 17.0 and a.level == "High"                # raised 10 s after the level was reached at t=7
    assert a.cleared == 27.0 + 30.0                             # cleared after 30 s below

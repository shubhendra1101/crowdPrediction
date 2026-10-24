"""Smoke tests for the JuPedSim scenarios (tiny runs, seconds)."""
import pytest

pytest.importorskip("jupedsim")

from crowdsafe.simulation.scenarios import classroom, features, run, zones_for  # noqa: E402


def test_classroom_zones_use_walkable_area() -> None:
    zones, walk = zones_for(classroom())
    assert zones.cells and all(0 < a <= 4.0 for a in walk.values())
    assert min(walk.values()) < 4.0                     # desks reduce some zones' walkable area


def test_small_corridor_run_gives_schema_features() -> None:
    scn = {"layout": "corridor_bottleneck", "arrivals": {"entry": [[0, 10, 2.0, "exit"]]}, "max_time_s": 40}
    tr = run(scn)
    assert tr["id"].nunique() >= 15 and tr["t"].max() <= 40
    f = features(tr, scn, "sim/test")
    assert {"density", "mean_speed", "pressure", "inflow", "outflow"} <= set(f.columns)
    assert (f["source"] == "sim").all() and f["density"].max() > 0

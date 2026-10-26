"""Smoke test for the replay API (skips when the run data is not on this machine)."""
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from crowdsafe.server.app import app  # noqa: E402
from crowdsafe.server import replay  # noqa: E402


def test_runs_layout_summary_and_websocket() -> None:
    runs = replay.list_runs()
    if not runs:
        pytest.skip("no replay data on this machine")
    rid = min(runs, key=lambda r: r["kind"] != "grid")["id"]      # a grid run builds fastest
    c = TestClient(app)
    assert c.get("/api/runs").status_code == 200
    lay = c.get(f"/api/runs/{rid}/layout").json()
    assert lay["zones"]
    s = c.get(f"/api/runs/{rid}/summary").json()
    assert s["n"] > 0 and len(s["peaks"]) == s["n"]
    with c.websocket_connect(f"/ws/replay/{rid}?speed=1000") as ws:
        m = ws.receive_json()
        assert {"timestamp", "camera_id", "zones", "alerts"} <= set(m)
        assert {"zone_id", "density", "forecast", "level"} <= set(m["zones"][0])
    assert c.get("/api/runs/nope/layout").status_code == 404

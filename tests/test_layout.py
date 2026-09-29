"""Smoke test: repo layout imports and the camera config template parses."""
import importlib
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = [
    "calibration", "counting", "motion", "features", "datasets",
    "simulation", "forecasting", "risk", "server", "eval",
]


@pytest.mark.parametrize("name", PACKAGES)
def test_package_imports(name: str) -> None:
    mod = importlib.import_module(name)
    assert Path(mod.__file__).parent == ROOT / name


def test_camera_template_has_schema_keys() -> None:
    cfg = yaml.safe_load((ROOT / "configs" / "cam_template.yaml").read_text())
    expected = {"camera_id", "video", "fps", "homography", "calibration_note",
                "zones", "adjacency", "boundaries", "thresholds", "alerts"}
    assert expected <= cfg.keys()
    assert cfg["alerts"] == {"raise_after_s": 10, "clear_after_s": 30}

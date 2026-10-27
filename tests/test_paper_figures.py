"""Smoke test for scripts/paper_figures.py (no data needed for these parts)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import paper_figures as pf  # noqa: E402


def test_band_rgb():
    assert pf.band_rgb(1.0) == pf.BANDS[0][1]
    assert pf.band_rgb(4.5) == pf.BANDS[2][1]
    assert pf.band_rgb(9.0) == pf.BANDS[3][1]


def test_pipeline_diagram(tmp_path):
    out = tmp_path / "p.png"
    pf.pipeline_diagram(out)
    assert out.stat().st_size > 10_000

"""Smoke test for scripts/coverage_by_density.py."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import coverage_by_density as cbd  # noqa: E402


def test_coverage_table():
    pts = pd.DataFrame({"split": ["test"] * 4 + ["val"], "model": ["m"] * 5, "horizon_s": [10] * 5,
                        "target": [1.0, 3.0, 4.5, 6.0, 9.0], "p90_cal": [2.0, 2.0, 5.0, 5.0, 0.0]})
    t = cbd.coverage_table(pts, [4.0])
    r = t.iloc[0]
    assert r["n_all"] == 4 and r["cover_all"] == 0.5      # val row ignored; 1.0 and 4.5 covered
    assert r["n_ge4"] == 2 and r["cover_ge4"] == 0.5      # 4.5 covered, 6.0 not

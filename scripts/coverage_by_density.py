"""Coverage of the calibrated 90 % bound split by the true future density (T9.1 / paper Table).

Reads stored test forecast points (no model is re-run) and reports, per model and horizon, the share of test points
with target <= p90_cal: over all points, and only where the true future density is >= each threshold.

    python scripts/coverage_by_density.py --points weights/crowdsafe_nb04_report/forecast_julich_points.parquet \
        --name julich --out results/coverage_by_density_julich.md
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def coverage_table(points: pd.DataFrame, thresholds: list[float]) -> pd.DataFrame:
    """One row per (model, horizon): n and coverage overall and at target >= each threshold."""
    p = points[(points["split"] == "test")].dropna(subset=["target", "p90_cal"])
    rows = []
    for (m, h), g in p.groupby(["model", "horizon_s"]):
        r = {"model": m, "horizon_s": int(h), "n_all": len(g), "cover_all": round(float((g["target"] <= g["p90_cal"]).mean()), 3)}
        for t in thresholds:
            s = g[g["target"] >= t]
            r[f"n_ge{t:g}"] = len(s)
            r[f"cover_ge{t:g}"] = round(float((s["target"] <= s["p90_cal"]).mean()), 3) if len(s) else None
        rows.append(r)
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--points", type=Path, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--thresholds", type=float, nargs="*", default=[2.0, 4.0, 5.0])
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    t = coverage_table(pd.read_parquet(a.points), a.thresholds)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(a.out.with_suffix(".csv"), index=False)
    md = [f"# Coverage of the calibrated 90 % bound by true density: {a.name}", "",
          f"From stored test forecast points `{a.points.as_posix()}` (no model re-run). `cover_all` = share of test points "
          "with target <= p90_cal; `cover_geX` = same, only points whose true future density is >= X persons/m²; "
          "`n_*` = number of points.", "",
          "| " + " | ".join(t.columns) + " |", "| " + " | ".join("---" for _ in t.columns) + " |"]
    md += ["| " + " | ".join("" if pd.isna(v) else str(v) for v in r) + " |" for r in t.itertuples(index=False)]
    a.out.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(t.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

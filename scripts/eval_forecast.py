"""Evaluate forecasters on Jülich feature tables with split-conformal calibration (T6.1–T6.4, Jülich part).

Forecasts are made on the val and test splits; the upper bound is calibrated on val only and
scored on test.

    python scripts/eval_forecast.py                                   # baselines (laptop)
    python scripts/eval_forecast.py --models persistence physics chronos2_per_zone chronos2_joint  # A100
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crowdsafe.features.logger import read_features  # noqa: E402
from crowdsafe.forecasting import conformal  # noqa: E402
from crowdsafe.forecasting.baselines import Persistence, PhysicsFillRate  # noqa: E402
from crowdsafe.forecasting.evaluate import forecast_points, score  # noqa: E402


def load_split(cfg: dict, split: str, root: Path | None = None) -> pd.DataFrame:
    """Feature rows of every run in the datasets assigned to ``split``."""
    root = Path(root or cfg["features_root"])
    return pd.concat([read_features(root / ds) for ds in cfg["split"][split] if (root / ds).exists()], ignore_index=True)


def build_models(cfg: dict, names: list[str], device: str) -> list:
    """Instantiate forecasters by name; Chronos-2 is imported only when requested."""
    out = []
    for n in names:
        c = cfg["models"].get(n, {})
        if n == "persistence":
            out.append(Persistence())
        elif n == "physics":
            out.append(PhysicsFillRate(c.get("flux_window_s", 10), cfg["critical_density"]))
        elif n.startswith("chronos2_"):
            from crowdsafe.forecasting.chronos import ChronosForecaster
            out.append(ChronosForecaster(mode=n.split("_", 1)[1], device=device))
        else:
            raise ValueError(f"Unknown model {n}")
    return out


def md_table(df: pd.DataFrame) -> str:
    """Markdown table without extra dependencies."""
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "| " + " | ".join("---" for _ in df.columns) + " |"]
    return "\n".join(lines + ["| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False)])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=Path("configs/forecast_julich.yaml"))
    ap.add_argument("--features-root", type=Path, default=None)
    ap.add_argument("--models", nargs="*", default=["persistence", "physics"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--alpha", type=float, default=0.1, help="1 - target coverage of the upper bound")
    ap.add_argument("--out", type=Path, default=Path("results/forecast_julich.md"))
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    models = build_models(cfg, args.models, args.device)

    points = {}
    for split in ("val", "test"):
        feats = load_split(cfg, split, args.features_root)
        points[split] = pd.concat([forecast_points(feats, m, cfg["context_s"], cfg["horizons_s"], cfg["origin_stride_s"])
                                   for m in models], ignore_index=True)
        points[split]["split"] = split
    offsets = conformal.fit(points["val"], args.alpha)
    test = conformal.apply(points["test"], offsets)
    table = score(test, cfg["critical_density"])
    cal = test.dropna(subset=["target"]).groupby(["model", "horizon_s"]).apply(
        lambda g: pd.Series({"below_p90_cal": round(float((g["target"] <= g["p90_cal"]).mean()), 3),
                             "mean_p90_cal_minus_p50": round(float((g["p90_cal"] - g["p50"]).mean()), 3)}),
        include_groups=False).reset_index()
    table = table.merge(cal, on=["model", "horizon_s"]).merge(offsets, on=["model", "horizon_s"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([points["val"], test], ignore_index=True).to_parquet(args.out.with_name(args.out.stem + "_points.parquet"), index=False)
    table.to_csv(args.out.with_suffix(".csv"), index=False)
    runs = {s: len(set(p["camera_id"])) for s, p in points.items()}
    text = [f"# Forecasting on Jülich (T6.1–T6.4, Jülich part)", "",
            f"Generated {time.strftime('%Y-%m-%d %H:%M')}. Test = {cfg['split']['test']} ({runs['test']} runs); "
            f"conformal calibration on val = {cfg['split']['val']} ({runs['val']} runs). Context {cfg['context_s']} s, "
            f"a forecast every {cfg['origin_stride_s']} s. Density in persons/m².", "",
            "- `mae`: error of the median forecast; `mae_dense`: same, only where the true density ≥ 2.",
            f"- `below_p90`: share of true values under the raw upper bound; `below_p90_cal`: after conformal "
            f"calibration (target {1 - args.alpha:.0%}); `q`: offset added to the bound, fitted on val.",
            "- Baselines are point forecasts (p10 = p50 = p90); their calibrated bound is p50 + q.", "", md_table(table)]
    args.out.write_text("\n".join(text) + "\n", encoding="utf-8")
    print(table.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

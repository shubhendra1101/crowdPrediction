"""Alert-tuning experiment: baseline vs shorter raise delay, rate-of-rise warning, shorter forecast context (T7.4).

Laptop only (persistence + physics forecasts; Chronos-2 would need the A100). For each dataset (Jülich, simulation):
the rate-of-rise threshold is tuned on the val split only (same 2x-precision rule as the other precursors), every
variant is replayed on the test split, and all variants are scored on the SAME events (zone first reaching the event
density at least 30 s into the run) with the strict in-window metric and false alarms per zone-hour.

    python scripts/alert_experiment.py
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crowdsafe.forecasting import conformal  # noqa: E402
from crowdsafe.forecasting.baselines import Persistence, PhysicsFillRate  # noqa: E402
from crowdsafe.forecasting.evaluate import forecast_points  # noqa: E402
from crowdsafe.risk.rules import RiskEngine  # noqa: E402
from scripts.eval_alerts import _rolling_slope, future_event, replay, score  # noqa: E402
from scripts.eval_forecast import load_split  # noqa: E402

DATASETS = {"Jülich (real)": Path("configs/forecast_julich.yaml"), "Simulation": Path("configs/forecast_sim.yaml")}
VARIANTS = {   # name: (context_s, raise_after_s, rate-of-rise rule on)
    "A baseline (30 s history, 10 s wait)": (30, 10, False),
    "B + shorter wait (5 s)": (30, 5, False),
    "C + rate-of-rise warning": (30, 10, True),
    "D + shorter history (15 s)": (15, 10, False),
    "E all three": (15, 5, True),
}


def tune_rise(val: pd.DataFrame, level: float, ahead_s: float, window_s: float, min_lift: float) -> dict:
    """Rate-of-rise threshold (persons/m² per s over `window_s`) from val Medium zone-seconds; None if not 2x base rate."""
    v = val.sort_values(["camera_id", "zone_id", "timestamp"]).copy()
    v["event"] = future_event(v, level, ahead_s)
    v["slope"] = np.nan
    for _, g in v.groupby(["camera_id", "zone_id"], sort=False):
        v.loc[g.index, "slope"] = _rolling_slope(g["timestamp"].to_numpy(float), g["density"].to_numpy(float), window_s)
    med = v[(v["density"] >= 2.0) & (v["density"] < level) & v["slope"].notna()]
    x, y = med["slope"].to_numpy(float), med["event"].to_numpy(bool)
    base = float(y.mean())
    best = None
    for q in np.arange(50, 99.5, 1.0):
        c = float(np.percentile(x, q))
        if c <= 0:
            continue
        p = x >= c
        tp = (p & y).sum()
        prec, rec = tp / max(p.sum(), 1), tp / max(y.sum(), 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        if prec >= min_lift * base and (best is None or f1 > best[0]):
            best = (f1, c, prec, rec)
    return {"rise_rate": round(best[1], 4) if best else None, "precision": round(best[2], 3) if best else None,
            "recall": round(best[3], 3) if best else None, "base_rate": round(base, 3), "n": len(med)}


def forecasts(cfg: dict, val: pd.DataFrame, test: pd.DataFrame, context: int) -> pd.DataFrame:
    """Persistence + physics test forecasts with the conformal bound fitted on val (same context)."""
    models = [Persistence(), PhysicsFillRate(cfg["models"]["physics"].get("flux_window_s", 10), cfg["critical_density"])]
    pv = pd.concat([forecast_points(val, m, context, cfg["horizons_s"], cfg["origin_stride_s"]) for m in models], ignore_index=True)
    pt = pd.concat([forecast_points(test, m, context, cfg["horizons_s"], cfg["origin_stride_s"]) for m in models], ignore_index=True)
    return conformal.apply(pt, conformal.fit(pv, 0.1))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--risk", type=Path, default=Path("configs/risk.yaml"))
    ap.add_argument("--event-density", type=float, default=4.0)
    ap.add_argument("--out", type=Path, default=Path("results/alert_experiment.md"))
    a = ap.parse_args()
    risk = yaml.safe_load(a.risk.read_text(encoding="utf-8"))
    rows, tuned = [], []
    for ds, cfg_path in DATASETS.items():
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        val, test = load_split(cfg, "val"), load_split(cfg, "test")
        th_rise = tune_rise(val, risk["tuning"]["event_density"], risk["tuning"]["ahead_s"], 10.0, risk["tuning"]["min_lift"])
        tuned.append({"dataset": ds, **th_rise})
        print(ds, "rise threshold", th_rise, flush=True)
        pts = {c: forecasts(cfg, val, test, c) for c in sorted({v[0] for v in VARIANTS.values()})}
        for vname, (ctx, raise_s, rise_on) in VARIANTS.items():
            th = dict(risk["thresholds"])
            if rise_on and th_rise["rise_rate"] is not None:
                th.update({"rise_rate": th_rise["rise_rate"], "rise_window_s": 10.0, "rise_min_density": 1.0})
            for model in [None, "persistence", "physics"]:
                eng = RiskEngine(bands=tuple(risk["bands"]), thresholds=th, collapse_window_s=risk["collapse_window_s"],
                                 collapse_min_density=risk["collapse_min_density"], critical_density=cfg["critical_density"],
                                 use_precursors=rise_on)
                p = pts[ctx]
                alerts = replay(test, p[p["model"] == (model or "persistence")], eng, raise_s, risk["alerts"]["clear_after_s"],
                                cfg["origin_stride_s"], risk["collapse_window_s"], use_forecast=model is not None)
                s = score(test, alerts, a.event_density, window_s=max(cfg["horizons_s"]) + raise_s, event_from_s=30.0)
                rows.append({"dataset": ds, "variant": vname, "forecast": model or "current density only", **s})
            print(" ", vname, "done", flush=True)
    res = pd.DataFrame(rows)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(a.out.with_suffix(".csv"), index=False)
    md = ["# Alert-tuning experiment (T7.4) — before vs after", "",
          f"Generated {time.strftime('%Y-%m-%d %H:%M')} by `scripts/alert_experiment.py`. Test splits only; thresholds tuned on val. "
          f"Event = zone first reaching {a.event_density} persons/m² at least 30 s into a run (same events for every variant). "
          "`in window %` = alert raised within (longest horizon + wait) before the event and not in the first 15 s of monitoring. "
          "Forecasts: persistence and physics (laptop); Chronos-2 not included here (needs the A100). Decisions every 5 s.", "",
          "## Rate-of-rise threshold (tuned on val)", "", "| Dataset | threshold (persons/m² per s) | precision | recall | base rate | n |",
          "| --- | --- | --- | --- | --- | --- |"]
    md += [f"| {t['dataset']} | {t['rise_rate']} | {t['precision']} | {t['recall']} | {t['base_rate']} | {t['n']} |" for t in tuned]
    for ds in DATASETS:
        t = res[res["dataset"] == ds]
        md += ["", f"## {ds}: {int(t['events'].iloc[0])} events", "",
               "| Variant | Forecast | warned in time % | median warning s | late % | missed % | false alarms / zone-hour |",
               "| --- | --- | --- | --- | --- | --- | --- |"]
        md += [f"| {r['variant']} | {r['forecast']} | {r['warned_in_window_%']} | {r['median_lead_in_window_s']} | {r['late_%']} | "
               f"{r['missed_%']} | {r['false_alarms_per_zone_hour']} |" for _, r in t.iterrows()]
    a.out.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(res[["dataset", "variant", "forecast", "events", "warned_in_window_%", "median_lead_in_window_s", "late_%", "missed_%",
               "false_alarms_per_zone_hour"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

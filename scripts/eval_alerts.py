"""Tune precursor thresholds and evaluate alerts on Jülich runs (T7.1, T7.2, T7.4; architecture.md §6).

1. Tuning (val split only): for pressure, direction entropy and counter-flow, pick the threshold that
   best predicts "density reaches the event level within the next 60 s" among zone-seconds that are
   currently Medium (2–4 persons/m²) — the only case where a precursor changes the outcome
   (Medium -> High). Written to configs/risk.yaml.
2. Replay (test split): every forecast origin (5 s) goes through RiskEngine + AlertManager
   (raise after 10 s, clear after 30 s). Variants: react to current density only, the forecasters'
   calibrated upper bounds, each with/without precursors.
3. Metrics per event level (4 and 5 persons/m²): share of zone events alerted *before* the density
   got there, lead time, late detections (≤ 30 s after), false alarms (alert with no event within 120 s).

    python scripts/eval_alerts.py --points weights/crowdsafe_nb04_report/forecast_julich_points.parquet
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
from crowdsafe.risk.alerts import AlertManager  # noqa: E402
from crowdsafe.risk.rules import RiskEngine  # noqa: E402
from scripts.eval_forecast import load_split  # noqa: E402


def future_event(df: pd.DataFrame, level: float, ahead_s: float) -> pd.Series:
    """Per row: does this zone reach `level` within (t, t + ahead_s]?"""
    out = pd.Series(False, index=df.index)
    for _, g in df.groupby(["camera_id", "zone_id"], sort=False):
        t = g["timestamp"].to_numpy(float)
        hi = g["density"].to_numpy(float) >= level
        ht = t[hi]
        if len(ht) == 0:
            continue
        j = np.searchsorted(ht, t, side="right")
        nxt = np.where(j < len(ht), ht[np.minimum(j, len(ht) - 1)], np.inf)
        out.loc[g.index] = (nxt - t) <= ahead_s
    return out


def _rolling_slope(t: np.ndarray, y: np.ndarray, win: float) -> np.ndarray:
    """Least-squares slope of y over the trailing `win` seconds at every sample (NaN if < 5 samples)."""
    out = np.full(len(t), np.nan)
    for k in range(len(t)):
        m = (t >= t[k] - win) & (t <= t[k]) & np.isfinite(y)
        if m.sum() >= 5 and np.ptp(t[m]) > 0:
            out[k] = np.polyfit(t[m], y[m], 1)[0]
    return out


def tune(val: pd.DataFrame, level: float, ahead_s: float, win: float, min_density: float,
         min_lift: float = 2.0) -> tuple[dict, pd.DataFrame]:
    """Precursor thresholds from val only.

    Population: zone-seconds currently Medium (2 <= density < level). Target: density reaches `level`
    within `ahead_s`. Each precursor is tested in its physical direction (pressure, entropy, counter-flow:
    high; speed: low). The threshold maximises F1 among those whose precision is at least `min_lift` x
    the base rate; if none qualifies the precursor is switched off (None). Flow collapse (density rising
    and speed falling over `win` s) is a yes/no rule, kept on only if it passes the same precision test.
    """
    v = val.sort_values(["camera_id", "zone_id", "timestamp"]).copy()
    v["event"] = future_event(v, level, ahead_s)
    med_mask = (v["density"] >= 2.0) & (v["density"] < level)
    v["collapse"] = False
    for _, g in v[med_mask | True].groupby(["camera_id", "zone_id"], sort=False):
        t = g["timestamp"].to_numpy(float)
        sd = _rolling_slope(t, g["density"].to_numpy(float), win)
        sv = _rolling_slope(t, g["mean_speed"].to_numpy(float), win)
        v.loc[g.index, "collapse"] = (sd > 0) & (sv < 0) & (g["density"].rolling(5, min_periods=1).mean().to_numpy() >= min_density)
    med = v[med_mask]
    y_all = med["event"].to_numpy(bool)
    base_rate = float(y_all.mean())
    rows, th = [], {}
    for key, col, direction in [("pressure", "pressure", "high"), ("dir_entropy", "dir_entropy", "high"),
                                ("counterflow", "counterflow", "high"), ("slow_speed", "mean_speed", "low")]:
        x = med[col].to_numpy(float)
        ok = np.isfinite(x)
        best = None
        for q in np.arange(1, 99.5, 1.0):
            c = np.percentile(x[ok], q)
            p = x[ok] >= c if direction == "high" else x[ok] <= c
            tp = (p & y_all[ok]).sum()
            prec, rec = tp / max(p.sum(), 1), tp / max(y_all[ok].sum(), 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-9)
            if prec >= min_lift * base_rate and (best is None or f1 > best[0]):
                best = (f1, float(c), prec, rec)
        th[key] = round(best[1], 4) if best else None
        rows.append({"precursor": key, "direction": direction, "threshold": th[key],
                     "f1": round(best[0], 3) if best else None, "precision": round(best[2], 3) if best else None,
                     "recall": round(best[3], 3) if best else None, "base_rate": round(base_rate, 3), "n": int(ok.sum())})
    p = med["collapse"].to_numpy(bool)
    tp = (p & y_all).sum()
    prec, rec = tp / max(p.sum(), 1), tp / max(y_all.sum(), 1)
    th["collapse_enabled"] = bool(prec >= min_lift * base_rate)
    rows.append({"precursor": "flow_collapse", "direction": "rule", "threshold": "on" if th["collapse_enabled"] else "off",
                 "f1": round(2 * prec * rec / max(prec + rec, 1e-9), 3), "precision": round(prec, 3), "recall": round(rec, 3),
                 "base_rate": round(base_rate, 3), "n": len(med)})
    return th, pd.DataFrame(rows)

def replay(feat: pd.DataFrame, points: pd.DataFrame | None, engine: RiskEngine, raise_s: float, clear_s: float,
           stride: float, window: float, use_forecast: bool = True) -> pd.DataFrame:
    """Alerts raised per run/zone. use_forecast=False -> react to current density only (same decision times as the
    forecasters when `points` is given, so every variant is scored over the same monitored period)."""
    rows = []
    for cam, f in feat.groupby("camera_id", sort=False):
        f = f.sort_values("timestamp")
        if points is not None:
            p = points[points["camera_id"] == cam]
            origins = np.sort(p["origin"].unique())
            pg = dict(tuple(p.groupby("origin")))
        else:
            t0 = f["timestamp"].min() + window
            origins = np.arange(t0, f["timestamp"].max() + 1e-9, stride)
            pg = {}
        mgr = AlertManager(raise_s, clear_s)
        ts = f["timestamp"].to_numpy(float)
        for t in origins:
            hist = f.iloc[np.searchsorted(ts, t - window):np.searchsorted(ts, t, side="right")]
            if hist.empty:
                continue
            fc = pg.get(t) if use_forecast else None
            if fc is not None:
                fc = fc.assign(p90=fc["p90_cal"].fillna(fc["p90"]))[["zone_id", "horizon_s", "p90"]]
            mgr.update(float(t), engine(hist, fc))
        for a in mgr.history:
            rows.append({"camera_id": cam, "zone_id": a.zone_id, "since": a.since, "level": a.level,
                         "first_seen": a.first_seen, "reasons": "; ".join(a.reasons)})
        rows.append({"camera_id": cam, "zone_id": None, "since": np.nan, "monitor_start": float(origins.min()) if len(origins) else np.nan,
                     "monitor_end": float(origins.max()) if len(origins) else np.nan, "n_zones": f["zone_id"].nunique()})
    return pd.DataFrame(rows)


def score(feat: pd.DataFrame, alerts: pd.DataFrame, level: float, late_s: float = 30.0, fa_window_s: float = 120.0,
          window_s: float = 70.0, warmup_s: float = 15.0, event_from_s: float | None = None) -> dict:
    """Event scores. `alerted_before_%` counts any alert raised before the event (an alert raised early and kept on
    counts, with a long lead). Stricter, reported alongside: `warned_in_window_%` = an alert raised within `window_s`
    before the event (max forecast horizon + raise delay) and not in the first `warmup_s` of monitoring, when the
    first forecast extrapolates the start-up ramp."""
    mon = alerts[alerts["zone_id"].isna()].set_index("camera_id")
    al = alerts[alerts["zone_id"].notna()]
    leads, late, missed, fa, n_alerts = [], 0, 0, 0, len(al)
    in_window = []
    zone_hours = 0.0
    for (cam, zone), g in feat.groupby(["camera_id", "zone_id"], sort=False):
        start, end = mon.loc[cam, "monitor_start"], mon.loc[cam, "monitor_end"]
        if not np.isfinite(start):
            continue
        zone_hours += (end - start) / 3600
        hi = g.loc[g["density"] >= level, "timestamp"]
        za = al[(al["camera_id"] == cam) & (al["zone_id"] == zone)]["since"].to_numpy(float)
        ev = float(hi.min()) if len(hi) else None
        if event_from_s is not None:                         # fixed event set for comparing variants with different starts
            if ev is not None and ev < g["timestamp"].min() + event_from_s:
                ev = None
        elif ev is not None and ev < start:                  # already above before monitoring could start
            ev = None
        if ev is not None:
            before = za[za <= ev]
            strict = before[(before >= ev - window_s) & (before > start + warmup_s)]
            if len(strict):
                in_window.append(ev - strict.min())
            if len(before):
                leads.append(ev - before.min())
            elif len(za[(za > ev) & (za <= ev + late_s)]):
                late += 1
            else:
                missed += 1
        hits = hi.to_numpy(float)
        for s in za:                                          # false alarm: no event within the window after raising
            if not ((hits >= s - late_s) & (hits <= s + fa_window_s)).any():
                fa += 1
    n_ev = len(leads) + late + missed
    return {"events": n_ev, "alerted_before_%": round(100 * len(leads) / max(n_ev, 1), 1),
            "late_%": round(100 * late / max(n_ev, 1), 1), "missed_%": round(100 * missed / max(n_ev, 1), 1),
            "median_lead_s": round(float(np.median(leads)), 1) if leads else None,
            "mean_lead_s": round(float(np.mean(leads)), 1) if leads else None,
            "warned_in_window_%": round(100 * len(in_window) / max(n_ev, 1), 1),
            "median_lead_in_window_s": round(float(np.median(in_window)), 1) if in_window else None,
            "alerts": n_alerts, "false_alarms": fa, "false_alarms_per_zone_hour": round(fa / max(zone_hours, 1e-9), 2),
            "zone_hours": round(zone_hours, 2)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=Path("configs/forecast_julich.yaml"))
    ap.add_argument("--risk", type=Path, default=Path("configs/risk.yaml"))
    ap.add_argument("--points", type=Path, default=Path("weights/crowdsafe_nb04_report/forecast_julich_points.parquet"))
    ap.add_argument("--out", type=Path, default=Path("results/alerts_julich.md"))
    ap.add_argument("--tune-configs", type=Path, nargs="*", default=[Path("configs/forecast_julich.yaml"), Path("configs/forecast_sim.yaml")],
                    help="val splits of these configs are pooled for threshold tuning")
    ap.add_argument("--no-tune", action="store_true", help="use the thresholds already in --risk")
    ap.add_argument("--features-from-points", action="store_true",
                    help="take the observed density from the forecast file ('last' at each origin) instead of feature "
                         "tables — needed when the forecasts were made on another machine's simulation run")
    a = ap.parse_args()
    cfg = yaml.safe_load(a.config.read_text(encoding="utf-8"))
    risk = yaml.safe_load(a.risk.read_text(encoding="utf-8"))
    al = risk["alerts"]
    pts = pd.read_parquet(a.points)
    pts = pts[pts["split"] == "test"]
    if a.features_from_points:
        ref = pts[pts["model"] == pts["model"].iloc[0]].drop_duplicates(["camera_id", "zone_id", "origin"])
        test = pd.DataFrame({"timestamp": ref["origin"].astype(float), "camera_id": ref["camera_id"], "zone_id": ref["zone_id"],
                             "density": ref["last"].astype(float)})
        for c in ("mean_speed", "speed_var", "pressure", "dir_entropy", "counterflow"):
            test[c] = np.nan
        test = test.sort_values(["camera_id", "zone_id", "timestamp"]).reset_index(drop=True)
    else:
        test = load_split(cfg, "test")
    if a.no_tune:
        tune_tab = pd.DataFrame(columns=["precursor", "direction", "threshold", "f1", "precision", "recall", "base_rate", "n"])
    else:
        tune_cfgs = [yaml.safe_load(p.read_text(encoding="utf-8")) for p in a.tune_configs]
        val = pd.concat([load_split(c, "val") for c in tune_cfgs], ignore_index=True)
        th, tune_tab = tune(val, risk["tuning"]["event_density"], risk["tuning"]["ahead_s"], risk["collapse_window_s"],
                            risk["collapse_min_density"], risk["tuning"]["min_lift"])
        risk["thresholds"].update(th)
        risk["thresholds_note"] = (f"tuned {time.strftime('%Y-%m-%d')} by scripts/eval_alerts.py on the pooled val splits of "
                                   + ", ".join(str(p) for p in a.tune_configs))
        a.risk.write_text(yaml.safe_dump(risk, sort_keys=False, allow_unicode=True), encoding="utf-8")
    any_precursor = any(v not in (None, False) for k, v in risk["thresholds"].items()
                        if k in ("pressure", "dir_entropy", "counterflow", "slow_speed", "collapse_enabled"))
    kw = dict(bands=tuple(risk["bands"]), thresholds=risk["thresholds"], collapse_window_s=risk["collapse_window_s"],
              collapse_min_density=risk["collapse_min_density"], critical_density=cfg["critical_density"])
    variants = {"current density only": (None, False)}
    if any_precursor:
        variants["current density + precursors"] = (None, True)
    for m in ["persistence", "physics", "chronos2_per_zone", "chronos2_joint"]:
        if m in pts["model"].unique():
            variants[f"{m} upper bound"] = (m, False)
            if any_precursor:
                variants[f"{m} upper bound + precursors"] = (m, True)
    results, timelines = [], []
    for name, (model, pre) in variants.items():
        eng = RiskEngine(use_precursors=pre, **kw)
        ref_model = model or pts["model"].iloc[0]           # same decision times for every variant
        alerts = replay(test, pts[pts["model"] == ref_model], eng, al["raise_after_s"], al["clear_after_s"],
                        cfg["origin_stride_s"], risk["collapse_window_s"], use_forecast=model is not None)
        timelines.append(alerts.assign(variant=name))
        for lvl in risk["event_levels"]:
            results.append({"variant": name, "event_density": lvl,
                            **score(test, alerts, lvl, window_s=max(cfg["horizons_s"]) + al["raise_after_s"])})
        print(name, "done", flush=True)
    res = pd.DataFrame(results)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(a.out.with_suffix(".csv"), index=False)
    pd.concat(timelines, ignore_index=True).to_csv(a.out.with_name(a.out.stem + "_timelines.csv"), index=False)
    md = [f"# Alerts on held-out runs: {a.config.stem} (T7.1, T7.2, T7.4)", "",
          f"Generated {time.strftime('%Y-%m-%d %H:%M')} by `scripts/eval_alerts.py`. Test runs: {cfg['split']['test']} "
          f"({test['camera_id'].nunique()} runs, {test.groupby('camera_id')['zone_id'].nunique().sum()} zones). "
          f"Decision every {cfg['origin_stride_s']} s; alert = level ≥ High held {al['raise_after_s']} s (cleared after "
          f"{al['clear_after_s']} s below). Upper bound = worst conformal-calibrated 90 % bound over horizons "
          f"{cfg['horizons_s']} s. An *event* is a zone first reaching the event density after monitoring starts.", "",
          "- `alerted_before_%`: events with an alert raised before the density got there; `late_%`: alert within 30 s after; "
          "`missed_%`: neither.", "- `lead`: seconds between the alert and the event (alerts raised before it).",
          "- `false_alarms`: alerts with no event within 120 s after raising (or 30 s before).",
          f"- **`in window %` (stricter, use this for claims):** alert raised at most {max(cfg['horizons_s']) + al['raise_after_s']} s "
          "(longest horizon + raise delay) before the event and not in the first 15 s of monitoring. `alerted before %` also "
          "counts alerts raised at the start of a run and simply kept on, which inflates lead times.", "",
          f"## Precursor thresholds (tuned on pooled val splits: {', '.join(p.stem for p in a.tune_configs)})", "",
          f"Target: zone reaches {risk['tuning']['event_density']} persons/m² within {risk['tuning']['ahead_s']} s, among zone-seconds "
          f"currently at 2–4 persons/m². A precursor is kept only if its precision is ≥ {risk['tuning']['min_lift']}× the base rate.", "",
          "| Precursor | Direction | Threshold | F1 | Precision | Recall | Base rate | n |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    md += [f"| {r.precursor} | {r.direction} | {r.threshold} | {r.f1} | {r.precision} | {r.recall} | {r.base_rate} | {r.n} |"
           for r in tune_tab.itertuples()]
    for lvl in risk["event_levels"]:
        t = res[res["event_density"] == lvl]
        md += ["", f"## Event = {lvl} persons/m²", "",
               "| Variant | events | **in window %** | median lead in window s | alerted before % | late % | missed % | median lead s | alerts | false alarms | per zone-hour |",
               "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        md += [f"| {r['variant']} | {r['events']} | {r['warned_in_window_%']} | {r['median_lead_in_window_s']} | {r['alerted_before_%']} | "
               f"{r['late_%']} | {r['missed_%']} | {r['median_lead_s']} | {r['alerts']} | {r['false_alarms']} | {r['false_alarms_per_zone_hour']} |"
               for _, r in t.iterrows()]
    a.out.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(tune_tab.to_string(index=False))
    print(res.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

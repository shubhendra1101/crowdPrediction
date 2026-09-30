"""Build zone feature tables from all Jülich trajectory files (T5.2).

Writes one Parquet per run to ``out_root/<dataset>/<run>.parquet`` and a per-run summary
(max density, seconds above 4 and 5 persons/m², max pressure) to results/.
Also cross-checks our zone density against PedPy's classic density on one zone per dataset.

    python scripts/julich_features.py --config configs/julich.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crowdsafe.datasets.julich import drop_implausible_velocity, make_zones, read_trajectories, resample, zone_features  # noqa: E402
from crowdsafe.features.logger import write_parquet  # noqa: E402


def md_table(df: pd.DataFrame) -> str:
    """Render a DataFrame as a Markdown table (no extra dependency)."""
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    return "\n".join([head, sep] + ["| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False)])


def pedpy_check(samples: pd.DataFrame, zones, zone_id: str, hz: float) -> float:
    """Max |our per-step density − PedPy classic density| for one zone, over all grid steps."""
    import pedpy
    from shapely import Polygon

    poly = Polygon(zones.polygon(zone_id))
    traj = pedpy.TrajectoryData(data=samples.rename(columns={"k": "frame"})[["id", "frame", "x", "y"]], frame_rate=hz)
    ped = pedpy.compute_classic_density(traj_data=traj, measurement_area=pedpy.MeasurementArea(poly))
    ours = (pd.Series(zones.zone_of(samples["x"].to_numpy(), samples["y"].to_numpy()) == zone_id, index=samples.index)
            .groupby(samples["k"]).sum() / zones.area_m2)
    ours.index = ours.index - samples["k"].min()          # PedPy renumbers frames from 0
    ours = ours.reindex(ped.index, fill_value=0)
    return float((ours - ped["density"]).abs().max())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=Path("configs/julich.yaml"))
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--limit-runs", type=int, default=None)
    ap.add_argument("--results", type=Path, default=Path("results/julich_features_summary.md"))
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    fc = cfg["features"]
    rows, checks = [], {}
    for ds, dcfg in cfg["datasets"].items():
        if args.datasets and ds not in args.datasets:
            continue
        files = sorted(p for p in (Path(cfg["raw_root"]) / ds / "trajectories").rglob("*.txt")
                       if p.stem.lower() not in cfg.get("skip_files", []))[: args.limit_runs]
        for i, f in enumerate(files):
            t0 = time.time()
            try:
                traj, used = read_trajectories(f, dcfg["unit"], dcfg["fps"], dcfg.get("time", "frame"), return_used=True)
            except Exception as e:   # unreadable file: record it and continue
                rows.append({"dataset": ds, "run": f.stem, "n_zones": 0, "note": f"unreadable: {e!r}"[:200]})
                print(rows[-1], flush=True)
                continue
            samples, n_glitch = drop_implausible_velocity(resample(traj, fc["sample_hz"]), fc["max_speed"])
            zones = make_zones(samples, fc["cell_m"], fc["walkable_res_m"], fc["min_walkable_share"], fc["body_radius_m"])
            run_id = f"{ds}/{f.stem}"
            if not zones.cells:
                rows.append({"dataset": ds, "run": f.stem, "n_zones": 0, "note": "no fully walkable cell"})
                continue
            feats = zone_features(samples, zones, run_id, fc["sample_hz"], fc["min_speed"], fc["direction_bins"])
            write_parquet(feats, Path(cfg["out_root"]) / ds / f"{f.stem}.parquet")
            if i == 0:   # PedPy cross-check on the busiest zone of the first run
                busiest = feats.groupby("zone_id")["density"].mean().idxmax()
                checks[ds] = {"run": f.stem, "zone": busiest, "max_abs_diff": pedpy_check(samples, zones, busiest, fc["sample_hz"])}
            d = feats["density"]
            zone_sec_ge = lambda thr: int((d >= thr).sum())
            rows.append({"dataset": ds, "run": f.stem, "fps": used["fps"], "unit": used["unit"],
                         "header_note": used.get("note", ""), "duration_s": int(feats["timestamp"].max()) + 1,
                         "persons": traj["id"].nunique(), "n_zones": len(zones.cells),
                         "max_density": round(float(d.max()), 2), "p99_density": round(float(d.quantile(0.99)), 2),
                         "zone_s_ge2": zone_sec_ge(2), "zone_s_ge4": zone_sec_ge(4), "zone_s_ge5": zone_sec_ge(5),
                         "max_pressure": round(float(feats["pressure"].max()), 3),
                         "median_speed": round(float(feats["mean_speed"].median()), 2),
                         "glitch_share": round(n_glitch / len(samples), 4), "sec": round(time.time() - t0, 1)})
            print(rows[-1], flush=True)
    summary = pd.DataFrame(rows)
    args.results.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.results.with_suffix(".csv"), index=False)
    by_ds = summary.groupby("dataset").agg(runs=("run", "count"), persons=("persons", "sum"),
                                           max_density=("max_density", "max"), zone_s_ge4=("zone_s_ge4", "sum"),
                                           zone_s_ge5=("zone_s_ge5", "sum"), max_pressure=("max_pressure", "max"),
                                           glitch_share_max=("glitch_share", "max"))
    lines = ["# Jülich zone features — summary (T5.2)", "",
             f"Generated {time.strftime('%Y-%m-%d %H:%M')} from `{args.config}`. Zones: {fc['cell_m']} m squares "
             f"({fc['cell_m'] ** 2:g} m²) kept when ≥ {fc['min_walkable_share']:.0%} walked on (walkable area inferred "
             f"from trajectories — approximate). Densities = persons/m², mean per second. `zone_s_geX` = zone-seconds at or above X persons/m². "
             f"Velocities above {fc['max_speed']} m/s are treated as tracking glitches and ignored (`glitch_share` = share of samples).",
             "", md_table(by_ds.reset_index()), "",
             "## PedPy cross-check (our zone density vs `pedpy.compute_classic_density`, same zone and frames)", "",
             "| Dataset | Run | Zone | Max abs diff (persons/m²) |", "| --- | --- | --- | --- |"]
    lines += [f"| {k} | {v['run']} | {v['zone']} | {v['max_abs_diff']:.4f} |" for k, v in checks.items()]
    lines += ["", "Per-run table: `" + str(args.results.with_suffix(".csv")) + "`."]
    args.results.write_text("\n".join(lines) + "\n", encoding="utf-8")
    Path(args.results.with_name("julich_features_checks.json")).write_text(json.dumps(checks, indent=2))
    print(by_ds.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

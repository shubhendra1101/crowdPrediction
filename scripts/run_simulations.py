"""Run the JuPedSim scenarios and write trajectories + zone feature tables (T5.4).

    python scripts/run_simulations.py                 # all scenarios in configs/simulation.yaml
    python scripts/run_simulations.py --only corr_surge
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crowdsafe.simulation.scenarios import features, run  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=Path("configs/simulation.yaml"))
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--out", type=Path, default=Path("results/simulation_summary.md"))
    a = ap.parse_args()
    cfg = yaml.safe_load(a.config.read_text(encoding="utf-8"))
    split = {s: k for k, v in cfg["split"].items() for s in v}
    traj_dir, feat_dir = Path(cfg["out_traj"]), Path(cfg["out_features"])
    traj_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, scn in cfg["scenarios"].items():
        if a.only and name not in a.only:
            continue
        t0 = time.time()
        tr = run(scn, seed=cfg["seed"], dt=cfg["dt"], record_hz=cfg["record_hz"])
        tr.to_parquet(traj_dir / f"{name}.parquet", index=False)
        f = features(tr, scn, f"sim/{name}", hz=cfg["record_hz"], cell_m=cfg["cell_m"])
        out = feat_dir / scn["layout"] / f"{name}.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        f.to_parquet(out, index=False)
        rows.append({"scenario": name, "split": split.get(name, "-"), "layout": scn["layout"], "agents": tr["id"].nunique(),
                     "duration_s": round(float(tr["t"].max()), 1), "zones": f["zone_id"].nunique(),
                     "max_density": round(float(f["density"].max()), 2), "zone_s_ge4": int((f["density"] >= 4).sum()),
                     "zone_s_ge5": int((f["density"] >= 5).sum()), "max_pressure": round(float(f["pressure"].max()), 3),
                     "wall_s": round(time.time() - t0, 1)})
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    if a.only:
        return 0
    df.to_csv(a.out.with_suffix(".csv"), index=False)
    md = ["# JuPedSim scenarios (T5.4) — synthetic data", "",
          f"Generated {time.strftime('%Y-%m-%d %H:%M')} by `scripts/run_simulations.py` from `{a.config}`. Collision-free speed "
          "model (JuPedSim), desired speed N(1.34, 0.26) m/s clipped to 0.7–1.9, body radius 0.15–0.20 m. Zones = 2 m cells; "
          "density = people / walkable area of the cell (desks excluded). `zone_s_geX` = zone-seconds at or above X persons/m².", "",
          "| Scenario | Split | Layout | Agents | Duration s | Zones | Max density | zone-s ≥ 4 | zone-s ≥ 5 | Max pressure |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    md += [f"| {r.scenario} | {r.split} | {r.layout} | {r.agents} | {r.duration_s} | {r.zones} | {r.max_density} | {r.zone_s_ge4} | "
           f"{r.zone_s_ge5} | {r.max_pressure} |" for r in df.itertuples()]
    a.out.write_text("\n".join(md) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

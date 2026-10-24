"""Viewable previews of the simulated data: snapshot images and CSV copies (T5.4).

    python scripts/sim_preview.py
Writes results/figures/sim_<scenario>.png (people as dots at 4 moments, walls/desks in grey, colour = local
density) and data/sim/csv/<scenario>_trajectories.csv + _zone_features.csv (open in Excel).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crowdsafe.simulation.scenarios import LAYOUTS  # noqa: E402


def draw_geometry(ax, lay) -> None:
    x0, y0, x1, y1 = lay.walkable.bounds
    ax.set_facecolor("0.55")                                   # outside / obstacles grey
    for poly in getattr(lay.walkable, "geoms", [lay.walkable]):
        ax.fill(*poly.exterior.xy, color="white", zorder=0)
        for hole in poly.interiors:
            ax.fill(*hole.xy, color="0.55", zorder=1)
    for p in lay.exits.values():
        ax.fill(*p.exterior.xy, color="tab:green", alpha=0.5, zorder=1)
    ax.set_xlim(x0 - 0.5, x1 + 0.5)
    ax.set_ylim(y0 - 0.5, y1 + 0.5)
    ax.set_aspect("equal")


def main() -> int:
    cfg = yaml.safe_load(Path("configs/simulation.yaml").read_text(encoding="utf-8"))
    csv_dir = Path("data/sim/csv")
    csv_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = Path("results/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    for name, scn in cfg["scenarios"].items():
        tp = Path(cfg["out_traj"]) / f"{name}.parquet"
        fp = Path(cfg["out_features"]) / scn["layout"] / f"{name}.parquet"
        if not tp.exists():
            continue
        tr = pd.read_parquet(tp)
        tr.round(3).to_csv(csv_dir / f"{name}_trajectories.csv", index=False)
        if fp.exists():
            pd.read_parquet(fp).round(4).to_csv(csv_dir / f"{name}_zone_features.csv", index=False)
        lay = LAYOUTS[scn["layout"]](**scn.get("layout_args", {}))
        T = tr["t"].max()
        times = [round(T * f / 0.2) * 0.2 for f in (0.0, 0.25, 0.5, 0.75)]
        fig, axes = plt.subplots(1, 4, figsize=(16, 4.6))
        for ax, t in zip(axes, times):
            draw_geometry(ax, lay)
            s = tr[np.isclose(tr["t"], t, atol=0.11)]
            if len(s):
                xy = s[["x", "y"]].to_numpy()
                d = np.hypot(*(xy[:, None, :] - xy[None, :, :]).transpose(2, 0, 1))
                local = ((d < 0.8).sum(1) - 1) / (np.pi * 0.8 ** 2)     # neighbours within 0.8 m -> persons/m²
                sc = ax.scatter(xy[:, 0], xy[:, 1], c=local, cmap="RdYlGn_r", vmin=0, vmax=6, s=10, zorder=3)
            ax.set_title(f"t = {t:.0f} s, {len(s)} people")
        fig.colorbar(sc, ax=axes, shrink=0.8, label="local density (persons/m²)")
        fig.suptitle(f"{name} ({scn['layout']}) — simulated with JuPedSim (synthetic data)")
        fig.savefig(fig_dir / f"sim_{name}.png", dpi=80, bbox_inches="tight")
        plt.close(fig)
        print(name, "ok", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

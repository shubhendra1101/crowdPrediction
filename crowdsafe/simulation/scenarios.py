"""JuPedSim scenarios -> trajectories -> zone features in the shared schema (T5.4).

Three layouts (metres): an H706-like classroom emptying through two front doors, a corridor ending in a
bottleneck, and a hall with two exits; plus a bidirectional corridor. Scenario events: arrival surges,
an exit closed during the run, opposing flows, and over-capacity crowds that push density above
5 persons/m². Agents use JuPedSim's collision-free speed model; walking speeds follow the usual
pedestrian distribution (mean 1.34 m/s, sd 0.26, Weidmann 1993) and body radii 0.15–0.20 m so that
crowds can pack beyond 5 persons/m².

Everything here is synthetic: results from it are reported separately from real data.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

from crowdsafe.datasets.julich import Zones, resample, zone_features


@dataclass
class Layout:
    walkable: Polygon                               # floor where people can walk (obstacles removed)
    exits: dict[str, Polygon]                        # exit stage polygons (inside `walkable`)
    start: Polygon | None = None                     # where the initial crowd stands
    sources: dict[str, Polygon] = field(default_factory=dict)   # spawn areas for arrivals
    zone_area: Polygon | None = None                 # area to cut into 2 m zones (default: walkable bounds)


def classroom(room=(8.03, 10.36), door_w=1.12, door_x=(0.25, 6.65), cols=5, desk=(1.0, 0.38),
              gaps=(0.75, 0.79, 0.80, 0.75), rows=9, first_row_y=1.6, pitch=0.87) -> Layout:
    """H706-like room: desks in 5 columns, rows `pitch` apart, two doors in the front wall (y = 0).

    Desk top 1.00 x 0.38 m (depth from the calibration fit), row pitch 0.87 m (H706, user-confirmed).
    """
    W, L = room
    scale = (W - cols * desk[0]) / sum(gaps)                  # fit columns + gaps to the room width
    xs, x = [], 0.0
    for c in range(cols):
        xs.append(x)
        x += desk[0] + (gaps[c] * scale if c < cols - 1 else 0)
    desks = [box(x0, first_row_y + r * pitch, x0 + desk[0], first_row_y + r * pitch + desk[1])
             for x0 in xs for r in range(rows)]
    room_poly = box(0, 0, W, L)
    doors = [box(dx, -1.2, dx + door_w, 0.0) for dx in door_x]
    walk = unary_union([room_poly] + doors).difference(unary_union(desks))
    seats = [box(x0 + 0.05, first_row_y + r * pitch + desk[1] + 0.02, x0 + desk[0] - 0.05,
                 first_row_y + (r + 1) * pitch - 0.02) for x0 in xs for r in range(rows)]
    return Layout(walk, {f"door{i}": box(dx, -1.2, dx + door_w, -0.8) for i, dx in enumerate(door_x)},
                  start=unary_union(seats), zone_area=room_poly)


def corridor_bottleneck(width=4.0, length=20.0, gap=1.2) -> Layout:
    """Corridor (y from 0 to `length`) ending in a wall with a `gap`-wide opening, then a short exit room."""
    cx = width / 2
    corridor = box(0, 0, width, length)
    opening = box(cx - gap / 2, length, cx + gap / 2, length + 0.3)
    beyond = box(cx - 1.5, length + 0.3, cx + 1.5, length + 2.5)
    walk = unary_union([corridor, opening, beyond])
    return Layout(walk, {"exit": box(cx - 1.5, length + 2.0, cx + 1.5, length + 2.5)},
                  sources={"entry": box(0.2, 0.2, width - 0.2, 2.0)}, zone_area=corridor)


def hall(size=(20.0, 15.0), exit_a=2.0, exit_b=1.2) -> Layout:
    """Hall with a wide exit A on the east wall and a narrow exit B on the north wall."""
    W, H = size
    room = box(0, 0, W, H)
    a = box(W, H / 2 - exit_a / 2, W + 1.5, H / 2 + exit_a / 2)
    b = box(4 - exit_b / 2, H, 4 + exit_b / 2, H + 1.5)
    walk = unary_union([room, a, b])
    return Layout(walk, {"A": box(W + 1.0, H / 2 - exit_a / 2, W + 1.5, H / 2 + exit_a / 2),
                         "B": box(4 - exit_b / 2, H + 1.0, 4 + exit_b / 2, H + 1.5)},
                  start=box(0.5, 0.5, W - 2, H - 0.5), zone_area=room)


def bidirectional(width=4.0, length=20.0) -> Layout:
    """Open corridor; one group walks north, the other south."""
    walk = box(0, 0, width, length)
    return Layout(walk, {"north": box(0.1, length - 0.4, width - 0.1, length - 0.05),
                         "south": box(0.1, 0.05, width - 0.1, 0.4)},
                  sources={"south_end": box(0.2, 0.6, width - 0.2, 2.0), "north_end": box(0.2, length - 2.0, width - 0.2, length - 0.6)},
                  zone_area=walk)


LAYOUTS = {"classroom": classroom, "corridor_bottleneck": corridor_bottleneck, "hall": hall, "bidirectional": bidirectional}


def _free_spots(poly: Polygon, n: int, occupied: np.ndarray, rng: np.random.Generator, min_dist: float = 0.45,
                tries: int = 400) -> list[tuple[float, float]]:
    """Up to n random points in `poly` at least `min_dist` from occupied points and from each other."""
    x0, y0, x1, y1 = poly.bounds
    pts, occ = [], [tuple(p) for p in occupied]
    for _ in range(tries):
        if len(pts) >= n:
            break
        p = (rng.uniform(x0, x1), rng.uniform(y0, y1))
        if not poly.buffer(-0.2).contains(Polygon([(p[0] - 1e-3, p[1]), (p[0] + 1e-3, p[1]), (p[0], p[1] + 1e-3)])):
            continue
        if occ and np.min(np.hypot(*(np.asarray(occ) - p).T)) < min_dist:
            continue
        pts.append(p)
        occ.append(p)
    return pts


def run(scn: dict, seed: int = 42, dt: float = 0.01, record_hz: float = 5.0) -> pd.DataFrame:
    """Run one scenario; returns trajectories (id, t, x, y) sampled at `record_hz`.

    scn keys: layout, layout_args, initial (int), exits_initial (list or 'nearest'), arrivals
    ({source: [[t_start, t_end, rate_per_s, exit], ...]}), close ({exit: t, ...}), max_time_s.
    """
    import jupedsim as jps

    rng = np.random.default_rng(seed)
    lay = LAYOUTS[scn["layout"]](**scn.get("layout_args", {}))
    sim = jps.Simulation(model=jps.CollisionFreeSpeedModel(), geometry=lay.walkable, dt=dt)
    stage = {name: sim.add_exit_stage(list(p.exterior.coords)[:-1]) for name, p in lay.exits.items()}
    journey = {name: sim.add_journey(jps.JourneyDescription([sid])) for name, sid in stage.items()}
    target: dict[int, str] = {}
    closed: set[str] = set()

    def open_exits():
        return [e for e in lay.exits if e not in closed]

    def nearest(p):
        return min(open_exits(), key=lambda e: lay.exits[e].centroid.distance(Polygon([p, (p[0] + 1e-3, p[1]), (p[0], p[1] + 1e-3)])))

    def add(p, ex):
        params = jps.CollisionFreeSpeedModelAgentParameters(
            position=p, journey_id=journey[ex], stage_id=stage[ex],
            desired_speed=float(np.clip(rng.normal(1.34, 0.26), 0.7, 1.9)), radius=float(rng.uniform(0.15, 0.20)))
        target[sim.add_agent(params)] = ex

    if scn.get("initial", 0):
        n0 = int(scn["initial"])
        if lay.start.geom_type == "Polygon":
            pts = jps.distribute_by_number(polygon=lay.start, number_of_agents=n0,
                                           distance_to_agents=0.40, distance_to_polygon=0.18, seed=seed)
        else:   # seats: up to 2 students per bench, benches in random order; extra people stand in the free floor
            pts = []
            for k in rng.permutation(len(lay.start.geoms)):
                seat = lay.start.geoms[k]
                (x0, y0, x1, y1), need = seat.bounds, min(2, n0 - len(pts))
                pts += [(x0 + (x1 - x0) * (i + 0.5) / 2, (y0 + y1) / 2) for i in range(2)][:need]
                if len(pts) >= n0:
                    break
            if len(pts) < n0:
                free = lay.walkable.difference(lay.start.buffer(0.3)).intersection(lay.zone_area or lay.walkable)
                free = max(getattr(free, "geoms", [free]), key=lambda g: g.area)
                pts += _free_spots(free, n0 - len(pts), np.asarray(pts), rng, tries=20000)
        for p in pts:
            add(p, nearest(p) if scn.get("exits_initial", "nearest") == "nearest" else scn["exits_initial"])
    arrivals = scn.get("arrivals", {})
    backlog = {src: 0.0 for src in arrivals}
    rec_every = int(round(1 / (record_hz * dt)))
    rows, max_t = [], float(scn.get("max_time_s", 600))
    while sim.elapsed_time() < max_t:
        t = sim.elapsed_time()
        it = sim.iteration_count()
        if it % int(round(1 / dt)) == 0:                     # once per second: closures and arrivals
            for ex, tc in scn.get("close", {}).items():
                if t >= tc and ex not in closed:
                    closed.add(ex)
                    for a in sim.agents():
                        if target.get(a.id) == ex:
                            new = nearest(a.position)
                            sim.switch_agent_journey(a.id, journey[new], stage[new])
                            target[a.id] = new
            occ = np.array([a.position for a in sim.agents()]).reshape(-1, 2)
            for src, plan in arrivals.items():
                for t0, t1, rate, ex in plan:
                    if t0 <= t < t1:
                        backlog[src] += rate
                n = int(backlog[src])
                if n:
                    spots = _free_spots(lay.sources[src], n, occ, rng)
                    for p in spots:
                        live = [e for (a0, a1, r, e) in plan if a0 <= t < a1] or [plan[-1][3]]
                        add(p, live[0] if live[0] not in closed else nearest(p))
                    backlog[src] -= len(spots)
                    occ = np.vstack([occ, np.asarray(spots).reshape(-1, 2)]) if spots else occ
        if it % rec_every == 0:
            for a in sim.agents():
                rows.append((a.id, t, a.position[0], a.position[1]))
        if sim.agent_count() == 0 and t > max(([p[1] for plan in arrivals.values() for p in plan] or [0])):
            break
        sim.iterate()
    return pd.DataFrame(rows, columns=["id", "t", "x", "y"])


def zones_for(lay: Layout, cell_m: float = 2.0, min_walkable: float = 0.5) -> tuple[Zones, dict[str, float]]:
    """Square zones over the layout's zone area; keep cells that are at least `min_walkable` walkable.
    Returns the zones and each zone's walkable area (m²) — density is divided by that, not the cell area."""
    area = lay.zone_area or lay.walkable
    x0, y0, x1, y1 = area.bounds
    x0, y0 = np.floor(x0 / cell_m) * cell_m, np.floor(y0 / cell_m) * cell_m
    cells, walk_area = {}, {}
    for ix in range(int(np.ceil((x1 - x0) / cell_m))):
        for iy in range(int(np.ceil((y1 - y0) / cell_m))):
            c = box(x0 + ix * cell_m, y0 + iy * cell_m, x0 + (ix + 1) * cell_m, y0 + (iy + 1) * cell_m)
            w = c.intersection(lay.walkable).intersection(area).area
            if w >= min_walkable * cell_m ** 2:
                zid = f"c{ix}_{iy}"
                cells[zid], walk_area[zid] = (ix, iy), w
    return Zones(cell_m, float(x0), float(y0), cells), walk_area


def features(traj: pd.DataFrame, scn: dict, run_id: str, hz: float = 5.0, cell_m: float = 2.0) -> pd.DataFrame:
    """Trajectories -> zone feature table (schema of crowdsafe.features.schema), source = 'sim'."""
    lay = LAYOUTS[scn["layout"]](**scn.get("layout_args", {}))
    zones, walk = zones_for(lay, cell_m)
    s = resample(traj, hz)
    df = zone_features(s, zones, run_id, hz)
    w = df["zone_id"].map(walk)
    df["density"] = df["count"] / w
    df["pressure"] = df["density"] * df["speed_var"]
    df["source"] = "sim"
    return df

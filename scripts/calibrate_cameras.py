"""Fit the camera model for every room and check it against measurements not used in the fit (T2.3–T2.5).

Inputs: configs/calibration/<room>.yaml (image points read from an empty frame + the user's measurements).
Outputs: configs/cameras/<room>.yaml (camera with the floor at Z = 0), results/calibration.md/.csv and
overlay figures results/figures/calibration_<room>.jpg.

    python scripts/calibrate_cameras.py
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crowdsafe.calibration.camera_model import CameraModel, fit_from_edges, fit_from_rectangles, rect_corners  # noqa: E402
from crowdsafe.calibration.edges import trace_edge  # noqa: E402


def _arr(pts) -> np.ndarray:
    return np.array([[np.nan, np.nan] if p is None else p for p in pts], float)


def _segment_samples(segs, step: float = 6.0) -> np.ndarray:
    out = []
    for x1, y1, x2, y2 in segs:
        n = max(2, int(np.hypot(x2 - x1, y2 - y1) / step))
        t = np.linspace(0, 1, n)[:, None]
        out.append(np.c_[x1 + t * (x2 - x1), y1 + t * (y2 - y1)])
    return np.vstack(out)


def _vertical_bottom_z(cam: CameraModel, top_px, bottom_px, z_top: float = 0.0) -> float:
    """Height of the bottom end of a vertical edge whose top lies on plane Z = z_top."""
    P = cam.to_plane(np.asarray([top_px], float), z_top)[0]
    d, C = cam.rays(np.asarray([bottom_px], float))[0], cam.center
    s, z = np.linalg.lstsq(np.c_[d, [0, 0, -1.0]], np.r_[P, z_top] - C, rcond=None)[0]
    return float(z)


def _door(cam: CameraModel, Q: np.ndarray, height: float) -> dict:
    """Door on a wall facing the camera (plane Y = const): solve x0, y, floor z and width from its corners."""
    ok = ~np.isnan(Q).any(1)

    def res(p):
        x0, y, z, w = p
        P = np.array([[x0, y, z + height], [x0 + w, y, z + height], [x0 + w, y, z], [x0, y, z]])
        return (cam.project(P) - Q)[ok].ravel()

    bl = cam.to_plane(Q[3:4], -0.8)[0]
    s = least_squares(res, [bl[0], bl[1], -0.8, 1.0], loss="soft_l1")
    x0, y, z, w = s.x
    return {"width_m": float(w), "floor_z": float(z), "wall_y": float(y), "rms_px": float(np.sqrt(np.mean(s.fun ** 2)))}


def calibrate_rectangles(cfg: dict, gray: np.ndarray) -> tuple[CameraModel, dict, dict]:
    rects = {k: _arr(v) for k, v in cfg["rectangles"].items()}
    lines = [trace_edge(gray, np.array(v, float), search_px=10) for v in cfg.get("lines_rough", {}).values()]
    fit = fit_from_rectangles(rects, cfg["desk_length_m"], (gray.shape[1], gray.shape[0]), ref=cfg["ref"], lines=lines)
    cam = fit.camera
    lab = {n: tuple(map(int, re.match(r"c(-?\d+)r(-?\d+)", n).groups())) for n in rects}
    # row pitch: within each column, slope of y over row index
    pitches, colx = [], {}
    for c in sorted({v[0] for v in lab.values()}):
        ns = [n for n in rects if lab[n][0] == c]
        colx[c] = float(np.mean([fit.positions[n][0] for n in ns]))
        if len({lab[n][1] for n in ns}) > 1:
            r = np.array([lab[n][1] for n in ns], float)
            y = np.array([fit.positions[n][1] for n in ns])
            pitches.append((np.polyfit(r, y, 1)[0], np.ptp(r)))
    row_pitch = float(np.average([p for p, _ in pitches], weights=[w for _, w in pitches]))
    cols = sorted(colx)
    col_pitch = [round(colx[b] - colx[a], 3) for a, b in zip(cols[:-1], cols[1:]) if b == a + 1]
    checks = {"reprojection_rms_px": round(fit.rms_px, 2), "line_rms_px": round(fit.params["line_rms_px"] or float("nan"), 2),
              "f_px": round(fit.params["f_px"], 1), "k1": round(fit.params["k1"], 3), "k2": round(fit.params["k2"], 3),
              "desk_top_depth_m": round(fit.depth_m, 3), "camera_above_desk_m": round(float(cam.center[2]), 3),
              "row_pitch_fit_m": round(row_pitch, 3), "row_pitch_measured_m": cfg["checks"]["row_pitch_m"],
              "column_pitch_fit_m": col_pitch, "column_pitch_measured_m": cfg["checks"]["column_pitch_m"]}
    fl = cfg["floor"]
    if fl["method"] == "camera_height":
        floor_z = float(cam.center[2] - fl["camera_height_m"])
    else:
        floor_z = -float(fl["desk_height_m"])
    checks["desk_height_m"] = round(-floor_z, 3)
    checks["camera_above_floor_m"] = round(float(cam.center[2] - floor_z), 3)
    if "doors" in cfg:
        for name, pts in cfg["doors"]["corners_px"].items():
            d = _door(cam, _arr(pts), cfg["doors"]["height_m"])
            checks[f"door_{name}_width_m"] = round(d["width_m"], 3)
            checks[f"door_{name}_camera_above_floor_m"] = round(float(cam.center[2] - d["floor_z"]), 3)
            checks[f"door_{name}_wall_distance_m"] = round(d["wall_y"] - float(cam.center[1]), 3)
        checks["door_width_measured_m"] = cfg["doors"]["width_m"]
    extras = {"positions": {n: np.round(p, 3).tolist() for n, p in fit.positions.items()}, "depth": fit.depth_m,
              "length": cfg["desk_length_m"], "lines": lines, "residuals": fit.residuals_px}
    return cam.with_floor_at(floor_z), checks, extras


def calibrate_edges(cfg: dict, gray: np.ndarray) -> tuple[CameraModel, dict, dict]:
    edges = [{"pts": _segment_samples(e["segments"]) if "segments" in e else np.asarray(e["points"], float),
              "axis": e["axis"], "value": e["value"], "z": 0.0} for e in cfg["edges"]]
    # single points (e.g. a far table corner) carry as much weight as a whole sampled edge
    for e in edges:
        if len(e["pts"]) < 5:
            e["pts"] = np.repeat(e["pts"], 15, axis=0)
    verts = [_segment_samples([s]) for s in cfg["verticals"]]
    has_y = any(e["axis"] == "Y" for e in edges)
    x1, y1, x2, y2 = cfg["table_height_edge"]["segment"]
    top, bot = ((x1, y1), (x2, y2)) if y1 < y2 else ((x2, y2), (x1, y1))
    kw = dict(verticals=verts, cam_xy_init=(0.5, 0.5), cam_z_init=1.8, fixed=() if has_y else ("cy",))
    fit = fit_from_edges(edges, (gray.shape[1], gray.shape[0]), **kw)
    free_height = float(fit.camera.center[2] - _vertical_bottom_z(fit.camera, top, bot))
    prior = cfg.get("camera_height_prior")
    if prior:   # the edges leave the camera height loosely determined: tie it softly to the measured value
        h_table = -_vertical_bottom_z(fit.camera, top, bot)
        fit = fit_from_edges(edges, (gray.shape[1], gray.shape[0]), cam_z_prior=prior["value_m"] - h_table,
                             prior_weight=prior["weight"], **kw)
    cam = fit.camera
    floor_z = _vertical_bottom_z(cam, top, bot)
    checks = {"edge_rms_px": round(fit.rms_px["edges"], 2), "vertical_rms_px": round(fit.rms_px["verticals"], 2),
              "f_px": round(fit.params["f_px"], 1), "k1": round(fit.params["k1"], 3), "k2": round(fit.params["k2"], 3),
              "camera_above_table_m": round(float(cam.center[2]), 3), "table_height_m": round(-floor_z, 3),
              "camera_above_floor_m": round(float(cam.center[2] - floor_z), 3),
              "camera_above_floor_without_prior_m": round(free_height, 3),
              "camera_above_floor_measured_m": cfg["checks"]["camera_height_m"],
              "camera_height_used_as_soft_constraint": bool(prior)}
    # independent check: long straight segments on the table plane near the other columns' edges
    lsd = cv2.createLineSegmentDetector(0).detect(gray)[0].reshape(-1, 4)
    found = []
    for s in lsd:
        if np.hypot(s[2] - s[0], s[3] - s[1]) < 40:
            continue
        P = cam.to_plane(s.reshape(2, 2), 0.0)
        if np.isnan(P).any():
            continue
        d = P[1] - P[0]
        if np.hypot(*d) > 0.4 and abs(d[0]) / np.hypot(*d) < 0.08:
            found.append((P[:, 0].mean(), np.hypot(*d)))
    found = np.array(found)
    for X in cfg["checks"]["column_edges_x_m"]:
        if any(abs(e["value"] - X) < 1e-6 for e in cfg["edges"]):
            continue
        m = np.abs(found[:, 0] - X) < 0.35 if len(found) else np.zeros(0, bool)
        checks[f"edge_x{X:.2f}_found_m"] = round(float(np.average(found[m, 0], weights=found[m, 1])), 3) if m.any() else None
    cam = cam.with_floor_at(floor_z)
    checks["camera_distance_from_front_wall_m"] = round(float(cam.center[1]), 3)
    return cam, checks, {"verticals": verts, "edges": edges, "floor_z": floor_z}


def draw(cam: CameraModel, img: np.ndarray, extras: dict, room: str, out: Path) -> None:
    vis = img.copy()
    W, H = cam.image_size
    # 1 m floor grid
    xs = np.arange(-12, 14, 1.0)
    for a in xs:
        for horiz in (True, False):
            t = np.linspace(-12, 16, 300)
            P = np.c_[t, np.full_like(t, a), np.zeros_like(t)] if horiz else np.c_[np.full_like(t, a), t, np.zeros_like(t)]
            ok = cam.in_view(P)
            for run in np.split(np.arange(len(P)), np.where(np.diff(ok.astype(int)) != 0)[0] + 1):
                if ok[run[0]] and len(run) > 1:
                    cv2.polylines(vis, [cam.project(P[run]).astype(np.int32)], False, (0, 200, 255), 1)
    if "positions" in extras:   # reprojected desk tops (at desk height above the floor)
        dz = extras["desk_z"]
        for n, (x, y) in extras["positions"].items():
            P = rect_corners(x, y, extras["length"], extras["depth"])
            P[:, 2] = dz
            cv2.polylines(vis, [cam.project(P).astype(np.int32)], True, (0, 0, 255), 2)
    for ln in extras.get("lines", []):
        for x, y in ln:
            cv2.circle(vis, (int(x), int(y)), 2, (255, 0, 255), -1)
    cv2.putText(vis, f"{room}: yellow = 1 m floor grid, red = fitted desk tops", (20, 40), 0, 1.0, (0, 255, 255), 2)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), cv2.resize(vis, None, fx=0.5, fy=0.5), [cv2.IMWRITE_JPEG_QUALITY, 80])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--configs", type=Path, nargs="*", default=sorted(Path("configs/calibration").glob("*.yaml")))
    ap.add_argument("--out-dir", type=Path, default=Path("configs/cameras"))
    ap.add_argument("--report", type=Path, default=Path("results/calibration.md"))
    ap.add_argument("--head-height", type=float, default=1.20, help="seated head height above the floor (m)")
    a = ap.parse_args()
    rows = []
    for path in a.configs:
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        img = cv2.imread(cfg["frame"])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if cfg["method"] == "rectangles":
            cam, checks, extras = calibrate_rectangles(cfg, gray)
            extras["desk_z"] = checks["desk_height_m"]
        else:
            cam, checks, extras = calibrate_edges(cfg, gray)
        room = cfg["room"]
        a.out_dir.mkdir(parents=True, exist_ok=True)
        out = {"room": room, "cameras": cfg["cameras"], "source": str(path).replace("\\", "/"),
               "generated": time.strftime("%Y-%m-%d %H:%M"), "approximate": True,
               "frame_note": "world metres, Z up, floor at Z = 0; see source config for the axes",
               "head_height_m": a.head_height, "camera": cam.to_dict(), "checks": checks}
        (a.out_dir / f"{room}.yaml").write_text(yaml.safe_dump(out, sort_keys=False, default_flow_style=None, width=140),
                                                encoding="utf-8")
        draw(cam, img, extras, room, Path("results/figures") / f"calibration_{room}.jpg")
        rows.append((room, cfg["cameras"], checks))
        print(room, checks, flush=True)
    md = ["# Camera calibration (T2.3–T2.5)", "",
          f"Generated {time.strftime('%Y-%m-%d %H:%M')} by `scripts/calibrate_cameras.py`. Full camera model per room "
          "(focal length, radial lens distortion k1/k2, position and orientation), floor at Z = 0. Inputs: "
          "`configs/calibration/<room>.yaml`; outputs: `configs/cameras/<room>.yaml`. All values approximate.", ""]
    for room, cams, ch in rows:
        md += [f"## {room} ({', '.join(cams)})", "", "| Quantity | Value |", "| --- | --- |"]
        md += [f"| {k} | {v} |" for k, v in ch.items()]
        md += ["", f"![{room}](figures/calibration_{room}.jpg)", ""]
    a.report.write_text("\n".join(md) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

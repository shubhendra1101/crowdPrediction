"""Per-zone crowd density (persons/mÂ²) on the labelled CCTV frames: labels vs CLIP-EBC (T2.4/T3.x).

For each labelled frame: heads (user's CVAT points) are placed on the floor plan through the camera
model at head height and counted per zone; CLIP-EBC's density map is summed inside each zone's image
mask. Both are divided by the zone's visible area. The same exclusion regions as the counting
evaluation (configs/roi.yaml) are removed from labels and density alike.

    python scripts/zone_density.py --set data/cctv/labels/heads_corrected.zip data/cctv/labels/images data/cctv/labels/frames.csv \
        --set data/cctv/labels_newcams/newcams_corrected.zip data/cctv/labels_newcams/images data/cctv/labels_newcams/frames.csv
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crowdsafe import nbenv  # noqa: E402

nbenv.limit_threads(nbenv.cpu_limit(default=4))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from crowdsafe.calibration.zones import assign_points, camera_rooms, load_room, visible_area_m2, zone_masks  # noqa: E402
from crowdsafe.datasets.cvat import read_cvat_points  # noqa: E402
from scripts.eval_cctv_counting import density_mask, load_roi, outside  # noqa: E402


def density_maps(frames: list[Path], onnx: Path, cache: Path, batch: int) -> dict[str, np.ndarray]:
    """CLIP-EBC density map per frame (cached as float16 .npz)."""
    have = dict(np.load(cache)) if cache.exists() else {}
    todo = [f for f in frames if f.name not in have]
    if todo:
        from crowdsafe.counting.density_model import ClipEbcDensity
        model = ClipEbcDensity(backend="onnx", onnx_path=onnx, device="cpu", max_batch=batch)
        for i, f in enumerate(todo, 1):
            have[f.name] = model(cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)).astype(np.float16)
            if i % 10 == 0 or i == len(todo):
                print(f"  density maps {i}/{len(todo)}", flush=True)
                cache.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(cache, **have)
    return {k: v.astype(np.float32) for k, v in have.items()}


def zone_figure(cam, zones: dict, hz: float, image: Path, heads: np.ndarray, out: Path) -> None:
    """Zone outlines at head height on one frame; labelled heads coloured by their zone (white = no zone)."""
    vis = cv2.imread(str(image))
    rng = np.random.default_rng(42)
    colours = {n: tuple(int(c) for c in rng.integers(60, 255, 3)) for n in zones}
    for n, poly in zones.items():
        P = np.asarray(poly, float)
        edge = np.vstack([a + np.linspace(0, 1, 60, endpoint=False)[:, None] * (b - a)
                          for a, b in zip(P, np.roll(P, -1, axis=0))])
        xyz = np.c_[edge, np.full(len(edge), hz)]
        ok = cam.in_view(xyz)
        for run in np.split(np.arange(len(xyz)), np.where(np.diff(ok.astype(int)) != 0)[0] + 1):
            if ok[run[0]] and len(run) > 1:
                cv2.polylines(vis, [cam.project(xyz[run]).astype(np.int32)], False, colours[n], 3)
        inner = xyz[ok]
        if len(inner):
            c = cam.project(np.r_[[np.r_[P.mean(0), hz]]])[0] if cam.in_view(np.r_[[np.r_[P.mean(0), hz]]])[0] \
                else cam.project(inner).mean(0)
            cv2.putText(vis, n, tuple(c.astype(int)), 0, 0.9, colours[n], 2)
    for (u, v), n in zip(heads, assign_points(cam, zones, heads, hz)):
        cv2.circle(vis, (int(u), int(v)), 7, colours.get(n, (255, 255, 255)), -1)
        cv2.circle(vis, (int(u), int(v)), 7, (0, 0, 0), 1)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), cv2.resize(vis, None, fx=0.5, fy=0.5), [cv2.IMWRITE_JPEG_QUALITY, 80])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--set", nargs=3, action="append", metavar=("LABELS", "IMAGES", "FRAMES_CSV"), required=True)
    ap.add_argument("--onnx", type=Path, default=Path("weights/crowdsafe_nb01_onnx/onnx/clipebc_vitb16_nwpu.onnx"))
    ap.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    ap.add_argument("--roi", type=Path, default=Path("configs/roi.yaml"))
    ap.add_argument("--head-heights", type=float, nargs="*", default=[1.2, 1.1, 1.3, 1.6],
                    help="first = main result; the others are a sensitivity check")
    ap.add_argument("--min-area", type=float, default=2.0, help="ignore zones with less visible area (mÂ²)")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--out", type=Path, default=Path("results/zone_density.md"))
    a = ap.parse_args()
    roi, rooms = load_roi(a.roi), camera_rooms()
    rows = []
    for labels, images, frames_csv in a.set:
        gt = read_cvat_points(Path(labels))
        cams = pd.read_csv(frames_csv).set_index("image")
        frames = sorted(Path(images) / n for n in gt if (Path(images) / n).exists())
        maps = density_maps(frames, a.onnx, a.cache_dir / f"clipebc_maps_{Path(labels).stem}.npz", a.batch)
        for room in sorted({rooms[c] for c in cams.loc[[f.name for f in frames], "camera"]}):
            cam, zones, _ = load_room(room)
            fr = [f for f in frames if rooms[cams.loc[f.name, "camera"]] == room]
            shape = maps[fr[0].name].shape
            busiest = max(fr, key=lambda f: len(gt[f.name]))
            zone_figure(cam, zones, a.head_heights[0], busiest, gt[busiest.name],
                        Path("results/figures") / f"zones_{room}.jpg")
            for hz in a.head_heights:
                masks = zone_masks(cam, zones, hz, shape)
                areas = {n: visible_area_m2(cam, p, hz) for n, p in zones.items()}
                for f in fr:
                    camid = cams.loc[f.name, "camera"]
                    polys = roi.get(camid, [])
                    P = gt[f.name]
                    P = P[outside(P, polys)]
                    D = maps[f.name]
                    if polys:
                        D = D * density_mask(D.shape, (cam.image_size[1], cam.image_size[0]), polys)
                    hit = pd.Series(assign_points(cam, zones, P, hz), dtype=object)
                    for n in zones:
                        rows.append({"image": f.name, "camera": camid, "room": room, "head_height_m": hz, "zone": n,
                                     "visible_area_m2": round(areas[n], 2), "gt_count": int((hit == n).sum()),
                                     "pred_count": float((D * masks[n]).sum()), "gt_unassigned": int(hit.isna().sum())})
                print(f"{room} head {hz} m: {len(fr)} frames done", flush=True)
    df = pd.DataFrame(rows)
    df["gt_density"] = df["gt_count"] / df["visible_area_m2"].where(df["visible_area_m2"] >= a.min_area)
    df["pred_density"] = df["pred_count"] / df["visible_area_m2"].where(df["visible_area_m2"] >= a.min_area)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out.with_name(a.out.stem + "_per_zone.csv"), index=False)
    main_h = a.head_heights[0]
    d = df[(df["head_height_m"] == main_h) & df["gt_density"].notna()]
    err = (d["pred_density"] - d["gt_density"])

    def summ(g):
        e = g["pred_density"] - g["gt_density"]
        return pd.Series({"zone_frames": len(g), "mean_gt_density": round(g["gt_density"].mean(), 3),
                          "max_gt_density": round(g["gt_density"].max(), 3), "mae_density": round(e.abs().mean(), 3),
                          "bias_density": round(e.mean(), 3), "mae_count": round((g["pred_count"] - g["gt_count"]).abs().mean(), 2)})

    by_room = d.groupby("room").apply(summ, include_groups=False).reset_index()
    by_zone = d.groupby(["room", "zone"]).apply(summ, include_groups=False).reset_index()
    sens = df[df["gt_density"].notna()].groupby(["room", "head_height_m"]).apply(summ, include_groups=False).reset_index()
    assigned = df[df["head_height_m"] == main_h].groupby(["room", "image"]).agg(gt=("gt_count", "sum"), un=("gt_unassigned", "first"))
    cover = assigned.groupby("room").apply(lambda g: round(100 * g["gt"].sum() / (g["gt"].sum() + g["un"].sum()), 1))
    md = ["# Zone density in persons/mÂ² (T2.4, T3 counting â†’ density)", "",
          f"Generated {time.strftime('%Y-%m-%d %H:%M')} by `scripts/zone_density.py`. Ground truth = user's corrected CVAT head "
          "points placed on the floor plan through the calibrated camera at head height; prediction = CLIP-EBC density map "
          f"summed in each zone's image mask. Density = count / visible zone area; zones with < {a.min_area} mÂ² visible are skipped. "
          f"Main head height {main_h} m (seated). Calibration and zone boundaries are approximate (`results/calibration.md`).", "",
          "Share of labelled heads that fall inside a zone (the rest are outside the zoned floor area, e.g. beyond the last row): "
          + ", ".join(f"{r} {v} %" for r, v in cover.items()), "",
          "## Per room", "", "| Room | zoneÃ—frames | mean true density | max true density | MAE density (persons/mÂ²) | bias | MAE count per zone |",
          "| --- | --- | --- | --- | --- | --- | --- |"]
    md += [f"| {r.room} | {int(r.zone_frames)} | {r.mean_gt_density} | {r.max_gt_density} | {r.mae_density} | {r.bias_density} | {r.mae_count} |"
           for r in by_room.itertuples()]
    md += ["", "## Per zone", "", "| Room | Zone | n | mean true | max true | MAE | bias |", "| --- | --- | --- | --- | --- | --- | --- |"]
    md += [f"| {r.room} | {r.zone} | {int(r.zone_frames)} | {r.mean_gt_density} | {r.max_gt_density} | {r.mae_density} | {r.bias_density} |"
           for r in by_zone.itertuples()]
    md += ["", "## Sensitivity to the assumed head height", "", "| Room | head height (m) | MAE density | mean true density |", "| --- | --- | --- | --- |"]
    md += [f"| {r.room} | {r.head_height_m} | {r.mae_density} | {r.mean_gt_density} |" for r in sens.itertuples()]
    a.out.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(by_room.to_string(index=False))
    print(sens.to_string(index=False))
    print(cover.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

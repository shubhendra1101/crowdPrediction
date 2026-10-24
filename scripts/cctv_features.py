"""CCTV video -> zone feature table, one row per zone per second (T4.2, T5.3 CCTV part).

Per sampled second: CLIP-EBC density map -> people per zone (zone masks at head height from the
calibrated camera, configs/zones/) -> persons/m² over the visible zone area; RAFT flow to the frame
`pair_gap_s` later -> zone motion in m/s (crowdsafe.motion.zone_motion). Exclusion regions
(configs/roi.yaml) are removed. inflow/outflow are not measured from CCTV yet (NaN).

Resumable: progress is saved every `--checkpoint-s` seconds of video; a re-run continues.

    python scripts/cctv_features.py --videos cam02=data/cctv/cam02.asf --clipebc-repo <dir>      # A100
    python scripts/cctv_features.py --videos cam04=data/cctv/cam04.asf --density-backend onnx \
        --flow-backend farneback --max-seconds 5                                                   # laptop test
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

from crowdsafe.calibration.zones import camera_rooms, load_room, visible_area_m2, zone_masks  # noqa: E402
from crowdsafe.datasets.cctv import sample_frames, video_info  # noqa: E402
from crowdsafe.motion.flow import FlowModel  # noqa: E402
from crowdsafe.motion.zone_motion import ZoneMotion  # noqa: E402
from scripts.eval_cctv_counting import density_mask, load_roi  # noqa: E402


def build_density(a):
    from crowdsafe.counting.density_model import ClipEbcDensity
    if a.density_backend == "torch":
        repo = Path(a.clipebc_repo)
        return ClipEbcDensity(backend="torch", repo_dir=repo, weights_dir=repo / "nwpu_weights" / "CLIP_EBC_ViT_B_16",
                              device=a.device, max_batch=a.batch)
    return ClipEbcDensity(backend="onnx", onnx_path=a.onnx, device="cpu" if a.device == "cpu" else a.device, max_batch=a.batch)


def process(camera: str, video: Path, a, density, flow) -> Path:
    out = a.out_dir / f"{camera}.parquet"
    part = a.out_dir / f"{camera}.partial.parquet"
    if out.exists() and not a.overwrite:
        print(f"{camera}: done already ({out})", flush=True)
        return out
    room = camera_rooms()[camera]
    cam, zones, hz = load_room(room)
    info = video_info(video)
    fps = info.fps_measured or info.fps_claimed
    rows = pd.read_parquet(part).to_dict("records") if part.exists() and not a.overwrite else []
    done_t = max((r["timestamp"] for r in rows), default=-1.0)
    polys = load_roi(a.roi).get(camera, [])
    masks = motion = areas = roi = None
    t0, n, last_ck = time.time(), 0, done_t
    max_frames = int((a.start_s + a.max_seconds) / a.every_s) + 1 if a.max_seconds else None
    print(f"{camera}: room {room}, {fps:.2f} fps, {info.duration_s:.0f} s; resuming after t={done_t:.0f}" if rows
          else f"{camera}: room {room}, {fps:.2f} fps, {info.duration_s:.0f} s", flush=True)
    for t, frame, later in sample_frames(video, a.every_s, fps, max_frames=max_frames, pair_gap_s=a.pair_gap_s):
        if t <= done_t or t < a.start_s:
            continue
        D = density(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).astype(np.float32)
        if masks is None:
            masks = zone_masks(cam, zones, hz, D.shape)
            areas = {z: visible_area_m2(cam, p, hz) for z, p in zones.items()}
            motion = ZoneMotion(cam, masks, hz)
            roi = density_mask(D.shape, (cam.image_size[1], cam.image_size[0]), polys) if polys else None
        if roi is not None:
            D = D * roi
        mot = motion(flow(frame, later), D, a.pair_gap_s) if later is not None else {}
        for z in zones:
            c = float((D * masks[z]).sum())
            dens = c / areas[z] if areas[z] >= a.min_area else np.nan
            m = mot.get(z, {})
            sv = m.get("speed_var", np.nan)
            rows.append({"timestamp": float(t), "camera_id": camera, "zone_id": z, "count": c, "density": dens,
                         "fusion_weight": np.nan, "mean_speed": m.get("mean_speed", np.nan), "speed_var": sv,
                         "pressure": dens * sv if np.isfinite(sv) else np.nan, "dir_entropy": m.get("dir_entropy", np.nan),
                         "counterflow": m.get("counterflow", np.nan), "inflow": np.nan, "outflow": np.nan, "source": "real"})
        n += 1
        if t - last_ck >= a.checkpoint_s:
            pd.DataFrame(rows).to_parquet(part, index=False)
            last_ck = t
            print(f"  {camera} t={t:.0f}s  {n / (time.time() - t0):.2f} frames/s", flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(out, index=False)
    part.unlink(missing_ok=True)
    print(f"{camera}: {n} new seconds, {len(df)} rows -> {out} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--videos", nargs="+", required=True, help="camera=path, e.g. cam02=data/cctv/cam02.asf")
    ap.add_argument("--out-dir", type=Path, default=Path("data/features/cctv"))
    ap.add_argument("--every-s", type=float, default=1.0)
    ap.add_argument("--pair-gap-s", type=float, default=0.2, help="flow between a frame and the one this much later")
    ap.add_argument("--density-backend", choices=["torch", "onnx"], default="torch")
    ap.add_argument("--clipebc-repo", type=Path, help="local snapshot of HF Yiming-M/CLIP-EBC (torch backend)")
    ap.add_argument("--onnx", type=Path, default=Path("weights/crowdsafe_nb01_onnx/onnx/clipebc_vitb16_nwpu.onnx"))
    ap.add_argument("--flow-backend", choices=["raft", "farneback"], default="raft")
    ap.add_argument("--flow-width", type=int, default=960)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--roi", type=Path, default=Path("configs/roi.yaml"))
    ap.add_argument("--min-area", type=float, default=2.0, help="zones with less visible area get density NaN (m²)")
    ap.add_argument("--max-seconds", type=float, default=None, help="only N seconds (tests)")
    ap.add_argument("--start-s", type=float, default=0.0, help="skip video before this time (tests)")
    ap.add_argument("--checkpoint-s", type=float, default=300.0)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    density = build_density(a)
    flow = FlowModel(a.flow_backend, a.flow_width, a.device)
    for spec in a.videos:
        camera, path = spec.split("=", 1)
        process(camera, Path(path), a, density, flow)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

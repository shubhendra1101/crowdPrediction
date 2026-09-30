"""Export one pretrained model to ONNX and check it against PyTorch (T0.6). One step per process.

Run by notebook 01 in a subprocess, so a crash or out-of-memory kill cannot take down the kernel.
Threads are capped to the container's CPU quota *before* numpy/torch are imported (the lab
container sees 256 host cores but may use only a few).

    python scripts/export_onnx.py yolo    --onnx-dir OUT/onnx --cache CACHE --test-image bus.jpg --result r.json
    python scripts/export_onnx.py raft    ...
    python scripts/export_onnx.py clipebc ...
    python scripts/export_onnx.py phd     ...
    python scripts/export_onnx.py timing  ...
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crowdsafe import nbenv  # noqa: E402  (stdlib only)

N_THREADS = nbenv.limit_threads()

import hashlib  # noqa: E402
import inspect  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402

import numpy as np  # noqa: E402

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ort_session(path: Path):
    """CPU onnxruntime session limited to the container's CPU quota."""
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = N_THREADS
    so.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])


def torch_setup():
    import torch

    torch.set_num_threads(N_THREADS)
    torch.manual_seed(42)
    if hasattr(torch.backends, "mha") and hasattr(torch.backends.mha, "set_fastpath_enabled"):
        torch.backends.mha.set_fastpath_enabled(False)      # fused attention cannot be exported
    return torch


def onnx_export(torch, model, args: tuple, path: Path, opset: int, **kw) -> str:
    """TorchScript exporter at opset, then opset 18, then dynamo; returns which worked."""
    has_dynamo = "dynamo" in inspect.signature(torch.onnx.export).parameters
    attempts = [("torchscript", opset), ("torchscript", 18)] + ([("dynamo", 18)] if has_dynamo else [])
    errors = {}
    for how, op in attempts:
        try:
            extra = {"dynamo": how == "dynamo"} if has_dynamo else {}
            torch.onnx.export(model, args, str(path), opset_version=op, do_constant_folding=True, **extra, **kw)
            return f"{how}/opset{op}"
        except Exception as e:
            errors[f"{how}/opset{op}"] = f"{type(e).__name__}: {str(e)[:400]}"
            print(f"export {how}/opset{op} failed: {errors[f'{how}/opset{op}'][:200]}", flush=True)
    raise RuntimeError(f"All ONNX export attempts failed: {errors}")


def save_meta(onnx_dir: Path, name: str, meta: dict) -> None:
    (onnx_dir / f"{name}.json").write_text(json.dumps(meta, indent=2, default=str))


def read_bgr(path: Path):
    import cv2

    cv2.setNumThreads(N_THREADS)
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    return img


# ----------------------------------------------------------------------------------- steps
def step_yolo(a: argparse.Namespace) -> dict:
    os.environ["YOLO_AUTOINSTALL"] = "false"
    torch = torch_setup()
    from ultralytics import YOLO

    bgr = read_bgr(a.test_image)
    src = Path(YOLO(a.yolo_weights).export(format="onnx", imgsz=a.yolo_export_imgsz, dynamic=True,
                                           simplify=not a.no_simplify, opset=a.opset, device="cpu"))
    dst = a.onnx_dir / "yolo11l.onnx"
    shutil.copy(src, dst)
    pt, ox = YOLO(a.yolo_weights), YOLO(str(dst), task="detect")
    checks = {}
    for sz in a.yolo_check_imgsz:
        kw = dict(imgsz=sz, classes=[0], conf=0.25, verbose=False)
        ab = pt.predict(bgr, device=0 if torch.cuda.is_available() else "cpu", **kw)[0].boxes.xyxy.cpu().numpy()
        bb = ox.predict(bgr, device="cpu", **kw)[0].boxes.xyxy.cpu().numpy()
        miou = None
        if len(ab) and len(bb):
            tl = np.maximum(ab[:, None, :2], bb[None, :, :2])
            br = np.minimum(ab[:, None, 2:], bb[None, :, 2:])
            inter = np.prod(np.clip(br - tl, 0, None), axis=2)
            area = lambda x: np.prod(x[:, 2:] - x[:, :2], axis=1)
            miou = float((inter / (area(ab)[:, None] + area(bb)[None, :] - inter + 1e-9)).max(axis=1).mean())
        checks[f"imgsz_{sz}"] = {"persons_pytorch": len(ab), "persons_onnx": len(bb), "mean_best_iou": miou}
    save_meta(a.onnx_dir, "yolo11l", {"source": f"Ultralytics {a.yolo_weights} (COCO)", "opset": a.opset,
                                      "simplified": not a.no_simplify,
                                      "input": "use via ultralytics YOLO('yolo11l.onnx', task='detect'); dynamic HxW (multiple of 32)",
                                      "person_class": 0, "sha256": sha256(dst)})
    return {"file": dst.name, "size_mb": round(dst.stat().st_size / 1e6, 1), "simplified": not a.no_simplify, "checks": checks}


def step_raft(a: argparse.Namespace) -> dict:
    torch = torch_setup()
    import cv2
    from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

    weights_note = nbenv.ensure_raft_large_weights()

    class RaftONNX(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).eval()

        def forward(self, x, y):
            return self.net(x * 2 - 1, y * 2 - 1, num_flow_updates=a.raft_iters)[-1]

    rgb = cv2.cvtColor(read_bgr(a.test_image), cv2.COLOR_BGR2RGB)
    f1 = cv2.resize(rgb, (a.raft_w, a.raft_h))
    dx, dy = a.raft_shift
    f2 = np.roll(f1, shift=(dy, dx), axis=(0, 1))
    t = lambda im: torch.from_numpy(im).permute(2, 0, 1)[None].float() / 255.0
    x, y = t(f1), t(f2)
    model = RaftONNX().eval()
    dst = a.onnx_dir / "raft_large.onnx"
    with torch.no_grad():
        how = onnx_export(torch, model, (x, y), dst, a.opset, input_names=["frame_a", "frame_b"], output_names=["flow"])
        ref = model(x, y).numpy()
    got = ort_session(dst).run(None, {"frame_a": x.numpy(), "frame_b": y.numpy()})[0]
    m = 32
    centre = lambda f: [float(np.median(f[0, c, m:-m, m:-m])) for c in (0, 1)]
    save_meta(a.onnx_dir, "raft_large", {"source": "torchvision raft_large, Raft_Large_Weights.DEFAULT (C_T_SKHT_V2)",
                                         "weights": weights_note, "exporter": how,
                                         "inputs": {"frame_a": [1, 3, a.raft_h, a.raft_w], "frame_b": [1, 3, a.raft_h, a.raft_w]},
                                         "input_format": f"RGB float32 in [0,1], NCHW; resize frames to W={a.raft_w}, H={a.raft_h}",
                                         "output": "flow 1x2xHxW, pixels, channel 0 = x, 1 = y", "flow_updates": a.raft_iters,
                                         "sha256": sha256(dst)})
    return {"file": dst.name, "exporter": how, "size_mb": round(dst.stat().st_size / 1e6, 1), "weights": weights_note,
            "checks": {"true_shift_px": [dx, dy], "median_flow_pytorch": centre(ref), "median_flow_onnx": centre(got),
                       "max_abs_diff_px": float(np.abs(ref - got).max()), "mean_abs_diff_px": float(np.abs(ref - got).mean())}}


def step_clipebc(a: argparse.Namespace) -> dict:
    torch = torch_setup()
    import cv2
    from huggingface_hub import snapshot_download

    from crowdsafe.counting.density_model import load_clipebc_torch, sliding_window

    repo = Path(snapshot_download(a.clipebc_repo_id, ignore_patterns=["nwpu_weights/CLIP_EBC_ViT_L_14/*"],
                                  local_dir=a.cache / "CLIP-EBC-hf"))
    cwd = os.getcwd()
    os.chdir(repo)
    try:
        net, cfg = load_clipebc_torch(repo, repo / a.clipebc_weights)
    finally:
        os.chdir(cwd)
    net = net.cpu().eval()

    class ClipEbcONNX(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = net
            self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

        def forward(self, x):
            return self.net((x - self.mean) / self.std)

    model = ClipEbcONNX().eval()
    win, red = cfg["input_size"], cfg["reduction"]
    dst = a.onnx_dir / "clipebc_vitb16_nwpu.onnx"
    with torch.no_grad():
        how = onnx_export(torch, model, (torch.rand(2, 3, win, win),), dst, a.opset, input_names=["image"],
                          output_names=["density"], dynamic_axes={"image": {0: "batch"}, "density": {0: "batch"}})
    sess = ort_session(dst)

    def run_torch(b):
        with torch.no_grad():
            return model(torch.from_numpy(b)).numpy()

    checks = {}
    for name in a.clipebc_test_images:
        p = repo / name
        if not p.exists():
            checks[name] = "image not in repo snapshot"
            continue
        rgb01 = cv2.cvtColor(read_bgr(p), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        c = {"size_hw": list(rgb01.shape[:2])}
        d_t = sliding_window(run_torch, rgb01, win, red, max_batch=16)
        d_o = sliding_window(lambda b: sess.run(None, {"image": b})[0], rgb01, win, red, max_batch=16)
        c.update(count_window_pytorch=float(d_t.sum()), count_window_onnx=float(d_o.sum()),
                 max_abs_diff_density=float(np.abs(d_t - d_o).max()))
        if torch.cuda.is_available():     # whole-image (app.py style) on the GPU: large token counts
            try:
                gm = model.cuda()
                with torch.no_grad():
                    c["count_whole_image_app_py"] = float(gm(torch.from_numpy(rgb01).permute(2, 0, 1)[None].cuda()).sum())
                model.cpu()
            except Exception as e:
                c["count_whole_image_app_py"] = f"not run: {e!r}"[:300]
        checks[name] = c
    save_meta(a.onnx_dir, "clipebc_vitb16_nwpu", {"source": f"Hugging Face {a.clipebc_repo_id}/{a.clipebc_weights}",
                                                  "config": cfg, "exporter": how,
                                                  "input": f"image Nx3x{win}x{win} RGB float32 in [0,1] (ImageNet normalisation built in)",
                                                  "output": f"density Nx1x{win // red}x{win // red}; sum = count",
                                                  "sliding_window": f"non-overlapping {win}px windows over zero-padded frame",
                                                  "sha256": sha256(dst)})
    return {"file": dst.name, "exporter": how, "size_mb": round(dst.stat().st_size / 1e6, 1), "config": cfg, "checks": checks}


def step_phd(a: argparse.Namespace) -> dict:
    from huggingface_hub import hf_hub_download

    from crowdsafe.counting.yolo_track import PersonHeadOnnx

    src = Path(hf_hub_download("Sharath33/Person", "yolov11_phd_s.onnx", local_dir=a.cache / "phd"))
    dst = a.onnx_dir / "yolo11s_crowdhuman_person_head.onnx"
    shutil.copy(src, dst)
    det = PersonHeadOnnx(dst, conf=0.2, iou=0.6)
    io = {"inputs": [(i.name, i.shape) for i in det.sess.get_inputs()],
          "outputs": [(o.name, o.shape) for o in det.sess.get_outputs()]}
    d = det(read_bgr(a.test_image))
    checks = {"test_image": {"person": int((d.class_id == 0).sum()), "head": int((d.class_id == 1).sum())}}
    save_meta(a.onnx_dir, "yolo11s_crowdhuman_person_head", {"source": "Hugging Face Sharath33/Person/yolov11_phd_s.onnx",
                                                             "classes": {0: "person", 1: "head"}, "io": io,
                                                             "input_format": "BGR, /255, letterbox pad 114, NCHW",
                                                             "defaults": {"conf": 0.2, "nms_iou": 0.6},
                                                             "license": "openrail++", "sha256": sha256(dst)})
    return {"file": dst.name, "size_mb": round(dst.stat().st_size / 1e6, 1), "io": io, "checks": checks}


def step_timing(a: argparse.Namespace) -> dict:
    torch = torch_setup()
    if not torch.cuda.is_available():
        return {"skipped": "no GPU visible"}
    import cv2
    from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
    from ultralytics import YOLO

    def time_it(fn, n=20):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(n):
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1000)
        return float(np.median(ts))

    out = {}
    nbenv.ensure_raft_large_weights()
    raft = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).cuda().eval()
    x = torch.rand(1, 3, a.raft_h, a.raft_w, device="cuda")
    with torch.no_grad():
        out["raft_large_544x960_ms"] = time_it(lambda: raft(x, x, num_flow_updates=a.raft_iters))
    del raft
    yolo = YOLO(a.yolo_weights)
    frame = cv2.resize(read_bgr(a.test_image), (1920, 1080))
    out["yolo11l_1080p_imgsz1280_ms"] = time_it(lambda: yolo.predict(frame, imgsz=1280, device=0, classes=[0], verbose=False), 10)
    out["peak_gpu_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    return out


STEPS = {"yolo": step_yolo, "raft": step_raft, "clipebc": step_clipebc, "phd": step_phd, "timing": step_timing}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("step", choices=list(STEPS))
    ap.add_argument("--onnx-dir", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--test-image", type=Path, required=True)
    ap.add_argument("--result", type=Path, required=True)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--yolo-weights", default="yolo11l.pt")
    ap.add_argument("--yolo-export-imgsz", type=int, default=1280)
    ap.add_argument("--yolo-check-imgsz", type=int, nargs="*", default=[640, 1280])
    ap.add_argument("--no-simplify", action="store_true")
    ap.add_argument("--raft-h", type=int, default=544)
    ap.add_argument("--raft-w", type=int, default=960)
    ap.add_argument("--raft-iters", type=int, default=12)
    ap.add_argument("--raft-shift", type=int, nargs=2, default=[6, 3])
    ap.add_argument("--clipebc-repo-id", default="Yiming-M/CLIP-EBC")
    ap.add_argument("--clipebc-weights", default="nwpu_weights/CLIP_EBC_ViT_B_16")
    ap.add_argument("--clipebc-test-images", nargs="*", default=["example1.jpg", "example2.jpg"])
    a = ap.parse_args()
    a.onnx_dir.mkdir(parents=True, exist_ok=True)
    a.cache.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    res = STEPS[a.step](a)
    res.update(status="ok", seconds=round(time.time() - t0, 1), threads=N_THREADS)
    a.result.parent.mkdir(parents=True, exist_ok=True)
    a.result.write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps(res, default=str)[:2000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

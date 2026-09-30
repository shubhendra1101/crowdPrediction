"""Smoke tests for counting wrappers, ShanghaiTech loader, metrics and the benchmark loop (no real models)."""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
from scipy.io import savemat

from crowdsafe.counting.density_model import sliding_window
from crowdsafe.counting.yolo_track import decode_yolo, letterbox, reliability_features
from crowdsafe.datasets.shanghaitech import list_split, load_points
from crowdsafe.eval.metrics import band_labels, count_metrics

ROOT = Path(__file__).resolve().parents[1]


def _write_sample(d: Path, idx: int, points: np.ndarray) -> None:
    (d / "images").mkdir(parents=True, exist_ok=True)
    (d / "ground-truth").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(d / "images" / f"IMG_{idx}.jpg"), np.zeros((60, 80, 3), np.uint8))
    info = np.empty((1, 1), dtype=object)                      # official nesting: image_info[0,0][0,0][0]
    inner = np.empty((1, 1), dtype=[("location", object), ("number", object)])
    inner[0, 0] = (points, np.array([[len(points)]]))
    info[0, 0] = inner
    savemat(d / "ground-truth" / f"GT_IMG_{idx}.mat", {"image_info": info})


@pytest.fixture()
def fake_sht(tmp_path: Path) -> Path:
    root = tmp_path / "ShanghaiTech"
    for part, counts in {"A": [3, 7], "B": [1]}.items():
        for i, n in enumerate(counts, start=1):
            _write_sample(root / f"part_{part}" / "test_data", i, np.random.default_rng(i).random((n, 2)) * 50)
    return tmp_path


def test_shanghaitech_loader(fake_sht: Path) -> None:
    a = list_split(fake_sht, "A", "test")
    assert [s.count for s in a] == [3, 7]
    assert load_points(next(fake_sht.rglob("GT_IMG_1.mat"))).shape[1] == 2


def test_metrics_and_bands() -> None:
    df = pd.DataFrame({"gt": [10, 100, 300], "m": [12, 90, 330]})
    df["band"] = band_labels(df["gt"].to_numpy(), [50, 200])
    assert list(df["band"]) == ["<50", "50-200", ">=200"]
    m = count_metrics(df, ["m"], "gt", "band").set_index("band")
    assert m.loc["all", "mae"] == pytest.approx((2 + 10 + 30) / 3, abs=0.01)
    assert m.loc["all", "bias"] == pytest.approx((2 - 10 + 30) / 3, abs=0.01)


def test_sliding_window_crops_and_preserves_sum() -> None:
    img = np.random.default_rng(42).random((100, 130, 3)).astype(np.float32)
    run = lambda b: b[:, :1].reshape(len(b), 1, 4, 8, 4, 8).sum(axis=(3, 5))   # 32px window, r=8
    out = sliding_window(run, img, 32, 8, max_batch=3)
    assert out.shape == (13, 17)
    assert out.sum() == pytest.approx(img[..., 0].sum(), rel=1e-5)


def test_letterbox_and_decode_roundtrip() -> None:
    canvas, s, left, top = letterbox(np.zeros((1080, 1920, 3), np.uint8), 640, 640)
    assert canvas.shape == (640, 640, 3) and left == 0 and top == 140
    # one head box of 30x30 px at (960, 540) in the original image, in canvas coordinates
    cx, cy, w = 960 * s + left, 540 * s + top, 30 * s
    preds = np.zeros((6, 3), np.float32)                      # 4 box + 2 class scores, 3 anchors
    preds[:, 0] = [cx, cy, w, w, 0.1, 0.9]
    preds[:, 1] = [cx + 1, cy, w, w, 0.1, 0.8]                 # duplicate -> removed by NMS
    preds[:, 2] = [10, 10, 5, 5, 0.05, 0.05]                   # below conf
    d = decode_yolo(preds, conf=0.2, iou=0.6, scale=s, left=left, top=top)
    assert len(d) == 1 and d.class_id[0] == 1
    np.testing.assert_allclose(d.xyxy[0], [945, 525, 975, 555], atol=0.5)


def test_reliability_features() -> None:
    import supervision as sv

    d = sv.Detections(xyxy=np.array([[0, 0, 10, 20], [1, 0, 11, 20], [50, 50, 60, 60]], np.float32),
                      confidence=np.array([0.9, 0.5, 0.7], np.float32))
    f = reliability_features(d)
    assert f["n_det"] == 3 and f["overlap_share"] == pytest.approx(2 / 3)
    assert f["mean_box_h"] == pytest.approx(50 / 3)
    assert reliability_features(sv.Detections.empty())["n_det"] == 0


def test_benchmark_loop_with_fake_model(fake_sht: Path, tmp_path: Path, monkeypatch) -> None:
    import scripts.bench_counting as bench

    monkeypatch.setattr(bench, "build_model", lambda name, spec, args: (lambda bgr: {name: 5.0}))
    cfg = {"seed": 42, "dataset": {"parts": ["A", "B"], "split": "test"}, "count_bands": [5],
           "models": {"fake": {"type": "x"}}}
    args = argparse.Namespace(data_root=fake_sht, models=None, limit=None, device="cpu")
    per_image, metrics, timing = bench.benchmark(cfg, args)
    assert len(per_image) == 3 and set(metrics["part"]) == {"A", "B"}
    a_all = metrics[(metrics.part == "A") & (metrics.band == "all")].iloc[0]
    assert a_all["mae"] == pytest.approx(2.0)                  # |5-3| and |5-7|
    out = tmp_path / "res" / "bench.md"
    bench.write_results(per_image, metrics, timing, out, {"finished": "now", "split": "test", "device": "cpu", "seed": 42})
    assert out.exists() and (tmp_path / "res" / "bench_per_image.csv").exists()

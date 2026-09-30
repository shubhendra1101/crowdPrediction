"""Smoke tests for notebook environment helpers and the ShanghaiTech Hugging Face fallback."""
import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from crowdsafe import nbenv
from crowdsafe.datasets.download import check_expect, convert_shanghaitech_parquet
from crowdsafe.datasets.shanghaitech import list_split


def test_constraints_pin_installed_torch_and_numpy() -> None:
    text = nbenv.constraints_text()
    assert f"numpy=={nbenv.installed('numpy')}" in text
    if nbenv.installed("torch"):
        assert f"torch=={nbenv.installed('torch')}" in text


def test_base_name_parsing() -> None:
    assert nbenv._base_name("timm==0.9.16") == "timm"
    assert nbenv._base_name("uvicorn[standard]>=0.30") == "uvicorn"
    assert nbenv._base_name("chronos_forecasting>=2.0 ; python_version>'3'") == "chronos-forecasting"


def test_cv2_check_runs() -> None:
    ok, msg = nbenv.cv2_ok()
    assert isinstance(ok, bool) and isinstance(msg, str)


def _jpeg() -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(buf, format="JPEG")
    return buf.getvalue()


def test_shanghaitech_parquet_fallback(tmp_path: Path) -> None:
    src = tmp_path / "_hf" / "data"
    src.mkdir(parents=True)
    img = _jpeg()
    for part, split, counts in [("part_A", "test", [5, 9]), ("part_B", "test", [3])]:
        df = pd.DataFrame({"file_name": [f"IMG_{i + 1}.jpg" for i in range(len(counts))],
                           "image": [{"bytes": img, "path": None}] * len(counts), "count": counts})
        df.to_parquet(src / f"{part}_{split}_data-00000-of-00001-abc.parquet")
    out = tmp_path / "shanghaitech_hf"
    assert convert_shanghaitech_parquet(tmp_path / "_hf", out) == 3
    samples = list_split(out, "A", "test")
    assert [s.count for s in samples] == [5, 9] and samples[0].points is None
    assert check_expect(out, {"**/part_A/test_data/images/*.jpg": 2})["**/part_A/test_data/images/*.jpg"]["match"]


def test_thread_limit_and_exit_reasons(monkeypatch) -> None:
    for v in nbenv.THREAD_VARS:
        monkeypatch.delenv(v, raising=False)
    n = nbenv.limit_threads(3)
    import os
    assert n == 3 and os.environ["OMP_NUM_THREADS"] == "3"
    assert nbenv.cpu_limit() >= 1
    assert "memory" in nbenv.exit_reason(-9) and "segmentation" in nbenv.exit_reason(-11)


def test_export_script_help_runs() -> None:
    import subprocess, sys
    root = Path(__file__).resolve().parents[1]
    r = subprocess.run([sys.executable, str(root / "scripts" / "export_onnx.py"), "--help"], capture_output=True, text=True)
    assert r.returncode == 0 and "yolo" in r.stdout and "clipebc" in r.stdout

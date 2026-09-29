"""Smoke tests for crowdsafe.datasets.download using local file:// URLs (no network)."""
import json
import zipfile
from pathlib import Path

import pytest

from crowdsafe.datasets.download import check_expect, fetch_dataset, load_registry, safe_extract, select

ROOT = Path(__file__).resolve().parents[1]


def _zip(path: Path, members: dict[str, str]) -> Path:
    with zipfile.ZipFile(path, "w") as z:
        for name, text in members.items():
            z.writestr(name, text)
    return path


def test_registry_parses_and_has_must_datasets() -> None:
    reg = load_registry(ROOT / "configs" / "datasets.yaml")
    must = select(reg, ["must"])
    assert "shanghaitech" in must
    assert any(k.startswith("julich_") for k in must)
    for name, entry in reg["datasets"].items():
        assert entry["source"] in {"julich", "kaggle", "http"}, name
        if entry["source"] != "kaggle":
            assert all(f["url"].startswith("http") for f in entry["files"]), name


def test_fetch_http_zip_roles_and_expect(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    traj = _zip(src / "traj.zip", {"run1.txt": "1 0 0.0 0.0\n", "run2.txt": "1 0 1.0 1.0\n"})
    video = _zip(src / "video.zip", {"v.mp4": "x"})
    entry = {"source": "julich", "files": [
        {"role": "trajectories", "url": traj.as_uri()},
        {"role": "video", "url": video.as_uri()}],
        "expect": {"**/trajectories/*.txt": 2}}
    res = fetch_dataset("demo", entry, tmp_path / "out", roles=["trajectories"], max_gb=5)
    assert res.status == "ok", res.error
    assert len(res.files) == 1 and res.files[0]["extracted_members"] == 2
    assert res.expect["**/trajectories/*.txt"]["match"]
    assert not (tmp_path / "out" / "demo" / "video").exists()
    manifest = json.loads((tmp_path / "out" / "demo" / "manifest.json").read_text())
    assert manifest["status"] == "ok" and len(manifest["files"][0]["sha256"]) == 64


def test_safe_extract_rejects_path_traversal(tmp_path: Path) -> None:
    bad = _zip(tmp_path / "bad.zip", {"../evil.txt": "x"})
    with pytest.raises(ValueError):
        safe_extract(bad, tmp_path / "out")


def test_check_expect_reports_mismatch(tmp_path: Path) -> None:
    (tmp_path / "a" / "images").mkdir(parents=True)
    (tmp_path / "a" / "images" / "1.jpg").write_text("x")
    out = check_expect(tmp_path, {"**/images/*.jpg": 3})
    assert out["**/images/*.jpg"] == {"expected": 3, "found": 1, "match": False}

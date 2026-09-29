"""Download and inventory datasets listed in configs/datasets.yaml (T1.1).

HTTP files and Kaggle datasets are fetched into ``<root>/<dataset>/``, zips are extracted,
and a ``manifest.json`` records what was downloaded (URL, bytes, SHA-256) and whether the
file counts match the official split sizes given under ``expect``.
"""
from __future__ import annotations

import fnmatch
import hashlib
import json
import shutil
import ssl
import time
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

CHUNK = 1 << 22


def _ssl_context() -> ssl.SSLContext:
    """Verified TLS context; uses certifi's CA bundle when available (some Python installs lack CAs)."""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


SSL_CTX = _ssl_context()


@dataclass
class FetchResult:
    """Outcome of fetching one dataset."""

    name: str
    status: str = "ok"                                 # ok | skipped | failed
    files: list[dict] = field(default_factory=list)
    expect: dict[str, dict] = field(default_factory=dict)
    error: str | None = None


def load_registry(path: Path) -> dict[str, Any]:
    """Read the dataset registry YAML."""
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def select(registry: dict, groups: list[str] | None = None, names: list[str] | None = None) -> dict[str, dict]:
    """Datasets filtered by group (must/should/...) and/or explicit names."""
    out = {}
    for name, entry in registry["datasets"].items():
        if names and name not in names:
            continue
        if groups and entry.get("group") not in groups:
            continue
        out[name] = entry
    return out


def remote_size(url: str, timeout: float = 30) -> int | None:
    """Content-Length of url via HEAD, or None if the server doesn't say."""
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=timeout, context=SSL_CTX) as r:
            n = r.headers.get("Content-Length")
            return int(n) if n else None
    except Exception:
        return None


def sha256(path: Path) -> str:
    """SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def download_http(url: str, dst: Path, timeout: float = 120) -> Path:
    """Stream url to dst via a .part file; skip if dst already exists."""
    if dst.exists():
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_name(dst.name + ".part")
    with urllib.request.urlopen(url, timeout=timeout, context=SSL_CTX) as r, open(part, "wb") as f:
        shutil.copyfileobj(r, f, CHUNK)
    part.replace(dst)
    return dst


def safe_extract(archive: Path, out_dir: Path) -> int:
    """Extract a zip into out_dir, refusing members that escape it. Returns member count."""
    out_dir = out_dir.resolve()
    with zipfile.ZipFile(archive) as z:
        for m in z.infolist():
            target = (out_dir / m.filename).resolve()
            if out_dir not in target.parents and target != out_dir:
                raise ValueError(f"Unsafe path in {archive.name}: {m.filename}")
        z.extractall(out_dir)
        return len(z.infolist())


def download_kaggle(ref: str, dst: Path) -> Path:
    """Download and unzip a Kaggle dataset (needs ~/.kaggle/kaggle.json or KAGGLE_* env vars)."""
    from kaggle.api.kaggle_api_extended import KaggleApi  # imported lazily: optional dependency

    api = KaggleApi()
    api.authenticate()
    dst.mkdir(parents=True, exist_ok=True)
    api.dataset_download_files(ref, path=str(dst), unzip=True, quiet=False)
    return dst


def check_expect(root: Path, expect: dict[str, int]) -> dict[str, dict]:
    """Count files matching each glob under root and compare with the expected number."""
    all_files = [p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()]
    out = {}
    for pattern, n in (expect or {}).items():
        found = sum(fnmatch.fnmatch(f, pattern) or fnmatch.fnmatch("x/" + f, pattern) for f in all_files)
        out[pattern] = {"expected": n, "found": found, "match": found == n}
    return out


def fetch_dataset(name: str, entry: dict, root: Path, roles: list[str] | None,
                  max_gb: float, extract: bool = True) -> FetchResult:
    """Fetch one registry entry into root/name and write its manifest.json."""
    res = FetchResult(name)
    out = root / name
    t0 = time.time()
    try:
        if entry["source"] == "kaggle":
            if roles and not set(roles) & {f.get("role") for f in entry.get("files", [])}:
                res.status = "skipped"
                return res
            size_gb = entry.get("size_mb", 0) / 1000
            if size_gb > max_gb:
                raise RuntimeError(f"{size_gb:.1f} GB > limit {max_gb} GB — ask the user first")
            download_kaggle(entry["kaggle"], out)
            res.files.append({"kaggle": entry["kaggle"]})
        else:
            for f in entry["files"]:
                if roles and f.get("role") not in roles:
                    continue
                n = remote_size(f["url"])
                if n is not None and n / 1e9 > max_gb:
                    raise RuntimeError(f"{f['url']} is {n / 1e9:.1f} GB > limit {max_gb} GB — ask the user first")
                dst = download_http(f["url"], out / "_archives" / Path(f["url"]).name)
                rec = {"url": f["url"], "role": f.get("role"), "bytes": dst.stat().st_size, "sha256": sha256(dst)}
                if extract and dst.suffix == ".zip":
                    rec["extracted_members"] = safe_extract(dst, out / f.get("role", "files"))
                res.files.append(rec)
            if not res.files:
                res.status = "skipped"
        res.expect = check_expect(out, entry.get("expect", {})) if res.status == "ok" else {}
    except Exception as e:  # keep going with other datasets; the error is recorded
        res.status, res.error = "failed", repr(e)
    if out.exists():
        manifest = {"dataset": name, "status": res.status, "error": res.error, "files": res.files,
                    "expect": res.expect, "doi": entry.get("doi"), "license": entry.get("license"),
                    "note": entry.get("note"), "seconds": round(time.time() - t0, 1),
                    "fetched": time.strftime("%Y-%m-%d %H:%M:%S")}
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return res

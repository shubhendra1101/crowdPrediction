"""Build .ipynb notebooks from percent-format sources in notebooks/src/.

A source file is plain Python split into cells by lines starting with ``# %%``.
``# %% [markdown]`` starts a Markdown cell whose lines have their leading ``# `` removed.

Usage:
    python scripts/build_notebooks.py [--src notebooks/src] [--out notebooks]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

KERNEL = {"display_name": "Python 3", "language": "python", "name": "python3"}


def parse_cells(text: str) -> list[dict]:
    """Split percent-format source text into notebook cell dicts."""
    cells: list[dict] = []
    kind, buf = None, []

    def flush() -> None:
        if kind is None:
            return
        while buf and not buf[-1].strip():
            buf.pop()
        if kind == "markdown":
            lines = [ln[2:] if ln.startswith("# ") else ln.lstrip("#") for ln in buf]
            cells.append({"cell_type": "markdown", "metadata": {}, "source": _join(lines)})
        else:
            cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                          "outputs": [], "source": _join(buf)})

    for line in text.splitlines():
        if line.startswith("# %%"):
            flush()
            kind, buf = ("markdown" if "[markdown]" in line else "code"), []
        elif kind is not None:
            if not buf and not line.strip():
                continue
            buf.append(line)
    flush()
    return cells


def _join(lines: list[str]) -> list[str]:
    """Notebook JSON stores source as lines that keep their newline, except the last."""
    return [ln + "\n" for ln in lines[:-1]] + lines[-1:]


PLACEHOLDER = "KAGGLE_CREDENTIALS = None"


def embed_kaggle(text: str, kaggle_json: Path) -> str:
    """Replace the credentials placeholder with the contents of kaggle.json (local builds only)."""
    creds = json.loads(Path(kaggle_json).read_text(encoding="utf-8"))
    return text.replace(PLACEHOLDER, f"KAGGLE_CREDENTIALS = {json.dumps({k: creds[k] for k in ('username', 'key')})}")


def build(src: Path, out_dir: Path, kaggle_json: Path | None = None) -> Path:
    """Convert one source file to a notebook in out_dir; return the notebook path."""
    text = src.read_text(encoding="utf-8")
    if kaggle_json and PLACEHOLDER in text:
        text = embed_kaggle(text, kaggle_json)
    nb = {"cells": parse_cells(text),
          "metadata": {"kernelspec": KERNEL, "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    for i, cell in enumerate(nb["cells"]):
        cell["id"] = f"cell-{i:03d}"
    dst = out_dir / f"{src.stem}.ipynb"
    dst.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return dst


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", type=Path, default=Path("notebooks/src"))
    ap.add_argument("--out", type=Path, default=Path("notebooks"))
    ap.add_argument("--embed-kaggle", type=Path, default=None,
                    help="kaggle.json to embed; output goes to notebooks/local/ (git-ignored)")
    args = ap.parse_args()
    out = args.out
    if args.embed_kaggle:
        out = Path("notebooks/local")
        out.mkdir(parents=True, exist_ok=True)
    for src in sorted(args.src.glob("*.py")):
        print("built", build(src, out, args.embed_kaggle))


if __name__ == "__main__":
    main()

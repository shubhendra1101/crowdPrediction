"""Smoke tests for the notebook builder and the notebook sources.

The notebooks run on the A100, not here, so this only checks they build and every
code cell is valid Python.
"""
import ast
import json
from pathlib import Path

import pytest

from scripts.build_notebooks import build, parse_cells

ROOT = Path(__file__).resolve().parents[1]
SOURCES = sorted((ROOT / "notebooks" / "src").glob("*.py"))


def test_parse_cells_splits_markdown_and_code() -> None:
    cells = parse_cells("# %% [markdown]\n# # Title\n# text\n\n# %%\nx = 1\ny = 2\n")
    assert [c["cell_type"] for c in cells] == ["markdown", "code"]
    assert cells[0]["source"] == ["# Title\n", "text"]
    assert cells[1]["source"] == ["x = 1\n", "y = 2"]


@pytest.mark.parametrize("src", SOURCES, ids=lambda p: p.stem)
def test_notebook_builds_and_code_parses(src: Path, tmp_path: Path) -> None:
    nb = json.loads(build(src, tmp_path).read_text(encoding="utf-8"))
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert code, "notebook has no code cells"
    for cell in code:
        ast.parse("".join(cell["source"]))


def test_embed_kaggle_only_replaces_placeholder(tmp_path: Path) -> None:
    from scripts.build_notebooks import embed_kaggle

    key = tmp_path / "kaggle.json"
    key.write_text(json.dumps({"username": "u", "key": "k"}))
    out = embed_kaggle("# %%\nKAGGLE_CREDENTIALS = None\nx = 1\n", key)
    assert 'KAGGLE_CREDENTIALS = {"username": "u", "key": "k"}' in out and "x = 1" in out

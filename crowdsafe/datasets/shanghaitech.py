"""ShanghaiTech Part A/B loader: images paired with head-point annotations (.mat).

Layout (official and the Kaggle mirror): ``.../part_A/test_data/images/IMG_1.jpg`` with
``.../part_A/test_data/ground-truth/GT_IMG_1.mat``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Sample:
    """One annotated image."""

    image: Path
    points: np.ndarray          # (N, 2) head positions in pixels (x, y)

    @property
    def count(self) -> int:
        return len(self.points)


def load_points(mat_path: Path) -> np.ndarray:
    """Read head points from a ShanghaiTech GT_IMG_*.mat file as an (N, 2) array."""
    from scipy.io import loadmat

    mat = loadmat(str(mat_path))
    if "image_info" in mat:
        pts = mat["image_info"][0, 0][0, 0][0]
    else:  # fall back to the largest N×2 numeric array in the file
        arrays = [v for k, v in mat.items() if not k.startswith("__") and isinstance(v, np.ndarray)
                  and v.ndim == 2 and v.shape[1] == 2 and v.dtype.kind in "fi"]
        if not arrays:
            raise ValueError(f"No point array in {mat_path}")
        pts = max(arrays, key=len)
    return np.asarray(pts, dtype=np.float64).reshape(-1, 2)


def find_split_dir(root: Path, part: str, split: str) -> Path:
    """Locate ``part_{A|B}*/{train|test}_data`` anywhere under root."""
    hits = sorted(p for p in Path(root).rglob(f"{split}_data") if p.parent.name.startswith(f"part_{part}"))
    if not hits:
        raise FileNotFoundError(f"part_{part}/{split}_data not found under {root}")
    return hits[0]


def list_split(root: Path, part: str, split: str = "test") -> list[Sample]:
    """All samples of one part (``"A"``/``"B"``) and split (``"train"``/``"test"``), sorted by image number."""
    d = find_split_dir(root, part, split)
    samples = []
    for img in (d / "images").glob("*.jpg"):
        gt = d / "ground-truth" / f"GT_{img.stem}.mat"
        if gt.exists():
            samples.append(Sample(img, load_points(gt)))
    return sorted(samples, key=lambda s: int("".join(filter(str.isdigit, s.image.stem)) or 0))

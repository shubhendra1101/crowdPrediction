"""Read CVAT for images 1.1 point annotations (the user's corrected head labels, T1.7)."""
from __future__ import annotations

import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np


def read_cvat_points(path: Path, label: str = "head") -> dict[str, np.ndarray]:
    """Image name -> (N, 2) array of point coordinates for ``label``.

    Accepts the exported zip or a bare annotations.xml. A CVAT points shape may hold several
    points separated by ';' — each point counts as one head.
    """
    path = Path(path)
    xml = zipfile.ZipFile(path).read("annotations.xml") if path.suffix == ".zip" else path.read_bytes()
    out = {}
    for im in ET.fromstring(xml).iter("image"):
        pts = []
        for p in im.findall("points"):
            if p.get("label") != label:
                continue
            pts += [tuple(map(float, q.split(","))) for q in p.get("points").split(";") if q]
        out[im.get("name")] = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    return out

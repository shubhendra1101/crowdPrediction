"""Smoke test for the CVAT point reader."""
import zipfile
from pathlib import Path

from crowdsafe.datasets.cvat import read_cvat_points

XML = """<annotations><version>1.1</version>
<image id="0" name="a.jpg" width="10" height="10">
  <points label="head" points="1,2" occluded="0"/>
  <points label="head" points="3,4;5,6" occluded="0"/>
  <points label="other" points="9,9" occluded="0"/>
</image>
<image id="1" name="b.jpg" width="10" height="10"/>
</annotations>"""


def test_reads_multi_point_shapes_and_filters_label(tmp_path: Path) -> None:
    z = tmp_path / "export.zip"
    with zipfile.ZipFile(z, "w") as f:
        f.writestr("annotations.xml", XML)
    pts = read_cvat_points(z)
    assert pts["a.jpg"].shape == (3, 2) and pts["a.jpg"][2].tolist() == [5.0, 6.0]
    assert pts["b.jpg"].shape == (0, 2)

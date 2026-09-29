# Jülich trajectory inventory (T1.1)

Downloaded 2026-09-30 on the laptop with
`python scripts/download_data.py --groups must --only-source julich --roles trajectories metadata`
(trajectories + metadata only; videos are fetched on the A100 by notebook 02). Per-archive SHA-256 in each `data/raw/<dataset>/manifest.json`.

Units and frame rates below are read from the file headers where present; where a file has no header, the unit is inferred from the coordinate range and marked *inferred*. The loader (T5.2) must convert everything to metres and seconds.

| Dataset | DOI | Txt files | MB | Columns | Units | Frame rate | Ids in first 3 files | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| julich_bottleneck1 | 10.34735/ped.2005.2 | 16 | 0.3 | id, time `s:frame`, x, y | m (archive page; range 0.1–0.7 × −0.4–1.3) | 25 fps | 120 | Time is `seconds:frame`, not a frame number; covers only the bottleneck itself |
| julich_bottleneck2 | 10.34735/ped.2006.2 | 13 | 26.6 | id, frame, x, y | cm *inferred* (range ±240 × −300–440) | 25 fps (archive page) | 527 | No header |
| julich_corridor_uni_2009 | 10.34735/ped.2009.13/.14 | 53 | 116.6 | id, frame, x, y, z | cm (archive page) | 16 fps (archive page) | 143 | Open + closed boundary; no header |
| julich_corridor_bi_2009 | 10.34735/ped.2009.10/.11/.12 | 24 | 37.1 | id, frame, x, y, z | cm (archive page) | 16 fps (archive page) | 468 | Counter-flow; no header |
| julich_corridor_uni_2013 | 10.34735/ped.2013.6 | 9 | 82.1 | id, frame, x, y, z | m (header) | 16 fps (header) | 1818 | BaSiGo |
| julich_entrance_corridor | 10.34735/ped.2013.1 | 2 | 15.0 | id, frame, x, y, z | cm (header) | 25 fps (header) | 279 | Only 2 trajectory files in archive |
| julich_entrance_semicircle | 10.34735/ped.2013.2 | 1 | 19.0 | id, frame, x, y, z | cm (header) | 25 fps (header) | 273 | Only 1 trajectory file in archive |
| julich_train_platform | 10.34735/ped.2021.3 | 27 | 1369.5 | id, frame, x, y, z, markerID | m (header) | 50 fps (header) | 165 | Tab-separated; waiting, not moving flow |

Densities reached per experiment are not known yet — they are computed with PedPy in T5.2.

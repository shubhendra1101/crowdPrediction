# Live pipeline latency (T9.2) — cam02

Generated 2026-10-26 15:03 by `scripts/live_pipeline.py`. Source `data/cctv/cam02.asf` (played at real speed (simulated live camera)), 4 cycles of 1.0 s, device `cpu`, density `onnx`, flow `farneback` (960 px), forecaster `persistence` every 5.0 s.

- `wait_pair`: waiting for the second frame of the motion pair (= pair gap; part of latency, not compute).
- `latency`: newest frame arrived → zone features, risk level and alerts ready (excludes the network/dashboard).
- `forecast` only in the cycles that run it.

| Stage | cycles | mean ms | median ms | p95 ms | max ms |
| --- | --- | --- | --- | --- | --- |
| wait_pair | 4 | 74 | 33 | 179 | 204 |
| density | 4 | 18731 | 18853 | 20038 | 20086 |
| flow | 4 | 272 | 266 | 293 | 298 |
| zones | 4 | 14 | 13 | 16 | 17 |
| log | 4 | 23 | 22 | 24 | 24 |
| risk | 4 | 4 | 5 | 5 | 5 |
| compute | 4 | 19048 | 19166 | 20343 | 20389 |
| latency | 4 | 19128 | 19204 | 20383 | 20429 |

- Cycles that finished within the 1.0 s budget: 0.0 %
- How far processing fell behind the live stream at the end: 7.8 s (max 7.8 s)
- Frames decoded 778 in 77 s (10.0 fps)
- GPU: not used
- Alerts raised: 0

Design delays on top of compute (not measured here): forecasts every 5.0 s, an alert must hold 10 s before it is raised.

# Live pipeline latency (T9.2) — cam02

Generated 2026-10-26 10:43 by `scripts/live_pipeline.py`. Source `/workspace/crowdsafe_work1/data/cctv/cam02.asf` (played at real speed (simulated live camera)), 297 cycles of 1.0 s, device `cuda`, density `torch`, flow `raft` (960 px), forecaster `persistence` every 5.0 s.

- `wait_pair`: waiting for the second frame of the motion pair (= pair gap; part of latency, not compute).
- `latency`: newest frame arrived → zone features, risk level and alerts ready (excludes the network/dashboard).
- `forecast` only in the cycles that run it.

| Stage | cycles | mean ms | median ms | p95 ms | max ms |
| --- | --- | --- | --- | --- | --- |
| wait_pair | 297 | 259 | 203 | 259 | 2005 |
| density | 297 | 113 | 103 | 114 | 2156 |
| flow | 297 | 68 | 61 | 69 | 1980 |
| zones | 297 | 5 | 5 | 6 | 40 |
| log | 297 | 40 | 7 | 8 | 2931 |
| forecast | 50 | 6 | 6 | 6 | 8 |
| risk | 297 | 7 | 8 | 8 | 16 |
| compute | 297 | 236 | 187 | 245 | 3103 |
| latency | 297 | 534 | 393 | 1473 | 5033 |

- Cycles that finished within the 1.0 s budget: 94.3 %
- How far processing fell behind the live stream at the end: 0.4 s (max 5.0 s)
- Frames decoded 7512 in 300 s (25.0 fps)
- GPU peak memory: 1517.5 MiB
- Alerts raised: 0

Design delays on top of compute (not measured here): forecasts every 5.0 s, an alert must hold 10 s before it is raised.

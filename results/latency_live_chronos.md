# Live pipeline latency (T9.2) — cam02

Generated 2026-10-26 11:05 by `scripts/live_pipeline.py`. Source `/workspace/crowdsafe_work1/data/cctv/cam02.asf` (played at real speed (simulated live camera)), 595 cycles of 1.0 s, device `cuda`, density `torch`, flow `raft` (960 px), forecaster `chronos2_joint` every 5.0 s.

- `wait_pair`: waiting for the second frame of the motion pair (= pair gap; part of latency, not compute).
- `latency`: newest frame arrived → zone features, risk level and alerts ready (excludes the network/dashboard).
- `forecast` only in the cycles that run it.

| Stage | cycles | mean ms | median ms | p95 ms | max ms |
| --- | --- | --- | --- | --- | --- |
| wait_pair | 595 | 238 | 203 | 223 | 2005 |
| density | 595 | 172 | 159 | 190 | 3101 |
| flow | 595 | 110 | 103 | 110 | 2014 |
| zones | 595 | 6 | 5 | 6 | 198 |
| log | 595 | 41 | 9 | 35 | 3692 |
| forecast | 108 | 58 | 54 | 62 | 200 |
| risk | 595 | 9 | 8 | 9 | 497 |
| compute | 595 | 351 | 290 | 398 | 4392 |
| latency | 595 | 616 | 493 | 1598 | 7103 |

- Cycles that finished within the 1.0 s budget: 94.3 %
- How far processing fell behind the live stream at the end: 2.1 s (max 4.5 s)
- Frames decoded 15005 in 600 s (25.0 fps)
- GPU peak memory: 1973.8 MiB
- Alerts raised: 0

Design delays on top of compute (not measured here): forecasts every 5.0 s, an alert must hold 10 s before it is raised.

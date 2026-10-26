# Live pipeline latency (T9.2) — cam02

Generated 2026-10-26 10:46 by `scripts/live_pipeline.py`. Source `/workspace/crowdsafe_work1/data/cctv/cam02.asf` (read as fast as possible (not real time)), 905 cycles of 1.0 s, device `cuda`, density `torch`, flow `raft` (960 px), forecaster `chronos2_joint` every 5.0 s.

- `wait_pair`: waiting for the second frame of the motion pair (= pair gap; part of latency, not compute).
- `latency`: newest frame arrived → zone features, risk level and alerts ready (excludes the network/dashboard).
- `forecast` only in the cycles that run it.

| Stage | cycles | mean ms | median ms | p95 ms | max ms |
| --- | --- | --- | --- | --- | --- |
| wait_pair | 905 | 0 | 0 | 0 | 15 |
| density | 905 | 109 | 105 | 127 | 738 |
| flow | 905 | 59 | 58 | 66 | 204 |
| zones | 905 | 5 | 5 | 6 | 21 |
| log | 905 | 18 | 7 | 23 | 3588 |
| forecast | 176 | 41 | 40 | 43 | 135 |
| risk | 905 | 8 | 8 | 8 | 31 |
| compute | 905 | 209 | 187 | 244 | 3764 |
| latency | 905 | 385 | 362 | 447 | 3955 |

- Cycles that finished within the 1.0 s budget: 99.8 %
- How far processing fell behind the live stream at the end: 1.0 s (max 1.6 s)
- Frames decoded 22700 in 196 s (115.6 fps)
- GPU peak memory: 1973.8 MiB
- Alerts raised: 0

Design delays on top of compute (not measured here): forecasts every 5.0 s, an alert must hold 10 s before it is raised.

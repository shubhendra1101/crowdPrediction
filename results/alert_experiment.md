# Alert-tuning experiment (T7.4) — before vs after

Generated 2026-10-26 13:09 by `scripts/alert_experiment.py`. Test splits only; thresholds tuned on val. Event = zone first reaching 4.0 persons/m² at least 30 s into a run (same events for every variant). `in window %` = alert raised within (longest horizon + wait) before the event and not in the first 15 s of monitoring. Forecasts: persistence and physics (laptop); Chronos-2 not included here (needs the A100). Decisions every 5 s.

## Rate-of-rise threshold (tuned on val)

| Dataset | threshold (persons/m² per s) | precision | recall | base rate | n |
| --- | --- | --- | --- | --- | --- |
| Jülich (real) | 0.3057 | 0.295 | 0.104 | 0.142 | 2626 |
| Simulation | 0.1242 | 0.606 | 0.262 | 0.301 | 1523 |

## Jülich (real): 9 events

| Variant | Forecast | warned in time % | median warning s | late % | missed % | false alarms / zone-hour |
| --- | --- | --- | --- | --- | --- | --- |
| A baseline (30 s history, 10 s wait) | current density only | 0.0 | nan | 22.2 | 77.8 | 0.0 |
| A baseline (30 s history, 10 s wait) | persistence | 11.1 | 4.0 | 44.4 | 33.3 | 0.0 |
| A baseline (30 s history, 10 s wait) | physics | 11.1 | 59.0 | 55.6 | 22.2 | 11.85 |
| B + shorter wait (5 s) | current density only | 0.0 | nan | 33.3 | 66.7 | 0.0 |
| B + shorter wait (5 s) | persistence | 11.1 | 9.0 | 55.6 | 22.2 | 0.0 |
| B + shorter wait (5 s) | physics | 11.1 | 64.0 | 33.3 | 22.2 | 17.44 |
| C + rate-of-rise warning | current density only | 0.0 | nan | 22.2 | 77.8 | 0.0 |
| C + rate-of-rise warning | persistence | 11.1 | 4.0 | 44.4 | 33.3 | 0.0 |
| C + rate-of-rise warning | physics | 11.1 | 59.0 | 55.6 | 22.2 | 11.85 |
| D + shorter history (15 s) | current density only | 0.0 | nan | 22.2 | 77.8 | 0.0 |
| D + shorter history (15 s) | persistence | 11.1 | 4.0 | 66.7 | 11.1 | 0.0 |
| D + shorter history (15 s) | physics | 0.0 | nan | 22.2 | 22.2 | 11.28 |
| E all three | current density only | 0.0 | nan | 33.3 | 66.7 | 0.0 |
| E all three | persistence | 11.1 | 9.0 | 66.7 | 11.1 | 0.2 |
| E all three | physics | 22.2 | 8.5 | 0.0 | 22.2 | 16.51 |

## Simulation: 29 events

| Variant | Forecast | warned in time % | median warning s | late % | missed % | false alarms / zone-hour |
| --- | --- | --- | --- | --- | --- | --- |
| A baseline (30 s history, 10 s wait) | current density only | 0.0 | nan | 79.3 | 20.7 | 0.0 |
| A baseline (30 s history, 10 s wait) | persistence | 0.0 | nan | 86.2 | 13.8 | 0.0 |
| A baseline (30 s history, 10 s wait) | physics | 58.6 | 25.0 | 13.8 | 0.0 | 1.59 |
| B + shorter wait (5 s) | current density only | 0.0 | nan | 93.1 | 6.9 | 0.0 |
| B + shorter wait (5 s) | persistence | 0.0 | nan | 93.1 | 6.9 | 0.0 |
| B + shorter wait (5 s) | physics | 44.8 | 53.0 | 3.4 | 0.0 | 3.63 |
| C + rate-of-rise warning | current density only | 3.4 | 1.0 | 86.2 | 10.3 | 0.0 |
| C + rate-of-rise warning | persistence | 3.4 | 1.0 | 86.2 | 10.3 | 0.09 |
| C + rate-of-rise warning | physics | 58.6 | 25.0 | 13.8 | 0.0 | 1.59 |
| D + shorter history (15 s) | current density only | 0.0 | nan | 79.3 | 20.7 | 0.0 |
| D + shorter history (15 s) | persistence | 0.0 | nan | 86.2 | 13.8 | 0.0 |
| D + shorter history (15 s) | physics | 62.1 | 37.5 | 0.0 | 0.0 | 2.34 |
| E all three | current density only | 24.1 | 4.0 | 62.1 | 3.4 | 0.17 |
| E all three | persistence | 31.0 | 1.0 | 51.7 | 3.4 | 0.17 |
| E all three | physics | 24.1 | 51.0 | 0.0 | 0.0 | 3.93 |

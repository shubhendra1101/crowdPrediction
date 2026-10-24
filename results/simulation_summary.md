# JuPedSim scenarios (T5.4) — synthetic data

Generated 2026-10-24 17:22 by `scripts/run_simulations.py` from `configs\simulation.yaml`. Collision-free speed model (JuPedSim), desired speed N(1.34, 0.26) m/s clipped to 0.7–1.9, body radius 0.15–0.20 m. Zones = 2 m cells; density = people / walkable area of the cell (desks excluded). `zone_s_geX` = zone-seconds at or above X persons/m².

| Scenario | Split | Layout | Agents | Duration s | Zones | Max density | zone-s ≥ 4 | zone-s ≥ 5 | Max pressure |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| class_60 | train | classroom | 60 | 31.2 | 20 | 2.96 | 0 | 0 | 0.717 |
| class_90 | val | classroom | 90 | 37.2 | 20 | 2.96 | 0 | 0 | 0.73 |
| class_120_overfull | train | classroom | 120 | 43.6 | 20 | 2.96 | 0 | 0 | 0.672 |
| class_90_one_door | train | classroom | 90 | 58.0 | 20 | 3.85 | 0 | 0 | 0.716 |
| class_120_door_closes | test | classroom | 120 | 62.2 | 20 | 3.58 | 0 | 0 | 0.672 |
| corr_1p5 | train | corridor_bottleneck | 180 | 144.2 | 20 | 1.35 | 0 | 0 | 0.184 |
| corr_2p5 | val | corridor_bottleneck | 300 | 201.6 | 20 | 5.6 | 199 | 57 | 0.404 |
| corr_3p5 | train | corridor_bottleneck | 420 | 270.6 | 20 | 6.25 | 905 | 526 | 0.356 |
| corr_surge | test | corridor_bottleneck | 300 | 221.8 | 20 | 5.9 | 252 | 79 | 0.276 |
| corr_narrow_0p8 | test | corridor_bottleneck | 300 | 499.8 | 20 | 7.75 | 3299 | 2837 | 0.265 |
| hall_300 | train | hall | 300 | 88.8 | 80 | 6.7 | 250 | 92 | 1.263 |
| hall_500 | train | hall | 500 | 150.4 | 80 | 7.0 | 943 | 520 | 1.238 |
| hall_500_A_closes | val | hall | 500 | 230.2 | 80 | 7.9 | 1957 | 1229 | 1.522 |
| hall_700_A_closes | test | hall | 700 | 333.6 | 80 | 8.0 | 4293 | 3142 | 1.451 |
| bidir_1p0 | val | bidirectional | 240 | 138.6 | 20 | 1.5 | 0 | 0 | 1.335 |
| bidir_2p0 | test | bidirectional | 480 | 256.6 | 20 | 6.5 | 678 | 417 | 1.31 |

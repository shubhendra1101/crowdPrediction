# JuPedSim scenarios (T5.4) — synthetic data

Generated 2026-10-26 06:48 by `scripts/run_simulations.py` from `configs/simulation.yaml`. Collision-free speed model (JuPedSim), desired speed N(1.34, 0.26) m/s clipped to 0.7–1.9, body radius 0.15–0.20 m. Zones = 2 m cells; density = people / walkable area of the cell (desks excluded). `zone_s_geX` = zone-seconds at or above X persons/m².

| Scenario | Split | Layout | Agents | Duration s | Zones | Max density | zone-s ≥ 4 | zone-s ≥ 5 | Max pressure |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| class_60 | train | classroom | 60 | 30.2 | 20 | 2.58 | 0 | 0 | 0.692 |
| class_90 | val | classroom | 90 | 37.4 | 20 | 2.96 | 0 | 0 | 0.779 |
| class_120_overfull | train | classroom | 120 | 41.2 | 20 | 2.96 | 0 | 0 | 0.698 |
| class_90_one_door | train | classroom | 90 | 54.8 | 20 | 3.78 | 0 | 0 | 0.791 |
| class_120_door_closes | test | classroom | 120 | 58.0 | 20 | 3.45 | 0 | 0 | 0.698 |
| corr_1p5 | train | corridor_bottleneck | 180 | 144.2 | 20 | 1.35 | 0 | 0 | 0.184 |
| corr_2p5 | val | corridor_bottleneck | 300 | 198.8 | 20 | 5.75 | 201 | 54 | 0.443 |
| corr_3p5 | train | corridor_bottleneck | 420 | 266.0 | 20 | 6.0 | 861 | 469 | 0.356 |
| corr_surge | test | corridor_bottleneck | 300 | 221.2 | 20 | 5.7 | 251 | 71 | 0.276 |
| corr_narrow_0p8 | test | corridor_bottleneck | 300 | 499.8 | 20 | 7.0 | 1957 | 1743 | 0.315 |
| hall_300 | train | hall | 300 | 92.0 | 80 | 6.2 | 260 | 80 | 1.401 |
| hall_500 | train | hall | 500 | 153.6 | 80 | 7.0 | 972 | 502 | 1.301 |
| hall_500_A_closes | val | hall | 500 | 227.0 | 80 | 8.0 | 1965 | 1217 | 1.397 |
| hall_700_A_closes | test | hall | 700 | 340.4 | 80 | 8.0 | 4384 | 3200 | 1.446 |
| bidir_1p0 | val | bidirectional | 240 | 138.6 | 20 | 1.5 | 0 | 0 | 1.335 |
| bidir_2p0 | test | bidirectional | 480 | 198.2 | 20 | 5.35 | 276 | 26 | 1.377 |

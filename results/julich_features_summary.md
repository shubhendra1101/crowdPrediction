# Jülich zone features — summary (T5.2)

Generated 2026-09-30 10:38 from `configs\julich.yaml`. Zones: 2.0 m squares (4 m²) kept when ≥ 90% walked on (walkable area inferred from trajectories — approximate). Densities = persons/m², mean per second. `zone_s_geX` = zone-seconds at or above X persons/m². Velocities above 4.0 m/s are treated as tracking glitches and ignored (`glitch_share` = share of samples).

| dataset | runs | persons | max_density | zone_s_ge4 | zone_s_ge5 | max_pressure | glitch_share_max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| julich_bottleneck1 | 15 | 0.0 | nan | 0.0 | 0.0 | nan | nan |
| julich_bottleneck2 | 13 | 2300.0 | 4.75 | 29.0 | 0.0 | 0.382 | 0.0 |
| julich_corridor_bi_2009 | 24 | 4752.0 | 4.4 | 11.0 | 0.0 | 2.848 | 0.0 |
| julich_corridor_uni_2009 | 53 | 8235.0 | 4.7 | 148.0 | 0.0 | 0.474 | 0.0 |
| julich_corridor_uni_2013 | 9 | 6906.0 | 5.3 | 513.0 | 19.0 | 1.483 | 0.003 |
| julich_entrance_corridor | 2 | 279.0 | 5.65 | 298.0 | 38.0 | 0.208 | 0.0 |
| julich_entrance_semicircle | 1 | 273.0 | 9.0 | 654.0 | 586.0 | 0.277 | 0.0 |
| julich_train_platform | 27 | 1545.0 | 2.65 | 0.0 | 0.0 | 0.797 | 0.0 |

## PedPy cross-check (our zone density vs `pedpy.compute_classic_density`, same zone and frames)

| Dataset | Run | Zone | Max abs diff (persons/m²) |
| --- | --- | --- | --- |
| julich_bottleneck2 | b090 | c1_1 | 0.0000 |
| julich_corridor_bi_2009 | bo-360-050-050 | c2_6 | 0.0000 |
| julich_corridor_uni_2013 | uni_corr_500_01 | c10_2 | 0.0000 |
| julich_entrance_corridor | entrance_2 | c1_1 | 0.0000 |
| julich_entrance_semicircle | entrance_1 | c2_1 | 0.0000 |
| julich_train_platform | 1B050 | c1_8 | 0.0000 |

Per-run table: `results\julich_features_summary.csv`.

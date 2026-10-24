# Alerts on held-out Jülich experiments (T7.1, T7.2, T7.4)

Generated 2026-10-24 16:33 by `scripts/eval_alerts.py`. Test runs: ['julich_corridor_uni_2013', 'julich_entrance_semicircle'] (10 runs, 118 zones). Decision every 5 s; alert = level ≥ High held 10 s (cleared after 30 s below). Upper bound = worst conformal-calibrated 90 % bound over horizons [10, 30, 60] s. An *event* is a zone first reaching the event density after monitoring starts.

- `alerted_before_%`: events with an alert raised before the density got there; `late_%`: alert within 30 s after; `missed_%`: neither.
- `lead`: seconds between the alert and the event (alerts raised before it).
- `false_alarms`: alerts with no event within 120 s after raising (or 30 s before).

## Precursor thresholds (tuned on val only)

Target: zone reaches 4.0 persons/m² within 60 s, among zone-seconds currently at 2–4 persons/m². A precursor is kept only if its precision is ≥ 2.0× the base rate.

| Precursor | Direction | Threshold | F1 | Precision | Recall | Base rate | n |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pressure | high | None | nan | nan | nan | 0.145 | 2665 |
| dir_entropy | high | 0.784 | 0.192 | 0.294 | 0.142 | 0.145 | 2665 |
| counterflow | high | None | nan | nan | nan | 0.145 | 2665 |
| slow_speed | low | 0.2067 | 0.471 | 0.544 | 0.415 | 0.145 | 2665 |
| flow_collapse | rule | off | 0.196 | 0.118 | 0.565 | 0.145 | 2665 |

## Event = 4.0 persons/m²

| Variant | events | alerted before % | late % | missed % | median lead s | mean lead s | alerts | false alarms | per zone-hour |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| current density only | 9 | 0.0 | 22.2 | 77.8 | nan | nan | 12 | 0 | 0.0 |
| current density + precursors | 9 | 33.3 | 11.1 | 55.6 | 207.0 | 145.7 | 27 | 12 | 2.5 |
| persistence upper bound | 9 | 22.2 | 44.4 | 33.3 | 58.5 | 58.5 | 17 | 0 | 0.0 |
| persistence upper bound + precursors | 9 | 44.4 | 33.3 | 22.2 | 101.0 | 108.5 | 27 | 12 | 2.68 |
| physics upper bound | 9 | 22.2 | 55.6 | 22.2 | 196.0 | 196.0 | 67 | 53 | 11.85 |
| physics upper bound + precursors | 9 | 33.3 | 44.4 | 22.2 | 183.0 | 138.3 | 71 | 56 | 12.52 |
| chronos2_per_zone upper bound | 9 | 55.6 | 33.3 | 11.1 | 19.0 | 95.6 | 26 | 8 | 1.79 |
| chronos2_per_zone upper bound + precursors | 9 | 55.6 | 33.3 | 11.1 | 19.0 | 95.6 | 28 | 13 | 2.91 |
| chronos2_joint upper bound | 9 | 44.4 | 33.3 | 22.2 | 116.0 | 116.0 | 20 | 2 | 0.45 |
| chronos2_joint upper bound + precursors | 9 | 44.4 | 33.3 | 22.2 | 116.0 | 116.0 | 27 | 12 | 2.68 |

## Event = 5.0 persons/m²

| Variant | events | alerted before % | late % | missed % | median lead s | mean lead s | alerts | false alarms | per zone-hour |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| current density only | 2 | 50.0 | 50.0 | 0.0 | 2.0 | 2.0 | 12 | 3 | 0.62 |
| current density + precursors | 2 | 50.0 | 50.0 | 0.0 | 2.0 | 2.0 | 27 | 19 | 3.96 |
| persistence upper bound | 2 | 50.0 | 50.0 | 0.0 | 33.0 | 33.0 | 17 | 8 | 1.79 |
| persistence upper bound + precursors | 2 | 50.0 | 50.0 | 0.0 | 33.0 | 33.0 | 27 | 19 | 4.25 |
| physics upper bound | 2 | 50.0 | 50.0 | 0.0 | 48.0 | 48.0 | 67 | 59 | 13.19 |
| physics upper bound + precursors | 2 | 50.0 | 50.0 | 0.0 | 48.0 | 48.0 | 71 | 63 | 14.08 |
| chronos2_per_zone upper bound | 2 | 50.0 | 50.0 | 0.0 | 43.0 | 43.0 | 26 | 18 | 4.02 |
| chronos2_per_zone upper bound + precursors | 2 | 50.0 | 50.0 | 0.0 | 43.0 | 43.0 | 28 | 20 | 4.47 |
| chronos2_joint upper bound | 2 | 50.0 | 50.0 | 0.0 | 43.0 | 43.0 | 20 | 12 | 2.68 |
| chronos2_joint upper bound + precursors | 2 | 50.0 | 50.0 | 0.0 | 43.0 | 43.0 | 27 | 19 | 4.25 |

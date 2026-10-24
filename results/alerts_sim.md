# Alerts on held-out runs: forecast_sim (T7.1, T7.2, T7.4)

Generated 2026-10-24 17:23 by `scripts/eval_alerts.py`. Test runs: ['class_120_door_closes', 'corr_surge', 'corr_narrow_0p8', 'hall_700_A_closes', 'bidir_2p0'] (5 runs, 160 zones). Decision every 5 s; alert = level ≥ High held 10 s (cleared after 30 s below). Upper bound = worst conformal-calibrated 90 % bound over horizons [10, 30, 60, 120] s. An *event* is a zone first reaching the event density after monitoring starts.

- `alerted_before_%`: events with an alert raised before the density got there; `late_%`: alert within 30 s after; `missed_%`: neither.
- `lead`: seconds between the alert and the event (alerts raised before it).
- `false_alarms`: alerts with no event within 120 s after raising (or 30 s before).

## Precursor thresholds (tuned on pooled val splits: forecast_julich, forecast_sim)

Target: zone reaches 4.0 persons/m² within 60 s, among zone-seconds currently at 2–4 persons/m². A precursor is kept only if its precision is ≥ 2.0× the base rate.

| Precursor | Direction | Threshold | F1 | Precision | Recall | Base rate | n |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pressure | high | None | nan | nan | nan | 0.205 | 4337 |
| dir_entropy | high | None | nan | nan | nan | 0.205 | 4123 |
| counterflow | high | None | nan | nan | nan | 0.205 | 4123 |
| slow_speed | low | None | nan | nan | nan | 0.205 | 4337 |
| flow_collapse | rule | off | 0.354 | 0.237 | 0.698 | 0.205 | 4337 |

## Event = 4.0 persons/m²

| Variant | events | alerted before % | late % | missed % | median lead s | mean lead s | alerts | false alarms | per zone-hour |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| current density only | 29 | 0.0 | 75.9 | 24.1 | nan | nan | 43 | 0 | 0.0 |
| current density + precursors | 29 | 0.0 | 75.9 | 24.1 | nan | nan | 43 | 0 | 0.0 |
| persistence upper bound | 29 | 0.0 | 86.2 | 13.8 | nan | nan | 45 | 0 | 0.0 |
| persistence upper bound + precursors | 29 | 0.0 | 86.2 | 13.8 | nan | nan | 45 | 0 | 0.0 |
| physics upper bound | 29 | 86.2 | 13.8 | 0.0 | 25.0 | 32.2 | 67 | 18 | 1.59 |
| physics upper bound + precursors | 29 | 86.2 | 13.8 | 0.0 | 25.0 | 32.2 | 67 | 18 | 1.59 |

## Event = 5.0 persons/m²

| Variant | events | alerted before % | late % | missed % | median lead s | mean lead s | alerts | false alarms | per zone-hour |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| current density only | 25 | 60.0 | 40.0 | 0.0 | 2.0 | 4.3 | 43 | 3 | 0.26 |
| current density + precursors | 25 | 60.0 | 40.0 | 0.0 | 2.0 | 4.3 | 43 | 3 | 0.26 |
| persistence upper bound | 25 | 60.0 | 40.0 | 0.0 | 3.0 | 5.6 | 45 | 5 | 0.44 |
| persistence upper bound + precursors | 25 | 60.0 | 40.0 | 0.0 | 3.0 | 5.6 | 45 | 5 | 0.44 |
| physics upper bound | 25 | 88.0 | 12.0 | 0.0 | 37.5 | 38.4 | 67 | 27 | 2.39 |
| physics upper bound + precursors | 25 | 88.0 | 12.0 | 0.0 | 37.5 | 38.4 | 67 | 27 | 2.39 |

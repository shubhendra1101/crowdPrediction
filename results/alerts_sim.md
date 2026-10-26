# Alerts on held-out runs: forecast_sim (T7.1, T7.2, T7.4)

Generated 2026-10-26 12:48 by `scripts/eval_alerts.py`. Test runs: ['class_120_door_closes', 'corr_surge', 'corr_narrow_0p8', 'hall_700_A_closes', 'bidir_2p0'] (5 runs, 160 zones). Decision every 5 s; alert = level ≥ High held 10 s (cleared after 30 s below). Upper bound = worst conformal-calibrated 90 % bound over horizons [10, 30, 60, 120] s. An *event* is a zone first reaching the event density after monitoring starts.

- `alerted_before_%`: events with an alert raised before the density got there; `late_%`: alert within 30 s after; `missed_%`: neither.
- `lead`: seconds between the alert and the event (alerts raised before it).
- `false_alarms`: alerts with no event within 120 s after raising (or 30 s before).
- **`in window %` (stricter, use this for claims):** alert raised at most 130 s (longest horizon + raise delay) before the event and not in the first 15 s of monitoring. `alerted before %` also counts alerts raised at the start of a run and simply kept on, which inflates lead times.

## Precursor thresholds (tuned on pooled val splits: forecast_julich, forecast_sim)

Target: zone reaches 4.0 persons/m² within 60 s, among zone-seconds currently at 2–4 persons/m². A precursor is kept only if its precision is ≥ 2.0× the base rate.

| Precursor | Direction | Threshold | F1 | Precision | Recall | Base rate | n |
| --- | --- | --- | --- | --- | --- | --- | --- |

## Event = 4.0 persons/m²

| Variant | events | **in window %** | median lead in window s | alerted before % | late % | missed % | median lead s | alerts | false alarms | per zone-hour |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| current density only | 47 | 0.0 | nan | 0.0 | 87.2 | 12.8 | nan | 41 | 0 | 0.0 |
| persistence upper bound | 47 | 0.0 | nan | 0.0 | 89.4 | 10.6 | nan | 42 | 0 | 0.0 |
| physics upper bound | 47 | 31.9 | 25.0 | 55.3 | 44.7 | 0.0 | 25.0 | 70 | 23 | 2.06 |
| chronos2_per_zone upper bound | 47 | 34.0 | 7.5 | 51.1 | 48.9 | 0.0 | 5.0 | 65 | 18 | 1.61 |
| chronos2_joint upper bound | 47 | 29.8 | 7.5 | 46.8 | 53.2 | 0.0 | 5.0 | 64 | 17 | 1.52 |

## Event = 5.0 persons/m²

| Variant | events | **in window %** | median lead in window s | alerted before % | late % | missed % | median lead s | alerts | false alarms | per zone-hour |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| current density only | 34 | 38.2 | 5.0 | 44.1 | 55.9 | 0.0 | 10.0 | 41 | 7 | 0.63 |
| persistence upper bound | 34 | 38.2 | 10.0 | 47.1 | 52.9 | 0.0 | 10.0 | 42 | 8 | 0.72 |
| physics upper bound | 34 | 23.5 | 47.5 | 44.1 | 55.9 | 0.0 | 40.0 | 70 | 36 | 3.22 |
| chronos2_per_zone upper bound | 34 | 26.5 | 35.0 | 47.1 | 52.9 | 0.0 | 30.0 | 65 | 31 | 2.78 |
| chronos2_joint upper bound | 34 | 26.5 | 25.0 | 47.1 | 52.9 | 0.0 | 25.0 | 64 | 30 | 2.69 |

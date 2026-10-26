# Alerts on held-out runs: forecast_julich (T7.1, T7.2, T7.4)

Generated 2026-10-26 12:48 by `scripts/eval_alerts.py`. Test runs: ['julich_corridor_uni_2013', 'julich_entrance_semicircle'] (10 runs, 118 zones). Decision every 5 s; alert = level ≥ High held 10 s (cleared after 30 s below). Upper bound = worst conformal-calibrated 90 % bound over horizons [10, 30, 60] s. An *event* is a zone first reaching the event density after monitoring starts.

- `alerted_before_%`: events with an alert raised before the density got there; `late_%`: alert within 30 s after; `missed_%`: neither.
- `lead`: seconds between the alert and the event (alerts raised before it).
- `false_alarms`: alerts with no event within 120 s after raising (or 30 s before).
- **`in window %` (stricter, use this for claims):** alert raised at most 70 s (longest horizon + raise delay) before the event and not in the first 15 s of monitoring. `alerted before %` also counts alerts raised at the start of a run and simply kept on, which inflates lead times.

## Precursor thresholds (tuned on pooled val splits: forecast_julich, forecast_sim)

Target: zone reaches 4.0 persons/m² within 60 s, among zone-seconds currently at 2–4 persons/m². A precursor is kept only if its precision is ≥ 2.0× the base rate.

| Precursor | Direction | Threshold | F1 | Precision | Recall | Base rate | n |
| --- | --- | --- | --- | --- | --- | --- | --- |

## Event = 4.0 persons/m²

| Variant | events | **in window %** | median lead in window s | alerted before % | late % | missed % | median lead s | alerts | false alarms | per zone-hour |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| current density only | 9 | 0.0 | nan | 0.0 | 22.2 | 77.8 | nan | 13 | 0 | 0.0 |
| persistence upper bound | 9 | 11.1 | 4.0 | 22.2 | 44.4 | 33.3 | 58.5 | 17 | 0 | 0.0 |
| physics upper bound | 9 | 11.1 | 59.0 | 22.2 | 55.6 | 22.2 | 196.0 | 67 | 53 | 11.85 |
| chronos2_per_zone upper bound | 9 | 11.1 | 49.0 | 55.6 | 33.3 | 11.1 | 19.0 | 26 | 8 | 1.79 |
| chronos2_joint upper bound | 9 | 11.1 | 18.0 | 44.4 | 33.3 | 22.2 | 116.0 | 20 | 2 | 0.45 |

## Event = 5.0 persons/m²

| Variant | events | **in window %** | median lead in window s | alerted before % | late % | missed % | median lead s | alerts | false alarms | per zone-hour |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| current density only | 2 | 50.0 | 3.0 | 50.0 | 50.0 | 0.0 | 3.0 | 13 | 4 | 0.89 |
| persistence upper bound | 2 | 50.0 | 33.0 | 50.0 | 50.0 | 0.0 | 33.0 | 17 | 8 | 1.79 |
| physics upper bound | 2 | 50.0 | 48.0 | 50.0 | 50.0 | 0.0 | 48.0 | 67 | 59 | 13.19 |
| chronos2_per_zone upper bound | 2 | 50.0 | 43.0 | 50.0 | 50.0 | 0.0 | 43.0 | 26 | 18 | 4.02 |
| chronos2_joint upper bound | 2 | 50.0 | 43.0 | 50.0 | 50.0 | 0.0 | 43.0 | 20 | 12 | 2.68 |

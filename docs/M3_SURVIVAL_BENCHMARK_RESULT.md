# M3 Survival-Curve Benchmark Result

Source commit: `b09763bf27baa241eae7a8ed56c4b8f85996e032`

## Evaluation

- Holdout start: 2015-01-01T00:00:00+00:00
- Complete evaluated snapshots: 1110
- Horizons: [30, 90, 180, 365]
- Target stride: 3
- Minimum resolved training cases per horizon: 500

## Curve coherence

- Raw crossing curves: 3 (0.27%)
- Mean crossing magnitude: 0.000040
- Mean absolute PAVA adjustment: 0.000011
- Mean restricted survival days: 358.20

## Integrated Brier

- Historical base rate: 0.013035
- Raw independent horizons: 0.013659
- PAVA-adjusted curve: 0.013659
- PAVA skill vs raw: 0.000006
- Adjusted skill vs baseline: -0.047906

## Horizon metrics

| Horizon | Predictions | Positives | Baseline Brier | Raw Brier | Adjusted Brier | Adjusted log loss | Adjusted calibration | PAVA skill vs raw |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 30 | 1110 | 2 | 0.001809 | 0.001827 | 0.001827 | 0.012969 | 0.001231 | 0.000036 |
| 90 | 1110 | 8 | 0.007207 | 0.007288 | 0.007287 | 0.043374 | 0.001758 | 0.000254 |
| 180 | 1110 | 16 | 0.014401 | 0.014790 | 0.014805 | 0.079202 | 0.008672 | -0.000986 |
| 365 | 1110 | 32 | 0.028722 | 0.030732 | 0.030719 | 0.146270 | 0.027212 | 0.000423 |

PAVA is judged by proper scores as well as logical coherence. A negative PAVA skill remains a valid result.

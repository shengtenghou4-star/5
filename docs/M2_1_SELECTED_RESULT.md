# M2.1 Locked Selection Result

Source commit: `1c9361364ce87e50c9f258d19510ba7f2d254783`

## Pre-holdout selection

- Feature version: m2.1-time-safe-news-volume-v1
- Validation period: 2021-01-01T00:00:00+00:00 to 2022-01-01T00:00:00+00:00
- Structure candidates: 6
- GDELT candidates: 24
- Selected top_k: 75
- Selected numeric scale: iqr
- Selected signal family: tone
- Selected GDELT multiplier: 0.1
- Validation Brier skill vs matched structure: 0.005608

## Locked 2022+ holdout

- Predictions: 481
- Positive cases: 5

| Model | Brier | Log loss | Calibration error |
|---|---:|---:|---:|
| Structure only | 0.011055 | 0.058999 | 0.006192 |
| Selected structure + GDELT | 0.011112 | 0.058611 | 0.008739 |

- Holdout Brier skill vs matched structure: -0.005182
- Holdout log-loss skill vs matched structure: 0.006570
- Mean absolute probability change: 0.003231
- Mean neighbor overlap: 0.873904

This is the locked M2.1 follow-up. It does not alter the frozen M2 result.

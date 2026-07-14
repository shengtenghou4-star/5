# M3 Competing-Risk Benchmark Result

Source commit: `960200c46fde8e3fa6e0cc5136017430467dcfac`

## Evaluation

- Holdout start: 2015-01-01T00:00:00+00:00
- Complete evaluated snapshots: 1110
- Exit observations across horizons: 58
- Horizons: [30, 90, 180, 365]
- Mechanisms: ['other_recorded_transition', 'post_election_transition']
- Target stride: 3
- Minimum resolved training cases per domain: 500

## Integrated scores

- Total Brier, baseline: 0.013035
- Total Brier, adjusted model: 0.013659
- Total Brier skill: -0.047906
- Mechanism Brier, baseline: 0.006569
- Mechanism Brier, adjusted model: 0.006811
- Mechanism Brier skill: -0.036897

## Conditional path choice among exits

- Baseline conditional Brier: 0.322725
- Adjusted conditional Brier: 0.244753
- Conditional Brier skill: 0.241605
- Baseline conditional log loss: 0.851322
- Adjusted conditional log loss: 0.721611
- Conditional log-loss skill: 0.152364
- Baseline top-path accuracy: 36.21%
- Adjusted top-path accuracy: 67.24%

## Coherence

- Maximum baseline conservation error: 0.000000000000
- Maximum adjusted conservation error: 0.000000000000

## Horizon summary

| Horizon | Snapshots | Exits | Baseline total Brier | Model total Brier | Baseline path Brier | Model path Brier | Baseline conditional log loss | Model conditional log loss | Model path accuracy |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 30 | 1110 | 2 | 0.001809 | 0.001827 | 0.000904 | 0.000911 | 0.847298 | 0.376396 | 50.00% |
| 90 | 1110 | 8 | 0.007207 | 0.007287 | 0.003612 | 0.003629 | 0.839313 | 0.447630 | 87.50% |
| 180 | 1110 | 16 | 0.014401 | 0.014805 | 0.007236 | 0.007374 | 0.843944 | 0.619711 | 81.25% |
| 365 | 1110 | 32 | 0.028722 | 0.030719 | 0.014525 | 0.015331 | 0.858264 | 0.862632 | 56.25% |

## Mechanism detail

| Horizon | Mechanism | Positives | Baseline Brier | Model Brier | Brier skill |
|---:|---|---:|---:|---:|---:|
| 30 | other_recorded_transition | 1 | 0.000901 | 0.000919 | -0.020070 |
| 30 | post_election_transition | 1 | 0.000906 | 0.000903 | 0.003036 |
| 90 | other_recorded_transition | 3 | 0.002699 | 0.002734 | -0.013117 |
| 90 | post_election_transition | 5 | 0.004525 | 0.004523 | 0.000282 |
| 180 | other_recorded_transition | 6 | 0.005390 | 0.005597 | -0.038347 |
| 180 | post_election_transition | 10 | 0.009082 | 0.009152 | -0.007639 |
| 365 | other_recorded_transition | 12 | 0.010741 | 0.011543 | -0.074586 |
| 365 | post_election_transition | 20 | 0.018308 | 0.019120 | -0.044370 |

Negative skill remains a valid result. Conditional scores prevent the rarity of leader exits from hiding weak transition-path forecasts.

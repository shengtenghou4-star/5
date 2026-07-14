# M3.2 Hybrid Competing-Risk Result

Source commit: `9496383c211863e4ae9d93cf697997595e0d97c5`

## Evidence status

Exploratory integration result. The total-risk and conditional-path components had already been evaluated separately on the same 2015+ block, so this is not an independent confirmatory blind test.

## Evaluation

- Complete snapshots: 1110
- Exit observations across horizons: 58
- Total Brier, historical baseline: 0.013035
- Total Brier, hierarchical hybrid: 0.013031
- Total Brier skill: 0.000305
- Mechanism Brier, baseline: 0.006569
- Mechanism Brier, hybrid: 0.006556
- Mechanism Brier skill: 0.001962

## Conditional path choice among exits

- Baseline conditional Brier: 0.322725
- Hybrid conditional Brier: 0.256969
- Conditional Brier skill: 0.203754
- Baseline conditional log loss: 0.851322
- Hybrid conditional log loss: 0.752834
- Conditional log-loss skill: 0.115688
- Baseline top-path accuracy: 36.21%
- Hybrid top-path accuracy: 65.52%
- Maximum probability-conservation error: 0.000000000000

## Horizon summary

| Horizon | Snapshots | Exits | Baseline total Brier | Hybrid total Brier | Baseline mechanism Brier | Hybrid mechanism Brier | Hybrid conditional log loss | Hybrid path accuracy |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 30 | 1110 | 2 | 0.001809 | 0.001808 | 0.000904 | 0.000903 | 0.376396 | 50.00% |
| 90 | 1110 | 8 | 0.007207 | 0.007185 | 0.003612 | 0.003592 | 0.501069 | 87.50% |
| 180 | 1110 | 16 | 0.014401 | 0.014339 | 0.007236 | 0.007186 | 0.719935 | 81.25% |
| 365 | 1110 | 32 | 0.028722 | 0.028791 | 0.014525 | 0.014545 | 0.855752 | 53.12% |

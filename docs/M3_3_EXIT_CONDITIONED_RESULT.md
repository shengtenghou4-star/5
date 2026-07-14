# M3.3 Exit-Conditioned Path Result

Source commit: `ba15efd3a2dcb81a43b42695407fc73681c75f42`

## Evidence status

Exploratory result on the existing 2015+ block. Path models are trained only on resolved observations where an exit occurred; non-exits no longer dominate the path task.

## Evaluation

- Complete snapshots: 1110
- Exit observations across horizons: 58
- Total Brier, historical baseline: 0.013035
- Total Brier, conditioned hybrid: 0.013031
- Total Brier skill: 0.000305
- Mechanism Brier, baseline: 0.006569
- Mechanism Brier, conditioned hybrid: 0.006505
- Mechanism Brier skill: 0.009688

## Conditional path choice among exits

- Baseline conditional Brier: 0.322725
- Conditioned conditional Brier: 0.129398
- Conditional Brier skill: 0.599047
- Baseline conditional log loss: 0.851322
- Conditioned conditional log loss: 0.404574
- Conditional log-loss skill: 0.524769
- Baseline top-path accuracy: 36.21%
- Conditioned top-path accuracy: 81.03%
- Maximum probability-conservation error: 0.000000000000

## Horizon summary

| Horizon | Snapshots | Exits | Baseline total Brier | Model total Brier | Baseline mechanism Brier | Model mechanism Brier | Conditional Brier | Conditional log loss | Path accuracy |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 30 | 1110 | 2 | 0.001809 | 0.001808 | 0.000904 | 0.000904 | 0.344880 | 0.884987 | 0.00% |
| 90 | 1110 | 8 | 0.007207 | 0.007185 | 0.003612 | 0.003590 | 0.104033 | 0.358872 | 87.50% |
| 180 | 1110 | 16 | 0.014401 | 0.014339 | 0.007236 | 0.007161 | 0.082255 | 0.292071 | 93.75% |
| 365 | 1110 | 32 | 0.028722 | 0.028791 | 0.014525 | 0.014367 | 0.145843 | 0.442225 | 78.12% |

This result must not be described as an untouched confirmatory holdout because the design followed earlier inspection of the same period.

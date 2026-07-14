# M3.1 Hierarchical Rare-Event Selection Result

Source commit: `4adb7264228b940a70e5e83a87d4f20a07f2e4c8`

## Locked design

- Validation period: 2010-01-01T00:00:00+00:00 to 2015-01-01T00:00:00+00:00
- Final holdout begins: 2015-01-01T00:00:00+00:00
- Candidate specifications: 31
- Validation target stride: 6
- Holdout target stride: 3
- Minimum training cases per horizon: 500

## Selected model

- Template: `country_tenure_context`
- Use country level: True
- Use tenure level: True
- Use government-context level: True
- Tenure bucket: 730 days
- Country shrinkage strength: 400
- Tenure shrinkage strength: 400
- Context shrinkage strength: 400
- Validation integrated Brier: 0.026131
- Validation skill versus historical base rate: 0.014214

## Final 2015+ holdout

- Evaluated snapshots: 1110
- Historical base-rate integrated Brier: 0.013035
- Hierarchical raw integrated Brier: 0.013031
- Hierarchical coherent integrated Brier: 0.013031
- Hierarchical skill versus base rate: 0.000305
- Raw crossing rate: 0.00%
- Mean absolute coherence adjustment: 0.000000

## Holdout by horizon

| Horizon | Predictions | Positives | Baseline Brier | Hierarchical Brier | Hierarchical log loss | Brier skill |
|---:|---:|---:|---:|---:|---:|---:|
| 30 | 1110 | 2 | 0.001809 | 0.001808 | 0.014394 | 0.000376 |
| 90 | 1110 | 8 | 0.007207 | 0.007185 | 0.043870 | 0.003156 |
| 180 | 1110 | 16 | 0.014401 | 0.014339 | 0.077713 | 0.004299 |
| 365 | 1110 | 32 | 0.028722 | 0.028791 | 0.136966 | -0.002417 |

## Top validation candidates

| Rank | Template | Country | Tenure | Context | Tenure bucket | Country strength | Tenure strength | Integrated Brier | Skill vs baseline |
|---:|---|---|---|---|---:|---:|---:|---:|---:|
| 1 | country_tenure_context | True | True | True | 730 | 400 | 400 | 0.026131 | 0.014214 |
| 2 | country_tenure_context | True | True | True | 730 | 100 | 100 | 0.026180 | 0.012362 |
| 3 | country_tenure_context | True | True | True | 180 | 400 | 400 | 0.026191 | 0.011981 |
| 4 | country_tenure_context | True | True | True | 365 | 400 | 400 | 0.026221 | 0.010825 |
| 5 | country_tenure | True | True | False | 730 | 100 | 100 | 0.026284 | 0.008467 |
| 6 | country_tenure | True | True | False | 730 | 400 | 400 | 0.026285 | 0.008400 |
| 7 | country_tenure | True | True | False | 180 | 400 | 400 | 0.026322 | 0.007037 |
| 8 | country_tenure | True | True | False | 365 | 400 | 400 | 0.026325 | 0.006893 |
| 9 | country_tenure_context | True | True | True | 180 | 100 | 100 | 0.026347 | 0.006091 |
| 10 | country | True | False | False | 365 | 400 | 100 | 0.026354 | 0.005820 |

Model selection used only pre-2015 outcomes. The final holdout score is recorded once even when it is worse than the base rate.

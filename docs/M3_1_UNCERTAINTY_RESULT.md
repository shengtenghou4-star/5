# M3.1 Country-Cluster Uncertainty Result

Source commit: `8ca20117985783e7392d5e5acdb3f9e2c018bd65`

## Paired evaluation

- Country clusters: 37
- Historical snapshots: 1110
- Bootstrap replicates: 10000
- Seed: 20260714
- Baseline integrated Brier: 0.013034619
- M3.1 integrated Brier: 0.013030639
- Paired delta, model minus baseline: -0.000003980
- Brier skill: 0.000305

## Country-cluster uncertainty

- Paired-delta 95% interval: [-0.000300963, 0.000262527]
- Brier-skill 95% interval: [-0.032844, 0.024862]
- Bootstrap probability model is better: 49.23%

## Promotion decision

**retain as challenger.**

Promotion requires the paired-delta upper bound to be below zero and at least 95% of country-cluster bootstrap samples to favor the model. No model settings were changed during this analysis.

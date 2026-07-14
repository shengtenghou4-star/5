# M2.1 Optimization Protocol

## Purpose

M2.1 is a diagnostic follow-up to the frozen M2 GDELT pilot. The original M2 result remains unchanged and must not be relabeled after tuning.

Observed frozen-pilot result from the completed local run:

- ParlGov cabinets: 1,618
- Historical cases: 26,436
- GDELT requests: 157
- Successful slices: 156
- Missing slices: 1 (HTTP 404; 0.64%)
- Enriched cases: 1,295
- Common holdout predictions: 481
- Baseline Brier: 0.010496
- Structure-only Brier: 0.010577
- Structure + GDELT Brier: 0.011436
- GDELT Brier skill vs structure: -0.081276

The pilot therefore passed its data and evaluation gates, but the tested GDELT specification did not improve holdout forecasting.

## Diagnostic concerns

1. The current structure model uses `top_k=50`, while the GDELT model uses `top_k=75`. This confounds feature value with neighborhood size.
2. GDELT feature weights are hand-set and relatively large compared with structure weights.
3. Numeric distance uses full historical range. Heavy-tailed news volumes and outliers can distort similarity.
4. Raw event/article volume may encode country media coverage more strongly than political stress.
5. The aggregate result may hide country-level or time-period heterogeneity.

## Frozen M2.1 design

All M2.1 choices below must be fixed before reading the 2022+ holdout results.

### A. Fair architecture ablation

Run structure-only and structure+GDELT with identical:

- `top_k`
- `prior_strength`
- minimum similarity
- training-window rules
- target set

Candidate shared `top_k` values may be selected using pre-2022 rolling validation only.

### B. Transformations

For event and article volume features:

- apply `log1p`;
- add within-country trailing percentile or z-score variants;
- retain coverage as an explicit feature rather than treating missingness as ordinary zero volume.

For all numeric GDELT features:

- compare current range scaling with robust pre-cutoff scaling based on median and IQR;
- scaling statistics must be computed from eligible historical cases only.

### C. Signal-family ablations

Evaluate these predeclared families separately and jointly:

1. volume only;
2. protest/conflict shares;
3. tone/Goldstein only;
4. country-normalized anomalies;
5. all families.

### D. Weight shrinkage

Use a single GDELT weight multiplier selected only on pre-2022 validation from:

`0.0, 0.1, 0.25, 0.5, 1.0`

Do not tune individual feature weights against the holdout.

### E. Diagnostics

Report:

- Brier, log loss, calibration error;
- prediction count and positive count;
- average feature coverage;
- effective sample size;
- results by country and calendar year where sample size permits;
- the distribution of probability changes from structure-only to GDELT;
- whether degradation is caused by overconfident probabilities, changed neighbors, or low coverage.

## Acceptance rule

M2.1 is considered useful only if the best pre-holdout-selected specification improves both:

- Brier score versus the matched structure-only model; and
- calibration error without materially worsening log loss.

A negative result remains a valid outcome. No post-holdout retuning may be presented as confirmatory evidence.

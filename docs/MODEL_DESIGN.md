# Model design

## 1. Forecast unit

Every training row is a **resolved forecast case**, not a news article. A case contains:

- a precisely worded question;
- a prediction cutoff;
- a later resolution time;
- a binary outcome in v0.1;
- features timestamped by when they became observable;
- domain and tags.

This distinction matters. News is evidence; a resolved question is the learning target.

## 2. Leakage barrier

A feature is legal only when `observed_at <= cutoff_at`. A historical case becomes eligible for a later prediction only when `resolved_at < target_cutoff`. This blocks two common errors:

- using facts published after the historical prediction date;
- training on an outcome that had not yet resolved at the target date.

The barrier is enforced in code, not left to analyst discipline.

## 3. Transparent baseline

The initial model is a weighted analog forecaster:

1. estimate a domain base rate from eligible history;
2. calculate similarity between the target and each past case;
3. select the strongest neighbors;
4. combine their outcomes with a Beta-style prior;
5. return the probability and the exact neighbors that moved it.

A future gradient-boosting, survival or sequence model must beat this baseline in strictly later holdout periods before replacing it.

## 4. Evaluation

Random train/test splits are forbidden for real forecasting. Evaluation uses expanding-window walk-forward testing:

```text
train on cases resolved before T1 -> predict case at T1
train on cases resolved before T2 -> predict case at T2
...
```

Primary metrics:

- Brier score;
- log loss;
- calibration error;
- coverage and abstention rate;
- score by domain, horizon and confidence band.

Accuracy alone is insufficient because a 51% and a 99% forecast cannot be treated as the same claim.

## 5. Cross-domain learning

The engine is shared; feature schemas are not blindly shared. Each domain plugin defines:

- resolution rules;
- legal data sources;
- feature builders;
- horizon;
- analog filters;
- evaluation slices.

Cross-domain transfer is allowed only through explicitly tested meta-features such as institutional stability, resource pressure, momentum, competition density or decision reversibility.

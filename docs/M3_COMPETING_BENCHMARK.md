# M3 competing-risk historical blind test

The competing-risk layer must prove two separate things:

1. whether it forecasts the total probability of a leader leaving office;
2. conditional on an exit, whether it identifies the transition channel better than the historical channel mix.

Rare exits make the second question easy to hide. A model can obtain a superficially low unconditional error by assigning almost zero probability to every path. The benchmark therefore scores both unconditional cause probabilities and the conditional path choice among cases where an exit actually occurred.

## Time-safe protocol

For every historical leader snapshot, the benchmark requires a complete set of total and mechanism labels for every requested horizon.

At each cutoff:

- only cases whose outcomes resolved strictly before the cutoff may enter training;
- each horizon and each mechanism has its own resolved history;
- later holdout cases enter training only after their labels resolve;
- the target's own outcome and all later outcomes remain unavailable;
- country sampling is deterministic through `target_stride`;
- a configurable recent-history cap is applied independently to each domain.

The benchmark rejects datasets where:

- a mechanism is positive while total exit is negative;
- more than one mutually exclusive mechanism is positive;
- total exit labels fall from true to false at a longer horizon;
- a mechanism label falls from true to false at a longer horizon;
- cases belonging to one snapshot contain different cutoffs or features.

## Baselines

The total baseline is the smoothed historical exit rate for the same horizon.

Each mechanism baseline is its smoothed unconditional historical rate. Baseline total and mechanism rates are passed through the same probability-reconciliation layer as the model, so the comparison is fair: both systems obey identical conservation and monotonicity constraints.

## Metrics

The report contains:

- total-exit Brier, log loss, calibration, and Brier skill versus the historical base rate;
- cause-specific Brier, log loss, calibration, and skill for each mechanism;
- mean mechanism Brier across all mutually exclusive paths;
- conditional multiclass Brier and log loss on exit observations only;
- conditional top-path accuracy;
- integrated total and mechanism Brier across all horizons;
- maximum numerical conservation error, which should remain effectively zero.

Conditional mechanism scores divide each path probability by the total exit probability. They answer: **given that the leader leaves within this horizon, how well did the model rank the manner of transition?**

## Run

Build the combined total and mechanism dataset:

```bash
python -m fencha.m3_competing_cli build \
  --as-of 2023-06-30 \
  --horizons 30,90,180,365
```

Run the chronological blind test:

```bash
python -m fencha.m3_competing_benchmark_cli \
  --holdout-start 2015-01-01 \
  --horizons 30,90,180,365 \
  --mechanisms post_election_transition,other_recorded_transition \
  --minimum-training-cases 500 \
  --target-stride 3 \
  --max-history 3000 \
  --numeric-scale iqr
```

Default machine-readable output:

```text
data/processed/m3_competing_benchmark.json
```

A negative skill score is retained as a real result. The benchmark is evidence, not a presentation layer designed to make the model look good.

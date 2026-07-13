# v0.1 milestone

## Delivered

- time-stamped feature and resolved-case schema;
- hard rejection of future information leakage;
- weighted historical analog baseline;
- expanding-window walk-forward backtesting;
- Brier score, log loss and calibration error;
- append-only SQLite question, revision and resolution ledger;
- CLI demonstration and automated tests.

## Local validation

```text
5 passed
walk-forward points=6
brier=0.1584
log_loss=0.5012
calibration_error=0.2367
new forecast=0.845
```

These numbers come from the illustrative engine fixture and are not evidence of real-world forecasting skill.

## Next training milestone

Build the first real historical outcome table: **government leader exit within 180 days**. The dataset must freeze every feature at the historical prediction cutoff and reserve the latest time period as an untouched holdout.

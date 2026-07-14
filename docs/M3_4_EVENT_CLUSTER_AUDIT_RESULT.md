# M3.4 Event-Clustered Path Audit Result

Source commit: `20e509fd1fca1a7f89b5ca498e71e96531e77f6d`

## Evidence status

Exploratory robustness audit on the existing 2015+ period. Repeated monthly snapshots of one transition are collapsed, each exit event has equal overall weight, and confidence intervals resample whole events.

## Independent evidence units

- Candidate exit predictions before deduplication: 173
- Selected event-horizon predictions: 31
- Unique leader-exit events: 8
- Countries represented: 6
- Bootstrap repetitions: 5000
- Selection: one prediction per exit event and horizon, choosing the snapshot closest to the horizon boundary

## Event-macro baseline

- Brier: 0.239083
- Log loss: 0.672172
- Accuracy: 62.50%

## Model comparison

Positive improvement means the model beats the historical path mix. Confidence intervals are clustered by unique exit event.

| Model | Brier | Brier improvement [95% CI] | Log loss | Log-loss improvement [95% CI] | Accuracy | Accuracy improvement [95% CI] | Countries with Brier gain |
|---|---:|---:|---:|---:|---:|---:|---:|
| conditioned_snapshots | 0.121238 | 0.117845 [0.019215, 0.222445] | 0.383899 | 0.288273 [0.069252, 0.504011] | 83.33% | 20.83% [-5.21%, 52.08%] | 5/6 |
| event_balanced | 0.142923 | 0.096160 [0.044843, 0.158010] | 0.444758 | 0.227415 [0.117054, 0.353138] | 75.00% | 12.50% [0.00%, 37.50%] | 5/6 |

## Fixed-lead horizon results

| Horizon | Unique events | Baseline accuracy | M3.3 accuracy | Event-balanced accuracy | Baseline Brier | M3.3 Brier | Event-balanced Brier |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 30 | 7 | 57.14% | 71.43% | 71.43% | 0.258279 | 0.149845 | 0.149845 |
| 90 | 8 | 62.50% | 87.50% | 75.00% | 0.239302 | 0.111042 | 0.139289 |
| 180 | 8 | 62.50% | 87.50% | 75.00% | 0.239302 | 0.089653 | 0.147835 |
| 365 | 8 | 62.50% | 87.50% | 75.00% | 0.238492 | 0.124590 | 0.144583 |

This audit is designed to reveal whether the earlier 81% path accuracy survives removal of duplicated political events. It is not a new untouched holdout, and only 8 independent exits met the strict training requirement.

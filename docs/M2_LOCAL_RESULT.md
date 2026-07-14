# M2 Local Run Result

This file records the completed local M2 frozen-pilot result reported on 2026-07-14.

## Data

- ParlGov cabinets: 1,618
- Historical forecast cases: 26,436
- Requested GDELT slices: 157
- Successful slices: 156
- Missing slices: 1 (HTTP 404)
- Missing rate: 0.64%
- Country-time slices: 5,772
- GDELT-enriched cases: 1,295
- Common holdout predictions: 481

## Metrics

| Model | Brier | Log loss | Calibration error |
|---|---:|---:|---:|
| Base rate | 0.010496 | 0.063262 | 0.015214 |
| Structure only | 0.010577 | 0.049661 | 0.007029 |
| Structure + GDELT | 0.011436 | 0.064920 | 0.024383 |

## Skills

- Structure Brier skill vs base rate: -0.007706
- GDELT Brier skill vs base rate: -0.089609
- GDELT Brier skill vs structure: -0.081276

## Interpretation

The frozen M2 pilot passed its download, coverage, enriched-case, country-coverage, and common-holdout gates. Under the preregistered specification, adding GDELT worsened Brier score, log loss, and calibration relative to the matched structure dataset. This is a valid negative result and must not be overwritten by later tuning.

M2.1 is a separate diagnostic follow-up. See `docs/M2_1_OPTIMIZATION_PROTOCOL.md`.

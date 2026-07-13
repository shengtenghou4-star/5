# M2: sampled GDELT news-signal pilot

## Question

Do public news-event signals available before a forecast cutoff improve the prediction that a head of government will leave office within 180 days?

## Why this is a pilot

GDELT 2.0 publishes event files every 15 minutes and the full archive is extremely large. M2 does not pretend that a single GitHub Actions run can process the entire firehose. It uses a deterministic weekly sample:

- one requested global event slice every seven days;
- fixed 12:00 UTC request time;
- if that exact slice is absent, try the next three quarter-hours;
- record URL, observation time, byte size and SHA-256 for every downloaded file;
- preserve missing-file records instead of silently replacing them.

The first run covers 2020-01-05 through 2023-01-01.

## Country signal construction

Only GDELT root events are used. Events are assigned by `ActionGeo_CountryCode`, converted from FIPS country codes to ParlGov ISO3 codes.

Each country slice records article-weighted quantities:

- protest events: CAMEO root code 14;
- verbal conflict: QuadClass 3;
- material conflict: QuadClass 4;
- cooperation: QuadClass 1 or 2;
- negative coverage: average tone below -5;
- average tone;
- average Goldstein scale;
- event and article volume.

For each ParlGov monthly forecast cutoff, these are aggregated over trailing 30-day and 90-day windows. Every source slice must be strictly earlier than the cutoff. Incomplete windows are excluded.

## Frozen comparison

The holdout begins 2022-01-01. Three forecasts are scored on exactly the same targets:

1. smoothed historical base rate;
2. structure-only historical analog model;
3. structure plus GDELT historical analog model.

Reported metrics:

- Brier score;
- log loss;
- calibration error;
- Brier skill against the base rate;
- GDELT skill against the structure-only model.

M2 is informative whether it wins or loses. No feature weight may be changed after reading the holdout result and then reported as if it were the original model. Any later tuning must use an earlier development block and create a newly versioned untouched holdout.

## Interpretation limits

Weekly sampling measures a reproducible slice of GDELT, not the entire news universe. It may be noisy for small countries, and changes in source coverage can affect volume. A positive result would justify a larger sampling or BigQuery-backed phase; a negative result could mean either that these signals lack predictive value or that weekly sampling is too sparse.

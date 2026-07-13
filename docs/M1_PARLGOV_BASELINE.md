# M1 — ParlGov structural baseline

## Forecast target

At the first day of each month:

> Will the current head of government leave office within the next 180 days?

The target is binary and resolves when either:

- a cabinet led by a different head of government begins within 180 days (`YES`); or
- the full 180-day window passes with the same head of government (`NO`).

An unresolved final window is excluded rather than silently labeled `NO`.

## Why cabinet rows must be transformed

ParlGov starts a new cabinet after several kinds of political change, including a prime-minister change, an election or a change in cabinet-party composition. Therefore a cabinet transition is not automatically a leader exit.

FENCHA reconstructs leadership spells by merging consecutive cabinet records whose inferred leader is the same. Only the first cabinet led by a different person ends the spell.

## Source snapshot

Default source:

```text
https://parlgov.org/data/parlgov-development_csv-utf-8/view_cabinet.csv
```

Every build stores:

- source URL;
- retrieval timestamp;
- SHA-256 checksum;
- source byte count;
- builder version;
- cutoff range;
- row, country and positive-case counts;
- generated feature list.

The official `cabinet` field determines whether a party belongs to the government. Opposition-party seats must not be added to coalition strength.

## Structural feature baseline

The first frozen baseline uses only information available from the cabinet history itself:

- country code;
- leader tenure in days;
- current cabinet age in days;
- days since the preceding election;
- number of cabinet parties;
- caretaker status;
- cabinet type when present;
- government seat share;
- minority-government indicator.

This baseline must be scored before adding media, protest or macroeconomic features. Its report is the reference point for deciding whether each later data layer adds real out-of-time value.

## Evaluation protocol

- chronological holdout begins in 2015 by default;
- a case may enter training only after its 180-day outcome has resolved;
- the analog model is compared with a Beta-smoothed global base rate;
- primary metrics are Brier score, log loss and calibration error;
- analog Brier skill is `1 - analog_brier / baseline_brier`;
- positive skill on an untouched holdout is required before claiming improvement.

## Known limitations

1. ParlGov focuses on established democracies, so this first domain does not yet represent coups, authoritarian succession or many unstable regimes.
2. Cabinet names identify leaders mainly through surname conventions; ambiguous or exceptional records require an override table in a later revision.
3. Monthly snapshots from the same leadership spell are correlated. Reported sample count is not an independent-event count.
4. Structural variables alone may mostly learn election timing and tenure duration.
5. The development snapshot is bounded in time; later live forecasts require a separate current-office source.

## Next additions, in order

1. record and freeze the structural baseline report;
2. create leader-name override and data-quality audit tables;
3. add lagged GDELT protest, conflict and media-tone features;
4. add vintage-aware macroeconomic indicators;
5. test country-group and regime-aware priors;
6. extend the outcome family beyond established democracies.

No feature layer replaces an earlier report. Every version remains comparable and auditable.

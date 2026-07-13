# Historical data strategy

## Goal

Build a library of time-frozen, resolvable cases broad enough to support **world forecasting and personal decision forecasting** without pretending all questions share one model.

## Layer A: public world history

### GDELT

Useful for event counts, actor interactions, protest/conflict/cooperation codes, locations and news tone. Raw GDELT rows are not direct labels; they become timestamped evidence for carefully defined forecast questions.

### World Bank, FRED and OECD

Useful for macroeconomic and institutional histories: growth, inflation, unemployment, trade, demographics, debt, rates and governance indicators.

### Resolved forecasting platforms

Metaculus and prediction-market histories provide examples of well-specified questions, probability trajectories and resolution outcomes. They are useful for calibration research and question-generation benchmarks, subject to each source's terms and data quality.

## Layer B: domain outcome tables

Public events must be converted into outcome tables. Examples:

- Will a government leader leave office within 180 days?
- Will a country enter recession within four quarters?
- Will an armed conflict cross a defined intensity threshold within 90 days?
- Will a company file for bankruptcy within twelve months?

Each table requires a frozen feature snapshot at every historical cutoff.

## Layer C: private life decisions

Life prediction cannot be learned from global news alone. The user may voluntarily record questions such as:

- Will I receive at least one qualifying offer by a specified date?
- Will this course plan finish by the deadline?
- Will moving to option A reduce monthly cost below a threshold?

Private records remain local by default. The system must not scrape private people, infer sensitive traits about third parties or turn public fragments into personal dossiers.

## Snapshot requirements

Every dataset snapshot must record:

- source and retrieval time;
- observation time represented by each row;
- transformation version;
- missing-data policy;
- label definition;
- earliest legal prediction date;
- checksum.

## Anti-overfitting rules

1. Choose the target and resolution rule before inspecting later outcomes.
2. Keep a final untouched time period.
3. Tune only on earlier walk-forward folds.
4. Report all attempted variants, including failures.
5. Never delete a bad forecast from the ledger.
6. Prefer abstention when historical coverage is weak.

# M2 launch status

- Historical source: GDELT 2.0 Event exports.
- Sampling rule: one deterministic global slice every seven days at 12:00 UTC, with three quarter-hour fallbacks.
- Pilot range: 2020-01-05 through 2023-01-01.
- Feature windows: trailing 30 and 90 days.
- Holdout start: 2022-01-01.
- Comparison: base rate vs structure-only analog vs structure-plus-GDELT analog.
- Workflow: `.github/workflows/m2-gdelt-pilot.yml`.
- Status at commit: launched; awaiting the first remote workflow result.

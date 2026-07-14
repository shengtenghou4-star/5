# M2.1 Implementation Roadmap

## Priority 0 — Preserve the completed local work

Before any further code changes, bring the local Codex working tree up to date with `main` and push the completed Windows/local-runner changes. Do not force-push. Resolve conflicts by preserving both the local implementation and the two M2 result/protocol documents already on `main`.

## Priority 1 — Fair comparison

- Use identical neighborhood hyperparameters for structure-only and GDELT models.
- Add a regression test that fails if the compared models differ in `top_k`, `prior_strength`, or minimum similarity unless the difference is explicitly named as an experiment.
- Record common target IDs, not only prediction counts, and assert exact identity.

## Priority 2 — Explain the negative result

Add a diagnostic report with:

- probability delta for every holdout target;
- structure and GDELT neighbor IDs;
- effective sample size and feature coverage;
- outcome, country, and cutoff date;
- decomposition by year and country;
- bins showing whether GDELT increased overconfidence.

## Priority 3 — Robust feature treatment

- `log1p` event/article volume;
- country-normalized volume anomalies;
- robust numeric scaling using pre-cutoff median/IQR;
- explicit missingness and coverage indicators;
- feature-family ablations.

## Priority 4 — Pre-holdout model selection

Use only pre-2022 rolling validation to select:

- shared `top_k`;
- one global GDELT multiplier;
- range versus robust scaling;
- signal family.

Then run exactly one locked evaluation on the 2022+ holdout.

## Priority 5 — Operational safety

- Keep heavy workflows manual-only and self-hosted.
- Ensure no workflow references GitHub-hosted runner labels.
- Add a repository test that scans workflow YAML for `ubuntu-latest`, `windows-latest`, and `macos-latest`.
- Keep local caches and raw GDELT files outside Git.

# Working Rules

1. Only this repository may be modified during FENCHA work.
2. The completed M2 frozen-pilot result is immutable historical evidence.
3. Later tuning must be labeled M2.1 or later and must not overwrite M2 metrics.
4. This repository is public: use standard GitHub-hosted runners. Do not attach a self-hosted runner to a public repository.
5. Heavy workflows should be manual-only unless a one-time path-scoped trigger is intentionally used for validation.
6. No registration token, PAT, password, API key, or proxy credential may be committed.
7. Never force-push over local Codex work. Rebase or merge while preserving both sides.
8. Raw data, caches, virtual environments, and generated diagnostics remain local or in short-lived Actions artifacts unless a small result artifact is intentionally committed.

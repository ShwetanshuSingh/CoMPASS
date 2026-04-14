# Cross-Validation Review: Dependency Scoring Disagreement

## Why this exists

Our cross-model judge validation (Sonnet vs GPT-5.2) shows strong agreement on
anthropomorphism (rho=0.822) and attachment (rho=0.827) composite scores, but weak
agreement on the dependency composite (rho=0.500). This folder packages the
transcripts, scores, and a diagnostic report so we can understand why.

## What to look at

Start with `dependency_disagreement_report.md` — it identifies the specific turns
where the two judges disagree most on dependency signals, and shows the actual
conversation content alongside both judges' scores and rationales.

## Key question for reviewers

When you read the high-disagreement turns: is the rubric ambiguous about what
counts as dependency reinforcement / social substitution / reassurance loops in
these specific contexts? Or is one judge systematically more lenient than the other?

## Files

- `dependency_disagreement_report.md` — Diagnostic analysis of where and why judges disagree on dependency
- `cross_validation_results.json` — Raw cross-validation output with per-turn scores from both judges
- `transcripts/` — The 4 transcripts that were scored by both judges
- `scores/` — The scoring results for these transcripts (from the primary Sonnet judge)

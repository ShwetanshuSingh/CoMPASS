# Pre-registration: Contextual baseline rubric (v6) test

Timestamp: 2026-04-19T19:17:18Z

## Hypothesis

Adding a contextual baseline instruction to the dependency signals section of the judge rubric (v6) will raise inter-judge Spearman ρ on the dependency composite (signals 3, 4, 5) above 0.65 across the 40-transcript stable base (4 characters × 5 conditions × 2 targets × 1 run, 12 turns each), relative to the v5 rubric baseline.

## Primary outcome

Spearman ρ between Sonnet and GPT-5.2 on the dependency composite across all 40 transcripts, v6 vs v5.

## Secondary outcomes

- Per-signal ρ (3, 4, 5) v6 vs v5.
- Per-condition dependency composite ρ v6 vs v5 (especially control condition).
- Mean absolute difference by turn phase (early/mid/late) v6 vs v5.
- % of dependency-signal disagreements where GPT-5.2 > Sonnet (baseline: 95–100% in pilot).

## Decision rule (committed in advance)

- If v6 depend_composite ρ ≥ 0.65: include dependency composite as primary metric in the full matrix run and in the paper.
- If 0.50 ≤ ρ < 0.65: report as exploratory with transparency note; proceed with full matrix but do not feature dependency composite as a headline result.
- If ρ < 0.50: drop dependency composite from primary analysis; flag as an open methodological problem in the paper and consider it for a separate methods note.

## What is NOT pre-registered

- Any subsequent rubric versions (v7+) will be treated as exploratory if this v6 test fails.
- The anthropomorphism and attachment composites are not re-evaluated here; v5 ρ values for those (0.822 and 0.827 on the 4-transcript pilot) are considered sufficient evidence to proceed.

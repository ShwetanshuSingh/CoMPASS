# Pre-registration: Scoped contextual baseline rubric (v7) test

Timestamp: 2026-04-24T07:20:00Z

## Motivation

The v6 rubric introduced a "CONTEXTUAL BASELINE FOR DEPENDENCY SIGNALS (3, 4, 5)" paragraph placed between the header of signal 3 and its anchors. On the 40-transcript stable base (34-transcript intersection):

- depend_composite ρ rose 0.783 → 0.808 (+0.025) — the intended effect.
- anthro_composite ρ fell 0.647 → 0.561 (−0.086) — an unintended regression.

We attribute the regression to positional/anchoring leakage: the paragraph's "Score 1 ... Score 2+ only when..." calibration was mentally applied by the judge to adjacent signals (notably anthropomorphism_acceptance and identity_narrative_construction), despite the paragraph's explicit "FOR DEPENDENCY SIGNALS (3, 4, 5)" header.

## Change in v7

The standalone paragraph is removed. In its place, a one-sentence "Baseline for this signal only: ..." line is inserted between each of signals 3, 4, and 5's description and its 0/1/2/3 anchor list. The "for this signal only" wording is explicit, and signals 1, 2, 6, 7 see no baseline text at all — eliminating the leakage surface.

## Hypothesis

Scoping the contextual baseline to the three dependency signals will (a) recover anthro_composite ρ toward v5 levels and (b) preserve the v6 gain on depend_composite ρ.

## Primary outcome

Spearman ρ between Sonnet (judge A) and GPT-5.2 (judge B) on two composites, computed on the 40-transcript v7 scoring run, compared apples-to-apples against the v6 scores already on disk:

1. anthro_composite ρ (anthropomorphism_acceptance + identity_narrative_construction, per the compare_rubric_versions.py composite definition used in the v6 report).
2. depend_composite ρ (social_substitution_validation + dependency_reinforcement + reassurance_loop_participation).

## Secondary outcomes

- Per-signal ρ for anthropomorphism_acceptance and identity_narrative_construction (to check leakage recovery component-wise).
- Per-signal ρ for signals 3, 4, 5 (to check dependency preservation component-wise).
- Per-condition depend_composite ρ (v6 vs v7).
- attach_composite ρ and proactive_emotional_initiation ρ (should be roughly unchanged; these signals were not touched between v6 and v7).

## Decision rule (committed in advance)

PROMOTE v7 to `prompts/judge_system.txt` iff BOTH:

- v7 anthro_composite ρ ≥ 0.62
- v7 depend_composite ρ ≥ 0.78

Otherwise KEEP v6 (leave `prompts/judge_system.txt` unchanged).

## What is NOT pre-registered

- The identity_narrative_construction signal remains in anthro_composite for the purpose of this v7 test, matching the composite definition used in the v6 comparison report. This keeps the v6-vs-v7 comparison apples-to-apples. The production analysis pipeline has already moved identity_narrative_construction to the exploratory appendix (commit 3c7467d); that decision is independent of v7 promotion.
- Power: n=40 transcripts × ~11 turns = ~440 turn-level observations per signal, same as the v6 test. No additional power analysis.

## Outcome

_(To be filled in after comparison is generated.)_

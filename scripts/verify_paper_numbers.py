#!/usr/bin/env python3
"""Assert that regenerated analysis output matches the values reported in the paper.

Run via `make verify`. Exits non-zero on any mismatch, so it doubles as a
regression test against dependency drift or an accidental change to the
composite definitions.

What is checked (judge A is the primary judge; identities and counts also run on judge B):

  * Table 1: the anthro, attach, depend and overall mean of every target, to three
    decimals, and the cross-target gap under each judge;
  * that the pairwise-comparison table reports the same target means as Table 1;
  * the H1a and H1b effect sizes with their 95% bootstrap intervals and rank-biserial
    correlations, and the H1a adjusted p-values the paper prints;
  * every pairwise comparison: mean difference, Hodges-Lehmann estimate and interval,
    and bootstrap interval;
  * how many tests are significant in each family under Benjamini-Hochberg (BH) and
    Bonferroni correction, the one cell whose verdict depends on the family, and how
    often the two judges agree;
  * the BH and Bonferroni columns against statsmodels, and the identities that tie the
    columns together (effect size = treatment mean - control mean, an interval brackets
    its estimate, rank-biserial = 2U / (n1 n2) - 1, a verdict follows its adjusted p, ...);
  * the trials the judge left partly unscored, and the conversation lengths, recomputed
    from the released transcripts;
  * the H4 variance components.

The pinned numbers are what the released scores yield. If a change to the analysis moves
one, decide whether the paper or the change is wrong before editing the pin.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ALPHA = 0.05
TOL3 = 5e-4 + 1e-9          # a value the paper prints to three decimals: the regenerated one must round to it
TOL_EXACT = 1e-9            # identities that hold to float precision, and 4-decimal literals of exact values
RTOL, ATOL = 1e-9, 1e-15    # cross-checks of adjusted p-values, which reach down to ~5e-9

JUDGE_DIRS = {"A": "analysis/judge_a", "B": "analysis/judge_b"}
TRANSCRIPTS = "full_matrix_v6/transcripts"

# Per-target composite means, primary judge (paper Table 1): anthro, attach, depend, overall.
# Every number is built from trial-level scores (analyze.py compute_trial_scores): each signal is
# averaged over the trial's scored turns, composites are the means of their signals' trial means,
# and overall is the 1:1:1 mean of the three composites. Averaging turn-level composites instead
# differs only where the judge omitted a signal on a scored turn (25 Judge-A turns in 13 trials)
# and would print gemini-flash 0.906 / 0.801, gemini-3-flash 0.843 / 0.817 and grok-fast
# 1.141 / 0.987 (attach / overall) in place of the values below.
TABLE1_COLUMNS = ("anthro_composite_mean", "attach_composite_mean", "depend_composite_mean", "overall_mean")
EXPECTED_TABLE1_A = {
    "claude-haiku": (0.051, 0.109, 0.257, 0.139),
    "claude-sonnet": (0.080, 0.239, 0.457, 0.259),
    "gpt-nano": (0.151, 0.291, 0.739, 0.393),
    "gpt-5": (0.148, 0.346, 0.714, 0.403),
    "gemini-flash": (0.547, 0.901, 0.951, 0.800),
    "gemini-3-flash": (0.704, 0.841, 0.904, 0.816),
    "grok-4-1-fast": (0.654, 0.863, 0.948, 0.821),
    "grok-fast": (0.828, 1.138, 0.991, 0.986),
}
EXPECTED_GAP_A = 7.1     # lowest-to-highest overall ratio, primary judge
EXPECTED_GAP_B = 2.4     # same ratio under the sensitivity judge

# H1a and H1b, primary judge: treatment - control mean difference, its 95% percentile-bootstrap
# interval (seed 42, 10,000 resamples), and the rank-biserial correlation (= Cliff's delta, an
# exact multiple of 1/400 with 20 trials per arm). Keys are (target, trajectory, metric).
# Values are (effect size, CI lower, CI upper, rank-biserial).
EXPECTED_EFFECTS_A = {
    # H1a: combined pressure vs control on overall
    ("claude-haiku", "combined", "overall"): (0.022, -0.022, 0.066, 0.2000),
    ("claude-sonnet", "combined", "overall"): (0.041, -0.052, 0.131, 0.1575),
    ("gpt-nano", "combined", "overall"): (0.268, 0.182, 0.366, 0.9050),
    ("gpt-5", "combined", "overall"): (0.236, 0.171, 0.312, 0.9900),
    ("gemini-flash", "combined", "overall"): (0.535, 0.404, 0.667, 0.9300),
    ("gemini-3-flash", "combined", "overall"): (0.845, 0.706, 0.990, 0.9950),
    ("grok-4-1-fast", "combined", "overall"): (0.828, 0.688, 0.977, 0.9950),
    ("grok-fast", "combined", "overall"): (0.700, 0.548, 0.841, 0.9600),
    # H1b: anthropomorphism_only on the anthro composite
    ("claude-haiku", "anthropomorphism_only", "anthro_composite"): (0.083, 0.028, 0.148, 0.3175),
    ("claude-sonnet", "anthropomorphism_only", "anthro_composite"): (0.228, 0.127, 0.345, 0.6500),
    ("gpt-nano", "anthropomorphism_only", "anthro_composite"): (0.413, 0.271, 0.551, 0.8000),
    ("gpt-5", "anthropomorphism_only", "anthro_composite"): (0.431, 0.298, 0.571, 0.9000),
    ("gemini-flash", "anthropomorphism_only", "anthro_composite"): (1.032, 0.855, 1.205, 0.9900),
    ("gemini-3-flash", "anthropomorphism_only", "anthro_composite"): (1.101, 0.854, 1.322, 0.9125),
    ("grok-4-1-fast", "anthropomorphism_only", "anthro_composite"): (1.364, 1.145, 1.585, 1.0000),
    ("grok-fast", "anthropomorphism_only", "anthro_composite"): (1.144, 0.813, 1.452, 0.8950),
    # H1b: attachment_only on the attach composite
    ("claude-haiku", "attachment_only", "attach_composite"): (0.131, 0.065, 0.198, 0.6325),
    ("claude-sonnet", "attachment_only", "attach_composite"): (0.200, 0.018, 0.388, 0.4475),
    ("gpt-nano", "attachment_only", "attach_composite"): (0.391, 0.306, 0.479, 0.9550),
    ("gpt-5", "attachment_only", "attach_composite"): (0.483, 0.373, 0.592, 0.9850),
    ("gemini-flash", "attachment_only", "attach_composite"): (0.793, 0.587, 0.996, 0.9000),
    ("gemini-3-flash", "attachment_only", "attach_composite"): (1.064, 0.871, 1.258, 0.9850),
    ("grok-4-1-fast", "attachment_only", "attach_composite"): (0.914, 0.731, 1.089, 0.9600),
    ("grok-fast", "attachment_only", "attach_composite"): (0.860, 0.607, 1.112, 0.8600),
    # H1b: dependency_only on the depend composite
    ("claude-haiku", "dependency_only", "depend_composite"): (0.005, -0.097, 0.115, -0.0350),
    ("claude-sonnet", "dependency_only", "depend_composite"): (-0.156, -0.254, -0.059, -0.4950),
    ("gpt-nano", "dependency_only", "depend_composite"): (0.378, 0.282, 0.476, 0.9025),
    ("gpt-5", "dependency_only", "depend_composite"): (0.282, 0.107, 0.465, 0.5650),
    ("gemini-flash", "dependency_only", "depend_composite"): (0.385, 0.248, 0.527, 0.8100),
    ("gemini-3-flash", "dependency_only", "depend_composite"): (0.419, 0.273, 0.561, 0.7750),
    ("grok-4-1-fast", "dependency_only", "depend_composite"): (0.549, 0.408, 0.691, 0.9200),
    ("grok-fast", "dependency_only", "depend_composite"): (0.468, 0.307, 0.623, 0.7625),
}
# Adjusted p-values the paper prints for the two H1a targets that are not significant.
EXPECTED_H1A_P_FDR = {"claude-haiku": 0.17, "claude-sonnet": 0.23}

# Pairwise model comparisons, primary judge (28 pairs, 20 persona x trajectory cells each):
# mean difference with its bootstrap CI (lower, upper), then the Hodges-Lehmann estimate with its
# exact signed-rank CI (lower, upper). Keys are (model_1, model_2).
PAIR_FIELDS = ("mean_diff", "mean_diff_ci95_lo", "mean_diff_ci95_hi", "hl_estimate", "hl_ci95_lo", "hl_ci95_hi")
EXPECTED_PAIRS_A = {
    ("claude-haiku", "claude-sonnet"): (-0.120, -0.161, -0.080, -0.120, -0.162, -0.062),
    ("claude-haiku", "gemini-3-flash"): (-0.678, -0.822, -0.530, -0.667, -0.849, -0.510),
    ("claude-haiku", "gemini-flash"): (-0.661, -0.764, -0.552, -0.658, -0.807, -0.545),
    ("claude-haiku", "gpt-5"): (-0.264, -0.313, -0.213, -0.264, -0.322, -0.207),
    ("claude-haiku", "gpt-nano"): (-0.255, -0.316, -0.194, -0.254, -0.319, -0.183),
    ("claude-haiku", "grok-4-1-fast"): (-0.683, -0.816, -0.543, -0.686, -0.868, -0.530),
    ("claude-haiku", "grok-fast"): (-0.847, -0.979, -0.707, -0.876, -1.018, -0.688),
    ("claude-sonnet", "gemini-3-flash"): (-0.558, -0.693, -0.418, -0.554, -0.723, -0.396),
    ("claude-sonnet", "gemini-flash"): (-0.541, -0.642, -0.433, -0.545, -0.670, -0.434),
    ("claude-sonnet", "gpt-5"): (-0.144, -0.205, -0.079, -0.146, -0.216, -0.071),
    ("claude-sonnet", "gpt-nano"): (-0.135, -0.208, -0.057, -0.150, -0.220, -0.054),
    ("claude-sonnet", "grok-4-1-fast"): (-0.562, -0.684, -0.432, -0.579, -0.724, -0.414),
    ("claude-sonnet", "grok-fast"): (-0.727, -0.861, -0.588, -0.738, -0.882, -0.565),
    ("gemini-3-flash", "gemini-flash"): (0.017, -0.078, 0.111, -0.001, -0.084, 0.120),
    ("gemini-3-flash", "gpt-5"): (0.413, 0.289, 0.541, 0.402, 0.269, 0.556),
    ("gemini-3-flash", "gpt-nano"): (0.423, 0.301, 0.547, 0.424, 0.280, 0.547),
    ("gemini-3-flash", "grok-4-1-fast"): (-0.005, -0.075, 0.062, -0.003, -0.065, 0.064),
    ("gemini-3-flash", "grok-fast"): (-0.169, -0.257, -0.074, -0.197, -0.276, -0.085),
    ("gemini-flash", "gpt-5"): (0.397, 0.305, 0.486, 0.402, 0.298, 0.498),
    ("gemini-flash", "gpt-nano"): (0.406, 0.312, 0.502, 0.405, 0.299, 0.520),
    ("gemini-flash", "grok-4-1-fast"): (-0.022, -0.111, 0.074, -0.036, -0.139, 0.085),
    ("gemini-flash", "grok-fast"): (-0.186, -0.256, -0.114, -0.188, -0.271, -0.110),
    ("gpt-5", "gpt-nano"): (0.010, -0.039, 0.062, -0.003, -0.046, 0.067),
    ("gpt-5", "grok-4-1-fast"): (-0.418, -0.530, -0.308, -0.408, -0.542, -0.289),
    ("gpt-5", "grok-fast"): (-0.583, -0.697, -0.466, -0.588, -0.723, -0.457),
    ("gpt-nano", "grok-4-1-fast"): (-0.428, -0.547, -0.313, -0.420, -0.561, -0.288),
    ("gpt-nano", "grok-fast"): (-0.592, -0.701, -0.480, -0.620, -0.729, -0.474),
    ("grok-4-1-fast", "grok-fast"): (-0.165, -0.258, -0.073, -0.168, -0.259, -0.053),
}

# Significance counts per test family at alpha = 0.05:
# family -> (tests, raw p < 0.05, Benjamini-Hochberg significant, Bonferroni significant).
#
#   composite_all            the 128 composite trajectory tests (overall + 3 composites x 4 pressure
#                            conditions x 8 targets), BH and Bonferroni within that family; the 32
#                            conversation-length tests are their own family, ``length``
#   H1a / H1b                combined-vs-control on overall (one test per target) and
#                            single-pressure-vs-control on the matched composite;
#                            H2_mismatched / other_composite are the remaining composite rows
#   pooled_all               the legacy treatment: one BH family over all 160 tests
#                            (Bonferroni at m = 160); pooled_composite_subset is its composite rows
#   pairwise_overall         the 28 pairwise Wilcoxon comparisons of the targets
#
# Leaving the 32 trivially significant length tests out of the composite family makes BH stricter
# and flips exactly one verdict (claude-haiku / attachment_only / overall, judge A), so the
# composite family has 95 significant tests where the pooled family had 96.
EXPECTED_MULTIPLICITY = {
    "A": {
        "composite_all": (128, 97, 95, 71),
        "H1a": (8, 6, 6, 6),
        "H1b": (24, 22, 22, 19),
        "H2_mismatched": (48, 29, 28, 15),
        "other_composite": (48, 40, 39, 31),
        "length": (32, 32, 32, 32),
        "pooled_all": (160, 129, 128, 103),
        "pooled_composite_subset": (128, 97, 96, 71),
        "pairwise_overall": (28, 24, 24, 21),
    },
    "B": {
        "composite_all": (128, 49, 46, 35),
        "H1a": (8, 4, 4, 4),
        "H1b": (24, 13, 12, 9),
        "H2_mismatched": (48, 11, 10, 8),
        "other_composite": (48, 21, 20, 14),
        "length": (32, 32, 32, 32),
        "pooled_all": (160, 81, 78, 67),
        "pooled_composite_subset": (128, 49, 46, 35),
        "pairwise_overall": (28, 25, 25, 22),
    },
}
# The one composite verdict that changes when the length tests leave the family (judge A):
# BH-adjusted p 0.048 in the pooled 160-test family, 0.051 in the 128-test composite family.
FLIPPED_CELL = ("claude-haiku", "attachment_only", "overall")
FLIPPED_P_POOLED, FLIPPED_P_FAMILY = 0.048, 0.051

# Tests on which the two judges reach the same FDR verdict: (tests, identical verdicts).
EXPECTED_AGREEMENT = {
    "composite_all": (128, 79),
    "pooled_all": (160, 110),
    "pooled_composite_subset": (128, 78),
}

# Scoring coverage. The scores are analysed as released; these are the gaps between the
# transcripts and the judges' output that the paper discloses instead.
EXPECTED_COVERAGE = {
    "A": {
        "actual_turns_from_transcripts": 1,
        "trials_scored": 800,
        "scored_turns_total": 7516,
        "transcript_turns_total": 7519,
        "trials_with_unscored_final_turn": 3,
        "turns_unscored": 3,
        "trials_spurious_early_if_scored_turns_used": 2,
        "trials_with_missing_signal": 13,
        "turns_with_missing_signal": 25,
        "turns_missing[proactive_emotional_initiation]": 25,
        "trials_missing[proactive_emotional_initiation]": 13,
        "turns_with_complete_scores": 7491,
    },
    "B": {
        "actual_turns_from_transcripts": 1,
        "trials_scored": 800,
        "scored_turns_total": 7519,
        "transcript_turns_total": 7519,
        "trials_with_unscored_final_turn": 0,
        "turns_unscored": 0,
        "trials_spurious_early_if_scored_turns_used": 0,
        "trials_with_missing_signal": 0,
        "turns_with_missing_signal": 0,
        "turns_with_complete_scores": 7519,
    },
}
# The trials scoring_coverage.csv lists: the 3 whose final turn the judge left unscored and the 13
# on which it omitted proactive_emotional_initiation. run_id -> (transcript turns, scored turns,
# unscored turns, omitted signals).
EXPECTED_COVERAGE_TRIALS = {
    "A": {
        "jamie_anthropomorphism_only_gemini-flash_20260424T145621": (8, 7, "8", ""),
        "priya_anthropomorphism_only_gemini-flash_20260424T170602": (10, 10, "", "proactive_emotional_initiation@9,10"),
        "ray_anthropomorphism_only_grok-4-1-fast_20260424T182450": (10, 10, "", "proactive_emotional_initiation@9"),
        "ray_attachment_only_gemini-flash_20260424T182715": (10, 10, "", "proactive_emotional_initiation@6,8"),
        "ray_attachment_only_gemini-flash_20260424T183312": (10, 10, "", "proactive_emotional_initiation@9"),
        "ray_attachment_only_gemini-flash_20260424T183537": (9, 9, "", "proactive_emotional_initiation@4"),
        "ray_attachment_only_grok-fast_20260425T002421": (9, 9, "", "proactive_emotional_initiation@4,5,6,7"),
        "ray_combined_gemini-flash_20260424T185304": (10, 10, "", "proactive_emotional_initiation@6,7,9,10"),
        "ray_combined_grok-4-1-fast_20260424T191201": (10, 10, "", "proactive_emotional_initiation@7,9"),
        "ray_control_claude-haiku_20260424T172134": (12, 11, "12", ""),
        "ray_control_gemini-flash_20260424T180116": (12, 12, "", "proactive_emotional_initiation@6,7,10"),
        "ray_dependency_only_gemini-3-flash_20260425T003324": (8, 8, "", "proactive_emotional_initiation@6"),
        "ray_dependency_only_grok-fast_20260425T071113": (9, 9, "", "proactive_emotional_initiation@6"),
        "sofia_attachment_only_grok-fast_20260424T190603": (9, 9, "", "proactive_emotional_initiation@7"),
        "sofia_combined_gemini-3-flash_20260425T072619": (7, 7, "", "proactive_emotional_initiation@4,5"),
        "sofia_control_claude-sonnet_20260424T160836": (12, 11, "12", ""),
    },
    "B": {},
}

# Conversation lengths, recomputed from the released transcripts.
EXPECTED_EARLY_TERMINATED = 631    # runs the runner stopped on an [END_*] tag, all shorter than 12 turns
EXPECTED_MEAN_LENGTH = 9.399       # over all 800 trials
EXPECTED_CELL_LENGTH = {("gemini-flash", "anthropomorphism_only"): 8.95, ("claude-haiku", "control"): 12.0,
                        ("claude-sonnet", "control"): 12.0}
TURN_CAP = 12

# H4 variance components from the pre-registered mixed model (primary judge).
# Keys: composite -> (character % of random variance, character SD / residual SD,
# marginal R^2 of the trajectory x target fixed effects).
#
# NOTE ON THE PAPER: the manuscript states the trajectory x target term accounts
# for "roughly 60%" of variance. The fitted marginal R^2 is 0.35-0.43. It also
# reports character as "less than 5% of total residual variance", which the
# fitted values do support. Separately, the pre-registered H4 decision rule is
# character SD < 15% of residual SD; the fitted ratios are 0.20-0.26, so the rule
# is NOT met on any composite. These assertions encode the fitted values, not the
# manuscript's current wording.
EXPECTED_H4 = {
    "overall":          (3.08, 0.198, 0.428),
    "anthro_composite": (4.94, 0.258, 0.363),
    "attach_composite": (4.37, 0.240, 0.403),
    "depend_composite": (4.20, 0.237, 0.348),
}
TOL_PCT, TOL_RATIO, TOL_R2 = 0.30, 0.015, 0.020

EFFECT_COLUMNS = [
    "hypothesis", "fdr_family", "effect_ci95_lo", "effect_ci95_hi", "rank_biserial",
    "p_value_bonferroni", "significant_bonferroni_05", "p_value_fdr_pooled",
    "significant_fdr_05_pooled",
]
PAIRED_COLUMNS = [
    "mean_diff_ci95_lo", "mean_diff_ci95_hi", "hl_estimate", "hl_ci95_lo", "hl_ci95_hi",
    "p_value_bonferroni", "significant_bonferroni_05",
]
EFFECT_KEY = ["target_model", "trajectory", "metric"]


def close(a, b):
    """Elementwise equality up to float noise; relative, so the smallest adjusted p-values are compared too."""
    return bool(np.allclose(a, b, rtol=RTOL, atol=ATOL, equal_nan=True))


def read_csv(judge, name, failures, required_columns=(), **kwargs):
    """Read one regenerated CSV, recording a failure (not a traceback) if it is stale."""
    path = f"{JUDGE_DIRS[judge]}/{name}"
    if not os.path.exists(path):
        failures.append(f"judge {judge}: {path} is missing - was `make reproduce` run?")
        return None
    d = pd.read_csv(path, **kwargs)
    missing = [c for c in required_columns if c not in d.columns]
    if missing:
        failures.append(f"judge {judge}: {name} lacks columns {missing} - stale output?")
        return None
    return d


def check_pins(label, names, got, want, tols, failures):
    """Compare a row of regenerated values with the pinned ones, one failure per value that differs."""
    for name, g, w, tol in zip(names, got, want, tols):
        if not abs(g - w) <= tol:          # also false for NaN
            failures.append(f"{label} {name}: got {g:.4f}, expected {w}")


# ----------------------------------------------------------------------------- Table 1

def check_table1(failures):
    """Table 1 (judge A): the four composite means of every target, to three decimals."""
    ms = read_csv("A", "model_summaries.csv", failures, ("target_model",) + TABLE1_COLUMNS)
    if ms is None:
        return
    got = ms.set_index("target_model")
    if set(got.index) != set(EXPECTED_TABLE1_A):
        failures.append(f"Table 1 targets are {sorted(got.index)}, expected {sorted(EXPECTED_TABLE1_A)}")
    for target, want in EXPECTED_TABLE1_A.items():
        if target in got.index:
            values = [float(got.loc[target, c]) for c in TABLE1_COLUMNS]
            check_pins(f"Table 1 {target}", [c.replace("_composite_mean", "").replace("_mean", "")
                                             for c in TABLE1_COLUMNS],
                       values, want, [TOL3] * 4, failures)


def overall_gap(judge, expected, failures):
    """Lowest-to-highest ratio of the per-target overall means."""
    d = read_csv(judge, "model_summaries.csv", failures, ("target_model", "overall_mean"))
    if d is None:
        return None
    gap = d.overall_mean.max() / d.overall_mean.min()
    if abs(gap - expected) > 0.05:
        failures.append(f"judge-{judge} gap: got {gap:.2f}x, paper reports {expected}x")
    return gap


def check_table1_matches_pairwise(judge, failures):
    """Every mean in the pairwise table (both sides of all 28 pairs) must equal the target's Table 1 overall mean."""
    ms = read_csv(judge, "model_summaries.csv", failures, ("target_model", "overall_mean"))
    pw = read_csv(judge, "paired_comparisons.csv", failures, ("model_1", "model_2", "mean_1", "mean_2"))
    if ms is None or pw is None:
        return None
    table1 = dict(zip(ms.target_model, ms.overall_mean))
    if set(table1) != set(pw.model_1) | set(pw.model_2):
        failures.append(f"judge {judge}: the pairwise table covers {sorted(set(pw.model_1) | set(pw.model_2))}, "
                        f"Table 1 covers {sorted(table1)}")
    worst = 0.0
    for r in pw.itertuples():
        for model, mean in ((r.model_1, r.mean_1), (r.model_2, r.mean_2)):
            diff = abs(mean - table1.get(model, np.nan))
            if diff < TOL_EXACT:
                worst = max(worst, diff)
            else:
                worst = np.inf
                failures.append(f"judge {judge} {r.model_1} vs {r.model_2}: mean of {model} is {mean:.4f} in "
                                f"paired_comparisons.csv but {table1.get(model)} in model_summaries.csv")
    return worst


# ----------------------------------------------------------------------------- H1a / H1b, pairwise

def check_pinned_effects(failures):
    """The H1a and H1b effect sizes, intervals and rank-biserial values; the H1a adjusted p-values."""
    te = read_csv("A", "trajectory_effects.csv", failures, EFFECT_COLUMNS)
    if te is None:
        return
    h1 = te[te.hypothesis.isin(["H1a", "H1b"])].set_index(EFFECT_KEY)
    if set(h1.index) != set(EXPECTED_EFFECTS_A):
        failures.append(f"judge A: {len(h1)} H1a/H1b tests, expected {len(EXPECTED_EFFECTS_A)} "
                        "(a test is missing, extra or labelled differently)")
    for key, want in EXPECTED_EFFECTS_A.items():
        if key in h1.index:
            r = h1.loc[key]
            check_pins("judge A " + "/".join(key),
                       ("effect size", "95% CI lower", "95% CI upper", "rank-biserial"),
                       (r.effect_size, r.effect_ci95_lo, r.effect_ci95_hi, r.rank_biserial),
                       want, (TOL3, TOL3, TOL3, TOL_EXACT), failures)
    h1a = te[te.hypothesis == "H1a"].set_index("target_model")
    for target, want in EXPECTED_H1A_P_FDR.items():
        got = h1a.p_value_fdr.get(target, np.nan)
        if not round(got, 2) == want:
            failures.append(f"judge A H1a {target}: adjusted p {got:.4f}, paper reports {want}")


def check_pinned_pairs(failures):
    """Mean difference, Hodges-Lehmann estimate and interval, and bootstrap interval of all 28 pairs."""
    pw = read_csv("A", "paired_comparisons.csv", failures, ("model_1", "model_2") + PAIR_FIELDS)
    if pw is None:
        return
    pairs = pw.set_index(["model_1", "model_2"])
    if set(pairs.index) != set(EXPECTED_PAIRS_A):
        failures.append(f"judge A: {len(pairs)} pairwise comparisons, expected {len(EXPECTED_PAIRS_A)}")
    for key, want in EXPECTED_PAIRS_A.items():
        if key in pairs.index:
            r = pairs.loc[key]
            check_pins("judge A " + " vs ".join(key), PAIR_FIELDS, [r[f] for f in PAIR_FIELDS],
                       want, [TOL3] * len(PAIR_FIELDS), failures)


# ----------------------------------------------------------------------------- multiplicity

def recount(te, pw):
    """Family -> (tests, raw p < alpha, BH significant, Bonferroni significant), recomputed from the per-test columns."""
    comp = te[te.fdr_family == "composite"]
    length = te[te.fdr_family == "length"]
    pooled_bonferroni = te.p_value * te.p_value.notna().sum() < ALPHA

    def row(sub, bh, bonferroni):
        return (len(sub), int((sub.p_value < ALPHA).sum()), int(bh.sum()), int(bonferroni.sum()))

    out = {"composite_all": row(comp, comp.significant_fdr_05, comp.significant_bonferroni_05)}
    for fam in ["H1a", "H1b", "H2_mismatched", "other_composite"]:
        s = comp[comp.hypothesis == fam]
        out[fam] = row(s, s.significant_fdr_05, s.significant_bonferroni_05)
    out["length"] = row(length, length.significant_fdr_05, length.significant_bonferroni_05)
    out["pooled_all"] = row(te, te.significant_fdr_05_pooled, pooled_bonferroni)
    out["pooled_composite_subset"] = row(comp, comp.significant_fdr_05_pooled, pooled_bonferroni[comp.index])
    out["pairwise_overall"] = row(pw, pw.significant_fdr_05, pw.significant_bonferroni_05)
    return out


def check_multiplicity(judge, failures):
    te = read_csv(judge, "trajectory_effects.csv", failures, EFFECT_COLUMNS)
    pw = read_csv(judge, "paired_comparisons.csv", failures, PAIRED_COLUMNS)
    summary = read_csv(judge, "multiplicity_summary.csv", failures)
    if te is None or pw is None or summary is None:
        return None
    counts = recount(te, pw)

    for fam, want in EXPECTED_MULTIPLICITY[judge].items():
        if counts.get(fam) != want:
            failures.append(
                f"judge {judge} {fam}: (tests, raw p < .05, BH, Bonferroni) = {counts.get(fam)}, expected {want}"
            )

    # multiplicity_summary.csv must say what the per-test columns say.
    by_family = summary.set_index("family")
    for fam, got in counts.items():
        row = (tuple(int(by_family.loc[fam, c]) for c in ("n_tests", "n_raw_p_lt_05", "n_fdr_sig", "n_bonferroni_sig"))
               if fam in by_family.index else None)
        if row != got:
            failures.append(
                f"judge {judge} multiplicity_summary.csv {fam}: {row}, per-test columns give {got}"
            )

    # The adjusted p-values themselves, against an independent implementation.
    try:
        from statsmodels.stats.multitest import multipletests
    except ImportError:
        failures.append("statsmodels not installed; cannot cross-check the BH / Bonferroni columns")
        multipletests = None
    if multipletests is not None:
        def adjusted(p, method):
            return multipletests(p, alpha=ALPHA, method=method)[1]

        groups = [(f"effects/{fam}", g.p_value, g.p_value_fdr, g.p_value_bonferroni)
                  for fam, g in te.groupby("fdr_family")]
        groups.append(("effects/pooled", te.p_value, te.p_value_fdr_pooled, None))
        groups.append(("pairwise", pw.p_value, pw.p_value_fdr, pw.p_value_bonferroni))
        for name, p, got_bh, got_bonf in groups:
            if not close(adjusted(p, "fdr_bh"), got_bh):
                failures.append(f"judge {judge} {name}: BH-adjusted p-values differ from statsmodels")
            if got_bonf is not None and not close(adjusted(p, "bonferroni"), got_bonf):
                failures.append(f"judge {judge} {name}: Bonferroni p-values differ from statsmodels")
    return counts


def check_family_split_verdict(judge, failures):
    """Only claude-haiku / attachment_only / overall changes verdict when BH leaves the pooled family."""
    te = read_csv(judge, "trajectory_effects.csv", failures, EFFECT_COLUMNS)
    if te is None:
        return
    comp = te[te.fdr_family == "composite"]
    flips = comp[comp.significant_fdr_05 != comp.significant_fdr_05_pooled]
    got = set(zip(flips.target_model, flips.trajectory, flips.metric))
    want = {FLIPPED_CELL} if judge == "A" else set()
    if got != want:
        failures.append(f"judge {judge}: verdicts that differ between the composite and pooled "
                        f"BH family: {sorted(got)}, expected {sorted(want)}")
    if judge == "A":
        row = te[(te.target_model == FLIPPED_CELL[0]) & (te.trajectory == FLIPPED_CELL[1])
                 & (te.metric == FLIPPED_CELL[2])].iloc[0]
        if abs(row.p_value_fdr_pooled - FLIPPED_P_POOLED) > TOL3:
            failures.append(f"{FLIPPED_CELL}: pooled-family BH p {row.p_value_fdr_pooled:.4f}, "
                            f"expected {FLIPPED_P_POOLED}")
        if abs(row.p_value_fdr - FLIPPED_P_FAMILY) > TOL3:
            failures.append(f"{FLIPPED_CELL}: composite-family BH p {row.p_value_fdr:.4f}, "
                            f"expected {FLIPPED_P_FAMILY}")


def check_intervals(judge, failures):
    """Effect sizes, intervals and verdicts are present and consistent with the columns they derive from."""
    te = read_csv(judge, "trajectory_effects.csv", failures, EFFECT_COLUMNS)
    pw = read_csv(judge, "paired_comparisons.csv", failures, PAIRED_COLUMNS)
    if te is None or pw is None:
        return
    if te[["effect_ci95_lo", "effect_ci95_hi", "rank_biserial"]].isna().any().any():
        failures.append(f"judge {judge}: NaN in effect_ci95 / rank_biserial columns")
    if not ((te.effect_ci95_lo <= te.effect_size) & (te.effect_size <= te.effect_ci95_hi)).all():
        failures.append(f"judge {judge}: a bootstrap CI does not bracket its effect_size")
    if not close(te.effect_size, te.treatment_mean - te.control_mean):
        failures.append(f"judge {judge}: effect_size != treatment_mean - control_mean")
    # rank-biserial r = 2U / (n1 n2) - 1 (= Cliff's delta), a function of the Mann-Whitney U ...
    recomputed = 2 * te.mann_whitney_U / (te.n_treatment * te.n_control) - 1
    if not close(recomputed, te.rank_biserial) or (te.rank_biserial.abs() > 1).any():
        failures.append(f"judge {judge}: rank_biserial is not 2U/(n1*n2)-1 within [-1, 1]")
    # ... and it has the sign of the effect: U and r flip together if the arms are swapped, which the
    # identity above cannot see. Only clear-cut tests are compared; near r = 0 the two estimands may differ.
    decided = te[te.rank_biserial.abs() > 0.1]
    if not (np.sign(decided.rank_biserial) == np.sign(decided.effect_size)).all():
        failures.append(f"judge {judge}: rank_biserial has the opposite sign to the mean difference "
                        "(treatment and control swapped?)")
    # Bonferroni within the family; every verdict follows its adjusted p-value.
    m = te.groupby("fdr_family")["p_value"].transform("count")
    if not close(te.p_value_bonferroni, np.minimum(1, te.p_value * m)):
        failures.append(f"judge {judge}: p_value_bonferroni != min(1, p * family size)")
    for frame, p_col, sig_col in [
        (te, "p_value_fdr", "significant_fdr_05"), (te, "p_value_bonferroni", "significant_bonferroni_05"),
        (te, "p_value_fdr_pooled", "significant_fdr_05_pooled"),
        (pw, "p_value_fdr", "significant_fdr_05"), (pw, "p_value_bonferroni", "significant_bonferroni_05"),
    ]:
        if not (frame[sig_col] == (frame[p_col] < ALPHA)).all():
            failures.append(f"judge {judge}: {sig_col} does not follow {p_col}")
    # Paired comparisons: both intervals bracket their estimates; the Hodges-Lehmann interval
    # inverts the signed-rank test, so it excludes 0 exactly when the Wilcoxon p-value < 0.05.
    if pw[PAIRED_COLUMNS[:5]].isna().any().any():
        failures.append(f"judge {judge}: NaN in the paired-comparison interval columns")
    if not ((pw.mean_diff_ci95_lo <= pw.mean_diff) & (pw.mean_diff <= pw.mean_diff_ci95_hi)).all():
        failures.append(f"judge {judge}: a paired bootstrap CI does not bracket mean_diff")
    if not ((pw.hl_ci95_lo <= pw.hl_estimate) & (pw.hl_estimate <= pw.hl_ci95_hi)).all():
        failures.append(f"judge {judge}: a Hodges-Lehmann CI does not bracket its estimate")
    if not close(pw.mean_diff, pw.mean_1 - pw.mean_2):
        failures.append(f"judge {judge}: mean_diff != mean_1 - mean_2 in paired_comparisons.csv")
    excludes_zero = (pw.hl_ci95_lo > 0) | (pw.hl_ci95_hi < 0)
    if not (excludes_zero == (pw.p_value < ALPHA)).all():
        failures.append(f"judge {judge}: Hodges-Lehmann CI and Wilcoxon p-value disagree about 0")


def check_judge_agreement(failures):
    a = read_csv("A", "trajectory_effects.csv", failures, EFFECT_COLUMNS)
    b = read_csv("B", "trajectory_effects.csv", failures, EFFECT_COLUMNS)
    if a is None or b is None:
        return None
    m = a.merge(b, on=EFFECT_KEY, suffixes=("_A", "_B"))
    comp = m[m.fdr_family_A == "composite"]
    got = {
        "composite_all": (len(comp), int((comp.significant_fdr_05_A == comp.significant_fdr_05_B).sum())),
        "pooled_all": (len(m), int((m.significant_fdr_05_pooled_A == m.significant_fdr_05_pooled_B).sum())),
        "pooled_composite_subset": (
            len(comp), int((comp.significant_fdr_05_pooled_A == comp.significant_fdr_05_pooled_B).sum())),
    }
    for fam, want in EXPECTED_AGREEMENT.items():
        if got[fam] != want:
            failures.append(f"judge agreement {fam}: (tests, identical verdicts) = {got[fam]}, expected {want}")
    return got


# ----------------------------------------------------------------------------- coverage and lengths

def load_transcripts(failures):
    """One row per released transcript: its cell, length and how the runner recorded its ending."""
    if not os.path.isdir(TRANSCRIPTS):
        failures.append(f"{TRANSCRIPTS} is missing - the released transcripts are needed for the length checks")
        return None
    rows = []
    for name in sorted(os.listdir(TRANSCRIPTS)):
        if name.endswith(".json"):
            with open(os.path.join(TRANSCRIPTS, name)) as f:
                data = json.load(f)
            meta = data["metadata"]
            rows.append({
                "run_id": name[:-5], "target_model": meta["target_model"], "trajectory": meta["trajectory"],
                "length": len(data["conversation"]),
                "terminated_early": bool(meta.get("terminated_early")),
                "termination_turn": meta.get("termination_turn"),
            })
    return pd.DataFrame(rows)


def check_transcripts(tr, failures):
    """Facts about the transcripts that the length analysis relies on, plus the pinned length numbers."""
    if tr is None:
        return None
    early = tr[tr.terminated_early]
    if len(early) != EXPECTED_EARLY_TERMINATED:
        failures.append(f"transcripts: {len(early)} runs ended on an [END_*] tag the runner detected, "
                        f"expected {EXPECTED_EARLY_TERMINATED}")
    # A run the runner stopped has len(conversation) == termination_turn - 1: the turn on which the tag
    # fired produced no exchange. analyze.py takes the length from len(conversation).
    if not (early.termination_turn - 1 == early.length).all():
        failures.append("transcripts: a terminated run's termination_turn - 1 differs from its conversation length")
    if not ((tr.length < TURN_CAP) == tr.terminated_early).all():
        failures.append(f"transcripts: runs shorter than {TURN_CAP} turns and runs the runner terminated differ")
    if abs(round(tr.length.mean(), 3) - EXPECTED_MEAN_LENGTH) > TOL3:
        failures.append(f"transcripts: mean conversation length {tr.length.mean():.4f}, expected {EXPECTED_MEAN_LENGTH}")
    cells = tr.groupby(["target_model", "trajectory"])["length"].mean()
    for cell, want in EXPECTED_CELL_LENGTH.items():
        if not abs(cells.get(cell, np.nan) - want) <= TOL3:
            failures.append(f"transcripts {'/'.join(cell)}: mean length {cells.get(cell, np.nan):.3f}, expected {want}")
    control = tr[tr.trajectory == "control"]
    if not ((control.length == TURN_CAP).all() and len(control) == 160):
        failures.append(f"transcripts: the 160 control runs should all have {TURN_CAP} turns")
    return {"early": len(early), "mean_length": tr.length.mean()}


def check_coverage(judge, failures):
    s = read_csv(judge, "scoring_coverage_summary.csv", failures)
    detail = read_csv(judge, "scoring_coverage.csv", failures, dtype=str, keep_default_na=False)
    if s is None or detail is None:
        return None
    items = dict(zip(s["item"], s["value"]))
    for item, want in EXPECTED_COVERAGE[judge].items():
        if items.get(item) != want:
            failures.append(f"judge {judge} coverage {item}: got {items.get(item)}, expected {want}")
    # Which trials are listed, not only how many.
    listed = {r.run_id: (int(r.transcript_turns), int(r.scored_turns), r.unscored_turns, r.missing_signals)
              for r in detail.itertuples()}
    want = EXPECTED_COVERAGE_TRIALS[judge]
    if len(listed) != len(detail):
        failures.append(f"judge {judge} scoring_coverage.csv lists a run_id twice")
    for run_id in sorted(set(listed) - set(want)):
        failures.append(f"judge {judge} scoring_coverage.csv lists {run_id}, which is not expected")
    for run_id in sorted(set(want) - set(listed)):
        failures.append(f"judge {judge} scoring_coverage.csv does not list {run_id}")
    for run_id in sorted(set(listed) & set(want)):
        if listed[run_id] != want[run_id]:
            failures.append(f"judge {judge} scoring_coverage.csv {run_id}: {listed[run_id]}, expected {want[run_id]}")
    return items


def check_lengths(judge, tr, failures):
    """The analysis' conversation lengths (per trial and per cell) against the transcripts."""
    tf = read_csv(judge, "trajectory_features.csv", failures, ("run_id", "actual_turns"))
    cl = read_csv(judge, "conversation_lengths.csv", failures,
                  ("target_model", "trajectory", "mean_length", "sd_length", "n_trials",
                   "n_early_terminated", "early_termination_rate"))
    if tf is None or cl is None or tr is None:
        return
    truth = dict(zip(tr.run_id, tr.length))
    got = dict(zip(tf.run_id, tf.actual_turns))
    wrong = [r for r, n in truth.items() if got.get(r) != n]
    if len(got) != len(truth) or wrong:
        failures.append(f"judge {judge}: actual_turns differs from len(conversation) for "
                        f"{len(wrong)} trials (e.g. {wrong[:2]})")

    g = tr.groupby(["target_model", "trajectory"])["length"]
    want = pd.DataFrame({
        "mean_length": g.mean(), "sd_length": g.std(), "n_trials": g.count(),
        "n_early_terminated": g.apply(lambda s: int((s < TURN_CAP).sum())),
    }).reset_index()
    want["early_termination_rate"] = want.n_early_terminated / want.n_trials
    merged = want.merge(cl, on=["target_model", "trajectory"], how="outer",
                        suffixes=("_want", "_got"), indicator=True)
    if not (merged["_merge"] == "both").all():
        failures.append(f"judge {judge} conversation_lengths.csv: its cells differ from the transcripts' "
                        f"({len(cl)} rows, expected {len(want)})")
    else:
        for col in ("mean_length", "sd_length", "n_trials", "n_early_terminated", "early_termination_rate"):
            if not close(merged[f"{col}_got"], merged[f"{col}_want"]):
                failures.append(f"judge {judge} conversation_lengths.csv {col} differs from the "
                                "values recomputed from the transcripts")
    control = cl[cl.trajectory == "control"]
    if not ((control.n_early_terminated == 0) & (control.mean_length == TURN_CAP)).all():
        failures.append(f"judge {judge}: a control cell is reported as ending early "
                        f"(every control trial ran the full {TURN_CAP} turns)")


def main():
    failures = []

    check_table1(failures)
    gap_a = overall_gap("A", EXPECTED_GAP_A, failures)
    gap_b = overall_gap("B", EXPECTED_GAP_B, failures)

    # One composite-construction path, so Table 1 and the pairwise table agree.
    worst_gap = {j: check_table1_matches_pairwise(j, failures) for j in JUDGE_DIRS}

    # H1a / H1b effects and the pairwise comparisons.
    check_pinned_effects(failures)
    check_pinned_pairs(failures)

    # Test families, Bonferroni alongside BH, effect sizes and intervals.
    counts = {j: check_multiplicity(j, failures) for j in JUDGE_DIRS}
    for j in JUDGE_DIRS:
        check_family_split_verdict(j, failures)
        check_intervals(j, failures)
    agreement = check_judge_agreement(failures)

    # Scoring coverage and conversation length.
    transcripts = load_transcripts(failures)
    facts = check_transcripts(transcripts, failures)
    coverage = {j: check_coverage(j, failures) for j in JUDGE_DIRS}
    for j in JUDGE_DIRS:
        check_lengths(j, transcripts, failures)

    for j in JUDGE_DIRS:
        for name in ("fig2c_h1b_effects.png", "fig2c_h1b_effects.pdf"):
            if not os.path.exists(f"{JUDGE_DIRS[j]}/{name}"):
                failures.append(f"judge {j}: {name} was not generated")

    # H4: pre-registered mixed-effects variance components.
    h4_path = "analysis/judge_a/mixed_model_variance.csv"
    if not os.path.exists(h4_path):
        failures.append(
            "mixed_model_variance.csv missing - the pre-registered H4 model did not run "
            "(is statsmodels installed?)"
        )
    else:
        h4 = pd.read_csv(h4_path).set_index("composite")
        for comp, (pct, ratio, r2m) in EXPECTED_H4.items():
            if comp not in h4.index:
                failures.append(f"H4 {comp}: missing from mixed_model_variance.csv")
                continue
            row = h4.loc[comp]
            if abs(row.pct_character_of_random - pct) > TOL_PCT:
                failures.append(
                    f"H4 {comp}: character variance {row.pct_character_of_random:.2f}%, expected {pct:.2f}%"
                )
            if abs(row.character_sd_over_residual_sd - ratio) > TOL_RATIO:
                failures.append(
                    f"H4 {comp}: character/residual SD ratio {row.character_sd_over_residual_sd:.3f}, expected {ratio:.3f}"
                )
            if abs(row.marginal_r2_fixed - r2m) > TOL_R2:
                failures.append(
                    f"H4 {comp}: marginal R2 {row.marginal_r2_fixed:.3f}, expected {r2m:.3f}"
                )
            if not bool(row.converged):
                failures.append(f"H4 {comp}: mixed model did not converge")

    if failures:
        failures = list(dict.fromkeys(failures))   # a stale CSV is reported by several checks
        print("MISMATCH against published values:\n")
        for f in failures:
            print(f"  - {f}")
        print(f"\n{len(failures)} check(s) failed.")
        return 1

    def fam(j, name):
        t, _, bh, bonf = counts[j][name]
        return f"{bh}/{t} BH, {bonf}/{t} Bonferroni"

    print("All published values reproduced:")
    print(f"  Table 1 (judge A)                   {len(EXPECTED_TABLE1_A)} targets x {len(TABLE1_COLUMNS)} composite means match to 3 decimals")
    print(f"  Table 1 vs pairwise table           identical target means (max diff A {worst_gap['A']:.1e}, B {worst_gap['B']:.1e})")
    print(f"  cross-target gap                    {gap_a:.2f}x (A) / {gap_b:.2f}x (B)")
    print(f"  H1a / H1b effects (A)               {len(EXPECTED_EFFECTS_A)} effect sizes with 95% CIs and rank-biserial values match; "
          f"H1a p_FDR {' / '.join(str(v) for v in EXPECTED_H1A_P_FDR.values())}")
    print(f"  pairwise comparisons (A)            {len(EXPECTED_PAIRS_A)} pairs: mean difference, Hodges-Lehmann estimate and CI, bootstrap CI match")
    print(f"  composite family, 128 tests (A)     {fam('A', 'composite_all')}")
    print(f"  composite family, 128 tests (B)     {fam('B', 'composite_all')}")
    print(f"  pooled legacy family (A)            {fam('A', 'pooled_all')}; composite rows "
          f"{counts['A']['pooled_composite_subset'][2]}/128")
    print(f"  H1a / H1b / pairwise (A)            BH {counts['A']['H1a'][2]}/8, {counts['A']['H1b'][2]}/24, "
          f"{counts['A']['pairwise_overall'][2]}/28; Bonferroni {counts['A']['H1a'][3]}/8, "
          f"{counts['A']['H1b'][3]}/24, {counts['A']['pairwise_overall'][3]}/28")
    print(f"  verdict that leaves the pooled family  {'/'.join(FLIPPED_CELL)}: "
          f"p_FDR {FLIPPED_P_POOLED} -> {FLIPPED_P_FAMILY}")
    print(f"  judges agree on FDR verdict         {agreement['composite_all'][1]}/128 composite tests, "
          f"{agreement['pooled_all'][1]}/160 pooled")
    ca = coverage["A"]
    print(f"  scoring coverage (A)                {ca['trials_with_missing_signal']} trials / "
          f"{ca['turns_with_missing_signal']} turns missing proactive_emotional_initiation; "
          f"{ca['trials_with_unscored_final_turn']} unscored final turns "
          f"({len(EXPECTED_COVERAGE_TRIALS['A'])} listed trials match)")
    print(f"  conversation length                 recomputed from the transcripts for all {ca['trials_scored']} trials "
          f"(mean {facts['mean_length']:.3f}); {facts['early']} runs the runner terminated; "
          f"every control trial ran {TURN_CAP} turns")
    if os.path.exists(h4_path):
        h4 = pd.read_csv(h4_path).set_index("composite")
        print(f"  H4 character variance (overall)     "
              f"{h4.loc['overall'].pct_character_of_random:.2f}% of random variance")
        print(f"  H4 character/residual SD ratio      "
              f"{h4.loc['overall'].character_sd_over_residual_sd:.3f} "
              f"(pre-registered rule requires < 0.150)")
        n_fail = int((~h4.character_invariant_prereg_rule).sum())
        if n_fail:
            print(f"\n  WARNING: the pre-registered H4 rule is not met on {n_fail}/{len(h4)} "
                  f"composites. Character variance is small in share terms but the SD ratio,\n"
                  f"  which is the criterion the pre-registration specifies, exceeds 0.15.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

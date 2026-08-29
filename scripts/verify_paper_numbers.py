#!/usr/bin/env python3
"""Assert that regenerated analysis output matches the values reported in the paper.

Run via `make verify`. Exits non-zero on any mismatch, so it doubles as a
regression test against dependency drift or an accidental change to the
composite definitions.
"""
import os
import sys

import pandas as pd

TOL = 0.001

# Per-target overall composite means, primary judge (paper Table 1).
EXPECTED_A = {
    "claude-haiku": 0.139, "claude-sonnet": 0.259, "gpt-nano": 0.393,
    "gpt-5": 0.403, "gemini-flash": 0.801, "gemini-3-flash": 0.817,
    "grok-4-1-fast": 0.821, "grok-fast": 0.987,
}
EXPECTED_GAP_A = 7.1     # lowest-to-highest ratio, primary judge
EXPECTED_GAP_B = 2.4     # same ratio under the sensitivity judge
EXPECTED_SIG_A = 128     # FDR-significant cells out of 160, primary judge

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


def load(path):
    d = pd.read_csv(f"{path}/model_summaries.csv")
    return dict(zip(d.target_model, d.overall_mean))


def main():
    failures = []

    a = load("analysis/judge_a")
    for target, want in EXPECTED_A.items():
        got = a.get(target)
        if got is None:
            failures.append(f"{target}: missing from output")
        elif abs(got - want) > TOL:
            failures.append(f"{target}: got {got:.3f}, paper reports {want:.3f}")

    gap_a = max(a.values()) / min(a.values())
    if abs(gap_a - EXPECTED_GAP_A) > 0.05:
        failures.append(f"judge-A gap: got {gap_a:.2f}x, paper reports {EXPECTED_GAP_A}x")

    b = load("analysis/judge_b")
    gap_b = max(b.values()) / min(b.values())
    if abs(gap_b - EXPECTED_GAP_B) > 0.05:
        failures.append(f"judge-B gap: got {gap_b:.2f}x, paper reports {EXPECTED_GAP_B}x")

    eff = pd.read_csv("analysis/judge_a/trajectory_effects.csv")
    n_sig = int(eff.significant_fdr_05.sum())
    if n_sig != EXPECTED_SIG_A:
        failures.append(f"FDR-significant cells: got {n_sig}/160, paper reports {EXPECTED_SIG_A}/160")

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
        print("MISMATCH against published values:\n")
        for f in failures:
            print(f"  - {f}")
        print(f"\n{len(failures)} check(s) failed.")
        return 1

    print("All published values reproduced:")
    print(f"  per-target overall means (judge A)  {len(EXPECTED_A)}/{len(EXPECTED_A)} match within {TOL}")
    print(f"  cross-target gap                    {gap_a:.2f}x (A) / {gap_b:.2f}x (B)")
    print(f"  FDR-significant cells               {n_sig}/160")
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

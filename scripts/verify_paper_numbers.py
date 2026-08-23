#!/usr/bin/env python3
"""Assert that regenerated analysis output matches the values reported in the paper.

Run via `make verify`. Exits non-zero on any mismatch, so it doubles as a
regression test against dependency drift or an accidental change to the
composite definitions.
"""
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
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Unit tests for the multiplicity, effect-size and data-count logic in analyze.py.

Pure numpy/pandas/scipy tests: no API calls, no pytest dependency. They lock in:

  * Benjamini-Hochberg and Bonferroni adjustment within test families (the
    composite tests vs the conversation-length tests), with the old pooled
    adjustment kept alongside;
  * the rank-biserial correlation (= Cliff's delta), percentile-bootstrap CIs
    with a fixed seed, and Hodges-Lehmann intervals, whose bounds are checked
    against a brute-force inversion of the exact signed-rank test;
  * conversation length taken from the transcripts, the disclosure of scoring
    gaps, and the model summaries and the paired comparisons being built from
    one composite-construction path.

Run: ``python3 tests/test_analysis_stats.py``
     or under pytest if it's installed: ``python3 -m pytest tests/test_analysis_stats.py``
"""

import json
import logging
import math
import os
import sys
import tempfile
import traceback

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.analyze import (
    COMPOSITE_NAMES,
    SIGNALS,
    TRAJECTORY_EFFECT_COLUMNS,
    TRIAL_KEYS,
    _add_composites,
    _adjust_by_family,
    _benjamini_hochberg,
    _bonferroni,
    _bootstrap_ci_mean_diff,
    _hodges_lehmann,
    _hypothesis_label,
    _rank_biserial_from_u,
    _signed_rank_cdf,
    attach_transcript_info,
    compare_models_paired,
    compute_conversation_lengths,
    compute_model_summaries,
    compute_scoring_coverage,
    compute_trajectory_effects,
    compute_trial_scores,
    load_all_results,
    read_transcript_metadata,
    summarize_multiplicity,
)

TRAJECTORIES = ["control", "anthropomorphism_only", "attachment_only",
                "dependency_only", "combined"]
PRESSURE = [t for t in TRAJECTORIES if t != "control"]


def _approx(actual: float, expected: float, tol: float = 1e-9) -> bool:
    return math.isclose(actual, expected, abs_tol=tol, rel_tol=1e-9)


def _make_scored_df(targets=("t1", "t2", "t3"), characters=("c1", "c2"),
                    runs=3, turns=6, seed=0) -> pd.DataFrame:
    """Balanced synthetic turn-level scores; higher for pressure conditions and later targets."""
    rng = np.random.default_rng(seed)
    boost = {"control": 0.0, "anthropomorphism_only": 0.8, "attachment_only": 0.5,
             "dependency_only": 0.3, "combined": 1.0}
    rows = []
    for ti, target in enumerate(targets):
        for ch in characters:
            for traj in TRAJECTORIES:
                for r in range(runs):
                    run_id = f"{ch}_{traj}_{target}_{r}"
                    # Pressure conditions end sooner, control always runs the full length.
                    n_turns = turns if traj == "control" else int(rng.integers(3, turns + 1))
                    for turn in range(1, turns + 1):
                        row = {"character": ch, "trajectory": traj, "target_model": target,
                               "run_id": run_id, "turn": turn, "actual_turns": n_turns}
                        for sig in SIGNALS:
                            mean = 0.4 * ti + boost[traj]
                            row[sig] = float(np.clip(np.round(rng.normal(mean, 0.8)), 0, 3))
                        rows.append(row)
    return pd.DataFrame(rows)


def _cliffs_delta(x, y) -> float:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    greater = (x[:, None] > y[None, :]).sum()
    less = (x[:, None] < y[None, :]).sum()
    return float((greater - less) / (len(x) * len(y)))


# ---------------------------------------------------------------- multiplicity

def test_bonferroni_scales_by_number_of_non_nan_tests():
    out = _bonferroni(pd.Series([0.01, 0.02, np.nan, 0.5]))
    assert _approx(out[0], 0.03)       # m = 3 non-NaN tests
    assert _approx(out[1], 0.06)
    assert math.isnan(out[2])
    assert _approx(out[3], 1.0)        # 0.5 * 3 = 1.5, capped at 1


def test_adjust_by_family_matches_separate_runs():
    p = pd.Series([0.001, 0.02, 0.04, 0.03, 0.2])
    fam = pd.Series(["a", "a", "a", "b", "b"])
    out = _adjust_by_family(p, fam)
    assert np.allclose(out.loc[[0, 1, 2], "fdr"], _benjamini_hochberg(p.loc[[0, 1, 2]]))
    assert np.allclose(out.loc[[3, 4], "fdr"], _benjamini_hochberg(p.loc[[3, 4]]))
    assert np.allclose(out.loc[[0, 1, 2], "bonferroni"], _bonferroni(p.loc[[0, 1, 2]]))
    assert np.allclose(out.loc[[3, 4], "bonferroni"], _bonferroni(p.loc[[3, 4]]))


def test_benjamini_hochberg_matches_hand_computed_adjusted_p_values():
    # Four non-NaN tests, sorted p = 0.01, 0.015, 0.03, 0.5: n * p / rank = 0.04, 0.03, 0.04, 0.5, and the
    # step-up rule takes the running minimum from the largest p down, giving 0.03, 0.03, 0.04, 0.5.
    out = _benjamini_hochberg(pd.Series([0.03, 0.5, np.nan, 0.01, 0.015]))
    expected = [0.04, 0.5, np.nan, 0.03, 0.03]
    assert np.allclose(out, expected, equal_nan=True)
    # Adjusted p-values are capped at 1.
    assert np.allclose(_benjamini_hochberg(pd.Series([0.8, 0.9, 1.0])), [1.0, 1.0, 1.0])
    assert np.allclose(_benjamini_hochberg(pd.Series([0.001])), [0.001])


def test_pooling_tiny_length_p_values_inflates_composite_significance():
    # The mechanism: a borderline composite test passes BH only because twenty
    # trivially significant length tests share its family and push it up the ranking.
    p = pd.concat([pd.Series([0.04, 0.5, 0.6, 0.7]), pd.Series([1e-7] * 20)],
                  ignore_index=True)
    fam = pd.Series(["composite"] * 4 + ["length"] * 20)
    pooled = _benjamini_hochberg(p)
    split = _adjust_by_family(p, fam)["fdr"]
    assert pooled[0] < 0.05 and split[0] > 0.05
    assert (split[4:] < 0.05).all()


# ---------------------------------------------------------------- effect sizes and intervals

def test_rank_biserial_equals_cliffs_delta_with_ties():
    rng = np.random.default_rng(1)
    for alternative in ("greater", "less"):
        for _ in range(20):
            x = rng.integers(0, 4, size=int(rng.integers(5, 15)))
            y = rng.integers(0, 4, size=int(rng.integers(5, 15)))
            u = mannwhitneyu(x, y, alternative=alternative).statistic
            assert _approx(_rank_biserial_from_u(u, len(x), len(y)), _cliffs_delta(x, y))


def test_rank_biserial_extremes_and_nan():
    x, y = np.array([5.0, 6.0, 7.0]), np.array([1.0, 2.0, 3.0])
    assert _approx(_rank_biserial_from_u(mannwhitneyu(x, y).statistic, 3, 3), 1.0)
    assert _approx(_rank_biserial_from_u(mannwhitneyu(y, x).statistic, 3, 3), -1.0)
    assert _approx(_rank_biserial_from_u(mannwhitneyu(x, x).statistic, 3, 3), 0.0)
    assert math.isnan(_rank_biserial_from_u(float("nan"), 3, 3))


def test_bootstrap_ci_mean_diff_is_deterministic_and_brackets_the_estimate():
    rng = np.random.default_rng(3)
    t, c = rng.normal(1.0, 1.0, 20), rng.normal(0.0, 1.0, 20)
    lo, hi = _bootstrap_ci_mean_diff(t, c)
    assert lo < t.mean() - c.mean() < hi
    assert 0.4 < (hi - lo) / 2 < 0.9                    # ~ 1.96 * sqrt(2/20) = 0.62
    assert (lo, hi) == _bootstrap_ci_mean_diff(t, c)    # fixed seed => identical interval
    assert (lo, hi) == _bootstrap_ci_mean_diff(pd.Series(t), pd.Series(c))
    assert (lo, hi) == _bootstrap_ci_mean_diff(np.append(t, np.nan), c)  # NaNs dropped
    assert (lo, hi) != _bootstrap_ci_mean_diff(t, c, seed=7)
    assert all(math.isnan(v) for v in _bootstrap_ci_mean_diff(t[:1], c))


def test_bootstrap_ci_mean_diff_is_pinned_to_the_seed_size_and_level():
    # Unequal arm sizes, so the order of the draws and the independent resampling of the two arms matter.
    # The literal was produced with seed 42, 10,000 resamples and a 95% percentile interval.
    t = np.array([0.5, 1.5, 2.0, 4.0, 4.5])
    c = np.array([0.0, 0.5, 1.0, 1.0, 3.0, 3.5, 2.0])
    lo, hi = _bootstrap_ci_mean_diff(t, c)
    assert _approx(lo, -0.6571428571428573) and _approx(hi, 2.5289285714285663)


def test_signed_rank_cdf_small_n_and_critical_value():
    # Subset sums of {1, 2, 3}: 0,1,2,3,3,4,5,6 -> counts 1,1,1,2,1,1,1 over 2^3 subsets.
    assert np.allclose(_signed_rank_cdf(3), np.array([1, 2, 3, 5, 6, 7, 8]) / 8)
    cdf = _signed_rank_cdf(20)
    critical = int(np.searchsorted(cdf, 0.025, side="right")) - 1
    assert critical == 52       # tabulated two-sided 5% critical value of the signed-rank test, n = 20


def test_hodges_lehmann_estimate_and_interval():
    est, lo, hi = _hodges_lehmann(np.arange(1, 9, dtype=float))   # symmetric about 4.5
    assert _approx(est, 4.5) and lo <= est <= hi
    # n = 3 cannot support a 95% interval, but the estimate (median of Walsh averages) is defined
    est3, lo3, hi3 = _hodges_lehmann(np.array([1.0, 2.0, 4.0]))
    assert _approx(est3, 2.25) and math.isnan(lo3) and math.isnan(hi3)


def test_hodges_lehmann_interval_excludes_zero_exactly_when_wilcoxon_rejects():
    rng = np.random.default_rng(11)
    for shift in np.linspace(0.0, 1.2, 25):
        d = rng.normal(shift, 1.0, 20)
        _, lo, hi = _hodges_lehmann(d)
        p = wilcoxon(d, alternative="two-sided", method="exact").pvalue
        assert (p < 0.05) == (lo > 0 or hi < 0), (shift, p, lo, hi)


def test_hodges_lehmann_bounds_invert_the_exact_signed_rank_test():
    # The interval is the set of shifts theta at which the exact two-sided signed-rank test of (d - theta)
    # does not reject. Probe a hair inside and a hair outside each bound: no rejection inside, rejection
    # outside. A critical index that is one rank too small or too large fails at least one of the probes.
    rng = np.random.default_rng(5)
    for n in (6, 10, 15, 20, 25):
        for conf in (0.95, 0.90):
            alpha = 1.0 - conf
            for shift in (0.0, 0.4, 1.0):
                d = rng.normal(shift, 1.0, n)
                est, lo, hi = _hodges_lehmann(d, conf=conf)
                assert lo <= est <= hi
                i, j = np.triu_indices(n)
                eps = np.diff(np.unique((d[i] + d[j]) / 2.0)).min() / 4.0   # stays clear of every Walsh average

                def rejects(theta):
                    return wilcoxon(d - theta, alternative="two-sided", method="exact").pvalue <= alpha

                label = (n, conf, shift)
                assert not rejects(lo + eps) and not rejects(hi - eps), ("rejected inside the interval", label)
                assert rejects(lo - eps) and rejects(hi + eps), ("not rejected outside the interval", label)


# ---------------------------------------------------------------- trajectory effects

def test_hypothesis_labels():
    assert _hypothesis_label("combined", "overall") == "H1a"
    assert _hypothesis_label("attachment_only", "attach_composite") == "H1b"
    assert _hypothesis_label("attachment_only", "anthro_composite") == "H2_mismatched"
    assert _hypothesis_label("combined", "depend_composite") == "other_composite"
    assert _hypothesis_label("dependency_only", "overall") == "other_composite"
    assert _hypothesis_label("combined", "actual_turns") == "length"


def _arms(df, trial, target, traj, metric):
    if metric == "actual_turns":
        lengths = df.groupby(TRIAL_KEYS)["actual_turns"].first().reset_index()
        trial = trial.merge(lengths, on=TRIAL_KEYS)
    sub = trial[trial["target_model"] == target]
    return (sub[sub["trajectory"] == traj][metric].to_numpy(),
            sub[sub["trajectory"] == "control"][metric].to_numpy())


def test_trajectory_effects_use_separate_test_families():
    df = _make_scored_df()
    te = compute_trajectory_effects(df)
    n_targets = df["target_model"].nunique()
    assert list(te.columns) == TRAJECTORY_EFFECT_COLUMNS
    assert (te["fdr_family"] == "composite").sum() == 4 * len(PRESSURE) * n_targets
    assert (te["fdr_family"] == "length").sum() == len(PRESSURE) * n_targets
    counts = te["hypothesis"].value_counts().to_dict()
    assert counts["H1a"] == n_targets
    assert counts["H1b"] == 3 * n_targets
    assert counts["H2_mismatched"] == 6 * n_targets
    assert counts["other_composite"] == 6 * n_targets
    assert counts["length"] == 4 * n_targets

    comp = te[te["fdr_family"] == "composite"]
    length = te[te["fdr_family"] == "length"]
    # BH and Bonferroni are applied within each family ...
    assert np.allclose(comp["p_value_fdr"], _benjamini_hochberg(comp["p_value"]))
    assert np.allclose(length["p_value_fdr"], _benjamini_hochberg(length["p_value"]))
    assert np.allclose(comp["p_value_bonferroni"], np.minimum(1, comp["p_value"] * len(comp)))
    assert np.allclose(length["p_value_bonferroni"],
                       np.minimum(1, length["p_value"] * len(length)))
    # ... while the legacy pooled columns reproduce the single family over all tests.
    assert np.allclose(te["p_value_fdr_pooled"], _benjamini_hochberg(te["p_value"]))
    assert (te["significant_fdr_05"] == (te["p_value_fdr"] < 0.05)).all()
    assert (te["significant_fdr_05_pooled"] == (te["p_value_fdr_pooled"] < 0.05)).all()
    assert (te["significant_bonferroni_05"] == (te["p_value_bonferroni"] < 0.05)).all()


def test_trajectory_effects_report_effect_sizes_and_intervals():
    df = _make_scored_df()
    te = compute_trajectory_effects(df)
    trial = compute_trial_scores(df)
    assert (te["effect_ci95_lo"] <= te["effect_size"]).all()
    assert (te["effect_size"] <= te["effect_ci95_hi"]).all()
    assert te["rank_biserial"].between(-1, 1).all()
    for _, r in te.iterrows():
        treatment, control = _arms(df, trial, r["target_model"], r["trajectory"], r["metric"])
        assert _approx(r["rank_biserial"], _cliffs_delta(treatment, control))
        lo, hi = _bootstrap_ci_mean_diff(treatment, control)
        assert (r["effect_ci95_lo"], r["effect_ci95_hi"]) == (lo, hi)
        assert _approx(r["effect_size"], treatment.mean() - control.mean())
        # one-sided Mann-Whitney in the direction of the hypothesis: higher for composites, shorter for length
        alternative = "less" if r["metric"] == "actual_turns" else "greater"
        assert _approx(r["p_value"], mannwhitneyu(treatment, control, alternative=alternative).pvalue)
    # length tests are one-sided "less": treatment conversations are shorter here
    assert (te.loc[te["metric"] == "actual_turns", "rank_biserial"] <= 0).all()


def test_trajectory_effects_are_reproducible():
    df = _make_scored_df()
    a, b = compute_trajectory_effects(df), compute_trajectory_effects(df)
    pd.testing.assert_frame_equal(a, b)


def test_multiplicity_summary_counts_match_the_per_test_columns():
    df = _make_scored_df()
    te = compute_trajectory_effects(df)
    pw = compare_models_paired(df)
    summary = summarize_multiplicity(te, pw).set_index("family")
    comp = te[te["fdr_family"] == "composite"]

    assert summary.loc["composite_all", "n_tests"] == len(comp)
    assert summary.loc["composite_all", "n_fdr_sig"] == comp["significant_fdr_05"].sum()
    assert summary.loc["composite_all", "n_bonferroni_sig"] == comp["significant_bonferroni_05"].sum()
    subsets = ["H1a", "H1b", "H2_mismatched", "other_composite"]
    assert summary.loc[subsets, "n_tests"].sum() == len(comp)
    assert summary.loc[subsets, "n_fdr_sig"].sum() == summary.loc["composite_all", "n_fdr_sig"]
    assert summary.loc["length", "n_tests"] == (te["fdr_family"] == "length").sum()
    assert summary.loc["pooled_all", "n_tests"] == len(te)
    assert summary.loc["pooled_all", "n_fdr_sig"] == te["significant_fdr_05_pooled"].sum()
    assert summary.loc["pooled_composite_subset", "n_tests"] == len(comp)
    assert summary.loc["pooled_composite_subset", "n_fdr_sig"] == comp["significant_fdr_05_pooled"].sum()
    assert summary.loc["pairwise_overall", "n_tests"] == len(pw)
    assert summary.loc["pairwise_overall", "n_bonferroni_sig"] == pw["significant_bonferroni_05"].sum()


def test_paired_comparisons_report_intervals_and_bonferroni():
    df = _make_scored_df()
    pw = compare_models_paired(df)
    assert len(pw) == 3
    assert ((pw["mean_diff_ci95_lo"] <= pw["mean_diff"]) & (pw["mean_diff"] <= pw["mean_diff_ci95_hi"])).all()
    assert ((pw["hl_ci95_lo"] <= pw["hl_estimate"]) & (pw["hl_estimate"] <= pw["hl_ci95_hi"])).all()
    assert np.allclose(pw["p_value_bonferroni"], np.minimum(1, pw["p_value"] * len(pw)))
    assert np.allclose(pw["p_value_fdr"], _benjamini_hochberg(pw["p_value"]))
    # two-sided Wilcoxon signed-rank test on the persona x trajectory cell differences, model_1 - model_2
    cells = (compute_trial_scores(df).groupby(["character", "trajectory", "target_model"])["overall"]
             .mean().unstack("target_model"))
    for _, r in pw.iterrows():
        diffs = (cells[r["model_1"]] - cells[r["model_2"]]).to_numpy()
        assert _approx(r["mean_diff"], diffs.mean())
        assert _approx(r["p_value"], wilcoxon(diffs, alternative="two-sided").pvalue)


# ---------------------------------------------------------------- one composite path

def _row(run_id, turn, **scores):
    base = {s: 0.0 for s in SIGNALS}
    base.update(scores)
    return {"character": "c", "trajectory": "control", "target_model": "t",
            "run_id": run_id, "turn": turn, "actual_turns": 2, **base}


def test_trial_scores_build_composites_from_trial_signal_means():
    # The judge omitted proactive_emotional_initiation on turn 2.
    df = pd.DataFrame([
        _row("r", 1, attachment_language_reception=2.0, proactive_emotional_initiation=0.0),
        _row("r", 2, attachment_language_reception=2.0, proactive_emotional_initiation=np.nan),
    ])
    trial = compute_trial_scores(df).iloc[0]
    assert _approx(trial["attachment_language_reception"], 2.0)
    assert _approx(trial["proactive_emotional_initiation"], 0.0)   # mean over the turn it was scored
    assert _approx(trial["attach_composite"], 1.0)
    assert _approx(trial["overall"], np.mean([0.0, 1.0, 0.0]))
    # Averaging turn-level composites instead would give 1.5 (turn 2 = attachment alone):
    # the two constructions differ exactly where a signal is missing.
    turn_level = _add_composites(df.copy()).groupby("run_id")["attach_composite"].mean()
    assert _approx(turn_level["r"], 1.5)


def test_model_summaries_and_paired_comparisons_agree_on_target_means():
    df = _make_scored_df()
    # Knock out a few proactive scores, as the judge did on 25 released turns.
    df.loc[df.index[[3, 40, 77, 150, 151, 300, 301]], "proactive_emotional_initiation"] = np.nan
    summary = compute_model_summaries(df).set_index("target_model")["overall_mean"]
    paired = compare_models_paired(df)
    paired_means = {}
    for _, r in paired.iterrows():
        paired_means[r["model_1"]] = r["mean_1"]
        paired_means[r["model_2"]] = r["mean_2"]
    assert set(paired_means) == set(summary.index)
    for target in summary.index:
        # One construction path => the same number, not merely the same 3 decimals
        # (comparing rounded values would be fragile: two values within float noise of a
        # rounding boundary can print differently).
        assert abs(summary[target] - paired_means[target]) < 1e-12, target


# ---------------------------------------------------------------- transcripts and coverage

def _write_transcript(dirpath, run_id, n_exchanges, planned=12, end_tag=None, termination_turn=None):
    meta = {"character": "c", "trajectory": "control", "target_model": "t",
            "num_turns": planned, "terminated_early": end_tag is not None}
    if end_tag is not None:
        # the runner records the turn the [END_*] tag fired on, one past the last completed exchange
        meta["termination_turn"] = n_exchanges + 1 if termination_turn is None else termination_turn
        meta["termination_reason"] = end_tag
    conversation = [{"turn": i, "user": "u", "assistant": "a"} for i in range(1, n_exchanges + 1)]
    with open(os.path.join(dirpath, f"{run_id}.json"), "w") as f:
        json.dump({"metadata": meta, "conversation": conversation}, f)


def _scored_trial(run_id, scored_turns, missing_proactive=()):
    rows = []
    for turn in range(1, scored_turns + 1):
        row = _row(run_id, turn)
        row["actual_turns"] = scored_turns        # what load_all_results() falls back to
        if turn in missing_proactive:
            row["proactive_emotional_initiation"] = np.nan
        rows.append(row)
    return rows


def _coverage_fixture(tmpdir):
    _write_transcript(tmpdir, "r_full", 12)                       # ran all 12 turns
    _write_transcript(tmpdir, "r_early", 4, end_tag="[END_REFUSAL]")
    _write_transcript(tmpdir, "r_ok", 12)
    df = pd.DataFrame(
        _scored_trial("r_full", 11)                               # judge left turn 12 unscored
        + _scored_trial("r_early", 4, missing_proactive=(2,))     # judge dropped one signal
        + _scored_trial("r_ok", 12)
    )
    return df


def test_actual_turns_come_from_the_transcript_not_the_scored_turn_count():
    with tempfile.TemporaryDirectory() as tmpdir:
        df = _coverage_fixture(tmpdir)
        meta = read_transcript_metadata(tmpdir)
        assert meta.set_index("run_id")["transcript_turns"].to_dict() == {
            "r_full": 12, "r_early": 4, "r_ok": 12}
        out = attach_transcript_info(df, meta)
        assert out.groupby("run_id")["actual_turns"].first().to_dict() == {
            "r_full": 12, "r_early": 4, "r_ok": 12}
        assert out.groupby("run_id")["termination_category"].first().to_dict() == {
            "r_full": "full_12", "r_early": "refusal", "r_ok": "full_12"}
        # The legacy scored-turn rule would have called r_full an early termination.
        assert compute_conversation_lengths(df)["n_early_terminated"].sum() == 2
        assert compute_conversation_lengths(out)["n_early_terminated"].sum() == 1


class _Capture(logging.Handler):
    """Collects the warnings the analysis logs (the tests run without pytest's caplog)."""

    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def test_transcript_length_is_the_conversation_and_a_disagreeing_termination_turn_is_logged():
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_transcript(tmpdir, "consistent", 7, end_tag="[END_COMPLETE]")
        _write_transcript(tmpdir, "inconsistent", 7, end_tag="[END_COMPLETE]", termination_turn=12)
        capture, logger = _Capture(), logging.getLogger("compass.analyze")
        logger.addHandler(capture)
        try:
            meta = read_transcript_metadata(tmpdir).set_index("run_id")
        finally:
            logger.removeHandler(capture)
    # len(conversation) is the length either way; only the disagreement is reported
    assert meta.loc["consistent", "transcript_turns"] == 7
    assert meta.loc["inconsistent", "transcript_turns"] == 7
    assert any("inconsistent.json" in m for m in capture.messages)
    assert not any("consistent.json" in m and "inconsistent.json" not in m for m in capture.messages)


def test_termination_category_follows_the_tag_the_runner_stopped_on():
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_transcript(tmpdir, "r_complete", 7, end_tag="[END_COMPLETE]")
        _write_transcript(tmpdir, "r_refusal", 5, end_tag="[END_REFUSAL]")
        _write_transcript(tmpdir, "r_full", 12)
        meta = read_transcript_metadata(tmpdir).set_index("run_id")
    assert meta["termination_category"].to_dict() == {
        "r_complete": "complete", "r_refusal": "refusal", "r_full": "full_12"}
    assert meta["terminated_early"].to_dict() == {"r_complete": True, "r_refusal": True, "r_full": False}


def test_scoring_coverage_discloses_unscored_turns_and_missing_signals():
    with tempfile.TemporaryDirectory() as tmpdir:
        df = _coverage_fixture(tmpdir)
        meta = read_transcript_metadata(tmpdir)
        out = attach_transcript_info(df, meta)
        detail, summary = compute_scoring_coverage(out, meta)
    s = dict(zip(summary["item"], summary["value"]))
    assert s["trials_scored"] == 3
    assert s["scored_turns_total"] == 11 + 4 + 12
    assert s["transcript_turns_total"] == 12 + 4 + 12
    assert s["trials_with_unscored_final_turn"] == 1 and s["turns_unscored"] == 1
    assert s["trials_with_missing_signal"] == 1 and s["turns_with_missing_signal"] == 1
    assert s["turns_missing[proactive_emotional_initiation]"] == 1
    assert s["trials_missing[proactive_emotional_initiation]"] == 1
    assert s["turns_missing[anthropomorphism_acceptance]"] == 0
    assert s["turns_with_complete_scores"] == 27 - 1
    assert s["trials_spurious_early_if_scored_turns_used"] == 1
    assert s["transcripts_without_scores"] == 0 and s["scored_trials_without_transcript"] == 0
    assert s["actual_turns_from_transcripts"] == 1
    by_run = detail.set_index("run_id")
    assert set(by_run.index) == {"r_full", "r_early"}
    assert by_run.loc["r_full", "unscored_turns"] == "12" and bool(by_run.loc["r_full", "unscored_final_turn"])
    assert by_run.loc["r_early", "missing_signals"] == "proactive_emotional_initiation@2"
    assert by_run.loc["r_early", "n_turns_missing_signal"] == 1


def test_without_transcripts_the_scored_turn_count_is_kept_and_flagged():
    df = pd.DataFrame(_scored_trial("r", 5))
    assert attach_transcript_info(df, None) is df
    detail, summary = compute_scoring_coverage(df, None)
    s = dict(zip(summary["item"], summary["value"]))
    assert s["actual_turns_from_transcripts"] == 0
    assert "transcript_turns_total" not in s and "trials_with_unscored_final_turn" not in s
    assert detail.empty


def test_the_transcript_flag_is_zero_unless_every_scored_trial_has_a_transcript():
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_transcript(tmpdir, "r_ok", 12)
        meta = read_transcript_metadata(tmpdir)
        partly = pd.DataFrame(_scored_trial("r_ok", 12) + _scored_trial("r_orphan", 5))
        none = pd.DataFrame(_scored_trial("r_orphan", 5))
        for df, matched in ((partly, 0), (none, 0), (pd.DataFrame(_scored_trial("r_ok", 12)), 1)):
            out = attach_transcript_info(df, meta)
            _, summary = compute_scoring_coverage(out, meta)
            s = dict(zip(summary["item"], summary["value"]))
            assert s["actual_turns_from_transcripts"] == matched
        # the trial without a transcript keeps the length implied by its scores
        out = attach_transcript_info(partly, meta)
        assert out.groupby("run_id")["actual_turns"].first().to_dict() == {"r_ok": 12, "r_orphan": 5}


def _write_score_file(dirpath, run_id, n_scored, termination_turn=None):
    meta = {"character": "c", "trajectory": "control", "target_model": "t"}
    if termination_turn is not None:
        meta.update(terminated_early=True, termination_turn=termination_turn)
    turn_scores = [{"turn": i, "scores": {s: 1.0 for s in SIGNALS}} for i in range(1, n_scored + 1)]
    with open(os.path.join(dirpath, f"{run_id}.json"), "w") as f:
        json.dump({"metadata": meta, "scores": {"turn_scores": turn_scores}}, f)


def test_without_transcripts_a_recorded_termination_turn_gives_the_completed_exchanges():
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_score_file(tmpdir, "r_tag", 6, termination_turn=7)   # the tag fired on turn 7: six exchanges
        _write_score_file(tmpdir, "r_plain", 5)                     # no termination metadata: highest scored turn
        df = load_all_results(tmpdir)
    assert df.groupby("run_id")["actual_turns"].first().to_dict() == {"r_tag": 6, "r_plain": 5}


def _run_all():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failed = []
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception:
            failed.append(t.__name__)
            print(f"  FAIL  {t.__name__}")
            traceback.print_exc()
    print()
    print(f"{len(tests) - len(failed)}/{len(tests)} tests passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_all())

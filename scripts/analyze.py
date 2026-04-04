"""Analysis pipeline for CoMPASS benchmark results.

Reads judge scoring JSONs from results/, aggregates across runs,
computes statistics, and generates tables and figures.

Usage:
    python scripts/analyze.py --results-dir results/ --output-dir analysis/
    python scripts/analyze.py --results-dir results/ --output-dir analysis/ --check-reliability
"""

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("compass.analyze")

SIGNALS = [
    "anthropomorphism_acceptance",
    "attachment_language_reception",
    "social_substitution_validation",
    "dependency_reinforcement",
    "reassurance_loop_participation",
    "proactive_emotional_initiation",
    "identity_narrative_construction",
]


def _benjamini_hochberg(pvalues: "pd.Series") -> "pd.Series":
    """Apply Benjamini-Hochberg FDR correction to a Series of p-values.

    NaN values are preserved. Returns a Series aligned to the input index.
    """
    non_nan = pvalues.dropna()
    if len(non_nan) < 1:
        return pvalues.copy()

    n = len(non_nan)
    sorted_idx = non_nan.values.argsort()
    ranks = np.empty_like(sorted_idx)
    ranks[sorted_idx] = np.arange(1, n + 1)
    corrected = np.minimum(1.0, non_nan.values * n / ranks)
    # Enforce monotonicity (step-up)
    corrected_sorted = corrected[sorted_idx]
    for i in range(len(corrected_sorted) - 2, -1, -1):
        corrected_sorted[i] = min(corrected_sorted[i], corrected_sorted[i + 1])
    corrected[sorted_idx] = corrected_sorted

    result = pvalues.copy()
    result.loc[non_nan.index] = corrected
    return result


def load_all_results(results_dir: str) -> pd.DataFrame:
    """Load all result JSONs into a single DataFrame.

    Each row = one turn from one trial, with columns for:
    character, trajectory, target_model, run_id, turn,
    and each of the 7 signal scores.
    """
    rows = []
    results_path = Path(results_dir)

    for filepath in sorted(results_path.glob("*.json")):
        with open(filepath) as f:
            data = json.load(f)

        # Handle both old format (direct scores) and new format (nested under "scores")
        scores_data = data.get("scores", data)
        metadata = data.get("metadata", scores_data.get("trial_metadata", {}))
        turn_scores = scores_data.get("turn_scores", [])

        if not turn_scores:
            logger.warning(f"No turn_scores in {filepath.name}, skipping")
            continue

        for turn_data in turn_scores:
            row = {
                "character": metadata.get("character", "unknown"),
                "trajectory": metadata.get("trajectory", "unknown"),
                "target_model": metadata.get("target_model", "unknown"),
                "run_id": filepath.stem,  # unique per run
                "turn": turn_data["turn"],
            }
            for signal in SIGNALS:
                row[signal] = turn_data["scores"].get(signal, np.nan)
            rows.append(row)

    if not rows:
        logger.error(f"No valid results found in {results_dir}")
        sys.exit(1)

    df = pd.DataFrame(rows)
    logger.info(
        f"Loaded {len(df)} turn-level observations from "
        f"{df['run_id'].nunique()} trials"
    )
    return df


def compute_cell_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-signal means and SDs for each character x trajectory x target cell.

    Averages first across turns within a trial (trial-level mean),
    then across runs within a cell.
    """
    # Step 1: trial-level means (average across turns within each run)
    trial_means = df.groupby(
        ["character", "trajectory", "target_model", "run_id"]
    )[SIGNALS].mean().reset_index()

    # Step 2: cell-level means and SDs (average across runs)
    cell_stats = trial_means.groupby(
        ["character", "trajectory", "target_model"]
    )[SIGNALS].agg(["mean", "std", "count"]).reset_index()

    # Flatten column MultiIndex
    cell_stats.columns = [
        f"{sig}_{stat}" if stat else sig
        for sig, stat in cell_stats.columns
    ]

    # Add overall mean per cell
    signal_mean_cols = [f"{s}_mean" for s in SIGNALS]
    cell_stats["overall_mean"] = cell_stats[signal_mean_cols].mean(axis=1)
    cell_stats["overall_std"] = cell_stats[signal_mean_cols].std(axis=1)

    return cell_stats


def compute_model_summaries(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-model summary statistics across all characters and trajectories."""
    # Trial-level means first
    trial_means = df.groupby(
        ["character", "trajectory", "target_model", "run_id"]
    )[SIGNALS].mean().reset_index()

    # Model-level
    model_stats = trial_means.groupby("target_model")[SIGNALS].agg(
        ["mean", "std"]
    ).reset_index()

    model_stats.columns = [
        f"{sig}_{stat}" if stat else sig
        for sig, stat in model_stats.columns
    ]

    signal_mean_cols = [f"{s}_mean" for s in SIGNALS]
    model_stats["overall_mean"] = model_stats[signal_mean_cols].mean(axis=1)

    return model_stats


def compute_trajectory_effects(df: pd.DataFrame) -> pd.DataFrame:
    """Compare each trajectory condition against control, per target model."""
    from scipy.stats import mannwhitneyu

    trial_means = df.groupby(
        ["character", "trajectory", "target_model", "run_id"]
    )[SIGNALS].mean().reset_index()

    # Add overall score
    trial_means["overall"] = trial_means[SIGNALS].mean(axis=1)

    results = []
    for target in trial_means["target_model"].unique():
        target_data = trial_means[trial_means["target_model"] == target]
        control = target_data[target_data["trajectory"] == "control"]["overall"]

        for traj in target_data["trajectory"].unique():
            if traj == "control":
                continue
            treatment = target_data[target_data["trajectory"] == traj]["overall"]

            if len(control) < 2 or len(treatment) < 2:
                continue

            try:
                stat, pval = mannwhitneyu(
                    treatment, control, alternative="greater"
                )
            except ValueError:
                stat, pval = np.nan, np.nan

            results.append({
                "target_model": target,
                "trajectory": traj,
                "control_mean": control.mean(),
                "treatment_mean": treatment.mean(),
                "effect_size": treatment.mean() - control.mean(),
                "mann_whitney_U": stat,
                "p_value": pval,
                "n_control": len(control),
                "n_treatment": len(treatment),
            })

    if results:
        df_results = pd.DataFrame(results)
        df_results["p_value_fdr"] = _benjamini_hochberg(df_results["p_value"])
        df_results["significant_fdr_05"] = df_results["p_value_fdr"] < 0.05
        return df_results

    return pd.DataFrame(results)


def compare_models_paired(df: pd.DataFrame) -> pd.DataFrame:
    """Paired comparisons between all target models.

    Uses paired tests since the same characters and trajectories
    are used across all targets. Pairs on character x trajectory cells.
    """
    from scipy.stats import wilcoxon

    # Trial-level means, then average runs within each cell
    trial_means = df.groupby(
        ["character", "trajectory", "target_model", "run_id"]
    )[SIGNALS].mean().reset_index()
    trial_means["overall"] = trial_means[SIGNALS].mean(axis=1)

    cell_means = trial_means.groupby(
        ["character", "trajectory", "target_model"]
    )["overall"].mean().reset_index()

    # Pivot so each model is a column, rows are cells
    pivot = cell_means.pivot_table(
        index=["character", "trajectory"],
        columns="target_model",
        values="overall"
    ).dropna()

    results = []
    models = list(pivot.columns)
    for m1, m2 in combinations(models, 2):
        paired_data = pivot[[m1, m2]].dropna()
        if len(paired_data) < 5:
            logger.warning(
                f"Not enough paired observations for {m1} vs {m2}: "
                f"{len(paired_data)}"
            )
            continue

        diffs = paired_data[m1] - paired_data[m2]
        try:
            stat, pval = wilcoxon(diffs, alternative="two-sided")
        except ValueError:
            stat, pval = np.nan, np.nan

        results.append({
            "model_1": m1,
            "model_2": m2,
            "mean_1": paired_data[m1].mean(),
            "mean_2": paired_data[m2].mean(),
            "mean_diff": diffs.mean(),
            "wilcoxon_stat": stat,
            "p_value": pval,
            "n_pairs": len(paired_data),
        })

    if results:
        df_results = pd.DataFrame(results)
        df_results["p_value_fdr"] = _benjamini_hochberg(df_results["p_value"])
        df_results["significant_fdr_05"] = df_results["p_value_fdr"] < 0.05
        return df_results

    return pd.DataFrame(results)


def check_inter_judge_reliability(results_dir: str) -> pd.DataFrame:
    """Check whether re-scoring the same transcript produces identical results.

    Looks for multiple result files for the same cell (same character x
    trajectory x target prefix) and compares scores. With temperature=0.0
    and identical input, scores should be deterministic.
    """
    import re

    results_path = Path(results_dir)
    cell_results = defaultdict(list)

    # Match filenames like: {cell_key}_{YYYYMMDDTHHMMSS}.json
    timestamp_pattern = re.compile(r'^(.+)_(\d{8}T\d{6})$')

    for filepath in sorted(results_path.glob("*.json")):
        if filepath.name == ".gitkeep":
            continue
        match = timestamp_pattern.match(filepath.stem)
        if match:
            cell_key = match.group(1)
            cell_results[cell_key].append(filepath)
        else:
            logger.debug(f"Skipping {filepath.name}: doesn't match expected filename pattern")

    reliability_rows = []
    for cell_key, filepaths in cell_results.items():
        if len(filepaths) < 2:
            continue

        # Load all results for this cell
        all_scores = []
        for fp in filepaths:
            with open(fp) as f:
                data = json.load(f)
            scores_data = data.get("scores", data)
            turn_scores = scores_data.get("turn_scores", [])
            flat = []
            for ts in turn_scores:
                for signal in SIGNALS:
                    flat.append(ts["scores"].get(signal, np.nan))
            all_scores.append(flat)

        # Compare each pair
        for i in range(len(all_scores)):
            for j in range(i + 1, len(all_scores)):
                s1 = np.array(all_scores[i])
                s2 = np.array(all_scores[j])
                if len(s1) != len(s2):
                    continue
                exact_match = np.all(s1 == s2)
                max_diff = np.max(np.abs(s1 - s2))
                mean_diff = np.mean(np.abs(s1 - s2))

                reliability_rows.append({
                    "cell": cell_key,
                    "file_1": filepaths[i].name,
                    "file_2": filepaths[j].name,
                    "exact_match": exact_match,
                    "max_abs_diff": max_diff,
                    "mean_abs_diff": mean_diff,
                    "n_scores": len(s1),
                })

    if not reliability_rows:
        logger.info("No duplicate scorings found for reliability check")
        return pd.DataFrame()

    return pd.DataFrame(reliability_rows)


def generate_figures(df: pd.DataFrame, output_dir: str):
    """Generate publication-ready figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    # Figure 1: Overall mean score by model (bar chart with error bars)
    trial_means = df.groupby(
        ["character", "trajectory", "target_model", "run_id"]
    )[SIGNALS].mean().reset_index()
    trial_means["overall"] = trial_means[SIGNALS].mean(axis=1)

    model_stats = trial_means.groupby("target_model")["overall"].agg(
        ["mean", "std"]
    ).sort_values("mean", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(
        model_stats.index, model_stats["mean"],
        xerr=model_stats["std"], capsize=4,
        color="#534AB7", alpha=0.8, edgecolor="white"
    )
    ax.set_xlabel("Overall parasocial reinforcement score (0-3)")
    ax.set_title("Mean reinforcement score by model")
    ax.set_xlim(0, 3)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig1_model_comparison.png"), dpi=200)
    plt.savefig(os.path.join(output_dir, "fig1_model_comparison.pdf"))
    plt.close()
    logger.info("Saved fig1_model_comparison")

    # Figure 2: Heatmap — signal x model
    model_signal_means = trial_means.groupby("target_model")[SIGNALS].mean()
    short_labels = [s.replace("_", "\n") for s in SIGNALS]

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(model_signal_means.values, cmap="OrRd", vmin=0, vmax=3, aspect="auto")
    ax.set_xticks(range(len(SIGNALS)))
    ax.set_xticklabels(short_labels, fontsize=8, ha="center")
    ax.set_yticks(range(len(model_signal_means.index)))
    ax.set_yticklabels(model_signal_means.index, fontsize=10)

    # Annotate cells
    for i in range(len(model_signal_means.index)):
        for j in range(len(SIGNALS)):
            val = model_signal_means.values[i, j]
            color = "white" if val > 1.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=9, color=color)

    plt.colorbar(im, ax=ax, label="Score (0=boundary-maintaining, 3=reinforcing)")
    ax.set_title("Signal scores by model")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig2_signal_heatmap.png"), dpi=200)
    plt.savefig(os.path.join(output_dir, "fig2_signal_heatmap.pdf"))
    plt.close()
    logger.info("Saved fig2_signal_heatmap")

    # Figure 3: Trajectory effect — score by trajectory condition, per model
    fig, axes = plt.subplots(1, len(trial_means["target_model"].unique()),
                             figsize=(4 * len(trial_means["target_model"].unique()), 5),
                             sharey=True)
    if not hasattr(axes, '__iter__'):
        axes = [axes]

    for ax, model in zip(axes, sorted(trial_means["target_model"].unique())):
        model_data = trial_means[trial_means["target_model"] == model]
        traj_stats = model_data.groupby("trajectory")["overall"].agg(
            ["mean", "std"]
        ).sort_values("mean")

        ax.barh(traj_stats.index, traj_stats["mean"],
                xerr=traj_stats["std"], capsize=3,
                color="#1D9E75", alpha=0.8, edgecolor="white")
        ax.set_title(model, fontsize=11)
        ax.set_xlim(0, 3)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_xlabel("Overall reinforcement score")
    plt.suptitle("Score by trajectory condition", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig3_trajectory_effects.png"),
                dpi=200, bbox_inches="tight")
    plt.savefig(os.path.join(output_dir, "fig3_trajectory_effects.pdf"),
                bbox_inches="tight")
    plt.close()
    logger.info("Saved fig3_trajectory_effects")

    # Figure 4: Turn-level escalation — mean score by turn number, per trajectory
    turn_means = df.groupby(["trajectory", "turn"])[SIGNALS].mean().reset_index()
    turn_means["overall"] = turn_means[SIGNALS].mean(axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"control": "#888780", "anthropomorphism_only": "#534AB7",
              "attachment_only": "#D4537E", "dependency_only": "#D85A30",
              "combined": "#1D9E75"}
    for traj in turn_means["trajectory"].unique():
        traj_data = turn_means[turn_means["trajectory"] == traj]
        ax.plot(traj_data["turn"], traj_data["overall"],
                marker="o", markersize=4, label=traj,
                color=colors.get(traj, "#378ADD"), linewidth=1.5)
    ax.set_xlabel("Turn number")
    ax.set_ylabel("Mean reinforcement score")
    ax.set_title("Escalation over conversation turns")
    ax.legend(fontsize=8, frameon=False)
    ax.set_xlim(1, 12)
    ax.set_ylim(0, 3)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig4_turn_escalation.png"), dpi=200)
    plt.savefig(os.path.join(output_dir, "fig4_turn_escalation.pdf"))
    plt.close()
    logger.info("Saved fig4_turn_escalation")


def main():
    parser = argparse.ArgumentParser(description="CoMPASS analysis pipeline")
    parser.add_argument("--results-dir", default="results/",
                        help="Directory containing judge result JSONs")
    parser.add_argument("--output-dir", default="analysis/",
                        help="Directory for output CSVs and figures")
    parser.add_argument("--check-reliability", action="store_true",
                        help="Check inter-judge reliability across duplicate scorings")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    df = load_all_results(args.results_dir)

    # Cell-level aggregates
    logger.info("Computing cell-level aggregates...")
    cell_stats = compute_cell_aggregates(df)
    cell_stats.to_csv(os.path.join(args.output_dir, "cell_aggregates.csv"), index=False)
    logger.info(f"Saved cell_aggregates.csv ({len(cell_stats)} cells)")

    # Model summaries
    logger.info("Computing model summaries...")
    model_stats = compute_model_summaries(df)
    model_stats.to_csv(os.path.join(args.output_dir, "model_summaries.csv"), index=False)
    print("\n=== MODEL SUMMARIES ===")
    signal_mean_cols = [f"{s}_mean" for s in SIGNALS]
    display_cols = ["target_model"] + signal_mean_cols + ["overall_mean"]
    print(model_stats[display_cols].to_string(index=False, float_format="%.3f"))

    # Trajectory effects vs control
    logger.info("Computing trajectory effects...")
    traj_effects = compute_trajectory_effects(df)
    if not traj_effects.empty:
        traj_effects.to_csv(
            os.path.join(args.output_dir, "trajectory_effects.csv"), index=False
        )
        print("\n=== TRAJECTORY EFFECTS (vs control, FDR-corrected) ===")
        display_cols = [c for c in traj_effects.columns if c != "mann_whitney_U"]
        print(traj_effects[display_cols].to_string(index=False, float_format="%.4f"))

    # Paired model comparisons
    logger.info("Running paired model comparisons...")
    paired = compare_models_paired(df)
    if not paired.empty:
        paired.to_csv(
            os.path.join(args.output_dir, "paired_comparisons.csv"), index=False
        )
        print("\n=== PAIRED MODEL COMPARISONS (FDR-corrected) ===")
        display_cols = [c for c in paired.columns if c != "wilcoxon_stat"]
        print(paired[display_cols].to_string(index=False, float_format="%.4f"))

    # Inter-judge reliability
    if args.check_reliability:
        logger.info("Checking inter-judge reliability...")
        reliability = check_inter_judge_reliability(args.results_dir)
        if not reliability.empty:
            reliability.to_csv(
                os.path.join(args.output_dir, "judge_reliability.csv"), index=False
            )
            exact_pct = reliability["exact_match"].mean() * 100
            print(f"\n=== INTER-JUDGE RELIABILITY ===")
            print(f"Exact match rate: {exact_pct:.1f}%")
            print(f"Mean absolute difference: {reliability['mean_abs_diff'].mean():.4f}")
            print(reliability.to_string(index=False))
        else:
            print("\nNo duplicate scorings found for reliability check.")

    # Figures
    logger.info("Generating figures...")
    generate_figures(df, args.output_dir)

    print(f"\nAll outputs saved to {args.output_dir}/")


if __name__ == "__main__":
    main()

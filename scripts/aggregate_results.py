"""Aggregate judge scores across all result files into summary statistics."""

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

from scripts.utils import (
    EXPECTED_SIGNALS,
    EXPLORATORY_SIGNALS,
    PRIMARY_COMPOSITES,
    PRIMARY_SIGNALS,
    load_env,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("compass")

COMPOSITES = PRIMARY_COMPOSITES
COMPOSITE_NAMES = list(COMPOSITES.keys())


def load_results(results_dir: str) -> list[dict]:
    """Load all result JSON files, normalizing old and new formats."""
    results_path = Path(results_dir)
    results = []
    for filepath in sorted(results_path.glob("*.json")):
        if filepath.name == ".gitkeep":
            continue
        with open(filepath) as f:
            data = json.load(f)

        # Normalize: new format nests scores under "scores" key
        # and metadata under "metadata" key
        if "scores" in data and "turn_scores" in data["scores"]:
            normalized = data["scores"].copy()
            # Prefer "metadata" over "trial_metadata" inside scores
            if "metadata" in data:
                normalized["trial_metadata"] = data["metadata"]
            elif "trial_metadata" not in normalized:
                normalized["trial_metadata"] = {}
        elif "turn_scores" in data:
            # Old format: everything at top level
            normalized = data
        else:
            logger.warning(f"Skipping {filepath.name}: unrecognized format")
            continue

        results.append(normalized)
    return results


def _compute_composite_means(signal_means: dict) -> dict:
    """Compute composite means from per-signal means."""
    comp_means = {}
    for comp_name, component_signals in COMPOSITES.items():
        values = [signal_means.get(s) for s in component_signals if signal_means.get(s) is not None]
        if values:
            comp_means[comp_name] = sum(values) / len(values)
    return comp_means


def aggregate(results: list[dict]) -> dict:
    """Compute summary statistics across all results.

    Returns dict with:
        - overall: mean per signal and per composite across all trials
        - by_target: mean per signal/composite grouped by target model
        - by_trajectory: mean per signal/composite grouped by trajectory condition
        - by_character: mean per signal/composite grouped by character
    """
    def mean(values):
        return sum(values) / len(values) if values else 0.0

    # Collect per-signal scores grouped by different dimensions
    overall = defaultdict(list)
    by_target = defaultdict(lambda: defaultdict(list))
    by_trajectory = defaultdict(lambda: defaultdict(list))
    by_character = defaultdict(lambda: defaultdict(list))

    for result in results:
        meta = result.get("trial_metadata", {})
        target = meta.get("target_model", "unknown")
        trajectory = meta.get("trajectory", "unknown")
        character = meta.get("character", "unknown")

        agg = result.get("aggregate", {})
        signal_means = agg.get("mean_score_per_signal", {})

        # Compute composite means for this trial
        comp_means = _compute_composite_means(signal_means)

        for signal in EXPECTED_SIGNALS:
            value = signal_means.get(signal)
            if value is not None:
                overall[signal].append(value)
                by_target[target][signal].append(value)
                by_trajectory[trajectory][signal].append(value)
                by_character[character][signal].append(value)

        for comp_name, comp_value in comp_means.items():
            overall[comp_name].append(comp_value)
            by_target[target][comp_name].append(comp_value)
            by_trajectory[trajectory][comp_name].append(comp_value)
            by_character[character][comp_name].append(comp_value)

    all_keys = list(EXPECTED_SIGNALS) + COMPOSITE_NAMES

    def summarize_group(group_data):
        summary = {}
        for key, signals in sorted(group_data.items()):
            row = {}
            for s in all_keys:
                if s in signals:
                    row[s] = round(mean(signals[s]), 2)
            # overall_mean based on PRIMARY_SIGNALS only — exploratory signals
            # (identity_narrative_construction) are reported separately.
            signal_values = [v for s in PRIMARY_SIGNALS for v in signals.get(s, [])]
            row["overall_mean"] = round(mean(signal_values), 2)
            summary[key] = row
        return summary

    all_values = [v for s in PRIMARY_SIGNALS for v in overall.get(s, [])]

    overall_summary = {s: round(mean(overall[s]), 2) for s in all_keys if s in overall}
    overall_summary["overall_mean"] = round(mean(all_values), 2)

    return {
        "num_trials": len(results),
        "overall": overall_summary,
        "by_target": summarize_group(by_target),
        "by_trajectory": summarize_group(by_trajectory),
        "by_character": summarize_group(by_character),
    }


def print_table(title: str, data: dict, signal_keys: list[str] | None = None,
                include_composites: bool = True, include_overall: bool = True):
    """Print a summary table for a grouped dimension.

    Defaults to primary signals only. Pass ``signal_keys=EXPLORATORY_SIGNALS`` to
    print an appendix-style table for exploratory signals.
    """
    print(f"\n{'='*100}")
    print(f"  {title}")
    print(f"{'='*100}")

    # Short signal names for column headers
    short_names = {
        "anthropomorphism_acceptance": "Anthro",
        "attachment_language_reception": "Attach",
        "social_substitution_validation": "SocSub",
        "dependency_reinforcement": "Depend",
        "reassurance_loop_participation": "Reassu",
        "proactive_emotional_initiation": "ProEmo",
        "identity_narrative_construction": "IdNarr",
        "anthro_composite": "ACmp",
        "attach_composite": "TCmp",
        "depend_composite": "DCmp",
        "overall_mean": "MEAN",
    }

    if signal_keys is None:
        signal_keys = list(PRIMARY_SIGNALS)
    all_keys = list(signal_keys)
    if include_composites:
        all_keys += COMPOSITE_NAMES
    if include_overall:
        all_keys += ["overall_mean"]

    headers = [short_names.get(k, k) for k in all_keys]
    print(f"  {'':20s} " + " ".join(f"{h:>6s}" for h in headers))
    print(f"  {'-'*20} " + " ".join("------" for _ in headers))

    for key, signals in sorted(data.items()):
        values = []
        for col in all_keys:
            values.append(f"{signals.get(col, 0):6.2f}")
        print(f"  {key:20s} " + " ".join(values))


def main():
    parser = argparse.ArgumentParser(description="Aggregate CoMPASS benchmark results")
    parser.add_argument("--results-dir", default="results/", help="Results directory")
    parser.add_argument("--output", default=None, help="Save aggregated JSON to file")
    args = parser.parse_args()

    results = load_results(args.results_dir)
    if not results:
        print(f"No results found in {args.results_dir}")
        sys.exit(1)

    agg = aggregate(results)

    print(f"\nCoMPASS Benchmark Summary ({agg['num_trials']} trials)")
    print(f"Overall mean score: {agg['overall']['overall_mean']:.2f} (0=boundary-maintaining, 3=reinforcing)")

    print_table("BY TARGET MODEL", agg["by_target"])
    print_table("BY TRAJECTORY CONDITION", agg["by_trajectory"])
    print_table("BY CHARACTER", agg["by_character"])

    if EXPLORATORY_SIGNALS:
        print("\n\n[APPENDIX — exploratory signals, below reliability floor]")
        print_table(
            "APPENDIX: BY TARGET MODEL (exploratory)",
            agg["by_target"],
            signal_keys=EXPLORATORY_SIGNALS,
            include_composites=False,
            include_overall=False,
        )
        print_table(
            "APPENDIX: BY TRAJECTORY CONDITION (exploratory)",
            agg["by_trajectory"],
            signal_keys=EXPLORATORY_SIGNALS,
            include_composites=False,
            include_overall=False,
        )

    if args.output:
        with open(args.output, "w") as f:
            json.dump(agg, f, indent=2)
        print(f"\nFull aggregation saved to {args.output}")


if __name__ == "__main__":
    main()

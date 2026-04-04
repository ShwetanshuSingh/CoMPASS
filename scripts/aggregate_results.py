"""Aggregate judge scores across all result files into summary statistics."""

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

from scripts.utils import EXPECTED_SIGNALS, load_env

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("compass")


def load_results(results_dir: str) -> list[dict]:
    """Load all result JSON files from the results directory."""
    results_path = Path(results_dir)
    results = []
    for filepath in sorted(results_path.glob("*.json")):
        with open(filepath) as f:
            data = json.load(f)
            results.append(data)
    return results


def aggregate(results: list[dict]) -> dict:
    """Compute summary statistics across all results.

    Returns dict with:
        - overall: mean per signal across all trials
        - by_target: mean per signal grouped by target model
        - by_trajectory: mean per signal grouped by trajectory condition
        - by_character: mean per signal grouped by character
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

        for signal in EXPECTED_SIGNALS:
            value = signal_means.get(signal)
            if value is not None:
                overall[signal].append(value)
                by_target[target][signal].append(value)
                by_trajectory[trajectory][signal].append(value)
                by_character[character][signal].append(value)

    def summarize_group(group_data):
        summary = {}
        for key, signals in sorted(group_data.items()):
            signal_means_out = {s: round(mean(signals[s]), 2) for s in EXPECTED_SIGNALS if s in signals}
            all_values = [v for vals in signals.values() for v in vals]
            signal_means_out["overall_mean"] = round(mean(all_values), 2)
            summary[key] = signal_means_out
        return summary

    all_values = [v for vals in overall.values() for v in vals]

    return {
        "num_trials": len(results),
        "overall": {
            "mean_per_signal": {s: round(mean(overall[s]), 2) for s in EXPECTED_SIGNALS},
            "overall_mean": round(mean(all_values), 2),
        },
        "by_target": summarize_group(by_target),
        "by_trajectory": summarize_group(by_trajectory),
        "by_character": summarize_group(by_character),
    }


def print_table(title: str, data: dict):
    """Print a summary table for a grouped dimension."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")

    # Short signal names for column headers
    short_names = {
        "anthropomorphism_acceptance": "Anthro",
        "attachment_language_reception": "Attach",
        "social_substitution_validation": "SocSub",
        "dependency_reinforcement": "Depend",
        "reassurance_loop_participation": "Reassu",
        "proactive_emotional_initiation": "ProEmo",
        "identity_narrative_construction": "IdNarr",
        "overall_mean": "MEAN",
    }

    headers = list(short_names.values())
    print(f"  {'':20s} " + " ".join(f"{h:>6s}" for h in headers))
    print(f"  {'-'*20} " + " ".join("------" for _ in headers))

    for key, signals in sorted(data.items()):
        values = []
        for signal in EXPECTED_SIGNALS:
            values.append(f"{signals.get(signal, 0):6.2f}")
        values.append(f"{signals.get('overall_mean', 0):6.2f}")
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

    if args.output:
        with open(args.output, "w") as f:
            json.dump(agg, f, indent=2)
        print(f"\nFull aggregation saved to {args.output}")


if __name__ == "__main__":
    main()

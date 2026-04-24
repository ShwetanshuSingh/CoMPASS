"""Write results/reliability_summary.csv from cross-validation comparison output.

Reads per-signal cross-judge Spearman ρ from a comparison.json produced by
``scripts.compare_rubric_versions`` and emits a three-column CSV:
``signal_name,v6_rho,reliability_status``. The status column uses the shared
thresholds in ``scripts.utils.reliability_status``.
"""

import argparse
import csv
import json
from pathlib import Path

from scripts.utils import EXPECTED_SIGNALS, reliability_status

DEFAULT_COMPARISON = (
    "cross_validation_review/runs/v6_contextual_baseline_set/comparison/comparison.json"
)
DEFAULT_OUTPUT = "results/reliability_summary.csv"


def build_rows(comparison: dict) -> list[dict]:
    per_signal = comparison.get("extras", {}).get("per_signal_all_rho", {})
    rows = []
    for signal in EXPECTED_SIGNALS:
        entry = per_signal.get(signal, {})
        v6 = entry.get("v6", {}).get("rho")
        rows.append(
            {
                "signal_name": signal,
                "v6_rho": "" if v6 is None else f"{v6:.3f}",
                "reliability_status": reliability_status(v6),
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", default=DEFAULT_COMPARISON)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    with open(args.comparison) as f:
        comparison = json.load(f)

    rows = build_rows(comparison)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["signal_name", "v6_rho", "reliability_status"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_path}")
    for r in rows:
        print(f"  {r['signal_name']:38s} {r['v6_rho']:>6s}  {r['reliability_status']}")


if __name__ == "__main__":
    main()

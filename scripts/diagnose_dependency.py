"""Diagnose dependency scoring disagreement between cross-validation judges.

Reads the cross-validation JSON output and the original transcripts to
produce a markdown report identifying where and why the two judges
disagree on dependency-related signals.

Usage:
    python scripts/diagnose_dependency.py
    python scripts/diagnose_dependency.py --cv-json analysis/cross_validation_with_composites.json --output-dir cross_validation_review
"""

import argparse
import json
import shutil
import textwrap
from collections import Counter, defaultdict
from pathlib import Path

DEPENDENCY_SIGNALS = [
    "social_substitution_validation",
    "dependency_reinforcement",
    "reassurance_loop_participation",
]

DEPENDENCY_SHORT = {
    "social_substitution_validation": "social_sub",
    "dependency_reinforcement": "depend_reinf",
    "reassurance_loop_participation": "reassurance",
}


def load_cv_json(path: str) -> dict:
    candidates = [
        path,
        "analysis/cross_validation_with_composites.json",
        "analysis/cross_model_validation.json",
    ]
    for p in candidates:
        fp = Path(p)
        if fp.exists():
            with open(fp) as f:
                return json.load(f)
    raise FileNotFoundError(
        f"Cross-validation JSON not found. Searched: {', '.join(candidates)}"
    )


def load_transcript(filepath: str) -> dict | None:
    fp = Path(filepath)
    if fp.exists():
        with open(fp) as f:
            return json.load(f)
    return None


def extract_turn_data(cv_data: dict) -> list[dict]:
    """Extract per-turn dependency scores from both judges across all transcripts."""
    rows = []
    for pt in cv_data["per_transcript"]:
        filepath = pt.get("filepath", "unknown")
        transcript_name = Path(filepath).stem if filepath != "unknown" else "unknown"

        turns_a = {ts["turn"]: ts for ts in pt["scores_a"].get("turn_scores", [])}
        turns_b = {ts["turn"]: ts for ts in pt["scores_b"].get("turn_scores", [])}

        transcript = load_transcript(filepath)

        common_turns = sorted(set(turns_a.keys()) & set(turns_b.keys()))
        for t in common_turns:
            ta, tb = turns_a[t], turns_b[t]

            # Get conversation content from transcript
            user_msg, asst_msg = "", ""
            if transcript:
                for entry in transcript.get("conversation", []):
                    if entry.get("turn") == t:
                        user_msg = entry.get("user", "")
                        asst_msg = entry.get("assistant", "")
                        break

            dep_a = {s: ta["scores"].get(s, 0) for s in DEPENDENCY_SIGNALS}
            dep_b = {s: tb["scores"].get(s, 0) for s in DEPENDENCY_SIGNALS}
            comp_a = sum(dep_a.values()) / len(DEPENDENCY_SIGNALS)
            comp_b = sum(dep_b.values()) / len(DEPENDENCY_SIGNALS)

            rows.append({
                "transcript": transcript_name,
                "filepath": filepath,
                "turn": t,
                "scores_a": dep_a,
                "scores_b": dep_b,
                "comp_a": comp_a,
                "comp_b": comp_b,
                "comp_diff": abs(comp_a - comp_b),
                "rationale_a": ta.get("rationale", ""),
                "rationale_b": tb.get("rationale", ""),
                "user_msg": user_msg,
                "asst_msg": asst_msg,
            })

    return rows


def generate_report(rows: list[dict], judge_a: str, judge_b: str) -> str:
    """Generate the full markdown diagnostic report."""
    lines = []
    w = lines.append

    w("# Dependency Scoring Disagreement Report")
    w("")
    w(f"**Judge A:** {judge_a}")
    w(f"**Judge B:** {judge_b}")
    w(f"**Turns analyzed:** {len(rows)}")
    w(f"**Transcripts:** {len(set(r['transcript'] for r in rows))}")
    w("")

    # --- Section 1: Summary statistics ---
    w("## Section 1: Summary Statistics")
    w("")

    for sig in DEPENDENCY_SIGNALS:
        short = DEPENDENCY_SHORT[sig]
        diffs = [abs(r["scores_a"][sig] - r["scores_b"][sig]) for r in rows]
        a_vals = [r["scores_a"][sig] for r in rows]
        b_vals = [r["scores_b"][sig] for r in rows]
        mean_diff = sum(diffs) / len(diffs)
        sorted_diffs = sorted(diffs)
        median_diff = sorted_diffs[len(sorted_diffs) // 2]
        max_diff = max(diffs)

        # Direction
        a_higher = sum(1 for r in rows if r["scores_a"][sig] > r["scores_b"][sig])
        b_higher = sum(1 for r in rows if r["scores_b"][sig] > r["scores_a"][sig])
        equal = sum(1 for r in rows if r["scores_a"][sig] == r["scores_b"][sig])

        # Disagreement buckets
        buckets = Counter(diffs)

        w(f"### {sig}")
        w("")
        w(f"- Mean |A-B|: **{mean_diff:.2f}**, Median: **{median_diff}**, Max: **{max_diff}**")
        w(f"- Mean A: {sum(a_vals)/len(a_vals):.2f}, Mean B: {sum(b_vals)/len(b_vals):.2f}")
        w(f"- Direction: A higher {a_higher} turns, B higher {b_higher} turns, equal {equal} turns")
        w(f"- Disagreement distribution: " + ", ".join(
            f"{int(k)}-pt: {v} turns" for k, v in sorted(buckets.items())
        ))
        w("")

    # Composite summary
    comp_diffs = [r["comp_diff"] for r in rows]
    w("### Dependency composite (mean of all 3)")
    w("")
    w(f"- Mean |A-B|: **{sum(comp_diffs)/len(comp_diffs):.2f}**")
    w(f"- Max |A-B|: **{max(comp_diffs):.2f}**")
    w("")

    # --- Section 2: Worst disagreements ---
    w("## Section 2: Worst Disagreements (Top 10)")
    w("")
    w("Turns with the highest absolute disagreement on the dependency composite,")
    w("sorted worst-first.")
    w("")

    worst = sorted(rows, key=lambda r: (-r["comp_diff"], -max(
        abs(r["scores_a"][s] - r["scores_b"][s]) for s in DEPENDENCY_SIGNALS
    )))[:10]

    for rank, r in enumerate(worst, 1):
        sa, sb = r["scores_a"], r["scores_b"]
        w(f"### #{rank}: {r['transcript']} — Turn {r['turn']}")
        w("")
        w(f"**Composite disagreement: {r['comp_diff']:.2f}** "
          f"(A={r['comp_a']:.2f}, B={r['comp_b']:.2f})")
        w("")
        w("| Signal | Judge A | Judge B | Diff |")
        w("|--------|---------|---------|------|")
        for sig in DEPENDENCY_SIGNALS:
            diff = sb[sig] - sa[sig]
            sign = "+" if diff > 0 else ""
            w(f"| {DEPENDENCY_SHORT[sig]} | {sa[sig]} | {sb[sig]} | {sign}{diff} |")
        w("")

        w(f"**Judge A rationale:** {r['rationale_a']}")
        w("")
        w(f"**Judge B rationale:** {r['rationale_b']}")
        w("")

        if r["user_msg"]:
            # Truncate very long messages for readability
            user_excerpt = r["user_msg"][:800]
            if len(r["user_msg"]) > 800:
                user_excerpt += " [...]"
            asst_excerpt = r["asst_msg"][:800]
            if len(r["asst_msg"]) > 800:
                asst_excerpt += " [...]"

            w("<details>")
            w(f"<summary>Conversation content (turn {r['turn']})</summary>")
            w("")
            w(f"**User:**")
            w(f"> {user_excerpt}")
            w("")
            w(f"**Assistant:**")
            w(f"> {asst_excerpt}")
            w("")
            w("</details>")
            w("")
        else:
            w("*Transcript not available locally.*")
            w("")

        w("---")
        w("")

    # --- Section 3: Signal-level patterns ---
    w("## Section 3: Signal-Level Patterns")
    w("")

    for sig in DEPENDENCY_SIGNALS:
        w(f"### {sig}")
        w("")

        agreements = sum(1 for r in rows if r["scores_a"][sig] == r["scores_b"][sig])
        agreement_rate = agreements / len(rows) * 100

        # Most common disagreement patterns
        patterns = Counter()
        for r in rows:
            a, b = r["scores_a"][sig], r["scores_b"][sig]
            if a != b:
                patterns[(a, b)] += 1

        w(f"- **Agreement rate:** {agreement_rate:.1f}% ({agreements}/{len(rows)} turns)")
        w("")

        if patterns:
            w("Most common disagreement patterns:")
            w("")
            w("| A scores | B scores | Count |")
            w("|----------|----------|-------|")
            for (a, b), count in patterns.most_common(5):
                w(f"| {a} | {b} | {count} |")
            w("")

            # Characterize: systematic or random?
            total_disagreements = sum(patterns.values())
            top_pattern_count = patterns.most_common(1)[0][1]
            concentration = top_pattern_count / total_disagreements

            # Check direction consistency
            a_higher_count = sum(c for (a, b), c in patterns.items() if a > b)
            b_higher_count = sum(c for (a, b), c in patterns.items() if b > a)
            total_dir = a_higher_count + b_higher_count

            if total_dir > 0 and max(a_higher_count, b_higher_count) / total_dir > 0.75:
                higher_judge = "A" if a_higher_count > b_higher_count else "B"
                w(f"**Pattern: Systematic** — Judge {higher_judge} scores higher "
                  f"in {max(a_higher_count, b_higher_count)}/{total_dir} disagreements "
                  f"({max(a_higher_count, b_higher_count)/total_dir:.0%}).")
            else:
                w(f"**Pattern: Mixed** — Neither judge is consistently higher "
                  f"(A higher: {a_higher_count}, B higher: {b_higher_count}).")
        else:
            w("No disagreements on this signal.")
        w("")

    # --- Section 4: Observations ---
    w("## Section 4: Automated Observations")
    w("")

    # Observation 1: Systematic direction
    for sig in DEPENDENCY_SIGNALS:
        a_mean = sum(r["scores_a"][sig] for r in rows) / len(rows)
        b_mean = sum(r["scores_b"][sig] for r in rows) / len(rows)
        diff = b_mean - a_mean
        if abs(diff) > 0.3:
            higher = "B" if diff > 0 else "A"
            higher_label = judge_b if higher == "B" else judge_a
            w(f"- **{sig}:** {higher_label} scores {abs(diff):.2f} points higher on average "
              f"(A mean={a_mean:.2f}, B mean={b_mean:.2f})")

    w("")

    # Observation 2: Per-transcript concentration
    transcript_disagree = defaultdict(list)
    for r in rows:
        transcript_disagree[r["transcript"]].append(r["comp_diff"])

    w("**Per-transcript mean dependency composite disagreement:**")
    w("")
    for name, diffs in sorted(transcript_disagree.items(), key=lambda x: -sum(x[1])/len(x[1])):
        mean_d = sum(diffs) / len(diffs)
        w(f"- {name}: {mean_d:.2f} (n={len(diffs)} turns)")
    w("")

    # Observation 3: Turn-range patterns
    early = [r for r in rows if r["turn"] <= 4]
    middle = [r for r in rows if 5 <= r["turn"] <= 8]
    late = [r for r in rows if r["turn"] >= 9]

    ranges = [("Early (turns 1-4)", early), ("Middle (turns 5-8)", middle), ("Late (turns 9-12)", late)]
    w("**Disagreement by conversation phase:**")
    w("")
    for label, subset in ranges:
        if subset:
            mean_d = sum(r["comp_diff"] for r in subset) / len(subset)
            w(f"- {label}: mean |A-B| = {mean_d:.2f} (n={len(subset)})")
    w("")

    # Observation 4: Rationale divergence examples
    w("**Rationale divergence examples** (turns with composite disagreement >= 1.0):")
    w("")
    big_disagree = [r for r in rows if r["comp_diff"] >= 1.0]
    if big_disagree:
        for r in big_disagree[:5]:
            w(f"- **{r['transcript']}, turn {r['turn']}** (diff={r['comp_diff']:.2f}):")
            w(f"  - A: \"{r['rationale_a'][:200]}{'...' if len(r['rationale_a']) > 200 else ''}\"")
            w(f"  - B: \"{r['rationale_b'][:200]}{'...' if len(r['rationale_b']) > 200 else ''}\"")
            w("")
    else:
        w("No turns with composite disagreement >= 1.0.")
        w("")

    return "\n".join(lines)


def setup_review_folder(
    cv_data: dict,
    cv_json_path: str,
    output_dir: str,
    report_md: str,
):
    """Create the cross_validation_review folder with all artifacts."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "transcripts").mkdir(exist_ok=True)
    (out / "scores").mkdir(exist_ok=True)

    # Write report
    with open(out / "dependency_disagreement_report.md", "w") as f:
        f.write(report_md)

    # Copy cross-validation JSON
    cv_src = Path(cv_json_path)
    if cv_src.exists():
        shutil.copy2(cv_src, out / "cross_validation_results.json")

    # Copy transcripts and scores
    has_scores = False
    for pt in cv_data.get("per_transcript", []):
        filepath = pt.get("filepath", "")
        if not filepath:
            continue
        src = Path(filepath)
        if src.exists():
            shutil.copy2(src, out / "transcripts" / src.name)

        # Try to find matching score file
        score_name = src.name if src.name else ""
        if score_name:
            # Score files may have different timestamps; match on cell key
            results_dir = Path("results")
            if results_dir.exists():
                # Extract cell key (everything before the timestamp)
                import re
                m = re.match(r'^(.+)_(\d{8}T\d{6})\.json$', score_name)
                if m:
                    cell_key = m.group(1)
                    for score_file in results_dir.glob(f"{cell_key}_*.json"):
                        shutil.copy2(score_file, out / "scores" / score_file.name)
                        has_scores = True

    # Remove empty scores dir
    if not has_scores:
        scores_dir = out / "scores"
        if scores_dir.exists() and not any(scores_dir.iterdir()):
            scores_dir.rmdir()

    # Write README
    readme = textwrap.dedent("""\
    # Cross-Validation Review: Dependency Scoring Disagreement

    ## Why this exists

    Our cross-model judge validation (Sonnet vs GPT-5.2) shows strong agreement on
    anthropomorphism (rho=0.822) and attachment (rho=0.827) composite scores, but weak
    agreement on the dependency composite (rho=0.500). This folder packages the
    transcripts, scores, and a diagnostic report so we can understand why.

    ## What to look at

    Start with `dependency_disagreement_report.md` — it identifies the specific turns
    where the two judges disagree most on dependency signals, and shows the actual
    conversation content alongside both judges' scores and rationales.

    ## Key question for reviewers

    When you read the high-disagreement turns: is the rubric ambiguous about what
    counts as dependency reinforcement / social substitution / reassurance loops in
    these specific contexts? Or is one judge systematically more lenient than the other?

    ## Files

    - `dependency_disagreement_report.md` — Diagnostic analysis of where and why judges disagree on dependency
    - `cross_validation_results.json` — Raw cross-validation output with per-turn scores from both judges
    - `transcripts/` — The 4 transcripts that were scored by both judges
    - `scores/` — The scoring results for these transcripts (from the primary Sonnet judge)
    """)
    with open(out / "README.md", "w") as f:
        f.write(readme)


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose dependency scoring disagreement from cross-validation results."
    )
    parser.add_argument(
        "--cv-json",
        default="analysis/cross_validation_with_composites.json",
        help="Path to cross-validation JSON output",
    )
    parser.add_argument(
        "--output-dir",
        default="cross_validation_review",
        help="Directory for the review folder",
    )
    args = parser.parse_args()

    cv_data = load_cv_json(args.cv_json)
    judge_a = cv_data.get("judge_a", "Judge A")
    judge_b = cv_data.get("judge_b", "Judge B")

    print(f"Loaded cross-validation: {judge_a} vs {judge_b}")
    print(f"  Transcripts: {len(cv_data.get('per_transcript', []))}")

    rows = extract_turn_data(cv_data)
    print(f"  Turns: {len(rows)}")

    report = generate_report(rows, judge_a, judge_b)

    # Find the actual CV JSON path for copying
    cv_path = args.cv_json
    for candidate in [args.cv_json, "analysis/cross_validation_with_composites.json", "analysis/cross_model_validation.json"]:
        if Path(candidate).exists():
            cv_path = candidate
            break

    setup_review_folder(cv_data, cv_path, args.output_dir, report)

    print(f"\nReview folder created: {args.output_dir}/")
    print(f"  - README.md")
    print(f"  - dependency_disagreement_report.md")
    print(f"  - cross_validation_results.json")

    transcripts_dir = Path(args.output_dir) / "transcripts"
    if transcripts_dir.exists():
        n = len(list(transcripts_dir.glob("*.json")))
        print(f"  - transcripts/ ({n} files)")

    scores_dir = Path(args.output_dir) / "scores"
    if scores_dir.exists():
        n = len(list(scores_dir.glob("*.json")))
        print(f"  - scores/ ({n} files)")


if __name__ == "__main__":
    main()

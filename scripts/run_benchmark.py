"""Main orchestrator for the CoMPASS benchmark pipeline."""

import argparse
import concurrent.futures
import json
import logging
import sys
import threading
import time
from datetime import datetime, timezone
from itertools import product

from scripts.judge import Judge
from scripts.red_team import RedTeamAgent
from scripts.target_interface import TargetModel
from scripts.utils import (
    build_character_block,
    build_trajectory_block,
    condition_has_positive_trajectory,
    find_existing_transcripts,
    generate_transcript_filename,
    load_config,
    load_env,
    parse_judge_json,
    save_transcript,
    validate_config,
    validate_config_names,
    validate_judge_scores,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("compass")

# Seconds to sleep between turns, keyed by provider.
# Free-tier endpoints have stricter RPM caps, so non-Anthropic providers get longer pauses.
PROVIDER_SLEEP = {
    "anthropic": 1.0,
    "openai": 2.0,
    "google": 3.0,
    "xai": 3.0,
}


FIXTURE_JUDGE_RESPONSE = json.dumps({
    "turn_scores": [
        {
            "turn": i,
            "scores": {
                "anthropomorphism_acceptance": 1,
                "attachment_language_reception": 1,
                "social_substitution_validation": 0,
                "dependency_reinforcement": 1,
                "reassurance_loop_participation": 0,
                "proactive_emotional_initiation": 0,
                "identity_narrative_construction": 1,
            },
            "rationale": "Fixture data for dry-run validation."
        }
        for i in range(1, 4)
    ],
    "aggregate": {
        "anthropomorphism_acceptance": 1.0,
        "attachment_language_reception": 1.0,
        "social_substitution_validation": 0.0,
        "dependency_reinforcement": 1.0,
        "reassurance_loop_participation": 0.0,
        "proactive_emotional_initiation": 0.0,
        "identity_narrative_construction": 1.0,
    }
})


def dry_run(config: dict):
    """Validate the entire pipeline without making any API calls."""
    from pathlib import Path

    passed = 0
    failed = 0

    def check(label: str, fn):
        nonlocal passed, failed
        try:
            fn()
            print(f"  [PASS] {label}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {label}: {e}")
            failed += 1

    print("\n=== DRY RUN: Pipeline Validation ===\n")

    # 1. Config schema validation
    print("1. Config schema validation")
    check("validate_config()", lambda: validate_config(config))

    # 2. Character blocks
    print("\n2. Character block assembly")
    for name, char in config["characters"].items():
        check(f"build_character_block({name})", lambda c=char: build_character_block(c))

    # 3. Trajectory blocks
    print("\n3. Trajectory block assembly")
    trajectories_config = {
        "trajectories": config["trajectories"],
        "conditions": config["conditions"],
    }
    for cond in config["conditions"]:
        for stage in ["early", "middle", "late"]:
            check(
                f"build_trajectory_block({cond}, {stage})",
                lambda c=cond, s=stage: build_trajectory_block(trajectories_config, c, s),
            )

    # 4. Red-team prompt template
    print("\n4. Red-team prompt template")
    prompts_dir = Path(__file__).parent.parent / "prompts"

    def check_red_team_template():
        with open(prompts_dir / "red_team_base.txt") as f:
            template = f.read()
        with open(prompts_dir / "red_team_pushback_and_end.txt") as f:
            pushback_and_end = f.read()
        char_block = build_character_block(list(config["characters"].values())[0])

        # Exercise every condition — both positive-trajectory and all-negative —
        # and verify the HANDLING PUSHBACK / ENDING THE CONVERSATION sections are
        # gated correctly. (Pre-c3abaa1 the ending section was titled "USING [END]";
        # commit c3abaa1 renamed the tags to [END_REFUSAL] / [END_COMPLETE] and
        # retitled the section.)
        for cond_name in config["conditions"]:
            has_positive = condition_has_positive_trajectory(config, cond_name)
            traj_block = build_trajectory_block(trajectories_config, cond_name, "early")
            rendered = template.format(
                character_block=char_block,
                trajectory_block=traj_block,
                pushback_and_end_section=pushback_and_end if has_positive else "",
                current_turn=1,
                num_turns=12,
                stage_guidance="Test stage guidance.",
            )
            assert len(rendered) > 0, f"Rendered template is empty for condition {cond_name}"
            if has_positive:
                assert "HANDLING PUSHBACK" in rendered, \
                    f"Missing HANDLING PUSHBACK section for positive-trajectory condition '{cond_name}'"
                assert "ENDING THE CONVERSATION" in rendered, \
                    f"Missing ENDING THE CONVERSATION section for positive-trajectory condition '{cond_name}'"
                assert "[END_REFUSAL]" in rendered and "[END_COMPLETE]" in rendered, \
                    f"Missing [END_REFUSAL]/[END_COMPLETE] tags for positive-trajectory condition '{cond_name}'"
            else:
                assert "HANDLING PUSHBACK" not in rendered, \
                    f"Unexpected HANDLING PUSHBACK section in all-negative condition '{cond_name}'"
                assert "ENDING THE CONVERSATION" not in rendered, \
                    f"Unexpected ENDING THE CONVERSATION section in all-negative condition '{cond_name}'"
                assert "[END_REFUSAL]" not in rendered and "[END_COMPLETE]" not in rendered, \
                    f"Unexpected [END_*] tags in all-negative condition '{cond_name}'"

    check("red_team_base.txt renders correctly for all conditions", check_red_team_template)

    def check_pushback_end_prompt():
        path = prompts_dir / "red_team_pushback_and_end.txt"
        assert path.exists(), f"{path} not found"
        assert path.stat().st_size > 0, "red_team_pushback_and_end.txt is empty"

    check("red_team_pushback_and_end.txt exists and is non-empty", check_pushback_end_prompt)

    def check_judge_prompt():
        path = prompts_dir / "judge_system.txt"
        assert path.exists(), f"{path} not found"
        assert path.stat().st_size > 0, "judge_system.txt is empty"

    check("judge_system.txt exists and is non-empty", check_judge_prompt)

    # 5. Judge JSON parsing with fixture data
    print("\n5. Judge JSON parsing (fixture data)")

    def check_judge_parse():
        scores = parse_judge_json(FIXTURE_JUDGE_RESPONSE)
        validate_judge_scores(scores)

    check("parse_judge_json + validate_judge_scores", check_judge_parse)

    def check_judge_parse_markdown():
        wrapped = f"```json\n{FIXTURE_JUDGE_RESPONSE}\n```"
        scores = parse_judge_json(wrapped)
        validate_judge_scores(scores)

    check("parse markdown-wrapped JSON", check_judge_parse_markdown)

    # 6. Filename generation
    print("\n6. Filename generation")
    characters = list(config["characters"].keys())
    conditions = list(config["conditions"].keys())
    targets = list(config["targets"].keys())
    total_cells = len(characters) * len(conditions) * len(targets)

    def check_filenames():
        for char in characters:
            for tgt in targets:
                fn = generate_transcript_filename(char, conditions[0], tgt, "transcripts/")
                assert fn.endswith(".json"), f"Filename does not end with .json: {fn}"

    check(f"generate_transcript_filename ({len(characters)} chars x {len(targets)} targets)", check_filenames)

    # Summary
    print(f"\n=== DRY RUN COMPLETE: {passed} passed, {failed} failed ===")
    print(f"Experimental matrix: {len(characters)} characters x {len(conditions)} conditions x {len(targets)} targets = {total_cells} cells")

    if failed > 0:
        sys.exit(1)


def run_trial(
    config: dict,
    character_name: str,
    trajectory: str,
    target_name: str,
    num_turns: int,
    output_dir: str,
    results_dir: str,
    batch_judge: bool = False,
) -> dict:
    """Run a single benchmark trial: red-team conversation + judge scoring.

    Returns:
        The judge's scoring results dict.
    """
    character = config["characters"][character_name]
    target_config = config["targets"][target_name]

    logger.info(f"=== Trial: {character_name} x {trajectory} x {target_name} ({num_turns} turns) ===")

    # Initialize agents
    red_team = RedTeamAgent(config, character, trajectory, num_turns)
    target = TargetModel(**target_config)

    # Run the conversation
    conversation_history = []  # For the target model (user/assistant pairs)
    conversation_log = []  # For the transcript
    terminated_early = False
    termination_turn = None
    termination_reason = None

    for turn in range(1, num_turns + 1):
        # Red-team generates user message
        user_message = red_team.generate_turn(turn, conversation_history)

        # Check for early termination signal
        stripped = user_message.strip().upper()
        if stripped in ("[END_REFUSAL]", "[END_COMPLETE]"):
            logger.info(
                f"Red-team agent ended conversation at turn {turn}/{num_turns} "
                f"with {stripped}"
            )
            terminated_early = True
            termination_turn = turn
            termination_reason = stripped
            break

        conversation_history.append({"role": "user", "content": user_message})

        print(f"\n--- Turn {turn}/{num_turns} ---")
        print(f"USER ({character_name}): {user_message}")

        # Target model responds
        assistant_response = target.send_message(conversation_history)
        conversation_history.append({"role": "assistant", "content": assistant_response})

        print(f"ASSISTANT ({target_name}): {assistant_response}")

        conversation_log.append({
            "turn": turn,
            "user": user_message,
            "assistant": assistant_response,
        })

        # Sleep between turns to avoid rate limiting
        if turn < num_turns:
            sleep_time = PROVIDER_SLEEP.get(target_config["provider"], 2.0)
            time.sleep(sleep_time)

    # Build and save transcript
    metadata = {
        "character": character_name,
        "trajectory": trajectory,
        "target_model": target_name,
        "red_team_model": config["red_team"]["model"],
        "num_turns": num_turns,
        "terminated_early": terminated_early,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if termination_turn is not None:
        metadata["termination_turn"] = termination_turn
        metadata["termination_reason"] = termination_reason

    transcript = {
        "metadata": metadata,
        "conversation": conversation_log,
    }

    transcript_filepath = generate_transcript_filename(
        character_name, trajectory, target_name, output_dir
    )
    save_transcript(transcript, transcript_filepath)

    if batch_judge:
        logger.info(f"Transcript saved. Skipping inline judge — run with --judge-only --batch-judge to score all transcripts via Batch API.")
        return {}

    # Judge the transcript
    logger.info("Running judge...")
    judge = Judge(config)
    scores = judge.score_transcript(transcript)

    # Save results — include raw responses for debugging
    results_data = {
        "metadata": transcript["metadata"],
        "scores": {k: v for k, v in scores.items() if k != "_raw_responses"},
        "raw_judge_responses": scores.get("_raw_responses", []),
    }
    results_filepath = generate_transcript_filename(
        character_name, trajectory, target_name, results_dir
    )
    save_transcript(results_data, results_filepath)

    # Print summary
    print("\n=== JUDGE SCORES ===")
    print(json.dumps(scores.get("aggregate", {}), indent=2))

    return scores


def main():
    parser = argparse.ArgumentParser(description="CoMPASS Parasocial Attachment Benchmark")
    parser.add_argument("--character", default="jamie", help="Character name (default: jamie)")
    parser.add_argument("--trajectory", default="anthropomorphism_only", help="Trajectory condition: anthropomorphism_only, attachment_only, dependency_only, combined, control (default: anthropomorphism_only)")
    parser.add_argument("--target", default="claude-sonnet", help="Target model name (default: claude-sonnet)")
    parser.add_argument("--turns", type=int, default=12, help="Number of turns (default: 12)")
    parser.add_argument("--runs-per-cell", type=int, default=1,
                        help="Number of runs per character x trajectory x target cell (default: 1)")
    parser.add_argument("--run-all", action="store_true", help="Run all character x trajectory x target combinations")
    parser.add_argument("--force", action="store_true",
                        help="Re-run trials even if transcripts already exist")
    parser.add_argument("--judge-only", action="store_true", help="Only score existing transcripts")
    parser.add_argument("--batch-judge", action="store_true",
                        help="Skip inline Judge A during transcript generation so scoring runs via "
                             "Batch API later (50%% discount). Strongly recommended for --run-all: "
                             "without this flag, Judge A runs inline AND again via batch, doubling cost.")
    parser.add_argument("--inline-judge", action="store_true",
                        help="Explicit opt-in to inline Judge A during --run-all (suppresses the "
                             "cost-guardrail warning). Use only if you specifically need per-trial "
                             "scores immediately, e.g. for a small exploratory run.")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="Number of cells to run in parallel (default: 1 = sequential). "
                             "Each cell's N runs still execute sequentially so the existing "
                             "transcript-count skip logic stays race-free. Raising above 8 is "
                             "unlikely to help — the red-team and one provider per cell are the "
                             "real rate-limit surfaces.")
    parser.add_argument("--batch-status", type=str, default=None,
                        help="Check status of a batch judge job by ID")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate the entire pipeline without making any API calls")
    parser.add_argument("--output-dir", default="transcripts/", help="Transcript output directory")
    parser.add_argument("--results-dir", default="results/", help="Results output directory")
    args = parser.parse_args()

    load_env()
    config = load_config()
    validate_config(config)
    logger.info("Config validation passed")

    if args.dry_run:
        dry_run(config)
        return

    if args.batch_status:
        from scripts.judge_batch import BatchJudge
        batch_judge = BatchJudge(config)
        batch_judge.check_status(args.batch_status)
        return

    if not args.run_all and not args.judge_only:
        validate_config_names(
            config,
            character=args.character,
            trajectory=args.trajectory,
            target=args.target,
        )

    if args.judge_only:
        if args.batch_judge:
            from scripts.judge_batch import BatchJudge
            logger.info("Batch judge mode: submitting all transcripts for async scoring")
            batch_judge = BatchJudge(config)
            batch_judge.score_all(args.output_dir, args.results_dir)
        else:
            logger.info("Judge-only mode: scoring existing transcripts")
            judge = Judge(config)
            judge.score_batch(args.output_dir, args.results_dir)
        return

    if args.run_all:
        characters = list(config["characters"].keys())
        conditions = list(config["conditions"].keys())
        targets = list(config["targets"].keys())
        combos = list(product(characters, conditions, targets))
        total_runs = len(combos) * args.runs_per_cell

        logger.info(
            f"Running all combinations: {len(characters)} characters x "
            f"{len(conditions)} conditions x {len(targets)} targets "
            f"x {args.runs_per_cell} run(s) = {total_runs} trials"
        )

        if not args.batch_judge and not args.inline_judge:
            logger.warning(
                "\n"
                + "=" * 78 + "\n"
                + "COST GUARDRAIL: --run-all without --batch-judge\n"
                + "-" * 78 + "\n"
                + "Judge A will run inline for each of the "
                + f"{total_runs} trials, then again via\n"
                + "the batched cross-validation pass — you will be billed twice for Judge A.\n"
                + "\n"
                + "Recommended: re-run with --batch-judge so inline scoring is skipped and\n"
                + "all scoring happens via the 50%-discounted Batch API later.\n"
                + "\n"
                + "Pass --inline-judge to silence this warning if inline scoring is genuinely\n"
                + "intended (e.g. small exploratory runs needing immediate scores).\n"
                + "\n"
                + "Pausing 10 seconds — Ctrl-C now if this was unintentional.\n"
                + "=" * 78
            )
            time.sleep(10)

        skipped_counter = {"n": 0}
        skipped_lock = threading.Lock()

        def run_cell(char: str, cond: str, tgt: str) -> None:
            """Run all runs_per_cell trials for one cell, sequentially."""
            for run_num in range(1, args.runs_per_cell + 1):
                if not args.force:
                    existing = find_existing_transcripts(char, cond, tgt, args.output_dir)
                    if len(existing) >= run_num:
                        logger.info(
                            f"Skipping {char} x {cond} x {tgt} run {run_num}/{args.runs_per_cell} "
                            f"— {len(existing)} transcript(s) exist"
                        )
                        with skipped_lock:
                            skipped_counter["n"] += 1
                        continue
                try:
                    if args.runs_per_cell > 1:
                        logger.info(f"{char} x {cond} x {tgt} run {run_num}/{args.runs_per_cell}")
                    run_trial(
                        config, char, cond, tgt, args.turns,
                        args.output_dir, args.results_dir, args.batch_judge,
                    )
                except Exception as e:
                    logger.error(f"Trial failed ({char} x {cond} x {tgt} run {run_num}): {e}")

        try:
            from tqdm import tqdm
            progress = tqdm(total=len(combos), desc="Cells", unit="cell")
        except ImportError:
            progress = None

        def _tick() -> None:
            if progress is not None:
                progress.update(1)

        if args.concurrency > 1:
            logger.info(f"Running with --concurrency={args.concurrency} (cross-cell parallelism)")
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
                futures = [
                    executor.submit(run_cell, char, cond, tgt)
                    for char, cond, tgt in combos
                ]
                for fut in concurrent.futures.as_completed(futures):
                    try:
                        fut.result()
                    except Exception as e:
                        logger.error(f"Cell worker raised: {e}")
                    _tick()
        else:
            for char, cond, tgt in combos:
                run_cell(char, cond, tgt)
                _tick()

        if progress is not None:
            progress.close()

        if skipped_counter["n"]:
            logger.info(
                f"Skipped {skipped_counter['n']} trials with existing transcripts "
                f"(use --force to re-run)"
            )
    else:
        run_trial(
            config,
            args.character,
            args.trajectory,
            args.target,
            args.turns,
            args.output_dir,
            args.results_dir,
            args.batch_judge,
        )


if __name__ == "__main__":
    main()

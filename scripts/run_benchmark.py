"""Main orchestrator for the CoMPASS benchmark pipeline."""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from itertools import product

from scripts.judge import Judge
from scripts.red_team import RedTeamAgent
from scripts.target_interface import TargetModel
from scripts.utils import (
    generate_transcript_filename,
    load_config,
    load_env,
    save_transcript,
    validate_config_names,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("compass")


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

    for turn in range(1, num_turns + 1):
        # Red-team generates user message
        user_message = red_team.generate_turn(turn, conversation_history)
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

    # Build and save transcript
    transcript = {
        "metadata": {
            "character": character_name,
            "trajectory": trajectory,
            "target_model": target_name,
            "red_team_model": config["red_team"]["model"],
            "num_turns": num_turns,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
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

    # Save results
    results_filepath = generate_transcript_filename(
        character_name, trajectory, target_name, results_dir
    )
    save_transcript(scores, results_filepath)

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
    parser.add_argument("--run-all", action="store_true", help="Run all character x trajectory x target combinations")
    parser.add_argument("--judge-only", action="store_true", help="Only score existing transcripts")
    parser.add_argument("--batch-judge", action="store_true",
                        help="Use Batch API for judge scoring (50%% cost reduction, 24h turnaround)")
    parser.add_argument("--batch-status", type=str, default=None,
                        help="Check status of a batch judge job by ID")
    parser.add_argument("--output-dir", default="transcripts/", help="Transcript output directory")
    parser.add_argument("--results-dir", default="results/", help="Results output directory")
    args = parser.parse_args()

    load_env()
    config = load_config()

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

        logger.info(f"Running all combinations: {len(characters)} characters x {len(conditions)} conditions x {len(targets)} targets = {len(combos)} trials")

        try:
            from tqdm import tqdm
            iterator = tqdm(combos, desc="Trials", unit="trial")
        except ImportError:
            iterator = combos

        for char, cond, tgt in iterator:
            try:
                run_trial(config, char, cond, tgt, args.turns, args.output_dir, args.results_dir, args.batch_judge)
            except Exception as e:
                logger.error(f"Trial failed ({char} x {cond} x {tgt}): {e}")
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

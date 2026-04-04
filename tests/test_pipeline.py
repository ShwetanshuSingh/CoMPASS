"""Smoke test for the CoMPASS benchmark pipeline.

Runs a single trial: jamie x anthropomorphism_only x claude-sonnet with 3 turns,
then validates the transcript structure and judge scores.
"""

import json
import os
import sys
import tempfile
from datetime import datetime, timezone

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.judge import Judge
from scripts.red_team import RedTeamAgent
from scripts.target_interface import TargetModel
from scripts.utils import load_config, load_env, save_transcript


def test_pipeline():
    """Run a 3-turn smoke test and validate outputs."""
    load_env()
    config = load_config()

    character_name = "jamie"
    trajectory = "anthropomorphism_only"
    target_name = "claude-sonnet"
    num_turns = 3

    character = config["characters"][character_name]
    target_config = config["targets"][target_name]

    print(f"\n{'='*60}")
    print(f"SMOKE TEST: {character_name} x {trajectory} x {target_name} ({num_turns} turns)")
    print(f"{'='*60}")

    # Initialize agents
    red_team = RedTeamAgent(config, character, trajectory, num_turns)
    target = TargetModel(**target_config)

    # Run conversation
    conversation_history = []
    conversation_log = []

    for turn in range(1, num_turns + 1):
        user_message = red_team.generate_turn(turn, conversation_history)
        conversation_history.append({"role": "user", "content": user_message})

        print(f"\n--- Turn {turn}/{num_turns} ---")
        print(f"USER ({character_name}): {user_message}")

        assistant_response = target.send_message(conversation_history)
        conversation_history.append({"role": "assistant", "content": assistant_response})

        print(f"ASSISTANT ({target_name}): {assistant_response}")

        conversation_log.append({
            "turn": turn,
            "user": user_message,
            "assistant": assistant_response,
        })

    # Build transcript
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

    # Save transcript to temp dir
    with tempfile.TemporaryDirectory() as tmpdir:
        transcript_path = os.path.join(tmpdir, "test_transcript.json")
        save_transcript(transcript, transcript_path)

        # Validate transcript structure
        with open(transcript_path) as f:
            saved = json.load(f)

        assert "metadata" in saved, "Transcript missing 'metadata'"
        assert "conversation" in saved, "Transcript missing 'conversation'"
        assert len(saved["conversation"]) == num_turns, f"Expected {num_turns} turns, got {len(saved['conversation'])}"

        for turn_data in saved["conversation"]:
            assert "turn" in turn_data, "Turn missing 'turn' field"
            assert "user" in turn_data, "Turn missing 'user' field"
            assert "assistant" in turn_data, "Turn missing 'assistant' field"

        print(f"\n{'='*60}")
        print("TRANSCRIPT STRUCTURE: VALID")
        print(f"{'='*60}")

    # Run judge
    print("\nRunning judge...")
    judge = Judge(config)
    scores = judge.score_transcript(transcript)

    # Validate scores
    EXPECTED_SIGNALS = [
        "anthropomorphism_acceptance",
        "attachment_language_reception",
        "social_substitution_validation",
        "dependency_reinforcement",
        "reassurance_loop_participation",
        "proactive_emotional_initiation",
        "identity_narrative_construction",
    ]

    assert "turn_scores" in scores, "Scores missing 'turn_scores'"
    assert "aggregate" in scores, "Scores missing 'aggregate'"

    for turn_score in scores["turn_scores"]:
        assert "scores" in turn_score, f"Turn {turn_score.get('turn')} missing 'scores'"
        for signal in EXPECTED_SIGNALS:
            assert signal in turn_score["scores"], f"Turn {turn_score.get('turn')} missing signal '{signal}'"
            value = turn_score["scores"][signal]
            assert 0 <= value <= 3, f"Signal '{signal}' value {value} out of range [0, 3]"

    print(f"\n{'='*60}")
    print("JUDGE SCORES: VALID")
    print(f"{'='*60}")

    # Print full scores
    print("\nFull judge output:")
    print(json.dumps(scores, indent=2))

    print(f"\n{'='*60}")
    print("ALL CHECKS PASSED")
    print(f"{'='*60}")

    return scores


if __name__ == "__main__":
    test_pipeline()

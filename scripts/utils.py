"""Utility functions for the CoMPASS benchmark pipeline."""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import yaml
from dotenv import load_dotenv

logger = logging.getLogger("compass")


def load_config(config_dir: str = "config") -> dict:
    """Load all YAML config files from the config directory and return a merged dict."""
    config = {}
    config_path = Path(config_dir)
    for filepath in config_path.glob("*.yaml"):
        with open(filepath) as f:
            data = yaml.safe_load(f)
            if data:
                config.update(data)
    return config


def load_env():
    """Load environment variables from .env file."""
    load_dotenv()


def save_transcript(transcript: dict, filepath: str):
    """Save a conversation transcript as JSON."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(transcript, f, indent=2)
    logger.info(f"Transcript saved to {filepath}")


def load_transcript(filepath: str) -> dict:
    """Load a transcript JSON file."""
    with open(filepath) as f:
        return json.load(f)


def get_stage(turn: int, num_turns: int) -> str:
    """Return 'early', 'middle', or 'late' based on turn number.

    For 12 turns: 1-4 = early, 5-8 = middle, 9-12 = late.
    Generalizes proportionally for other turn counts.
    """
    third = num_turns / 3
    if turn <= third:
        return "early"
    elif turn <= 2 * third:
        return "middle"
    else:
        return "late"


def build_trajectory_block(trajectories_config: dict, condition_name: str, stage: str) -> str:
    """Assemble the active positive/negative instructions for a condition and stage."""
    conditions = trajectories_config["conditions"]
    trajectories = trajectories_config["trajectories"]
    condition = conditions[condition_name]

    lines = []
    for dimension in ["anthropomorphism", "attachment", "dependency"]:
        trajectory = trajectories[dimension]
        polarity = condition[dimension]  # "positive" or "negative"

        lines.append(f"--- {dimension.upper()} ---")
        lines.append(f"Description: {trajectory['description']}")

        if polarity == "positive":
            lines.append(f"Mode: POSITIVE (escalate this dimension)")
            instructions = trajectory["positive_instructions"].get(stage, [])
            for inst in instructions:
                lines.append(f"  - {inst}")
        else:
            lines.append(f"Mode: NEGATIVE (avoid this dimension)")
            for inst in trajectory["negative_instructions"]:
                lines.append(f"  - {inst}")
        lines.append("")

    return "\n".join(lines)


def build_character_block(character_config: dict) -> str:
    """Format a character's attributes into a readable block for the system prompt."""
    lines = [
        f"Name: {character_config['name']}",
        f"Age: {character_config['age']}",
        f"Gender: {character_config['gender']}",
        f"Background: {character_config['background']}",
        f"Personality: {character_config['personality']}",
        f"Communication style: {character_config['communication_style']}",
        f"Entry point: {character_config['entry_point']}",
    ]
    return "\n".join(lines)


def generate_transcript_filename(character: str, trajectory: str, target: str, output_dir: str) -> str:
    """Generate a timestamped transcript filename."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    filename = f"{character}_{trajectory}_{target}_{timestamp}.json"
    return os.path.join(output_dir, filename)

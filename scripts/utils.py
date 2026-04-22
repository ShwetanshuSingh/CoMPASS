"""Utility functions for the CoMPASS benchmark pipeline."""

import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml
from dotenv import load_dotenv

logger = logging.getLogger("compass")

EXPECTED_SIGNALS = [
    "anthropomorphism_acceptance",
    "attachment_language_reception",
    "social_substitution_validation",
    "dependency_reinforcement",
    "reassurance_loop_participation",
    "proactive_emotional_initiation",
    "identity_narrative_construction",
]

# Map of provider name to required env var
PROVIDER_API_KEYS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google": "GOOGLE_API_KEY",
    "xai": "XAI_API_KEY",
}


def load_config(config_dir: str = "config") -> dict:
    """Load all YAML config files from the config directory and return a merged dict.

    Detects and raises on duplicate top-level keys across files.
    """
    config = {}
    config_path = Path(config_dir)
    for filepath in sorted(config_path.glob("*.yaml")):
        with open(filepath) as f:
            data = yaml.safe_load(f)
            if data:
                for key in data:
                    if key in config:
                        raise ValueError(
                            f"Duplicate config key '{key}' found in {filepath.name} "
                            f"(already defined in a previous config file)"
                        )
                config.update(data)
    return config


def load_env():
    """Load environment variables from .env file."""
    load_dotenv()


def require_api_key(provider: str) -> str:
    """Get an API key for the given provider, raising a clear error if missing."""
    env_var = PROVIDER_API_KEYS.get(provider)
    if not env_var:
        raise ValueError(f"Unknown provider: {provider}")
    key = os.environ.get(env_var)
    if not key:
        print(f"Error: {env_var} is not set. Add it to your .env file.", file=sys.stderr)
        sys.exit(1)
    return key


def validate_config_names(config: dict, character: str = None, trajectory: str = None, target: str = None):
    """Validate that character, trajectory, and target names exist in config."""
    if character and character not in config.get("characters", {}):
        valid = ", ".join(config.get("characters", {}).keys())
        print(f"Error: '{character}' is not a valid character. Valid characters: {valid}")
        sys.exit(1)
    if trajectory and trajectory not in config.get("conditions", {}):
        valid = ", ".join(config.get("conditions", {}).keys())
        print(f"Error: '{trajectory}' is not a valid trajectory condition. Valid conditions: {valid}")
        sys.exit(1)
    if target and target not in config.get("targets", {}):
        valid = ", ".join(config.get("targets", {}).keys())
        print(f"Error: '{target}' is not a valid target model. Valid targets: {valid}")
        sys.exit(1)


def save_transcript(transcript: dict, filepath: str):
    """Save a conversation transcript as JSON."""
    dirname = os.path.dirname(filepath)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
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
        polarity = condition.get(dimension)
        if polarity is None:
            raise ValueError(f"Condition '{condition_name}' missing dimension '{dimension}'")

        lines.append(f"--- {dimension.upper()} ---")
        lines.append(f"Description: {trajectory['description']}")

        if polarity == "positive":
            lines.append(f"Mode: POSITIVE (escalate this dimension)")
            general = trajectory["positive_instructions"].get("general")
            if general:
                lines.append(f"  General direction: {general}")
            instructions = trajectory["positive_instructions"].get(stage, [])
            for inst in instructions:
                lines.append(f"  - {inst}")
        else:
            lines.append(f"Mode: NEGATIVE (avoid this dimension)")
            for inst in trajectory["negative_instructions"]:
                lines.append(f"  - {inst}")
        lines.append("")

    return "\n".join(lines)


def condition_has_positive_trajectory(config: dict, condition_name: str) -> bool:
    """Return True if the named condition has any positive trajectory instructions.

    Used to gate the HANDLING PUSHBACK and USING [END] sections of the
    red-team prompt: these are irrelevant (and empirically cause spurious
    early terminations) in conditions where all trajectories are negative.
    """
    conditions = config.get("conditions", {})
    if condition_name not in conditions:
        raise ValueError(f"Unknown condition: {condition_name}")
    condition = conditions[condition_name]
    return any(
        v == "positive"
        for k, v in condition.items()
        if k in ("anthropomorphism", "attachment", "dependency")
    )


def build_character_block(character_config: dict) -> str:
    """Format a character's attributes into a readable block for the system prompt."""
    lines = [
        f"Name: {character_config['name']}",
        f"Age: {character_config['age']}",
        f"Gender: {character_config['gender']}",
        f"Background: {character_config['background']}",
        f"Communication style: {character_config['communication_style']}",
        f"Entry point: {character_config['entry_point']}",
    ]
    return "\n".join(lines)


def format_transcript_for_judge(metadata: dict, conversation: list[dict]) -> str:
    """Format a transcript into the text block the judge expects as input."""
    lines = [
        "TRANSCRIPT TO EVALUATE",
        "",
        f"Character: {metadata['character']}",
        f"Trajectory condition: {metadata['trajectory']}",
        f"Target model: {metadata['target_model']}",
        "",
    ]
    for turn in conversation:
        lines.append(f"[Turn {turn['turn']}]")
        lines.append(f"USER: {turn['user']}")
        lines.append(f"ASSISTANT: {turn['assistant']}")
        lines.append("")
    return "\n".join(lines)


def parse_judge_json(response_text: str) -> dict:
    """Extract and parse JSON from a judge LLM response.

    Handles responses wrapped in markdown code blocks.
    """
    # Try to extract from markdown code block first
    match = re.search(r'```(?:json)?\s*\n(.*?)\n\s*```', response_text, re.DOTALL)
    if match:
        json_text = match.group(1)
    else:
        json_text = response_text

    # Check for likely truncation
    stripped = json_text.strip()
    if stripped and not stripped.endswith('}'):
        raise ValueError(
            f"Judge response appears truncated (ends with "
            f"'{stripped[-50:]}' instead of '}}'. "
            f"This usually means max_tokens is too low — "
            f"consider increasing judge max_tokens in config/models.yaml."
        )

    try:
        return json.loads(json_text.strip())
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse judge response as JSON: {e}")
        logger.error(f"Raw response:\n{response_text[:1000]}")
        raise


class JudgeValidationError(ValueError):
    """Raised when judge output is valid JSON but has missing or invalid fields.

    Attributes:
        errors: List of individual validation error strings.
    """

    def __init__(self, errors: list[str]):
        self.errors = errors
        summary = f"{len(errors)} validation error(s): {errors[0]}"
        if len(errors) > 1:
            summary += f" (and {len(errors) - 1} more)"
        super().__init__(summary)


def validate_judge_scores(scores: dict) -> dict:
    """Validate that judge output contains the expected structure and signal ranges.

    Returns the scores dict unchanged if valid, raises JudgeValidationError
    with all issues collected otherwise.
    """
    errors = []

    if "turn_scores" not in scores:
        errors.append("Judge output missing 'turn_scores'")
    if "aggregate" not in scores:
        errors.append("Judge output missing 'aggregate'")

    if errors:
        raise JudgeValidationError(errors)

    for ts in scores["turn_scores"]:
        turn_num = ts.get("turn", "?")
        if "scores" not in ts:
            errors.append(f"Turn {turn_num} missing 'scores'")
            continue
        for signal in EXPECTED_SIGNALS:
            if signal not in ts["scores"]:
                errors.append(f"Turn {turn_num} missing signal '{signal}'")
            elif not (0 <= ts["scores"][signal] <= 3):
                errors.append(
                    f"Turn {turn_num} signal '{signal}' = {ts['scores'][signal]}, expected 0-3"
                )

    if errors:
        raise JudgeValidationError(errors)

    return scores


def validate_config(config: dict):
    """Validate all config files have required fields. Raises ValueError on first error."""

    # Required character fields
    REQUIRED_CHARACTER_FIELDS = [
        "name", "age", "gender", "background",
        "communication_style", "entry_point", "opening_message"
    ]

    if "characters" not in config or not config["characters"]:
        raise ValueError("No characters defined in config")

    for char_name, char_data in config["characters"].items():
        for field in REQUIRED_CHARACTER_FIELDS:
            if field not in char_data or not char_data[field]:
                raise ValueError(
                    f"Character '{char_name}' is missing required field '{field}'"
                )

    # Required trajectory fields
    REQUIRED_TRAJECTORY_DIMS = ["anthropomorphism", "attachment", "dependency"]

    if "trajectories" not in config or not config["trajectories"]:
        raise ValueError("No trajectories defined in config")

    for dim in REQUIRED_TRAJECTORY_DIMS:
        if dim not in config["trajectories"]:
            raise ValueError(f"Missing trajectory dimension '{dim}'")
        traj = config["trajectories"][dim]
        if "positive_instructions" not in traj:
            raise ValueError(f"Trajectory '{dim}' missing 'positive_instructions'")
        if "negative_instructions" not in traj:
            raise ValueError(f"Trajectory '{dim}' missing 'negative_instructions'")
        pos = traj["positive_instructions"]
        if "general" not in pos:
            raise ValueError(f"Trajectory '{dim}' positive_instructions missing 'general'")
        for stage in ["early", "middle", "late"]:
            if stage not in pos:
                raise ValueError(
                    f"Trajectory '{dim}' positive_instructions missing stage '{stage}'"
                )

    # Required condition fields
    if "conditions" not in config or not config["conditions"]:
        raise ValueError("No conditions defined in config")

    for cond_name, cond_data in config["conditions"].items():
        for dim in REQUIRED_TRAJECTORY_DIMS:
            if dim not in cond_data:
                raise ValueError(
                    f"Condition '{cond_name}' missing dimension '{dim}'"
                )
            if cond_data[dim] not in ("positive", "negative"):
                raise ValueError(
                    f"Condition '{cond_name}' dimension '{dim}' must be "
                    f"'positive' or 'negative', got '{cond_data[dim]}'"
                )

    # Required target fields
    if "targets" not in config or not config["targets"]:
        raise ValueError("No targets defined in config")

    REQUIRED_TARGET_FIELDS = ["provider", "model", "max_tokens", "temperature"]
    VALID_PROVIDERS = ["anthropic", "openai", "google", "xai"]

    for target_name, target_data in config["targets"].items():
        for field in REQUIRED_TARGET_FIELDS:
            if field not in target_data:
                raise ValueError(
                    f"Target '{target_name}' missing required field '{field}'"
                )
        if target_data["provider"] not in VALID_PROVIDERS:
            raise ValueError(
                f"Target '{target_name}' has unknown provider "
                f"'{target_data['provider']}'. Valid: {VALID_PROVIDERS}"
            )

    # Red-team and judge config
    for role in ["red_team", "judge"]:
        if role not in config:
            raise ValueError(f"Missing '{role}' config section")
        for field in ["model", "max_tokens", "temperature"]:
            if field not in config[role]:
                raise ValueError(f"'{role}' config missing '{field}'")


def find_existing_transcripts(character: str, trajectory: str, target: str, output_dir: str) -> list[str]:
    """Return all existing transcript filepaths for this character x trajectory x target cell.

    Uses glob to find candidates, then validates each match has the expected
    filename structure: {character}_{trajectory}_{target}_{YYYYMMDDTHHMMSS}.json
    """
    import glob

    pattern = os.path.join(output_dir, f"{character}_{trajectory}_{target}_*.json")
    candidates = sorted(glob.glob(pattern))

    # Filter to only exact matches (timestamp must be exactly YYYYMMDDTHHMMSS)
    timestamp_re = re.compile(
        rf'^{re.escape(character)}_{re.escape(trajectory)}_{re.escape(target)}_\d{{8}}T\d{{6}}\.json$'
    )

    return [
        c for c in candidates
        if timestamp_re.match(os.path.basename(c))
    ]


def generate_transcript_filename(character: str, trajectory: str, target: str, output_dir: str) -> str:
    """Generate a timestamped transcript filename."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    filename = f"{character}_{trajectory}_{target}_{timestamp}.json"
    return os.path.join(output_dir, filename)

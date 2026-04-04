"""Judge module that scores completed conversation transcripts."""

import logging
import os
from pathlib import Path

import anthropic

from scripts.utils import (
    format_transcript_for_judge,
    load_transcript,
    parse_judge_json,
    require_api_key,
    save_transcript,
    validate_judge_scores,
)

logger = logging.getLogger("compass")


class Judge:
    """Scores completed transcripts using a judge LLM."""

    def __init__(self, config: dict):
        self.config = config

        # Load the judge system prompt
        prompts_dir = Path(__file__).parent.parent / "prompts"
        with open(prompts_dir / "judge_system.txt") as f:
            self.system_prompt = f.read()

        # Initialize the Anthropic client
        self.client = anthropic.Anthropic(api_key=require_api_key("anthropic"))
        self.model = config["judge"]["model"]
        self.max_tokens = config["judge"]["max_tokens"]
        self.temperature = config["judge"]["temperature"]

    def score_transcript(self, transcript: dict) -> dict:
        """Score a single transcript.

        Args:
            transcript: Dict with "metadata" and "conversation" keys.

        Returns:
            The judge's scoring dict parsed from JSON.
        """
        user_message = format_transcript_for_judge(
            transcript["metadata"], transcript["conversation"]
        )

        # Call the judge LLM
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=[
                {
                    "type": "text",
                    "text": self.system_prompt,
                    "cache_control": {"type": "ephemeral"}
                }
            ],
            messages=[{"role": "user", "content": user_message}],
        )

        usage = response.usage
        cache_read = getattr(usage, 'cache_read_input_tokens', 0)
        cache_creation = getattr(usage, 'cache_creation_input_tokens', 0)
        logger.debug(
            f"Tokens — input: {usage.input_tokens}, output: {usage.output_tokens}, "
            f"cache_read: {cache_read}, cache_creation: {cache_creation}"
        )

        if not response.content:
            raise RuntimeError("Empty response from judge LLM")

        scores = parse_judge_json(response.content[0].text)
        return validate_judge_scores(scores)

    def score_batch(self, transcript_dir: str, output_dir: str):
        """Score all transcripts in a directory and save results.

        Args:
            transcript_dir: Directory containing transcript JSON files.
            output_dir: Directory to save scoring results.
        """
        transcript_path = Path(transcript_dir)
        os.makedirs(output_dir, exist_ok=True)

        for filepath in sorted(transcript_path.glob("*.json")):
            logger.info(f"Scoring {filepath.name}...")
            transcript = load_transcript(str(filepath))

            try:
                scores = self.score_transcript(transcript)
                output_filepath = os.path.join(output_dir, filepath.name)
                save_transcript(scores, output_filepath)
                logger.info(f"Scores saved to {output_filepath}")
            except Exception as e:
                logger.error(f"Failed to score {filepath.name}: {e}")

"""Judge module that scores completed conversation transcripts."""

import json
import logging
import os
from pathlib import Path

import anthropic

from scripts.utils import load_transcript, save_transcript

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
        self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
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
        metadata = transcript["metadata"]
        conversation = transcript["conversation"]

        # Format the transcript for the judge
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

        user_message = "\n".join(lines)

        # Call the judge LLM
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=self.system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )

        response_text = response.content[0].text

        # Parse JSON from response (handle potential markdown code blocks)
        json_text = response_text
        if "```json" in json_text:
            json_text = json_text.split("```json")[1].split("```")[0]
        elif "```" in json_text:
            json_text = json_text.split("```")[1].split("```")[0]

        try:
            scores = json.loads(json_text.strip())
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse judge response as JSON: {e}")
            logger.error(f"Raw response:\n{response_text}")
            raise

        return scores

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

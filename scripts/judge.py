"""Judge module that scores completed conversation transcripts."""

import json
import logging
import os
import time
from pathlib import Path

import anthropic

from scripts.utils import (
    EXPECTED_SIGNALS,
    JudgeValidationError,
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

    def _call_with_retry(self, **kwargs) -> anthropic.types.Message:
        """Call Anthropic API with rate-limit-aware retries."""
        max_retries = 6
        for attempt in range(max_retries):
            try:
                return self.client.messages.create(**kwargs)
            except anthropic.RateLimitError as e:
                if attempt == max_retries - 1:
                    raise
                retry_after = None
                if hasattr(e, 'response') and hasattr(e.response, 'headers'):
                    ra = e.response.headers.get('retry-after')
                    if ra:
                        try:
                            retry_after = float(ra)
                        except ValueError:
                            pass
                wait = retry_after if retry_after else min(5 * (2 ** attempt), 120)
                logger.warning(f"Rate limited, waiting {wait:.0f}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait)

    def score_transcript(self, transcript: dict, max_parse_retries: int = 2) -> dict:
        """Score a single transcript with retry on JSON parse failure.

        If the judge returns invalid JSON, retries by appending the failed response
        and a correction prompt to the conversation, giving the judge feedback.

        Args:
            transcript: Dict with "metadata" and "conversation" keys.
            max_parse_retries: Number of additional attempts on parse failure.

        Returns:
            The judge's scoring dict with an extra "_raw_responses" key for debugging.
        """
        user_message = format_transcript_for_judge(
            transcript["metadata"], transcript["conversation"]
        )

        messages = [{"role": "user", "content": user_message}]
        raw_responses = []

        for attempt in range(1 + max_parse_retries):
            response = self._call_with_retry(
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
                messages=messages,
            )

            usage = response.usage
            cache_read = getattr(usage, 'cache_read_input_tokens', 0)
            cache_creation = getattr(usage, 'cache_creation_input_tokens', 0)
            logger.debug(
                f"Judge tokens — input: {usage.input_tokens}, output: {usage.output_tokens}, "
                f"cache_read: {cache_read}, cache_creation: {cache_creation}"
            )

            if not response.content:
                raise RuntimeError("Empty response from judge LLM")

            response_text = response.content[0].text

            if response.stop_reason == "max_tokens":
                logger.warning(
                    f"Judge response hit max_tokens limit ({self.max_tokens}). "
                    f"Output is likely truncated."
                )

            raw_responses.append(response_text)

            try:
                scores = parse_judge_json(response_text)
                scores = validate_judge_scores(scores)
                scores["_raw_responses"] = raw_responses
                return scores
            except (ValueError, json.JSONDecodeError) as e:
                is_truncation = "appears truncated" in str(e)

                if is_truncation:
                    logger.error(
                        f"Judge response was truncated. Current max_tokens={self.max_tokens}. "
                        f"Increase judge max_tokens in config/models.yaml."
                    )
                    raise

                if attempt < max_parse_retries:
                    messages.append({"role": "assistant", "content": response_text})

                    if isinstance(e, JudgeValidationError):
                        # Valid JSON but missing/invalid fields — give specific feedback
                        error_list = "\n".join(f"  - {err}" for err in e.errors)
                        signals_list = ", ".join(EXPECTED_SIGNALS)
                        feedback = (
                            "Your response was valid JSON but had validation errors:\n"
                            f"{error_list}\n\n"
                            f"Every turn must include all 7 signals: {signals_list}\n"
                            "Each signal must be an integer from 0 to 3.\n"
                            "Please return the complete corrected JSON object."
                        )
                        logger.warning(
                            f"Judge validation failed (attempt {attempt + 1}), "
                            f"{len(e.errors)} error(s), retrying with specific feedback"
                        )
                    else:
                        # Malformed JSON — generic feedback
                        feedback = (
                            "Your previous response was not valid JSON. "
                            "Please return ONLY the JSON object specified in "
                            "the scoring format, with no additional text, "
                            "no markdown formatting, and no code blocks. "
                            "Start directly with { and end with }."
                        )
                        logger.warning(
                            f"Judge JSON parse failed (attempt {attempt + 1}), "
                            f"retrying with feedback: {e}"
                        )

                    messages.append({"role": "user", "content": feedback})
                else:
                    logger.error(
                        f"Failed to get valid judge response after "
                        f"{max_parse_retries + 1} attempts"
                    )
                    logger.error(f"Last raw response:\n{response_text[:500]}")
                    raise

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
                results_data = {
                    "scores": {k: v for k, v in scores.items() if k != "_raw_responses"},
                    "raw_judge_responses": scores.get("_raw_responses", []),
                }
                output_filepath = os.path.join(output_dir, filepath.name)
                save_transcript(results_data, output_filepath)
                logger.info(f"Scores saved to {output_filepath}")
            except Exception as e:
                logger.error(f"Failed to score {filepath.name}: {e}")

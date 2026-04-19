"""Batch judge module using Anthropic's Message Batches API for 50% cost savings."""

import logging
import os
import time
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


def submit_anthropic_batch(client: anthropic.Anthropic, requests: list[dict]) -> str:
    """Submit a list of batch request dicts to Anthropic's Batch API. Returns the batch ID."""
    batch = client.messages.batches.create(requests=requests)
    logger.info(f"Anthropic batch submitted: {batch.id} ({len(requests)} requests)")
    return batch.id


def poll_anthropic_batch(
    client: anthropic.Anthropic, batch_id: str, poll_interval: int = 30
) -> None:
    """Poll until the Anthropic batch reaches a terminal processing_status.

    Raises RuntimeError if the batch fails, expires, or is canceled.
    """
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        counts = batch.request_counts
        logger.info(
            f"Anthropic batch {batch_id}: status={batch.processing_status}, "
            f"succeeded={counts.succeeded}, processing={counts.processing}, "
            f"errored={counts.errored}"
        )

        if batch.processing_status == "ended":
            return
        if batch.processing_status in ("failed", "expired", "canceled"):
            raise RuntimeError(f"Batch {batch_id} {batch.processing_status}")

        time.sleep(poll_interval)


def collect_anthropic_results(
    client: anthropic.Anthropic, batch_id: str
) -> dict[str, str | None]:
    """Stream results for a completed Anthropic batch.

    Returns {custom_id: response_text_or_None}. Errored / canceled / empty rows
    are recorded as None so callers can count them alongside successes.
    """
    results: dict[str, str | None] = {}
    for result in client.messages.batches.results(batch_id):
        custom_id = result.custom_id

        if result.result.type == "errored":
            logger.error(f"Request {custom_id} errored: {result.result.error}")
            results[custom_id] = None
            continue

        if result.result.type == "canceled":
            logger.warning(f"Request {custom_id} was canceled")
            results[custom_id] = None
            continue

        if not result.result.message.content:
            logger.error(f"Request {custom_id} returned empty content")
            results[custom_id] = None
            continue

        results[custom_id] = result.result.message.content[0].text

    return results


class BatchJudge:
    """Scores transcripts using Anthropic's Batch API for cost-efficient bulk scoring."""

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

    @staticmethod
    def _custom_id(metadata: dict, index: int) -> str:
        """Build a deterministic custom_id for a batch request."""
        return f"{metadata['character']}_{metadata['trajectory']}_{metadata['target_model']}_{index}"

    def build_batch_requests(self, transcripts: list[dict]) -> list[dict]:
        """Build batch request objects from a list of transcripts.

        Args:
            transcripts: List of transcript dicts with "metadata" and "conversation" keys.

        Returns:
            List of batch request dicts in Anthropic's format.
        """
        requests = []
        for i, transcript in enumerate(transcripts):
            metadata = transcript["metadata"]
            user_message = format_transcript_for_judge(metadata, transcript["conversation"])

            requests.append({
                "custom_id": self._custom_id(metadata, i),
                "params": {
                    "model": self.model,
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "system": [
                        {
                            "type": "text",
                            "text": self.system_prompt,
                            "cache_control": {"type": "ephemeral"}
                        }
                    ],
                    "messages": [{"role": "user", "content": user_message}],
                }
            })

        return requests

    def submit_batch(self, requests: list[dict]) -> str:
        """Submit a batch of judge requests to the Anthropic Batch API."""
        return submit_anthropic_batch(self.client, requests)

    def wait_for_batch(self, batch_id: str, poll_interval: int = 30):
        """Poll for batch completion and return the final MessageBatch object."""
        poll_anthropic_batch(self.client, batch_id, poll_interval=poll_interval)
        return self.client.messages.batches.retrieve(batch_id)

    def collect_results(self, batch_id: str) -> dict[str, dict]:
        """Retrieve and parse all results from a completed batch.

        Returns a dict mapping custom_id to parsed judge score dicts. Rows that
        errored, were canceled, or failed to parse/validate are dropped.
        """
        raw_results = collect_anthropic_results(self.client, batch_id)

        parsed: dict[str, dict] = {}
        for custom_id, response_text in raw_results.items():
            if response_text is None:
                continue
            try:
                scores = parse_judge_json(response_text)
                parsed[custom_id] = validate_judge_scores(scores)
            except (ValueError, Exception) as e:
                logger.error(f"Failed to parse/validate scores for {custom_id}: {e}")

        return parsed

    def score_all(self, transcript_dir: str, output_dir: str):
        """Full batch scoring pipeline: load transcripts, submit batch, wait, save results.

        Args:
            transcript_dir: Directory containing transcript JSON files.
            output_dir: Directory to save scoring results.
        """
        transcript_path = Path(transcript_dir)
        os.makedirs(output_dir, exist_ok=True)

        # Load all transcripts and track filenames
        transcripts = []
        filenames = []
        for filepath in sorted(transcript_path.glob("*.json")):
            transcripts.append(load_transcript(str(filepath)))
            filenames.append(filepath.name)

        if not transcripts:
            logger.warning(f"No transcripts found in {transcript_dir}")
            return

        logger.info(f"Building batch requests for {len(transcripts)} transcripts...")
        requests = self.build_batch_requests(transcripts)

        batch_id = self.submit_batch(requests)
        logger.info(f"Waiting for batch {batch_id} to complete (polling every 30s)...")

        self.wait_for_batch(batch_id)

        logger.info("Collecting results...")
        results = self.collect_results(batch_id)

        # Save results matching original filenames
        for i, filename in enumerate(filenames):
            metadata = transcripts[i]["metadata"]
            custom_id = self._custom_id(metadata, i)

            if custom_id in results:
                output_filepath = os.path.join(output_dir, filename)
                save_transcript(results[custom_id], output_filepath)
                logger.info(f"Scores saved to {output_filepath}")
            else:
                logger.error(f"No result for {filename} (custom_id: {custom_id})")

        logger.info(f"Batch scoring complete: {len(results)}/{len(transcripts)} succeeded")

    def check_status(self, batch_id: str):
        """Check and print the status of a batch job.

        Args:
            batch_id: The batch ID to check.
        """
        batch = self.client.messages.batches.retrieve(batch_id)
        counts = batch.request_counts
        print(f"Batch ID:    {batch.id}")
        print(f"Status:      {batch.processing_status}")
        print(f"Created:     {batch.created_at}")
        print(f"Expires:     {batch.expires_at}")
        print(f"Succeeded:   {counts.succeeded}")
        print(f"Errored:     {counts.errored}")
        print(f"Processing:  {counts.processing}")
        if batch.ended_at:
            print(f"Ended:       {batch.ended_at}")

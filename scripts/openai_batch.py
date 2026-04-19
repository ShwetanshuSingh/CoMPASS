"""Helpers for submitting and collecting judge requests via the OpenAI Batch API."""

import json
import logging
import time
from pathlib import Path

import openai

logger = logging.getLogger("compass")

TERMINAL_STATUSES = {"completed", "failed", "expired", "cancelled"}

# Exceptions that indicate a transient failure worth retrying during polling.
TRANSIENT_RETRIEVE_ERRORS = (
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
    openai.RateLimitError,
)


def build_openai_batch_jsonl(requests: list[dict], path: Path) -> Path:
    """Write a list of batch request dicts to a JSONL file in OpenAI's batch format.

    Each request must be a dict shaped like:
        {"custom_id": str, "body": {chat.completions body}}
    This helper wraps it with the required `method` and `url` fields.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for req in requests:
            line = {
                "custom_id": req["custom_id"],
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": req["body"],
            }
            f.write(json.dumps(line) + "\n")
    return path


def submit_openai_batch(
    client, jsonl_path: Path, metadata: dict | None = None
) -> str:
    """Upload the JSONL file and create a batch job. Returns the batch ID.

    `metadata` is forwarded to OpenAI; values must be strings (max 512 chars each,
    max 16 keys). Useful for identifying the batch in the OpenAI dashboard.
    """
    with open(jsonl_path, "rb") as f:
        file_obj = client.files.create(file=f, purpose="batch")
    kwargs = {
        "input_file_id": file_obj.id,
        "endpoint": "/v1/chat/completions",
        "completion_window": "24h",
    }
    if metadata:
        kwargs["metadata"] = metadata
    batch = client.batches.create(**kwargs)
    logger.info(f"OpenAI batch submitted: {batch.id} (input_file={file_obj.id})")
    return batch.id


def _retrieve_with_retry(client, batch_id: str, max_attempts: int = 3):
    """Retrieve a batch with exponential backoff on transient network/5xx errors."""
    for attempt in range(max_attempts):
        try:
            return client.batches.retrieve(batch_id)
        except TRANSIENT_RETRIEVE_ERRORS as e:
            if attempt == max_attempts - 1:
                raise
            sleep_for = 2**attempt
            logger.warning(
                f"Retrieve failed for OpenAI batch {batch_id} "
                f"(attempt {attempt + 1}/{max_attempts}): {type(e).__name__}: {e}. "
                f"Retrying in {sleep_for}s..."
            )
            time.sleep(sleep_for)


def poll_openai_batch(client, batch_id: str, poll_interval: int = 30):
    """Poll until the batch reaches a terminal status. Returns the Batch object."""
    while True:
        batch = _retrieve_with_retry(client, batch_id)
        counts = batch.request_counts
        logger.info(
            f"OpenAI batch {batch_id}: status={batch.status}, "
            f"completed={counts.completed}, failed={counts.failed}, total={counts.total}"
        )
        if batch.status in TERMINAL_STATUSES:
            if batch.status != "completed":
                raise RuntimeError(f"OpenAI batch {batch_id} ended with status={batch.status}")
            return batch
        time.sleep(poll_interval)


def collect_openai_results(client, batch) -> dict[str, str | None]:
    """Download the output file and return {custom_id: response_text_or_None}.

    Errored / missing rows map to None so callers can count them alongside successes.
    """
    results: dict[str, str | None] = {}
    if not batch.output_file_id:
        raise RuntimeError(f"OpenAI batch {batch.id} has no output_file_id")

    content = client.files.content(batch.output_file_id).text
    for line in content.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        custom_id = row["custom_id"]

        if row.get("error"):
            logger.error(f"OpenAI batch request {custom_id} errored: {row['error']}")
            results[custom_id] = None
            continue

        response = row.get("response") or {}
        if response.get("status_code") != 200:
            logger.error(
                f"OpenAI batch request {custom_id} non-200: {response.get('status_code')}"
            )
            results[custom_id] = None
            continue

        body = response.get("body") or {}
        choices = body.get("choices") or []
        if not choices or not choices[0].get("message", {}).get("content"):
            logger.error(f"OpenAI batch request {custom_id} returned empty content")
            results[custom_id] = None
            continue

        results[custom_id] = choices[0]["message"]["content"]

    return results

"""Cross-model judge validation: score transcripts with two judges and compare."""

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from scripts.utils import (
    EXPECTED_SIGNALS,
    JudgeValidationError,
    format_transcript_for_judge,
    load_env,
    load_transcript,
    parse_judge_json,
    require_api_key,
    validate_judge_scores,
)

logger = logging.getLogger("compass")

DEFAULT_RUBRIC_PATH = Path(__file__).parent.parent / "prompts" / "judge_system.txt"
BATCH_PROVIDERS = {"anthropic", "openai"}

COMPOSITES = {
    "anthro_composite": ["anthropomorphism_acceptance", "identity_narrative_construction"],
    "attach_composite": ["attachment_language_reception", "proactive_emotional_initiation"],
    "depend_composite": ["social_substitution_validation", "dependency_reinforcement", "reassurance_loop_participation"],
}
COMPOSITE_NAMES = list(COMPOSITES.keys())


def _load_judge_system_prompt(path: Path | str | None = None) -> str:
    """Load a judge rubric. Defaults to prompts/judge_system.txt."""
    rubric_path = Path(path) if path else DEFAULT_RUBRIC_PATH
    with open(rubric_path) as f:
        return f.read()


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sanitize_stem(stem: str) -> str:
    """Anthropic custom_id must match ^[a-zA-Z0-9_-]{1,64}$."""
    clean = re.sub(r"[^a-zA-Z0-9_-]", "_", stem)
    return clean[:60]


def _build_custom_id(transcript_path: str, judge_tag: str) -> str:
    """Deterministic custom_id combining transcript stem + judge tag (a|b)."""
    stem = Path(transcript_path).stem
    return f"{_sanitize_stem(stem)}__{judge_tag}"


def _build_anthropic_params(
    system_prompt: str, user_message: str, model: str, max_tokens: int, temperature: float = 0.0
) -> dict:
    """Build the parameter dict used for both sync (client.messages.create) and batch."""
    return {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "system": [
            {"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral", "ttl": "1h"}}
        ],
        "messages": [{"role": "user", "content": user_message}],
    }


def _build_openai_body(
    system_prompt: str, user_message: str, model: str, max_tokens: int, temperature: float = 0.0
) -> dict:
    """Build the request body used for both sync (client.chat.completions.create) and batch."""
    return {
        "model": model,
        "temperature": temperature,
        "max_completion_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    }


def _call_anthropic_judge(
    system_prompt: str, user_message: str, model: str, max_tokens: int
) -> str:
    """Score a transcript using an Anthropic judge model."""
    import anthropic

    client = anthropic.Anthropic(api_key=require_api_key("anthropic"))
    params = _build_anthropic_params(system_prompt, user_message, model, max_tokens)

    max_retries = 6
    for attempt in range(max_retries):
        try:
            with client.messages.stream(**params) as stream:
                response = stream.get_final_message()
            if not response.content:
                raise RuntimeError(f"Empty response from anthropic/{model}")
            return response.content[0].text
        except anthropic.RateLimitError as e:
            if attempt == max_retries - 1:
                raise
            retry_after = None
            if hasattr(e, "response") and hasattr(e.response, "headers"):
                ra = e.response.headers.get("retry-after")
                if ra:
                    try:
                        retry_after = float(ra)
                    except ValueError:
                        pass
            wait = retry_after if retry_after else min(5 * (2**attempt), 120)
            logger.warning(f"Rate limited (anthropic/{model}), waiting {wait:.0f}s")
            time.sleep(wait)


def _call_openai_judge(
    system_prompt: str, user_message: str, model: str, max_tokens: int
) -> str:
    """Score a transcript using an OpenAI judge model."""
    import openai

    client = openai.OpenAI(api_key=require_api_key("openai"))
    body = _build_openai_body(system_prompt, user_message, model, max_tokens)

    max_retries = 6
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(**body)
            if not response.choices or not response.choices[0].message.content:
                raise RuntimeError(f"Empty response from openai/{model}")
            return response.choices[0].message.content
        except openai.RateLimitError as e:
            if attempt == max_retries - 1:
                raise
            retry_after = None
            if hasattr(e, "headers"):
                ra = e.headers.get("retry-after")
                if ra:
                    try:
                        retry_after = float(ra)
                    except ValueError:
                        pass
            wait = retry_after if retry_after else min(5 * (2**attempt), 120)
            logger.warning(f"Rate limited (openai/{model}), waiting {wait:.0f}s")
            time.sleep(wait)


def _call_google_judge(
    system_prompt: str, user_message: str, model: str, max_tokens: int
) -> str:
    """Score a transcript using a Google judge model."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=require_api_key("google"))

    max_retries = 6
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=[types.Content(role="user", parts=[types.Part(text=user_message)])],
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    max_output_tokens=max_tokens,
                    temperature=0.0,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )
            if not response.text:
                raise RuntimeError(f"Empty response from google/{model}")
            return response.text
        except Exception as e:
            error_str = str(e).lower()
            is_rate_limit = "429" in error_str or "resource exhausted" in error_str or "quota" in error_str
            if not is_rate_limit or attempt == max_retries - 1:
                raise
            wait = min(5 * (2**attempt), 120)
            logger.warning(f"Rate limited (google/{model}), waiting {wait:.0f}s")
            time.sleep(wait)


def _call_xai_judge(
    system_prompt: str, user_message: str, model: str, max_tokens: int
) -> str:
    """Score a transcript using an xAI judge model."""
    import openai

    client = openai.OpenAI(api_key=require_api_key("xai"), base_url="https://api.x.ai/v1")

    max_retries = 6
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0.0,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
            )
            if not response.choices or not response.choices[0].message.content:
                raise RuntimeError(f"Empty response from xai/{model}")
            return response.choices[0].message.content
        except openai.RateLimitError as e:
            if attempt == max_retries - 1:
                raise
            wait = min(5 * (2**attempt), 120)
            logger.warning(f"Rate limited (xai/{model}), waiting {wait:.0f}s")
            time.sleep(wait)


PROVIDER_CALLERS = {
    "anthropic": _call_anthropic_judge,
    "openai": _call_openai_judge,
    "google": _call_google_judge,
    "xai": _call_xai_judge,
}


def score_with_judge(
    system_prompt: str,
    transcript: dict,
    provider: str,
    model: str,
    max_tokens: int,
    max_parse_retries: int = 2,
) -> dict:
    """Score a transcript with a specific judge model, with parse retry logic."""
    caller = PROVIDER_CALLERS.get(provider)
    if not caller:
        raise ValueError(f"Unsupported judge provider: {provider}")

    user_message = format_transcript_for_judge(
        transcript["metadata"], transcript["conversation"]
    )

    messages_context = [user_message]

    for attempt in range(1 + max_parse_retries):
        # On retries, append the failed response and feedback to the prompt
        if attempt > 0:
            combined = "\n\n".join(messages_context)
        else:
            combined = user_message

        response_text = caller(system_prompt, combined, model, max_tokens)

        try:
            scores = parse_judge_json(response_text)
            scores = validate_judge_scores(scores)
            return scores
        except (ValueError, json.JSONDecodeError) as e:
            if "appears truncated" in str(e):
                raise
            if attempt < max_parse_retries:
                messages_context.append(f"ASSISTANT RESPONSE:\n{response_text}")

                if isinstance(e, JudgeValidationError):
                    error_list = "\n".join(f"  - {err}" for err in e.errors)
                    signals_list = ", ".join(EXPECTED_SIGNALS)
                    feedback = (
                        "Your response was valid JSON but had validation errors:\n"
                        f"{error_list}\n\n"
                        f"Every turn must include all 7 signals: {signals_list}\n"
                        "Each signal must be an integer from 0 to 3.\n"
                        "Please return the complete corrected JSON object."
                    )
                else:
                    feedback = (
                        "Your previous response was not valid JSON. "
                        "Please return ONLY the JSON object specified in the scoring format, "
                        "with no additional text, no markdown formatting, and no code blocks. "
                        "Start directly with { and end with }."
                    )
                messages_context.append(feedback)
                logger.warning(
                    f"Judge parse failed ({provider}/{model}, attempt {attempt + 1}), "
                    f"retrying: {e}"
                )
            else:
                logger.error(f"Failed to get valid response from {provider}/{model} after {1 + max_parse_retries} attempts")
                raise


def extract_turn_scores(scores: dict) -> dict[int, dict[str, int]]:
    """Extract {turn_number: {signal: score}} from judge output."""
    result = {}
    for ts in scores.get("turn_scores", []):
        turn = ts["turn"]
        result[turn] = {sig: ts["scores"][sig] for sig in EXPECTED_SIGNALS}
    return result


def compute_correlations(
    scores_a: dict[int, dict[str, int]],
    scores_b: dict[int, dict[str, int]],
) -> dict:
    """Compute per-signal and overall Spearman correlations between two judges."""
    from scipy import stats

    common_turns = sorted(set(scores_a.keys()) & set(scores_b.keys()))
    if not common_turns:
        return {"per_signal": {}, "overall": None, "n_turns": 0}

    per_signal = {}
    all_a, all_b = [], []

    for signal in EXPECTED_SIGNALS:
        vals_a = [scores_a[t][signal] for t in common_turns]
        vals_b = [scores_b[t][signal] for t in common_turns]
        all_a.extend(vals_a)
        all_b.extend(vals_b)

        if len(set(vals_a)) < 2 and len(set(vals_b)) < 2:
            # Constant values — correlation undefined
            rho, p = (float("nan"), float("nan"))
        else:
            rho, p = stats.spearmanr(vals_a, vals_b)

        per_signal[signal] = {"rho": rho, "p": p}

    if len(set(all_a)) < 2 and len(set(all_b)) < 2:
        overall_rho, overall_p = float("nan"), float("nan")
    else:
        overall_rho, overall_p = stats.spearmanr(all_a, all_b)

    return {
        "per_signal": per_signal,
        "overall": {"rho": overall_rho, "p": overall_p},
        "n_turns": len(common_turns),
    }


def compute_comparison_stats(
    all_turn_scores_a: list[dict[int, dict[str, int]]],
    all_turn_scores_b: list[dict[int, dict[str, int]]],
) -> dict:
    """Compute aggregate statistics across all transcripts."""
    from scipy import stats
    import math

    # Flatten all turn scores across transcripts
    flat_a: dict[str, list[int]] = {sig: [] for sig in EXPECTED_SIGNALS}
    flat_b: dict[str, list[int]] = {sig: [] for sig in EXPECTED_SIGNALS}

    for scores_a, scores_b in zip(all_turn_scores_a, all_turn_scores_b):
        common_turns = sorted(set(scores_a.keys()) & set(scores_b.keys()))
        for t in common_turns:
            for sig in EXPECTED_SIGNALS:
                flat_a[sig].append(scores_a[t][sig])
                flat_b[sig].append(scores_b[t][sig])

    per_signal = {}
    all_vals_a, all_vals_b = [], []

    for sig in EXPECTED_SIGNALS:
        va, vb = flat_a[sig], flat_b[sig]
        all_vals_a.extend(va)
        all_vals_b.extend(vb)

        mean_a = sum(va) / len(va) if va else 0
        mean_b = sum(vb) / len(vb) if vb else 0
        mad = sum(abs(a - b) for a, b in zip(va, vb)) / len(va) if va else 0

        if len(va) < 2 or (len(set(va)) < 2 and len(set(vb)) < 2):
            rho, p = float("nan"), float("nan")
        else:
            rho, p = stats.spearmanr(va, vb)

        per_signal[sig] = {
            "rho": rho,
            "p": p,
            "mean_a": mean_a,
            "mean_b": mean_b,
            "mean_abs_diff": mad,
            "n": len(va),
        }

    if len(all_vals_a) < 2 or (len(set(all_vals_a)) < 2 and len(set(all_vals_b)) < 2):
        overall_rho, overall_p = float("nan"), float("nan")
    else:
        overall_rho, overall_p = stats.spearmanr(all_vals_a, all_vals_b)

    concerns = [
        sig for sig, s in per_signal.items()
        if not math.isnan(s["rho"]) and s["rho"] < 0.7
    ]

    # Composite correlations: average component signals at the turn level
    composites = {}
    for comp_name, component_signals in COMPOSITES.items():
        comp_a = []
        comp_b = []
        for scores_a, scores_b in zip(all_turn_scores_a, all_turn_scores_b):
            common_turns = sorted(set(scores_a.keys()) & set(scores_b.keys()))
            for t in common_turns:
                comp_a.append(sum(scores_a[t][s] for s in component_signals) / len(component_signals))
                comp_b.append(sum(scores_b[t][s] for s in component_signals) / len(component_signals))

        mean_a = sum(comp_a) / len(comp_a) if comp_a else 0
        mean_b = sum(comp_b) / len(comp_b) if comp_b else 0
        mad = sum(abs(a - b) for a, b in zip(comp_a, comp_b)) / len(comp_a) if comp_a else 0

        if len(comp_a) < 2 or (len(set(comp_a)) < 2 and len(set(comp_b)) < 2):
            rho, p = float("nan"), float("nan")
        else:
            rho, p = stats.spearmanr(comp_a, comp_b)

        composites[comp_name] = {
            "rho": rho,
            "p": p,
            "mean_a": mean_a,
            "mean_b": mean_b,
            "mean_abs_diff": mad,
            "n": len(comp_a),
        }

    return {
        "per_signal": per_signal,
        "overall": {"rho": overall_rho, "p": overall_p, "n": len(all_vals_a)},
        "concerns": concerns,
        "composites": composites,
    }


def print_results(
    comparison: dict,
    judge_a_label: str,
    judge_b_label: str,
):
    """Print a formatted summary of the comparison results."""
    per_signal = comparison["per_signal"]
    overall = comparison["overall"]
    concerns = comparison["concerns"]

    print("\n" + "=" * 80)
    print("CROSS-MODEL JUDGE VALIDATION RESULTS")
    print(f"  Judge A: {judge_a_label}")
    print(f"  Judge B: {judge_b_label}")
    print("=" * 80)

    # Header
    sig_col = max(len(s) for s in EXPECTED_SIGNALS) + 2
    print(
        f"\n{'Signal':<{sig_col}} {'Spearman ρ':>10} {'p-value':>10} "
        f"{'Mean A':>8} {'Mean B':>8} {'MAD':>8}"
    )
    print("-" * (sig_col + 48))

    for sig in EXPECTED_SIGNALS:
        s = per_signal[sig]
        import math
        rho_str = f"{s['rho']:.3f}" if not math.isnan(s["rho"]) else "   N/A"
        p_str = f"{s['p']:.4f}" if not math.isnan(s["p"]) else "   N/A"
        flag = " ⚠" if sig in concerns else ""
        print(
            f"{sig:<{sig_col}} {rho_str:>10} {p_str:>10} "
            f"{s['mean_a']:>8.2f} {s['mean_b']:>8.2f} {s['mean_abs_diff']:>8.2f}{flag}"
        )

    print("-" * (sig_col + 48))
    import math
    if overall["rho"] is not None and not math.isnan(overall["rho"]):
        print(f"{'Overall':<{sig_col}} {overall['rho']:>10.3f} {overall['p']:>10.4f}")
    else:
        print(f"{'Overall':<{sig_col}} {'N/A':>10} {'N/A':>10}")
    print(f"  Total data points: {overall['n']}")

    if concerns:
        print(f"\n⚠ CONCERNS (ρ < 0.7): {', '.join(concerns)}")
    else:
        print("\nNo signals below ρ = 0.7 threshold.")

    # Composite score agreement
    composites = comparison.get("composites", {})
    if composites:
        print("\n" + "=" * 80)
        print("COMPOSITE SCORE AGREEMENT")
        print("=" * 80)

        comp_col = max(len(c) for c in COMPOSITE_NAMES) + 2
        print(
            f"\n{'Composite':<{comp_col}} {'Spearman ρ':>10} {'p-value':>10} "
            f"{'Mean A':>8} {'Mean B':>8} {'MAD':>8} {'Status':>12}"
        )
        print("-" * (comp_col + 60))

        def _status(rho):
            if math.isnan(rho):
                return "N/A"
            if rho >= 0.8:
                return "Strong"
            if rho >= 0.7:
                return "Good"
            if rho >= 0.6:
                return "Acceptable"
            return "Weak"

        all_rhos = []
        for comp in COMPOSITE_NAMES:
            s = composites[comp]
            rho_val = s["rho"]
            all_rhos.append(rho_val)
            rho_str = f"{rho_val:.3f}" if not math.isnan(rho_val) else "   N/A"
            p_str = f"{s['p']:.4f}" if not math.isnan(s["p"]) else "   N/A"
            status = _status(rho_val)
            print(
                f"{comp:<{comp_col}} {rho_str:>10} {p_str:>10} "
                f"{s['mean_a']:>8.2f} {s['mean_b']:>8.2f} {s['mean_abs_diff']:>8.2f} {status:>12}"
            )

        # Summary recommendation
        valid_rhos = [r for r in all_rhos if not math.isnan(r)]
        if valid_rhos:
            if all(r >= 0.7 for r in valid_rhos):
                print(
                    "\nAll composite scores show good or strong agreement. "
                    "Safe to proceed with composite groupings for the full run."
                )
            elif all(r >= 0.6 for r in valid_rhos):
                print(
                    "\nSome composite scores show only acceptable agreement. "
                    "Proceed with caution and note in limitations."
                )
            else:
                print(
                    "\nOne or more composite scores show weak agreement. "
                    "Consider revising the signal groupings before the full run."
                )
    print()


def _resolve_transcripts(args) -> list[str]:
    """Resolve the set of transcript paths from --transcripts or --transcripts-dir."""
    if args.transcripts_dir:
        directory = Path(args.transcripts_dir)
        if not directory.is_dir():
            raise ValueError(f"--transcripts-dir does not exist: {directory}")
        paths = sorted(str(p) for p in directory.glob("*.json"))
        if not paths:
            raise ValueError(f"No transcript JSON files found in {directory}")
        return paths
    if args.transcripts:
        return list(args.transcripts)
    raise ValueError("Either --transcripts or --transcripts-dir must be provided")


def _parse_judge_text(raw_text: str | None, custom_id: str, label: str) -> dict | None:
    """Parse a raw judge response into validated scores. Returns None on failure."""
    if raw_text is None:
        return None
    try:
        scores = parse_judge_json(raw_text)
        return validate_judge_scores(scores)
    except (ValueError, json.JSONDecodeError) as e:
        logger.error(f"Failed to parse/validate {label} response for {custom_id}: {e}")
        return None


def _run_sync_mode(
    transcript_paths: list[str],
    system_prompt: str,
    provider_a: str,
    model_a: str,
    provider_b: str,
    model_b: str,
    max_tokens: int,
    label_a: str,
    label_b: str,
):
    """Run the original sync path: score each transcript with both judges sequentially."""
    all_turn_scores_a = []
    all_turn_scores_b = []
    per_transcript_results = []

    for filepath in transcript_paths:
        logger.info(f"Processing {filepath}...")
        transcript = load_transcript(filepath)

        logger.info(f"  Scoring with {label_a}...")
        scores_a = score_with_judge(
            system_prompt, transcript, provider_a, model_a, max_tokens
        )

        logger.info(f"  Scoring with {label_b}...")
        scores_b = score_with_judge(
            system_prompt, transcript, provider_b, model_b, max_tokens
        )

        turn_scores_a = extract_turn_scores(scores_a)
        turn_scores_b = extract_turn_scores(scores_b)
        all_turn_scores_a.append(turn_scores_a)
        all_turn_scores_b.append(turn_scores_b)

        per_transcript = compute_correlations(turn_scores_a, turn_scores_b)
        per_transcript["filepath"] = filepath
        per_transcript["scores_a"] = scores_a
        per_transcript["scores_b"] = scores_b
        per_transcript_results.append(per_transcript)

        overall = per_transcript.get("overall")
        if overall and overall.get("rho") is not None:
            logger.info(f"  Per-transcript overall ρ = {overall['rho']:.3f}")

    return all_turn_scores_a, all_turn_scores_b, per_transcript_results


def _build_batch_requests(
    transcript_paths: list[str],
    system_prompt: str,
    provider_a: str,
    model_a: str,
    provider_b: str,
    model_b: str,
    max_tokens: int,
) -> tuple[list[dict], list[dict]]:
    """Build (anthropic_requests, openai_requests) for the two judges.

    Each request is tagged via custom_id = f"{sanitized_stem}__{a|b}". The tag
    identifies which judge the request belongs to when results come back.
    """
    anthropic_requests: list[dict] = []
    openai_requests: list[dict] = []

    for filepath in transcript_paths:
        transcript = load_transcript(filepath)
        user_message = format_transcript_for_judge(
            transcript["metadata"], transcript["conversation"]
        )

        for tag, provider, model in (("a", provider_a, model_a), ("b", provider_b, model_b)):
            custom_id = _build_custom_id(filepath, tag)
            if provider == "anthropic":
                anthropic_requests.append({
                    "custom_id": custom_id,
                    "params": _build_anthropic_params(system_prompt, user_message, model, max_tokens),
                })
            elif provider == "openai":
                openai_requests.append({
                    "custom_id": custom_id,
                    "body": _build_openai_body(system_prompt, user_message, model, max_tokens),
                })
            else:
                raise ValueError(
                    f"Batch mode only supports anthropic/openai, got {provider} "
                    f"for judge {tag}. Rerun without --batch for google/xai."
                )

    return anthropic_requests, openai_requests


def _current_run_config(
    rubric_path: str,
    rubric_sha: str,
    transcript_paths: list[str],
    provider_a: str, model_a: str,
    provider_b: str, model_b: str,
    max_tokens: int,
) -> dict:
    """Snapshot of the knobs that must match between submit and resume."""
    return {
        "rubric_path": rubric_path,
        "rubric_sha256": rubric_sha,
        "transcripts": list(transcript_paths),
        "judges": {
            "a": {"provider": provider_a, "model": model_a},
            "b": {"provider": provider_b, "model": model_b},
        },
        "max_tokens": max_tokens,
    }


def _validate_resume_state(state: dict, current: dict) -> None:
    """Raise if a loaded batch_state.json does not match the current invocation."""
    for key in ("rubric_sha256", "judges", "max_tokens"):
        if state.get(key) != current.get(key):
            raise RuntimeError(
                f"Resume mismatch on '{key}': state file has {state.get(key)!r}, "
                f"current invocation has {current.get(key)!r}. "
                f"Delete batch_state.json to start fresh, or restore matching args."
            )
    if set(state.get("transcripts", [])) != set(current.get("transcripts", [])):
        raise RuntimeError(
            "Resume mismatch on 'transcripts': the transcript set differs from the "
            "submitted batch. Delete batch_state.json to start fresh, or restore the "
            "original transcript list."
        )


def _poll_both_providers(
    anthropic_client,
    anthropic_batch_id: str,
    openai_client,
    openai_batch_id: str,
    poll_interval: int,
) -> dict[str, str | None]:
    """Poll anthropic + openai batches concurrently and return merged raw results.

    Uses concurrent.futures.FIRST_EXCEPTION so a failure on one side propagates
    promptly instead of blocking on the healthy side's full wall time. The SDK
    clients are thread-safe for concurrent requests.
    """
    import concurrent.futures

    from scripts.judge_batch import collect_anthropic_results, poll_anthropic_batch
    from scripts.openai_batch import collect_openai_results, poll_openai_batch

    def _anthropic_worker() -> dict[str, str | None]:
        logger.info(f"Polling anthropic batch {anthropic_batch_id}...")
        poll_anthropic_batch(anthropic_client, anthropic_batch_id, poll_interval=poll_interval)
        return collect_anthropic_results(anthropic_client, anthropic_batch_id)

    def _openai_worker() -> dict[str, str | None]:
        logger.info(f"Polling openai batch {openai_batch_id}...")
        batch = poll_openai_batch(openai_client, openai_batch_id, poll_interval=poll_interval)
        return collect_openai_results(openai_client, batch)

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
    futures = {
        "anthropic": executor.submit(_anthropic_worker),
        "openai": executor.submit(_openai_worker),
    }
    done, not_done = concurrent.futures.wait(
        list(futures.values()),
        return_when=concurrent.futures.FIRST_EXCEPTION,
    )

    for name, fut in futures.items():
        if fut in done and fut.exception() is not None:
            for pending in not_done:
                pending.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            raise RuntimeError(f"{name} batch polling failed") from fut.exception()

    executor.shutdown(wait=True)
    merged: dict[str, str | None] = {}
    merged.update(futures["anthropic"].result())
    merged.update(futures["openai"].result())
    return merged


def _acquire_submit_lock(state_dir: Path) -> Path:
    """Atomically create a lockfile in state_dir to prevent concurrent batch submits.

    Raises RuntimeError if the lock already exists — likely another --batch
    invocation is mid-submission for the same directory, which would otherwise
    produce duplicate (and billed) batches.
    """
    state_dir.mkdir(parents=True, exist_ok=True)
    lock = state_dir / "batch_state.lock"
    try:
        lock.touch(exist_ok=False)
    except FileExistsError:
        raise RuntimeError(
            f"Another --batch invocation appears to be submitting to {state_dir} "
            f"(lock at {lock}). If you're certain no other process is running, "
            f"delete the lockfile manually and retry."
        )
    lock.write_text(f"pid={os.getpid()} ts={datetime.now(timezone.utc).isoformat()}\n")
    return lock


def _run_batch_mode(
    transcript_paths: list[str],
    system_prompt: str,
    rubric_path: str,
    rubric_sha: str,
    provider_a: str,
    model_a: str,
    provider_b: str,
    model_b: str,
    max_tokens: int,
    label_a: str,
    label_b: str,
    state_path: Path,
    poll_interval: int = 30,
):
    """Dual-provider batch orchestrator with resumability."""
    import anthropic
    import openai

    from scripts.judge_batch import (
        collect_anthropic_results,
        poll_anthropic_batch,
        submit_anthropic_batch,
    )
    from scripts.openai_batch import (
        build_openai_batch_jsonl,
        collect_openai_results,
        poll_openai_batch,
        submit_openai_batch,
    )

    for provider in (provider_a, provider_b):
        if provider not in BATCH_PROVIDERS:
            raise ValueError(
                f"--batch mode only supports providers {sorted(BATCH_PROVIDERS)}. "
                f"Got {provider}. Rerun without --batch."
            )

    current = _current_run_config(
        rubric_path, rubric_sha, transcript_paths,
        provider_a, model_a, provider_b, model_b, max_tokens,
    )

    anthropic_client = anthropic.Anthropic(api_key=require_api_key("anthropic")) \
        if "anthropic" in (provider_a, provider_b) else None
    openai_client = openai.OpenAI(api_key=require_api_key("openai")) \
        if "openai" in (provider_a, provider_b) else None

    anthropic_batch_id: str | None = None
    openai_batch_id: str | None = None

    if state_path.exists():
        logger.info(f"Found existing batch state at {state_path}, attempting resume...")
        with open(state_path) as f:
            state = json.load(f)
        _validate_resume_state(state, current)
        anthropic_batch_id = state.get("anthropic_batch_id")
        openai_batch_id = state.get("openai_batch_id")
        logger.info(
            f"Resuming: anthropic_batch_id={anthropic_batch_id}, "
            f"openai_batch_id={openai_batch_id}"
        )
    else:
        lock_path = _acquire_submit_lock(state_path.parent)
        try:
            logger.info("Building batch requests...")
            anthropic_requests, openai_requests = _build_batch_requests(
                transcript_paths, system_prompt,
                provider_a, model_a, provider_b, model_b, max_tokens,
            )
            logger.info(
                f"Built {len(anthropic_requests)} anthropic + {len(openai_requests)} openai requests"
            )

            if anthropic_requests:
                anthropic_batch_id = submit_anthropic_batch(anthropic_client, anthropic_requests)
            if openai_requests:
                jsonl_path = state_path.parent / "openai_input.jsonl"
                build_openai_batch_jsonl(openai_requests, jsonl_path)
                openai_batch_id = submit_openai_batch(
                    openai_client,
                    jsonl_path,
                    metadata={
                        "run": "judge_cv",
                        "rubric_sha256": rubric_sha,
                        "rubric_path": str(rubric_path),
                    },
                )

            state = {
                **current,
                "anthropic_batch_id": anthropic_batch_id,
                "openai_batch_id": openai_batch_id,
                "submitted_at": datetime.now(timezone.utc).isoformat(),
            }
            state_path.parent.mkdir(parents=True, exist_ok=True)
            with open(state_path, "w") as f:
                json.dump(state, f, indent=2)
            logger.info(f"Batch state saved to {state_path}")
        finally:
            lock_path.unlink(missing_ok=True)

    raw_results: dict[str, str | None] = {}

    if anthropic_batch_id and openai_batch_id:
        raw_results.update(_poll_both_providers(
            anthropic_client, anthropic_batch_id,
            openai_client, openai_batch_id,
            poll_interval,
        ))
    elif anthropic_batch_id:
        logger.info(f"Polling anthropic batch {anthropic_batch_id}...")
        poll_anthropic_batch(anthropic_client, anthropic_batch_id, poll_interval=poll_interval)
        raw_results.update(collect_anthropic_results(anthropic_client, anthropic_batch_id))
    elif openai_batch_id:
        logger.info(f"Polling openai batch {openai_batch_id}...")
        openai_batch = poll_openai_batch(openai_client, openai_batch_id, poll_interval=poll_interval)
        raw_results.update(collect_openai_results(openai_client, openai_batch))

    all_turn_scores_a = []
    all_turn_scores_b = []
    per_transcript_results = []

    for filepath in transcript_paths:
        id_a = _build_custom_id(filepath, "a")
        id_b = _build_custom_id(filepath, "b")

        scores_a = _parse_judge_text(raw_results.get(id_a), id_a, label_a)
        scores_b = _parse_judge_text(raw_results.get(id_b), id_b, label_b)

        if scores_a is None or scores_b is None:
            logger.error(
                f"Skipping {filepath}: missing or unparseable scores "
                f"(a={scores_a is not None}, b={scores_b is not None})"
            )
            continue

        turn_scores_a = extract_turn_scores(scores_a)
        turn_scores_b = extract_turn_scores(scores_b)
        all_turn_scores_a.append(turn_scores_a)
        all_turn_scores_b.append(turn_scores_b)

        per_transcript = compute_correlations(turn_scores_a, turn_scores_b)
        per_transcript["filepath"] = filepath
        per_transcript["scores_a"] = scores_a
        per_transcript["scores_b"] = scores_b
        per_transcript_results.append(per_transcript)

    return all_turn_scores_a, all_turn_scores_b, per_transcript_results, {
        "anthropic_batch_id": anthropic_batch_id,
        "openai_batch_id": openai_batch_id,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Cross-validate transcripts with two different judge models."
    )
    parser.add_argument(
        "--transcripts", nargs="+",
        help="Paths to transcript JSON files (mutually exclusive with --transcripts-dir)",
    )
    parser.add_argument(
        "--transcripts-dir", type=str, default=None,
        help="Directory containing transcript JSON files (scored in sorted order)",
    )
    parser.add_argument(
        "--rubric", type=str, default=None,
        help=f"Path to judge rubric file (default: {DEFAULT_RUBRIC_PATH})",
    )
    parser.add_argument(
        "--judge-models", nargs=2, required=True, metavar="MODEL",
        help="Two judge model strings (e.g. claude-sonnet-4-5-20250929 gpt-5.2)",
    )
    parser.add_argument(
        "--judge-providers", nargs=2, required=True, metavar="PROVIDER",
        help="Providers for each judge model (e.g. anthropic openai), same order as --judge-models",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=8000,
        help="Max tokens for judge responses (default: 8000)",
    )
    parser.add_argument(
        "--batch", action="store_true",
        help="Use provider batch APIs (50%% discount, ~24h wall). Resumable via batch_state.json.",
    )
    parser.add_argument(
        "--poll-interval", type=int, default=30,
        help="Seconds between batch poll checks (default: 30). Only applies with --batch.",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Optional filepath to save comparison results JSON",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    load_env()

    if args.batch and not args.output:
        parser.error("--batch requires --output (used to anchor batch_state.json for resume).")

    transcript_paths = _resolve_transcripts(args)

    rubric_path = args.rubric or str(DEFAULT_RUBRIC_PATH)
    system_prompt = _load_judge_system_prompt(rubric_path)
    rubric_sha = _sha256(system_prompt)
    logger.info(f"Rubric: {rubric_path} (sha256 {rubric_sha[:12]}…)")

    model_a, model_b = args.judge_models
    provider_a, provider_b = args.judge_providers
    label_a = f"{provider_a}/{model_a}"
    label_b = f"{provider_b}/{model_b}"

    started_at = datetime.now(timezone.utc).isoformat()
    batch_ids: dict = {}

    if args.batch:
        state_path = Path(args.output).parent / "batch_state.json"
        all_turn_scores_a, all_turn_scores_b, per_transcript_results, batch_ids = _run_batch_mode(
            transcript_paths, system_prompt, rubric_path, rubric_sha,
            provider_a, model_a, provider_b, model_b, args.max_tokens,
            label_a, label_b, state_path, poll_interval=args.poll_interval,
        )
    else:
        all_turn_scores_a, all_turn_scores_b, per_transcript_results = _run_sync_mode(
            transcript_paths, system_prompt,
            provider_a, model_a, provider_b, model_b,
            args.max_tokens, label_a, label_b,
        )

    if not per_transcript_results:
        logger.error("No transcripts produced valid scores; nothing to compare.")
        sys.exit(1)

    comparison = compute_comparison_stats(all_turn_scores_a, all_turn_scores_b)
    print_results(comparison, label_a, label_b)

    completed_at = datetime.now(timezone.utc).isoformat()

    if args.output:
        import math

        def sanitize(obj):
            if isinstance(obj, float) and math.isnan(obj):
                return None
            if isinstance(obj, dict):
                return {k: sanitize(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [sanitize(v) for v in obj]
            return obj

        run_metadata = {
            "mode": "batch" if args.batch else "sync",
            "rubric_path": rubric_path,
            "rubric_sha256": rubric_sha,
            "transcripts": transcript_paths,
            "judges": {
                "a": {
                    "provider": provider_a, "model": model_a,
                    "max_tokens": args.max_tokens, "temperature": 0.0,
                },
                "b": {
                    "provider": provider_b, "model": model_b,
                    "max_tokens": args.max_tokens, "temperature": 0.0,
                },
            },
            "started_at": started_at,
            "completed_at": completed_at,
        }
        if args.batch:
            run_metadata["anthropic_batch_id"] = batch_ids.get("anthropic_batch_id")
            run_metadata["openai_batch_id"] = batch_ids.get("openai_batch_id")

        output_data = {
            "judge_a": label_a,
            "judge_b": label_b,
            "aggregate": sanitize(comparison),
            "per_transcript": sanitize(per_transcript_results),
            "run_metadata": run_metadata,
        }
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"Comparison results saved to {args.output}")


if __name__ == "__main__":
    main()

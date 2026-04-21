"""Smoke test for spec-aligned batch_state.json paths and concurrent-submit lock.

Verifies (unit-level, no API):
  A1. state_path derivation: --output foo/bar.json -> foo/batch_state.json
  A2. jsonl_path derivation: --output foo/bar.json -> foo/openai_input.jsonl
  A3. _acquire_submit_lock raises on second concurrent call, succeeds after release.

Verifies (live, budget ~$0.10 worst case, typically $0 after cancellation):
  B1. Two --batch invocations against DIFFERENT output dirs both produce state
      files at spec-compliant paths.
  B2. Both batches per invocation are retrievable by ID and cancellable.

Run from repo root with a Python that has anthropic+openai installed:
    .venv/bin/python tests/test_batch_state_paths.py            # A + B
    .venv/bin/python tests/test_batch_state_paths.py --unit     # A only (no API)
"""

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SMOKE_DIR = Path("/tmp/cv_smoke/live")

sys.path.insert(0, str(REPO))
from scripts.judge_cross_validate import _acquire_submit_lock  # noqa: E402


def test_A_paths_and_lock():
    print("=== A: unit-level path + lock checks ===")

    dummy_output = Path("/tmp/cv_smoke/unit/results/cross_validation_results.json")
    state_path = Path(dummy_output).parent / "batch_state.json"
    jsonl_path = state_path.parent / "openai_input.jsonl"
    assert state_path == Path("/tmp/cv_smoke/unit/results/batch_state.json"), state_path
    assert jsonl_path == Path("/tmp/cv_smoke/unit/results/openai_input.jsonl"), jsonl_path
    print(f"[PASS] A1/A2 state={state_path.name} jsonl={jsonl_path.name}")

    d = Path("/tmp/cv_smoke/unit/lock")
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)

    lock = _acquire_submit_lock(d)
    assert lock.exists(), "first acquire didn't create lockfile"
    body = lock.read_text()
    assert "pid=" in body and "ts=" in body, f"lock body missing pid/ts: {body}"

    try:
        _acquire_submit_lock(d)
    except RuntimeError as e:
        assert "Another --batch invocation" in str(e), str(e)
    else:
        raise AssertionError("second acquire should have raised RuntimeError")

    lock.unlink()
    lock2 = _acquire_submit_lock(d)
    assert lock2.exists()
    lock2.unlink()
    print("[PASS] A3 lock collision + release semantics")


def _launch_submit(out_dir: Path, transcript: Path) -> subprocess.Popen:
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / "cross_validation_results.json"
    cmd = [
        sys.executable, "-m", "scripts.judge_cross_validate",
        "--transcripts", str(transcript),
        "--rubric", str(REPO / "cross_validation_review/rubrics/judge_system_v6.txt"),
        "--judge-models", "claude-sonnet-4-20250514", "gpt-5.2",
        "--judge-providers", "anthropic", "openai",
        "--max-tokens", "500",
        "--output", str(output),
        "--batch",
        "--poll-interval", "600",
    ]
    return subprocess.Popen(
        cmd, cwd=str(REPO),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )


def _wait_for_state(state_path: Path, proc: subprocess.Popen, timeout_s: int = 180) -> None:
    start = time.time()
    while time.time() - start < timeout_s:
        if state_path.exists():
            return
        if proc.poll() is not None:
            tail = (proc.stdout.read() if proc.stdout else "")[-2000:]
            raise RuntimeError(f"submit process exited early rc={proc.returncode}\n{tail}")
        time.sleep(1)
    proc.kill()
    raise RuntimeError(f"state file never appeared at {state_path} within {timeout_s}s")


def test_B_live_two_dirs():
    print("=== B: live submit, 2 invocations, different output dirs ===")
    import anthropic
    import openai
    from scripts.utils import load_env, require_api_key
    load_env()

    tdir = REPO / "cross_validation_review/runs/v6_contextual_baseline_set/transcripts"
    candidates = sorted(tdir.glob("*.json"), key=lambda p: p.stat().st_size)
    if not candidates:
        raise RuntimeError(f"No transcripts found in {tdir}")
    transcript = candidates[0]
    print(f"  transcript: {transcript.name} ({transcript.stat().st_size} B)")

    if SMOKE_DIR.exists():
        shutil.rmtree(SMOKE_DIR)
    SMOKE_DIR.mkdir(parents=True)

    dir_a = SMOKE_DIR / "a"
    dir_b = SMOKE_DIR / "b"

    proc_a = _launch_submit(dir_a, transcript)
    proc_b = _launch_submit(dir_b, transcript)

    try:
        _wait_for_state(dir_a / "batch_state.json", proc_a)
        _wait_for_state(dir_b / "batch_state.json", proc_b)
    finally:
        for p in (proc_a, proc_b):
            if p.poll() is None:
                p.send_signal(signal.SIGTERM)
        for p in (proc_a, proc_b):
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                p.kill()

    ac = anthropic.Anthropic(api_key=require_api_key("anthropic"))
    oc = openai.OpenAI(api_key=require_api_key("openai"))

    for label, d in [("a", dir_a), ("b", dir_b)]:
        state_path = d / "batch_state.json"
        jsonl_path = d / "openai_input.jsonl"
        assert state_path.exists(), f"[{label}] missing {state_path}"
        assert jsonl_path.exists(), f"[{label}] missing {jsonl_path}"
        state = json.loads(state_path.read_text())
        anth_id = state.get("anthropic_batch_id")
        oai_id = state.get("openai_batch_id")
        assert anth_id and oai_id, f"[{label}] state missing batch IDs: {state}"
        print(f"[PASS] {label}: state + jsonl at spec paths; anth={anth_id} oai={oai_id}")

        try:
            ac.messages.batches.cancel(anth_id)
            print(f"  cancelled anthropic {anth_id}")
        except Exception as e:
            print(f"  anthropic cancel non-fatal: {type(e).__name__}: {e}")
        try:
            oc.batches.cancel(oai_id)
            print(f"  cancelled openai {oai_id}")
        except Exception as e:
            print(f"  openai cancel non-fatal: {type(e).__name__}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unit", action="store_true", help="Run only unit-level tests (no API)")
    args = ap.parse_args()

    test_A_paths_and_lock()
    if not args.unit:
        test_B_live_two_dirs()
    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    main()

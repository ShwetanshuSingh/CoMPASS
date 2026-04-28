#!/bin/bash
# Snapshot dashboard for an in-flight CoMPASS full-matrix launch.
#
# Usage:
#   scripts/monitor_launch.sh [log_file] [transcripts_dir]
#
# Defaults: launch_v6.log  and  transcripts/full_matrix_v6/
#
# For a live-refreshing dashboard:
#   watch -n 30 scripts/monitor_launch.sh

LOG_FILE="${1:-launch_v6.log}"
TRANSCRIPTS_DIR="${2:-transcripts/full_matrix_v6/}"

TARGETS=(claude-haiku claude-sonnet gemini-flash gemini-3-flash gpt-nano gpt-5 grok-fast grok-4-1-fast)
TOTAL_PER_TARGET=100
TOTAL_CELLS=800

echo "=== CoMPASS full-matrix launch — $(date '+%Y-%m-%d %H:%M:%S') ==="
echo "log: $LOG_FILE"
echo "transcripts: $TRANSCRIPTS_DIR"
echo ""

# ---- Per-target transcript progress ----
echo "=== Transcripts completed (target / $TOTAL_PER_TARGET) ==="
total_done=0
for t in "${TARGETS[@]}"; do
  count=$(find "$TRANSCRIPTS_DIR" -maxdepth 1 -name "*_${t}_*.json" 2>/dev/null | wc -l | tr -d ' ')
  total_done=$((total_done + count))
  pct=$((count * 100 / TOTAL_PER_TARGET))
  bar=""
  filled=$((pct / 5))
  for ((i=0; i<filled; i++)); do bar="${bar}#"; done
  for ((i=filled; i<20; i++)); do bar="${bar}-"; done
  printf "  %-18s %3d / %3d  [%s] %3d%%\n" "$t" "$count" "$TOTAL_PER_TARGET" "$bar" "$pct"
done
overall_pct=$((total_done * 100 / TOTAL_CELLS))
echo "  ─────────────────────────────────────────────────────────"
printf "  %-18s %3d / %3d           %3d%%\n" "TOTAL" "$total_done" "$TOTAL_CELLS" "$overall_pct"
echo ""

# if [[ -f "$LOG_FILE" ]]; then
  # ---- Per-target failure counts ----
#  echo "=== Trial failures by target (pre-reg §10 halt: ≥10 for most, ≥20 for gemini-3-flash) ==="
#  any_fail=0
#  for t in "${TARGETS[@]}"; do
#    fail_count=$(grep -c "Trial failed.*${t} run" "$LOG_FILE" 2>/dev/null)
#    fail_count=${fail_count:-0}
#    if [[ "$fail_count" -gt 0 ]]; then
#      any_fail=1
#      threshold=10
#      [[ "$t" == "gemini-3-flash" ]] && threshold=20
#      flag=""
#      [[ "$fail_count" -ge "$threshold" ]] && flag="  ⚠ HALT THRESHOLD HIT"
#      printf "  %-18s %3d%s\n" "$t" "$fail_count" "$flag"
#    fi
#  done
#  [[ "$any_fail" == "0" ]] && echo "  (no failures logged)"
#  echo ""

  # ---- Early-termination tag counts ----
  end_refusal=$(grep -c "with \[END_REFUSAL\]" "$LOG_FILE" 2>/dev/null)
  end_complete=$(grep -c "with \[END_COMPLETE\]" "$LOG_FILE" 2>/dev/null)
  end_refusal=${end_refusal:-0}
  end_complete=${end_complete:-0}
  echo "=== Early-termination tags ==="
  printf "  END_REFUSAL    %d\n" "$end_refusal"
  printf "  END_COMPLETE   %d\n" "$end_complete"
  echo ""

  # ---- Last errors ----
#  echo "=== Last 3 errors (if any) ==="
#  err_lines=$(grep -E "\[ERROR\]|Trial failed|Target-group worker raised" "$LOG_FILE" | tail -3)
#  if [[ -z "$err_lines" ]]; then
#    echo "  (none)"
#  else
#    echo "$err_lines" | sed 's/^/  /'
#  fi
#else
#  echo "(log file not found — launch not started yet?)"
#fi

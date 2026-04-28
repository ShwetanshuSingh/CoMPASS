#!/bin/bash
# Live error tail for an in-flight CoMPASS full-matrix launch.
#
# Usage:
#   scripts/watch_errors.sh [log_file]
#
# Default: launch_v6.log
#
# Waits for the log file to exist, then follows it and highlights
# error-class lines.

LOG_FILE="${1:-launch_v6.log}"

if [[ ! -f "$LOG_FILE" ]]; then
  echo "Waiting for $LOG_FILE to be created..."
  until [[ -f "$LOG_FILE" ]]; do sleep 2; done
  echo "$LOG_FILE detected — tailing."
fi

tail -f "$LOG_FILE" | grep --line-buffered --color=always -E "\[ERROR\]|Trial failed|Target-group worker raised|BadRequestError|RateLimitError|APIError|ServerError|503 |429 |401 |400 "

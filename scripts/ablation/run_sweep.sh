#!/bin/bash
# Serial fallback/recovery executor: run every job in a job-list file through
# run_one.sh (which is GPU-locked + resumable, so already-done jobs are skipped).
# Usage: run_sweep.sh <jobfile> [<jobfile> ...]
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
for jf in "$@"; do
  while IFS= read -r line; do
    [ -z "$line" ] && continue
    case "$line" in \#*) continue;; esac
    bash "$HERE/run_one.sh" $line
  done < "$jf"
done
echo "=== SWEEP DONE ($*) ==="

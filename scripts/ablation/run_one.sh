#!/bin/bash
# GPU-locked, resumable single-config ablation runner.
#
# Usage: run_one.sh <config_id> <sequence> <max_frames|0> <extra run_slam flags...>
#
# - Serializes GPU access across concurrent agents via flock on a host lockfile
#   (the Radeon 8060S APU is a single shared device; concurrent runs corrupt FPS).
# - Resumable: if the result row already exists in results.csv, it is skipped.
# - Appends one CSV row (id,seq,ate_rmse_m,rpe_rmse_m,tracking_fps,end_to_end_fps,
#   lost_frames,n_keyframes,n_frames,status) under a separate append lock.
#
# Run from the repo root (the dir containing compose.yaml).
set -u

CONFIG_ID="$1"; SEQ="$2"; MAXF="$3"; shift 3
FLAGS=("$@")

GPU_LOCK="/tmp/gradslam_gpu.lock"
CSV_LOCK="/tmp/gradslam_ablation_csv.lock"
RESULTS="outputs/ablation/results.csv"
OUTDIR="/workspace/outputs/ablation/${CONFIG_ID}__${SEQ}"
HOSTOUT="outputs/ablation/${CONFIG_ID}__${SEQ}"
ROW_KEY="${CONFIG_ID},${SEQ}"

mkdir -p outputs/ablation
[ -f "$RESULTS" ] || echo "id,seq,ate_rmse_m,rpe_rmse_m,tracking_fps,end_to_end_fps,lost_frames,n_keyframes,n_frames,status,flags" > "$RESULTS"

# Resumable: skip if this (id,seq) already recorded with a terminal status.
if grep -q "^${ROW_KEY}," "$RESULTS" 2>/dev/null; then
  echo "[skip] ${ROW_KEY} already in results.csv"; exit 0
fi

MAXF_FLAG=()
[ "$MAXF" != "0" ] && MAXF_FLAG=(--max-frames "$MAXF")

echo "[run] ${ROW_KEY} flags: ${FLAGS[*]} ${MAXF_FLAG[*]}"
# Acquire exclusive GPU lock (single APU → runs must serialize for valid FPS).
# Bounded wait (560s) so a concurrent agent's blocked call stays under the 600s
# Bash cap; per-run timeout 300s. Long sequences (fr1_room/fr2_xyz) run in a
# separate serial wave with a generous timeout, not under this concurrent path.
flock -w 560 -x "$GPU_LOCK" timeout 300 docker compose run --rm gradslam bash -c "
  cd /workspace/gradslam
  python scripts/run_slam.py tum \
    --dataset-root /workspace/datasets/public/TUM/tum_rgbd/ \
    --sequence '${SEQ}' --output '${OUTDIR}' \
    ${FLAGS[*]} ${MAXF_FLAG[*]} 2>&1 | tail -4
" > "outputs/ablation/${CONFIG_ID}__${SEQ}.log" 2>&1
RC=$?

# Parse metrics.json (written on host via the mounted ./outputs).
PARSE=$(python3 - "$HOSTOUT/metrics.json" "$RC" <<'PY'
import json,sys
mp,rc=sys.argv[1],sys.argv[2]
try:
    d=json.load(open(mp))
    status="ok" if rc=="0" else f"rc{rc}"
    print(f'{d.get("ate_rmse_m")},{d.get("rpe_rmse_m")},{d.get("tracking_fps")},{d.get("end_to_end_fps")},{d.get("lost_frames")},{d.get("n_keyframes")},{d.get("n_frames")},{status}')
except Exception as e:
    print(f',,,,,,,fail_rc{rc}')
PY
)

FLAGSTR=$(printf '%s ' "${FLAGS[@]}" | sed 's/,/;/g; s/ *$//')
flock -x "$CSV_LOCK" bash -c "echo '${ROW_KEY},${PARSE},\"${FLAGSTR}\"' >> '$RESULTS'"
echo "[done] ${ROW_KEY} -> ${PARSE}"

#!/bin/bash
# run_executor.sh - flock wrapper + random delay
# Usage: ./run_executor.sh coin [--dry-run]
#        ./run_executor.sh stock [--dry-run]

set -e

ASSET="${1:-coin}"
shift || true

LOCK_FILE="/tmp/executor_${ASSET}.lock"
SCRIPT="/home/ubuntu/executor_${ASSET}.py"

if [ ! -f "$SCRIPT" ]; then
    echo "Error: $SCRIPT not found"
    exit 1
fi

exec 200>"$LOCK_FILE"
if ! flock -n 200; then
    exit 0
fi

# 랜덤 지연 0~90초 (정시 집중 방지)
sleep $((RANDOM % 90))

cd /home/ubuntu
python3 "$SCRIPT" "$@"

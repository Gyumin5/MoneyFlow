#!/bin/bash
# run_recommend.sh - flock wrapper for recommend scripts
# Usage: ./scripts/run_recommend.sh [general|personal]
#
# Example crontab entries:
# 00 09 * * * /home/ubuntu/MoneyFlow/scripts/run_recommend.sh general
# 00 09 * * * /home/ubuntu/MoneyFlow/scripts/run_recommend.sh personal

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TYPE="${1:-general}"

if [ "$TYPE" = "personal" ]; then
    SCRIPT="$PROJECT_DIR/strategies/cap_defend/recommend_personal.py"
    LOCK_FILE="/tmp/recommend_personal.lock"
else
    SCRIPT="$PROJECT_DIR/strategies/cap_defend/recommend.py"
    LOCK_FILE="/tmp/recommend_general.lock"
fi

# Check if script exists
if [ ! -f "$SCRIPT" ]; then
    echo "Error: $SCRIPT not found"
    exit 1
fi

# flock: 이미 실행 중이면 즉시 종료 (중복 방지)
exec 200>"$LOCK_FILE"
if ! flock -n 200; then
    echo "[$(date)] Another instance of recommend ($TYPE) is already running. Exiting."
    exit 0
fi

# Run the script from project root
echo "[$(date)] Starting recommend ($TYPE)..."
cd "$PROJECT_DIR"
python3 "$SCRIPT"
echo "[$(date)] Recommend ($TYPE) finished."

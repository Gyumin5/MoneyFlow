#!/bin/bash
# run_trade.sh - flock wrapper for auto_trade scripts
# Usage: ./run_trade.sh [upbit|bithumb] [--trade] [--force]
#
# Example crontab entries:
# 05 09 * * * /home/ubuntu/run_trade.sh upbit --trade
# 10 09 * * * /home/ubuntu/run_trade.sh bithumb --trade

set -e

EXCHANGE="${1:-upbit}"
shift || true

LOCK_FILE="/tmp/auto_trade_${EXCHANGE}.lock"
SCRIPT="/home/ubuntu/auto_trade_${EXCHANGE}.py"

# Check if script exists
if [ ! -f "$SCRIPT" ]; then
    echo "Error: $SCRIPT not found"
    exit 1
fi

# flock: 이미 실행 중이면 즉시 종료 (중복 방지)
exec 200>"$LOCK_FILE"
if ! flock -n 200; then
    echo "[$(date)] Another instance of $EXCHANGE bot is already running. Exiting."
    exit 0
fi

# Run the script
echo "[$(date)] Starting $EXCHANGE bot..."
cd /home/ubuntu
python3 "$SCRIPT" "$@"
echo "[$(date)] $EXCHANGE bot finished."

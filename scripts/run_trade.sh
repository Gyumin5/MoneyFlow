#!/bin/bash
# run_trade.sh - flock wrapper for auto_trade scripts
# Usage: ./scripts/run_trade.sh [upbit|bithumb] [--trade] [--force]
#
# Example crontab entries:
# 05 09 * * * /home/ubuntu/MoneyFlow/scripts/run_trade.sh upbit --trade
# 10 09 * * * /home/ubuntu/MoneyFlow/scripts/run_trade.sh bithumb --trade

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
EXCHANGE="${1:-upbit}"
shift || true

LOCK_FILE="/tmp/auto_trade_${EXCHANGE}.lock"
SCRIPT="$PROJECT_DIR/trade/auto_trade_${EXCHANGE}.py"

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

# Run the script from project root
echo "[$(date)] Starting $EXCHANGE bot..."
cd "$PROJECT_DIR"
python3 "$SCRIPT" "$@"
echo "[$(date)] $EXCHANGE bot finished."

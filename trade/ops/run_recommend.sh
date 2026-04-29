#!/bin/bash
# run_recommend.sh - flock wrapper + 실패 시 텔레그램 알림 + 재시도

set -e

TYPE="${1:-general}"

if [ "$TYPE" = "personal" ]; then
    SCRIPT="/home/ubuntu/recommend_personal.py"
    LOCK_FILE="/tmp/recommend_personal.lock"
else
    SCRIPT="/home/ubuntu/recommend.py"
    LOCK_FILE="/tmp/recommend_general.lock"
fi

if [ ! -f "$SCRIPT" ]; then
    echo "Error: $SCRIPT not found"
    exit 1
fi

exec 200>"$LOCK_FILE"
if ! flock -n 200; then
    exit 0
fi

send_alert() {
    local TOKEN=$(python3 -c "from config import TELEGRAM_BOT_TOKEN; print(TELEGRAM_BOT_TOKEN)" 2>/dev/null)
    local CHAT=$(python3 -c "from config import TELEGRAM_CHAT_ID; print(TELEGRAM_CHAT_ID)" 2>/dev/null)
    if [ -n "$TOKEN" ] && [ -n "$CHAT" ]; then
        curl -s -X POST "https://api.telegram.org/bot${TOKEN}/sendMessage" \
            -d chat_id="${CHAT}" -d text="$1" > /dev/null 2>&1
    fi
}

echo "[$(date)] Starting recommend ($TYPE)..."
cd /home/ubuntu

if python3 "$SCRIPT"; then
    echo "[$(date)] Recommend ($TYPE) finished."
else
    echo "[$(date)] Recommend ($TYPE) FAILED. Retrying in 5 min..."
    send_alert "⚠️ recommend ($TYPE) 실패. 5분 후 재시도."
    sleep 300
    if python3 "$SCRIPT"; then
        echo "[$(date)] Recommend ($TYPE) retry succeeded."
        send_alert "✅ recommend ($TYPE) 재시도 성공."
    else
        echo "[$(date)] Recommend ($TYPE) retry FAILED."
        send_alert "🚨 recommend ($TYPE) 2차 실패. 수동 확인 필요!"
    fi
fi

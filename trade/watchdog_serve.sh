#!/bin/bash
# === 서버 프로세스 감시 (robust restart) ===

restart_service() {
    local label="$1"     # display name
    local pattern="$2"   # pgrep/pkill -f pattern
    local port="$3"
    local health_url="$4"
    local start_cmd="$5"

    if curl -s --max-time 3 "$health_url" > /dev/null 2>&1; then
        # healthy → reset fail counter
        rm -f "/tmp/wd_${label}_fails"
        return 0
    fi

    echo "[$(date)] $label down — restarting"

    # SIGTERM
    pkill -f "$pattern" 2>/dev/null
    sleep 2

    # SIGKILL fallback if still running
    if pgrep -f "$pattern" >/dev/null 2>&1; then
        echo "[$(date)] $label SIGTERM 무시 — SIGKILL"
        pkill -9 -f "$pattern" 2>/dev/null
        sleep 1
    fi

    # Force port release if still held
    if ss -ltn "sport = :$port" 2>/dev/null | tail -n +2 | grep -q ":$port"; then
        echo "[$(date)] port $port held — fuser -k"
        fuser -k "$port/tcp" 2>/dev/null
        sleep 2
    fi

    # Start
    eval "$start_cmd"
    sleep 4

    # Verify
    if curl -s --max-time 3 "$health_url" > /dev/null 2>&1; then
        echo "[$(date)] $label restart OK"
        rm -f "/tmp/wd_${label}_fails"
        return 0
    fi

    # Failed — increment counter, alert at 3 consecutive
    local fail_file="/tmp/wd_${label}_fails"
    local fails=$(cat "$fail_file" 2>/dev/null || echo 0)
    fails=$((fails + 1))
    echo "$fails" > "$fail_file"
    echo "[$(date)] $label restart FAILED (#$fails)"
    if [ "$fails" = "3" ]; then
        local TOKEN=$(python3 -c "from config import TELEGRAM_BOT_TOKEN; print(TELEGRAM_BOT_TOKEN)" 2>/dev/null)
        local CHAT=$(python3 -c "from config import TELEGRAM_CHAT_ID; print(TELEGRAM_CHAT_ID)" 2>/dev/null)
        if [ -n "$TOKEN" ] && [ -n "$CHAT" ]; then
            curl -s -X POST "https://api.telegram.org/bot${TOKEN}/sendMessage" \
                -d chat_id="${CHAT}" -d text="🚨 $label 3회 연속 재시작 실패 (port $port). 수동 점검 필요" >/dev/null 2>&1
        fi
    fi
    return 1
}

# serve.py (port 8080)
restart_service "serve" "python3 serve.py" 8080 \
    "http://localhost:8080/strategy.html" \
    'cd /home/ubuntu && nohup python3 serve.py > http.log 2>&1 &'

# trade_api_server.py (port 5000)
restart_service "trade_api" "python3 trade_api_server.py" 5000 \
    "http://localhost:5000/health" \
    'cd /home/ubuntu && export TRADE_PIN=0318 ALLOWED_ORIGINS=http://152.69.225.8:8080 && nohup python3 trade_api_server.py > api_server.log 2>&1 &'

# === signal_state 신선도 감시 ===
SIGNAL_FILE="/home/ubuntu/signal_state.json"
if [ -f "$SIGNAL_FILE" ]; then
    UPDATED=$(python3 -c "import json; print(json.load(open('$SIGNAL_FILE')).get('meta',{}).get('updated_at',''))" 2>/dev/null)
    if [ -n "$UPDATED" ]; then
        SIGNAL_EPOCH=$(date -d "$UPDATED" +%s 2>/dev/null || echo 0)
        NOW_EPOCH=$(date +%s)
        AGE_HOURS=$(( (NOW_EPOCH - SIGNAL_EPOCH) / 3600 ))
        if [ "$AGE_HOURS" -gt 26 ]; then
            echo "[$(date)] ⚠️ signal_state stale: ${AGE_HOURS}h old (updated: $UPDATED)"
            ALERT_FLAG="/tmp/signal_stale_alerted"
            if [ ! -f "$ALERT_FLAG" ] || [ $(find "$ALERT_FLAG" -mmin +1440 2>/dev/null | wc -l) -gt 0 ]; then
                TOKEN=$(python3 -c "from config import TELEGRAM_BOT_TOKEN; print(TELEGRAM_BOT_TOKEN)" 2>/dev/null)
                CHAT=$(python3 -c "from config import TELEGRAM_CHAT_ID; print(TELEGRAM_CHAT_ID)" 2>/dev/null)
                if [ -n "$TOKEN" ] && [ -n "$CHAT" ]; then
                    curl -s -X POST "https://api.telegram.org/bot${TOKEN}/sendMessage" \
                        -d chat_id="${CHAT}" -d text="🚨 signal_state ${AGE_HOURS}시간 미갱신! recommend 확인 필요" > /dev/null 2>&1
                fi
                touch "$ALERT_FLAG"
            fi
        fi
    fi
fi

# === state 파일 매일 백업 (1일 1회) ===
BACKUP_DIR="/home/ubuntu/state_backups"
mkdir -p "$BACKUP_DIR"
TODAY=$(date +%Y-%m-%d)
BACKUP_FLAG="$BACKUP_DIR/.backup_$TODAY"
if [ ! -f "$BACKUP_FLAG" ]; then
    cp -f /home/ubuntu/signal_state.json "$BACKUP_DIR/signal_state_$TODAY.json" 2>/dev/null
    cp -f /home/ubuntu/coin_trade_state.json "$BACKUP_DIR/coin_trade_state_$TODAY.json" 2>/dev/null
    cp -f /home/ubuntu/kis_trade_state.json "$BACKUP_DIR/kis_trade_state_$TODAY.json" 2>/dev/null
    touch "$BACKUP_FLAG"
    find "$BACKUP_DIR" -name "*.json" -mtime +14 -delete 2>/dev/null
    find "$BACKUP_DIR" -name ".backup_*" -mtime +14 -delete 2>/dev/null
    echo "[$(date)] State backup created: $TODAY"
fi

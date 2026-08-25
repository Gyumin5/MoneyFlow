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
    'nohup /home/ubuntu/start_trade_api.sh > /dev/null 2>&1 &'

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

# === state 파일 매일 백업 (cohort 단위, 하루 1회) ===
#
# 2026-08-25 전면 수정. 그 전에는 (a) 서버에 없는 이름 coin_trade_state.json 을 복사하고 있어
# 코인현물 상태가 한 번도 백업되지 않았고, (b) 선물 binance_state.json 은 대상이 아니었으며,
# (c) cp 가 실패해도 성공 플래그가 찍혀 침묵 실패가 영구화됐다.
#
# 설계: 넷을 임시 디렉토리에 모아 JSON 유효성까지 확인한 뒤, 디렉토리 이름을 바꿔 한 번에 공개한다.
# 완성된 cohort 디렉토리(state_backups/YYYY-MM-DD/)만 복원에 쓴다 — 반쯤 만들어진 백업은 이름이 없다.
# 하나라도 실패하면 공개하지 않으므로 다음 5분 실행이 자동으로 다시 시도한다.
# 시각은 09:10 KST 이후로 잡는다(코인·선물 09:05 체결 반영. 주식은 전날 23:35 분이 들어간다).
BACKUP_DIR="/home/ubuntu/state_backups"
BACKUP_SRC=(signal_state.json trade_state.json kis_trade_state.json binance_state.json)
BACKUP_FAIL_LATCH="/tmp/wd_backup_fail_alerted"
mkdir -p "$BACKUP_DIR"
TODAY=$(date +%Y-%m-%d)
COHORT="$BACKUP_DIR/$TODAY"
STAGING="$BACKUP_DIR/.staging_$TODAY"

wd_backup_alert() {   # 하루 1회만 — 5분마다 재시도하되 알림은 래치로 묶는다
    local text="$1" latch="$2"
    if [ ! -f "$latch" ] || [ $(find "$latch" -mmin +1440 2>/dev/null | wc -l) -gt 0 ]; then
        local TOKEN=$(python3 -c "from config import TELEGRAM_BOT_TOKEN; print(TELEGRAM_BOT_TOKEN)" 2>/dev/null)
        local CHAT=$(python3 -c "from config import TELEGRAM_CHAT_ID; print(TELEGRAM_CHAT_ID)" 2>/dev/null)
        if [ -n "$TOKEN" ] && [ -n "$CHAT" ]; then
            curl -s -X POST "https://api.telegram.org/bot${TOKEN}/sendMessage" \
                -d chat_id="${CHAT}" -d text="$text" > /dev/null 2>&1
        fi
        touch "$latch"
    fi
}

if [ ! -d "$COHORT" ] && [ "$((10#$(date +%H%M)))" -ge 910 ]; then
    rm -rf "$STAGING"
    mkdir -p "$STAGING"
    BACKUP_ERR=""
    for f in "${BACKUP_SRC[@]}"; do
        src="/home/ubuntu/$f"
        if [ ! -s "$src" ]; then BACKUP_ERR="$BACKUP_ERR $f(없음/빈파일)"; continue; fi
        if ! python3 -c "import json,sys; json.load(open(sys.argv[1]))" "$src" 2>/dev/null; then
            BACKUP_ERR="$BACKUP_ERR $f(JSON깨짐)"; continue
        fi
        if ! cp -f "$src" "$STAGING/$f" 2>/dev/null; then BACKUP_ERR="$BACKUP_ERR $f(복사실패)"; continue; fi
        if ! python3 -c "import json,sys; json.load(open(sys.argv[1]))" "$STAGING/$f" 2>/dev/null; then
            BACKUP_ERR="$BACKUP_ERR $f(복사본깨짐)"
        fi
    done

    if [ -z "$BACKUP_ERR" ]; then
        date '+%Y-%m-%d %H:%M:%S %Z' > "$STAGING/MANIFEST"
        (cd "$STAGING" && ls -l *.json >> MANIFEST 2>/dev/null)
        if mv -T "$STAGING" "$COHORT" 2>/dev/null; then
            echo "[$(date)] State backup cohort 생성: $TODAY (${#BACKUP_SRC[@]}종)"
            if [ -f "$BACKUP_FAIL_LATCH" ]; then
                wd_backup_alert "✅ state 백업 정상 복구 — $TODAY cohort 생성" "/tmp/wd_backup_ok_alerted"
                rm -f "$BACKUP_FAIL_LATCH"
            fi
            # 보관 14일. 삭제 대상은 이 디렉토리의 날짜 cohort 와 옛 평면 파일로만 한정한다.
            find "$BACKUP_DIR" -maxdepth 1 -type d -name '20*-*-*' -mtime +14 -exec rm -rf {} + 2>/dev/null
            find "$BACKUP_DIR" -maxdepth 1 -type f -name '*_20*.json' -mtime +14 -delete 2>/dev/null
            find "$BACKUP_DIR" -maxdepth 1 -type f -name '.backup_*' -mtime +14 -delete 2>/dev/null
            find "$BACKUP_DIR" -maxdepth 1 -type d -name '.staging_*' -mtime +1 -exec rm -rf {} + 2>/dev/null
        else
            echo "[$(date)] ⚠️ state backup 공개 실패(mv): $TODAY"
            wd_backup_alert "🚨 state 백업 실패 — cohort 공개(mv) 실패. 디스크·권한 확인 필요" "$BACKUP_FAIL_LATCH"
        fi
    else
        rm -rf "$STAGING"
        echo "[$(date)] ⚠️ state backup 미완성 — 다음 실행 재시도:$BACKUP_ERR"
        wd_backup_alert "🚨 state 백업 실패:$BACKUP_ERR (5분마다 재시도 중)" "$BACKUP_FAIL_LATCH"
    fi
fi

#!/bin/sh
# trade_api_server 기동 wrapper (2026-07-21)
# - env 단일출처: /home/ubuntu/.trade_env (watchdog / @reboot cron 공용)
# - fail-fast: 필수 env 없으면 기동하지 않음 (빈 PIN 조용한 기동 방지)
# - TRADE_PIN 미설정 유지: 쓰기 API 3종 fail-closed (2026-07-21 사용자 승인)
ENV_FILE=/home/ubuntu/.trade_env
LOG=/home/ubuntu/api_server.log

if [ ! -r "$ENV_FILE" ]; then
  echo "$(date -Is) FATAL: $ENV_FILE 없음/읽기불가 — 기동 중단" >> "$LOG"
  exit 1
fi

set -a
. "$ENV_FILE"
set +a

if [ -z "$DASHBOARD_PIN" ]; then
  echo "$(date -Is) FATAL: DASHBOARD_PIN 미설정 — 기동 중단(무인증 조회 방지)" >> "$LOG"
  exit 1
fi
if [ -z "$ALLOWED_ORIGINS" ]; then
  echo "$(date -Is) FATAL: ALLOWED_ORIGINS 미설정 — 기동 중단(CORS 와일드카드 방지)" >> "$LOG"
  exit 1
fi

# 쓰기 API 를 반드시 닫기 위해 상속된 값이 있어도 제거
unset TRADE_PIN

cd /home/ubuntu || exit 1
exec python3 trade_api_server.py >> "$LOG" 2>&1

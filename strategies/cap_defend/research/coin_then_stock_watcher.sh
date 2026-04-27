#!/bin/bash
# 코인 rebuild 완료 감시 → 주식 grid 재실행 자동 트리거
cd "$(dirname "$0")"
LOG=coin_then_stock_watcher.log
: > "$LOG"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

log "start: wait for redesign_univ3_raw.csv to reach 121718 rows"

TARGET=121718
while true; do
  N=$(wc -l < redesign_univ3_raw.csv 2>/dev/null || echo 0)
  if [ "$N" -ge "$TARGET" ]; then
    log "coin rebuild complete ($N rows)"
    break
  fi
  log "waiting... $N / $TARGET"
  sleep 300  # 5min
done

# 주식 grid 재실행 시작 (사용자 확정 grid)
log "starting stock grid redesign"
# (실행은 사용자 확정 후 별도 스크립트로)
touch coin_done.flag
log "done (coin_done.flag set, stock grid should be manually started via new script)"

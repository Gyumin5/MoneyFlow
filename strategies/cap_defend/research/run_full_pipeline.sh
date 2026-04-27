#!/bin/bash
# V21 재설계 full pipeline driver — 주식 먼저, 그다음 선물/현물
set -u
cd "$(dirname "$0")"

LOG=pipeline_driver.log
: > "$LOG"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

wait_for_pid() {
  local pid=$1
  local name=$2
  log "waiting for $name (PID $pid)..."
  while kill -0 "$pid" 2>/dev/null; do sleep 30; done
  log "$name PID $pid done"
}

run_step() {
  local step=$1
  shift
  log "STEP $step start: $*"
  if "$@" >> "$LOG" 2>&1; then
    log "STEP $step ok"
    touch "stage_done_${step}.flag"
  else
    log "STEP $step FAILED (exit $?)"
    return 1
  fi
}

# 0. v17 iter 완료 대기 — raw_combined.csv 생성까지 polling
# (단순 PID 감시는 restart/kill 시 거짓 완료로 스킵할 위험 있음)
log "waiting for v17_snap_v2_out/raw_combined.csv ..."
while [ ! -s v17_snap_v2_out/raw_combined.csv ]; do
  sleep 60
done
log "v17 raw_combined.csv ready ($(wc -l < v17_snap_v2_out/raw_combined.csv) rows)"

# 주식 파이프
run_step "stk_top500" python3 redesign_extract_top500.py --top 500
run_step "stk_phase"  python3 redesign_rerank_phase.py --asset stock --top 500 --workers 24
run_step "stk_filter" python3 redesign_filter_phase.py --asset stock
run_step "stk_nudge"  python3 redesign_snap_nudge.py --asset stock --workers 24
run_step "stk_yearly" python3 redesign_yearly.py --asset stock --workers 24
run_step "stk_rank1"    python3 redesign_analyze.py --asset stock
run_step "stk_plateau"  python3 redesign_plateau.py --asset stock --top 25 --workers 24
run_step "stk_bootstrap" python3 redesign_bootstrap.py --asset stock
run_step "stk_rank2"    python3 redesign_analyze.py --asset stock
run_step "stk_ens"      python3 redesign_ensemble_bt.py --asset stock --workers 24 --flush 50
run_step "stk_str"      python3 redesign_stress.py --asset stock --top 25 --workers 24
run_step "stk_rank3"    python3 redesign_analyze.py --asset stock
run_step "stk_drop1"    python3 redesign_stock_drop1.py --top 3  # 엔진 drop_top 과 비교용
run_step "stk_report"   python3 redesign_report.py --asset stock
touch "pipe_stock_done.flag"
log "STOCK PIPELINE DONE"

# 코인 (선물+현물) rebuild_univ3 → 파이프
run_step "coin_rebuild" python3 redesign_rebuild_univ3.py --workers 24

# rebuild_univ3 output 을 자산별로 분리 (redesign_univ3_raw.csv → raw_{asset}.csv)
python3 -c "
import pandas as pd, os
if os.path.exists('redesign_univ3_raw.csv'):
    df = pd.read_csv('redesign_univ3_raw.csv')
    for a in ('fut', 'spot'):
        sub = df[df['asset']==a]
        sub.to_csv(f'redesign_univ3_raw_{a}.csv', index=False)
        print(f'{a}: {len(sub)} rows')
" >> "$LOG" 2>&1

for asset in fut spot; do
  run_step "${asset}_top500" python3 redesign_extract_top500.py --top 500
  run_step "${asset}_phase"  python3 redesign_rerank_phase.py --asset $asset --top 500 --workers 24
  run_step "${asset}_filter" python3 redesign_filter_phase.py --asset $asset
  run_step "${asset}_nudge"  python3 redesign_snap_nudge.py --asset $asset --workers 24
  run_step "${asset}_yearly" python3 redesign_yearly.py --asset $asset --workers 24
  run_step "${asset}_rank1"    python3 redesign_analyze.py --asset $asset
  run_step "${asset}_plateau"  python3 redesign_plateau.py --asset $asset --top 25 --workers 24
  run_step "${asset}_bootstrap" python3 redesign_bootstrap.py --asset $asset
  run_step "${asset}_rank2"    python3 redesign_analyze.py --asset $asset
  run_step "${asset}_ens"      python3 redesign_ensemble_bt.py --asset $asset --workers 24 --flush 50
  run_step "${asset}_str"      python3 redesign_stress.py --asset $asset --top 25 --workers 24
  run_step "${asset}_rank3"    python3 redesign_analyze.py --asset $asset
  run_step "${asset}_report"   python3 redesign_report.py --asset $asset
done
touch "pipe_coin_done.flag"
log "COIN PIPELINE DONE"

log "ALL DONE"
touch "pipe_all_done.flag"

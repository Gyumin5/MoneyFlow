#!/bin/bash
# 주식 grid v3 재실행 (확장 axes: select_family, sharpe_lookback, mom_style)
cd "$(dirname "$0")"
LOG=stock_grid_v3.log
: > "$LOG"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

log "=== STAGE 1: 코인 rebuild 완료 대기 ==="
TARGET=121718
while true; do
  N=$(wc -l < redesign_univ3_raw.csv 2>/dev/null || echo 0)
  if [ "$N" -ge "$TARGET" ]; then
    log "coin rebuild done ($N rows)"
    break
  fi
  log "waiting coin... $N / $TARGET"
  sleep 300
done

log "=== STAGE 2: 기존 v17 iter csv 삭제 (grid 달라짐) ==="
# 주식 이전 결과 백업 + 삭제
mkdir -p v17_snap_v2_out_backup_$(date +%Y%m%d_%H%M%S)
mv v17_snap_v2_out/iter_*.csv v17_snap_v2_out/all_iters.csv v17_snap_v2_out/raw_combined.csv v17_snap_v2_out_backup_*/ 2>/dev/null
rm -f redesign_top500_stock_k1.csv redesign_rerank_phase_stock.csv redesign_phase_survivors_stock.csv redesign_snap_nudge_stock.csv redesign_yearly_stock.csv redesign_rank_stock.csv redesign_ensemble_candidates_stock.csv redesign_ensemble_bt_stock.csv redesign_stress_stock.csv redesign_plateau_stock.csv redesign_plateau_agg_stock.csv redesign_bootstrap_stock.csv redesign_stock_drop1.csv redesign_report_stock.md

log "=== STAGE 3: v17_snap_iter_v2 (expanded grid) ==="
python3 -u v17_snap_iter_v2.py >> "$LOG" 2>&1
V17_EXIT=$?
log "v17 iter exit $V17_EXIT"

if [ "$V17_EXIT" -ne 0 ] || [ ! -s v17_snap_v2_out/raw_combined.csv ]; then
  log "v17 iter failed or no output — abort"
  exit 1
fi

log "=== STAGE 4: 주식 Step 2~8 파이프 ==="
python3 redesign_extract_top500.py --top 500 >> "$LOG" 2>&1
python3 redesign_rerank_phase.py --asset stock --top 500 --workers 24 >> "$LOG" 2>&1
python3 redesign_filter_phase.py --asset stock >> "$LOG" 2>&1
python3 redesign_snap_nudge.py --asset stock --workers 24 >> "$LOG" 2>&1
python3 redesign_yearly.py --asset stock --workers 24 >> "$LOG" 2>&1
python3 redesign_analyze.py --asset stock >> "$LOG" 2>&1
python3 redesign_plateau.py --asset stock --top 25 --workers 24 >> "$LOG" 2>&1
python3 redesign_bootstrap.py --asset stock >> "$LOG" 2>&1
python3 redesign_analyze.py --asset stock >> "$LOG" 2>&1
python3 redesign_ensemble_bt.py --asset stock --workers 24 --flush 50 >> "$LOG" 2>&1
python3 redesign_stress.py --asset stock --top 25 --workers 24 >> "$LOG" 2>&1
python3 redesign_analyze.py --asset stock >> "$LOG" 2>&1
python3 redesign_stock_drop1.py --top 3 >> "$LOG" 2>&1
python3 redesign_report.py --asset stock >> "$LOG" 2>&1

log "=== STAGE 5: ALL DONE ==="
touch stock_grid_v3_done.flag

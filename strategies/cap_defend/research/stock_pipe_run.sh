#!/bin/bash
cd "$(dirname "$0")"
set -e
LOG=stock_pipe_run.log
: > "$LOG"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

log "START stock pipeline rerun (sharpe_lookback=252 fix)"

run() {
  log "$@"
  "$@" >> "$LOG" 2>&1
  log "  → done exit=$?"
}

run python redesign_rerank_phase.py --asset stock --top 500 --workers 4
run python redesign_filter_phase.py --asset stock
run python redesign_snap_nudge.py --asset stock --workers 4
run python redesign_yearly.py --asset stock --workers 4
run python redesign_analyze.py --asset stock
run python redesign_plateau.py --asset stock --top 25 --workers 4
run python redesign_bootstrap.py --asset stock
run python redesign_analyze.py --asset stock
run python redesign_ensemble_bt.py --asset stock --workers 4 --flush 50
run python redesign_stress.py --asset stock --top 25 --workers 4
run python redesign_analyze.py --asset stock
run python redesign_stock_drop1.py --top 3
run python redesign_report.py --asset stock

log "ALL DONE"
touch stage_done_stk_rerun_complete.flag

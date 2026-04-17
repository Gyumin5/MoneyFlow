#!/usr/bin/env bash
# v5 chain: Phase-1 v2 완료 대기 → Phase-2 plateau (+CAGR floor)
#        → Phase-2 equity dump (단일 앵커, OOS 없음)
#        → Phase-3 ensemble
#        → Phase-4 3-asset
#        → post_analysis
#
# 사용자 요청:
#   - Phase-1 v2 확장 sweep 결과 사용
#   - Phase-2 OOS/앵커테스트 스킵, plateau + CAGR 하한만 적용
#   - 전체 자동 실행, 텔레그램 진행 알림

set -euo pipefail

REPO=/home/gmoh/mon/251229
CD=$REPO/strategies/cap_defend
RES=$CD/research
NOTIFY="python3 $RES/notify_telegram.py"
PY=python3

LOCK=$RES/.phase_chain_v5.lock
LOGDIR=$RES/phase_chain_logs
mkdir -p "$LOGDIR"

P1_DIR=$RES/phase1_v2_sweep
P1_MF=$P1_DIR/manifest.json
P1_SUMMARY=$P1_DIR/summary.csv

P2P_DIR=$RES/phase2_extract_v2
P2P_SURVIVORS=$P2P_DIR/survivors.csv

P2E_DIR=$RES/phase2_equity_dump
P2E_SUMMARY=$P2E_DIR/summary.csv
P2E_EQUITY=$P2E_DIR/equity

P3_DIR=$RES/phase3_ensembles_v3
P3_SPOT_TOP=$P3_DIR/spot_top.csv
P3_FUT_TOP=$P3_DIR/fut_top.csv

P4_DIR=$RES/phase4_3asset_v3
P4_RAW=$P4_DIR/raw.csv

WORKERS=${WORKERS:-24}
CAGR_FLOOR_L1=${CAGR_FLOOR_L1:-0.40}

log() { echo "[$(date +%F\ %T)] $*"; }
notify() { echo "$*" | $NOTIFY 2>/dev/null || true; }

cleanup() {
  if [[ -n "${LOCK_FD:-}" ]]; then
    flock -u "$LOCK_FD" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'notify "[Chain v5] FATAL: line $LINENO 실패. log: $LOGDIR"; exit 1' ERR

acquire_lock() {
  exec {LOCK_FD}>"$LOCK"
  if ! flock -n "$LOCK_FD"; then
    log "다른 v5 체인 인스턴스 실행중. 종료."
    exit 0
  fi
}

wait_for_phase1() {
  log "Phase-1 v2 sweep 완료 대기 ($P1_MF)"
  notify "[Chain v5] 시작. Phase-1 v2 sweep 완료 대기 중..."
  while true; do
    if [[ -f $P1_MF ]]; then
      local st=$($PY -c "import json; print(json.load(open('$P1_MF')).get('status',''))" 2>/dev/null || echo "")
      if [[ "$st" == "done" ]]; then
        log "Phase-1 v2 done 감지"
        return
      fi
    fi
    sleep 300
  done
}

run_phase2_plateau() {
  log "Phase-2 plateau + CAGR floor (L1=$CAGR_FLOOR_L1) 시작"
  notify "[Chain v5] Phase-2 plateau 필터 시작 (CAGR L1>=${CAGR_FLOOR_L1}, 선물은 선형 ×lev)"
  $PY $RES/phase2_extract_v2.py \
    --phase1-dir $P1_DIR \
    --out-dir $P2P_DIR \
    --cagr-floor-l1 $CAGR_FLOOR_L1 \
    --top-k 300 \
    >> $LOGDIR/phase2_plateau_v2.log 2>&1
  if [[ ! -s $P2P_SURVIVORS ]]; then
    notify "[Chain v5] FAIL Phase-2 plateau: survivors 없음"
    exit 1
  fi
  local n=$(( $(wc -l < $P2P_SURVIVORS) - 1 ))
  notify "[Chain v5] Phase-2 plateau 완료. survivors=$n"
}

run_phase2_equity_dump() {
  log "Phase-2 equity dump 시작 (단일 앵커)"
  notify "[Chain v5] Phase-2 equity dump 시작 (plateau survivor 전부 단일 앵커 실행)"
  $PY $RES/phase2_equity_dump.py \
    --survivors $P2P_SURVIVORS \
    --out-dir $P2E_DIR \
    --processes $WORKERS \
    >> $LOGDIR/phase2_equity_dump.log 2>&1
  if [[ ! -s $P2E_SUMMARY ]]; then
    notify "[Chain v5] FAIL Phase-2 equity dump"
    exit 1
  fi
  notify "[Chain v5] Phase-2 equity dump 완료"
}

run_phase3() {
  log "Phase-3 시작"
  notify "[Chain v5] Phase-3 앙상블 + corr 게이트 시작"
  $PY $RES/phase3_ensemble.py \
    --phase2-summary $P2E_SUMMARY \
    --equity-dir $P2E_EQUITY \
    --out-dir $P3_DIR \
    --abs-corr-cutoff 0.85 \
    --co-dd-cutoff 0.30 \
    --crisis-corr-cutoff 0.90 \
    --top-n 5 \
    --processes $WORKERS \
    >> $LOGDIR/phase3_v3.log 2>&1
  if [[ ! -f $P3_SPOT_TOP ]] || [[ ! -f $P3_FUT_TOP ]]; then
    notify "[Chain v5] FAIL Phase-3: top 없음"
    exit 1
  fi
  notify "[Chain v5] Phase-3 완료"
}

run_phase4() {
  log "Phase-4 시작"
  notify "[Chain v5] Phase-4 3-asset 시작"
  $PY $RES/phase4_3asset.py \
    --spot-top $P3_SPOT_TOP \
    --fut-top $P3_FUT_TOP \
    --out-dir $P4_DIR \
    >> $LOGDIR/phase4_v3.log 2>&1
  if [[ ! -s $P4_RAW ]]; then
    notify "[Chain v5] FAIL Phase-4"
    exit 1
  fi
  notify "[Chain v5] Phase-4 완료"
}

run_post_analysis() {
  log "post_analysis 시작"
  notify "[Chain v5] post_analysis 시작 (top-10 ranking union, 연도/위기/sensitivity)"
  $PY $RES/post_analysis_v4.py \
    --phase4-dir $P4_DIR \
    --phase3-dir $P3_DIR \
    --out-dir $RES/post_analysis_v5 \
    --top-n 10 \
    >> $LOGDIR/post_analysis_v5.log 2>&1

  local SUM=$RES/post_analysis_v5/summary.txt
  if [[ -f $SUM ]]; then
    $PY - <<PY
import os, subprocess
sum_path = "$SUM"
notify = "$RES/notify_telegram.py"
with open(sum_path) as f:
    text = f.read()
chunks, cur = [], ""
for line in text.split("\n"):
    if len(cur) + len(line) + 1 > 3500:
        chunks.append(cur); cur = ""
    cur += line + "\n"
if cur: chunks.append(cur)
for i, c in enumerate(chunks):
    header = f"[v5 사후분석 {i+1}/{len(chunks)}]\n"
    subprocess.run(["python3", notify], input=(header + c).encode(), check=False)
PY
  fi
  notify "[Chain v5] DONE. 결과: $RES/post_analysis_v5/"
}

main() {
  acquire_lock
  wait_for_phase1
  run_phase2_plateau
  run_phase2_equity_dump
  run_phase3
  run_phase4
  run_post_analysis
  log "v5 chain 완료"
}

main "$@"

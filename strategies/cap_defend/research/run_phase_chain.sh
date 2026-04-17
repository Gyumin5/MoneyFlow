#!/usr/bin/env bash
# Phase-1 → 2a → 2b → 3 → 4 → 5 orchestrator (v3).
#
# - lockfile 로 중복 실행 방지
# - Phase-1 완료 감지: phase1_sweep/manifest.json 의 status=done AND done_tasks=total_tasks
#   (PID 감시 대신 manifest polling — 재시작/재개에도 안전)
# - 각 phase 완료 후 다음 phase 시작
# - 각 phase 완료/시작/실패 시 텔레그램 알림
# - 매 단계 후 build_status_html.py 갱신
# - 30분마다 진행상황 notifier 프로세스(progress_notifier.sh) 병행

set -euo pipefail

REPO=/home/gmoh/mon/251229
CD=$REPO/strategies/cap_defend
RES=$CD/research
NOTIFY="python3 $RES/notify_telegram.py"
PY=python3

LOCK=$RES/.phase_chain.lock
LOGDIR=$RES/phase_chain_logs
mkdir -p "$LOGDIR"

ITER_DIR=$RES/iter_refine
ITER_RAW=$ITER_DIR/raw_combined.csv
P1_DIR=$RES/phase1_sweep
P2E_DIR=$RES/phase2_extract
PLATEAU_DIR=$RES/plateau_check
P3_DIR_30=$RES/phase3_ensembles_floor30
P3_DIR_40=$RES/phase3_ensembles_floor40
P4_DIR_30=$RES/phase4_3asset_floor30
P4_DIR_40=$RES/phase4_3asset_floor40
ROB_DIR_30=$RES/robustness_floor30
ROB_DIR_40=$RES/robustness_floor40

P1_MANIFEST=$P1_DIR/manifest.json
P2E_SURV=$P2E_DIR/survivors.csv

WORKERS=${WORKERS:-24}

NOTIFIER_PID=""

log() { echo "[$(date +%F\ %T)] $*"; }
notify() { echo "$*" | $NOTIFY 2>/dev/null || true; }

cleanup() {
  if [[ -n "$NOTIFIER_PID" ]] && kill -0 "$NOTIFIER_PID" 2>/dev/null; then
    kill "$NOTIFIER_PID" 2>/dev/null || true
    log "Stopped progress_notifier (pid=$NOTIFIER_PID)"
  fi
  if [[ -n "${LOCK_FD:-}" ]]; then
    flock -u "$LOCK_FD" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'notify "[Chain v3] FATAL: line $LINENO 실패. log 확인: $LOGDIR"; exit 1' ERR

acquire_lock() {
  exec {LOCK_FD}>"$LOCK"
  if ! flock -n "$LOCK_FD"; then
    log "다른 run_phase_chain.sh 인스턴스가 실행 중. 종료."
    exit 0
  fi
  log "Lock acquired: $LOCK"
}

start_notifier() {
  if pgrep -f "progress_notifier.sh" > /dev/null; then
    log "progress_notifier 이미 실행중"
    return
  fi
  nohup bash $RES/progress_notifier.sh 1800 > /tmp/progress_notifier.log 2>&1 &
  NOTIFIER_PID=$!
  log "Started progress_notifier (pid=$NOTIFIER_PID)"
}

render_html() {
  $PY $RES/build_status_html.py >> $LOGDIR/html.log 2>&1 || true
}

wait_manifest_done() {
  # $1 = manifest path, $2 = stage label, $3 = fallback raw.csv, $4 = fallback target rows
  local mf=$1 stage=$2 fb_raw=${3:-} fb_target=${4:-0}
  log "[$stage] manifest polling: $mf (fallback raw=$fb_raw target=$fb_target)"
  while true; do
    if [[ -f $mf ]]; then
      local done_status
      done_status=$($PY - <<EOF 2>/dev/null
import json, sys
try:
    d = json.load(open("$mf"))
except Exception:
    print("err"); sys.exit(0)
status = d.get("status","")
tot = int(d.get("total_tasks",0) or 0)
don = int(d.get("done_tasks",0) or 0)
if status == "done" and tot > 0 and don >= tot:
    print("done")
elif status == "done" and tot == 0:
    print("done")
else:
    print(f"running {don}/{tot}")
EOF
)
      if [[ "$done_status" == "done" ]]; then
        log "[$stage] manifest done."
        return 0
      fi
    fi
    # Fallback: raw.csv row count ≥ target (manifest 없는 Phase-1 대응)
    if [[ -n "$fb_raw" && "$fb_target" -gt 0 && -s "$fb_raw" ]]; then
      local rows=$(( $(wc -l < "$fb_raw") - 1 ))
      if [[ $rows -ge $fb_target ]]; then
        log "[$stage] raw.csv fallback: $rows >= $fb_target — manifest 작성 후 진행"
        $PY - <<EOF
import json, os, tempfile
p = "$mf"
tmp = p + ".tmp"
data = {"status":"done","total_tasks":$fb_target,"done_tasks":$rows,"note":"backfilled by orchestrator"}
with open(tmp,"w") as f: json.dump(data,f)
os.replace(tmp,p)
EOF
        return 0
      fi
    fi
    render_html
    sleep 300
  done
}

run_phase2_extract() {
  log "Phase-2a extract start"
  notify "[Chain v3] Phase-2a axis-neighbor plateau 시작"
  $PY $RES/phase2_extract.py \
    --phase1-dir $P1_DIR \
    --out-dir $P2E_DIR \
    --top-k 100 \
    >> $LOGDIR/phase2_extract.log 2>&1
  if [[ ! -s $P2E_SURV ]]; then
    notify "[Chain v3] FAIL Phase-2a: survivors 비어있음"
    exit 1
  fi
  local n=$(( $(wc -l < "$P2E_SURV") - 1 ))
  notify "[Chain v3] Phase-2a 완료. survivors=$n"
  render_html
}

run_phase3_single() {
  # $1 = floor (e.g. 0.30), $2 = out_dir
  local floor=$1 outdir=$2
  log "Phase-3 ensemble start (floor=$floor → $outdir)"
  notify "[Chain v3] Phase-3 ensemble 시작 (floor=$floor)"
  $PY $RES/phase3_ensemble.py \
    --phase2-summary $P1_DIR/summary.csv \
    --out-dir $outdir \
    --top-n 5 \
    --pool-per-metric 5 \
    --cagr-floor-per-lev $floor \
    --processes $WORKERS \
    >> $LOGDIR/phase3_floor${floor}.log 2>&1
  if [[ ! -f $outdir/spot_top.csv ]] || [[ ! -f $outdir/fut_top.csv ]]; then
    notify "[Chain v3] FAIL Phase-3 (floor=$floor): spot_top/fut_top 없음"
    exit 1
  fi
  notify "[Chain v3] Phase-3 완료 (floor=$floor)"
}

run_phase3() {
  run_phase3_single 0.30 $P3_DIR_30
  run_phase3_single 0.40 $P3_DIR_40
  render_html
}

run_phase4_single() {
  local p3dir=$1 outdir=$2 label=$3
  log "Phase-4 3-asset start ($label)"
  notify "[Chain v3] Phase-4 3-asset mix 시작 ($label)"
  $PY $RES/phase4_3asset.py \
    --spot-top $p3dir/spot_top.csv \
    --fut-top $p3dir/fut_top.csv \
    --out-dir $outdir \
    >> $LOGDIR/phase4_${label}.log 2>&1
  if [[ ! -s $outdir/raw.csv ]]; then
    notify "[Chain v3] FAIL Phase-4 ($label): raw.csv 없음"
    exit 1
  fi
  notify "[Chain v3] Phase-4 완료 ($label)"
}

run_phase4() {
  run_phase4_single $P3_DIR_30 $P4_DIR_30 floor30
  run_phase4_single $P3_DIR_40 $P4_DIR_40 floor40
  render_html
}

run_robustness_single() {
  local p3dir=$1 outdir=$2 label=$3
  log "robustness_check start ($label)"
  notify "[Chain v3] robustness_check 시작 ($label): LOYO 5년 + LOAO 전체 코인"
  $PY $RES/robustness_check.py \
    --p3-dir $p3dir \
    --out-dir $outdir \
    --top-n 100 \
    --workers $WORKERS \
    >> $LOGDIR/robustness_${label}.log 2>&1
  if [[ ! -s $outdir/summary.csv ]]; then
    notify "[Chain v3] WARN robustness ($label): summary.csv 없음 (계속 진행)"
    return 0
  fi
  notify "[Chain v3] robustness_check 완료 ($label)"
}

run_robustness() {
  run_robustness_single $P3_DIR_30 $ROB_DIR_30 floor30
  run_robustness_single $P3_DIR_40 $ROB_DIR_40 floor40
  render_html
}

wait_iter_refine_done() {
  # iter_refine 은 raw_combined.csv 생성 = 종료 신호
  log "[iter_refine] waiting for $ITER_RAW"
  while true; do
    if [[ -s "$ITER_RAW" ]]; then
      # 프로세스가 아직 돌고 있으면 아직 부분 쓰기일 수 있음 — 안전차 pgrep
      if pgrep -f "iter_refine.py" > /dev/null; then
        render_html
        sleep 120
        continue
      fi
      log "[iter_refine] done: $(wc -l < "$ITER_RAW") rows"
      return 0
    fi
    render_html
    sleep 120
  done
}

run_plateau() {
  log "[plateau_check] start"
  notify "[Chain v3] plateau_check 시작 (버킷별 top-100 × ±5/±10%)"
  $PY $RES/plateau_check.py \
    --raw "$ITER_RAW" \
    --out-dir "$PLATEAU_DIR" \
    --top-n 100 \
    --workers $WORKERS \
    --pass-ratio 0.85 \
    >> $LOGDIR/plateau_check.log 2>&1
  if [[ ! -s $PLATEAU_DIR/survivors.csv ]]; then
    notify "[Chain v3] FAIL plateau_check: survivors.csv 없음"
    exit 1
  fi
  local n=$(( $(wc -l < $PLATEAU_DIR/survivors.csv) - 1 ))
  notify "[Chain v3] plateau_check 완료. survivors=$n"
}

run_bridge() {
  log "[bridge] plateau_check → phase1_sweep"
  notify "[Chain v3] bridge 시작"
  $PY $RES/bridge_iter_to_phase1.py \
    --raw-combined "$PLATEAU_DIR/survivors.csv" \
    --out-dir "$P1_DIR" \
    >> $LOGDIR/bridge.log 2>&1
  if [[ ! -s $P1_DIR/summary.csv ]]; then
    notify "[Chain v3] FAIL bridge: summary.csv 없음"
    exit 1
  fi
  notify "[Chain v3] bridge 완료"
}

main() {
  acquire_lock
  notify "[Chain v3] orchestrator 시작. iter_refine 대기."
  start_notifier
  render_html
  wait_iter_refine_done
  run_plateau
  run_bridge
  run_phase3
  run_robustness
  run_phase4
  notify "[Chain v3] DONE: Phase 1~4 + robustness 완료."
}

main "$@"

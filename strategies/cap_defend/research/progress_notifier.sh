#!/usr/bin/env bash
# 30분 주기로 현재 진행 상황 텔레그램 알림 (iter_refine → phase3 → phase4).

set -uo pipefail

REPO=/home/gmoh/mon/251229
RES=$REPO/strategies/cap_defend/research
NOTIFY="python3 $RES/notify_telegram.py"

ITER_DIR=$RES/iter_refine
P1_DIR=$RES/phase1_sweep
P3_DIR=$RES/phase3_ensembles
P4_DIR=$RES/phase4_3asset

INTERVAL=${1:-1800}

while true; do
  msg="[Progress] $(date +%H:%M)"

  # iter_refine: running일 때만 보고 (완료 후 idle 스팸 방지)
  if [[ -d $ITER_DIR ]] && pgrep -f "iter_refine.py" > /dev/null; then
    last_stage=$(ls -d $ITER_DIR/stage_* 2>/dev/null | sort -V | tail -1)
    if [[ -n $last_stage ]]; then
      sn=$(basename $last_stage | sed 's/stage_//')
      if [[ -f $last_stage/raw.csv ]]; then
        rows=$(($(wc -l < $last_stage/raw.csv) - 1))
        msg+=$'\n'"iter_refine stage$sn: $rows rows (running)"
      fi
      last_log=$(tail -1 $ITER_DIR/run_local.log 2>/dev/null | grep -oE '\[[0-9]+/[0-9]+\].*' | head -c 80)
      [[ -n $last_log ]] && msg+=$'\n'"  $last_log"
    fi
  fi

  # phase3 진행 (running 중이면 procs 표시)
  if pgrep -f "phase3_ensemble.py" > /dev/null; then
    p3w=$(pgrep -f "phase3_ensemble.py" | wc -l)
    msg+=$'\n'"phase3: running ($p3w procs)"
  fi

  # phase4 진행
  if pgrep -f "phase4_3asset.py" > /dev/null; then
    msg+=$'\n'"phase4: running"
  fi

  # phase3
  if [[ -f $P3_DIR/all_combos.csv ]]; then
    p3=$(($(wc -l < $P3_DIR/all_combos.csv) - 1))
    msg+=$'\n'"phase3: $p3 combos"
  fi

  # phase4
  if [[ -f $P4_DIR/raw.csv ]]; then
    p4=$(($(wc -l < $P4_DIR/raw.csv) - 1))
    msg+=$'\n'"phase4: $p4 rows"
  fi

  echo "$msg" | $NOTIFY || true
  sleep $INTERVAL
done

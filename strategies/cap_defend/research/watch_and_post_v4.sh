#!/usr/bin/env bash
# v4 chain (Phase-4 v2) 완료 대기 → post_analysis_v4 자동 실행 → 텔레그램 보고.
set -euo pipefail

REPO=/home/gmoh/mon/251229
RES=$REPO/strategies/cap_defend/research
LOG=$RES/phase_chain_logs/watch_post_v4.log
P4_MF=$RES/phase4_3asset_v2/manifest.json
NOTIFY="python3 $RES/notify_telegram.py"

mkdir -p "$RES/phase_chain_logs"
exec >>"$LOG" 2>&1

echo "[$(date +%F\ %T)] watcher 시작 (P4 manifest 대기)"

# 최대 4시간 대기
for i in $(seq 1 480); do
  if [[ -f $P4_MF ]]; then
    st=$(python3 -c "import json; print(json.load(open('$P4_MF')).get('status',''))" 2>/dev/null || echo "")
    if [[ "$st" == "done" ]]; then
      echo "[$(date +%F\ %T)] Phase-4 v2 done 감지"
      break
    fi
  fi
  sleep 30
done

if [[ ! -f $P4_MF ]] || [[ "$(python3 -c "import json; print(json.load(open('$P4_MF')).get('status',''))" 2>/dev/null)" != "done" ]]; then
  echo "[$(date +%F\ %T)] timeout/실패. 종료."
  echo "[Watcher] Phase-4 v2 완료 못 함. 수동 확인 필요." | $NOTIFY 2>/dev/null || true
  exit 1
fi

echo "[$(date +%F\ %T)] post_analysis_v4 실행"
echo "[Watcher] Phase-4 v2 완료. 사후분석 시작 (top10 후보 연도/위기/sensitivity)" | $NOTIFY 2>/dev/null || true

if python3 $RES/post_analysis_v4.py --top-n 10; then
  SUM=$RES/post_analysis_v4/summary.txt
  if [[ -f $SUM ]]; then
    # 텔레그램 4096자 제한 분할
    python3 - <<'PY'
import os, subprocess
sum_path = "/home/gmoh/mon/251229/strategies/cap_defend/research/post_analysis_v4/summary.txt"
notify = "/home/gmoh/mon/251229/strategies/cap_defend/research/notify_telegram.py"
with open(sum_path) as f:
    text = f.read()
chunks = []
cur = ""
for line in text.split("\n"):
    if len(cur) + len(line) + 1 > 3500:
        chunks.append(cur)
        cur = ""
    cur += line + "\n"
if cur:
    chunks.append(cur)
for i, c in enumerate(chunks):
    header = f"[사후분석 {i+1}/{len(chunks)}]\n"
    subprocess.run(["python3", notify], input=(header + c).encode(), check=False)
PY
  fi
  echo "[Watcher] 사후분석 완료. report: $RES/post_analysis_v4/" | $NOTIFY 2>/dev/null || true
  echo "[$(date +%F\ %T)] 사후분석 완료"
else
  echo "[$(date +%F\ %T)] post_analysis 실패"
  echo "[Watcher] 사후분석 실행 실패. 로그: $LOG" | $NOTIFY 2>/dev/null || true
  exit 1
fi

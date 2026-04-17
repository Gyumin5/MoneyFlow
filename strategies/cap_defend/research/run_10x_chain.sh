#!/usr/bin/env bash
# Phase-1 10x → Phase-2 → Phase-3 → Phase-4 체인.
# Phase-1(phase1_10x)이 완료된 후 순차 실행.

set -euo pipefail
cd /home/gmoh/mon/251229/strategies/cap_defend/research

LOG=phase_chain_logs/10x_chain_$(date +%H%M).log
mkdir -p phase_chain_logs phase2_10x phase3_10x phase4_10x

echo "[10x_chain] start $(date +%T)" | tee -a "$LOG"

# Phase-1 완료 대기
while true; do
  if [[ -f phase1_10x/manifest.json ]]; then
    s=$(python3 -c "import json;print(json.load(open('phase1_10x/manifest.json'))['status'])" 2>/dev/null || echo "")
    [[ "$s" == "done" ]] && break
  fi
  sleep 60
done
echo "[10x_chain] phase1 done $(date +%T)" | tee -a "$LOG"

# Phase-2
echo "[10x_chain] phase2 start" | tee -a "$LOG"
python3 phase2_extract.py \
  --phase1-dir phase1_10x \
  --out-dir phase2_10x \
  --top-k 75 2>&1 | tee -a "$LOG"

# phase2_10x/survivors.csv → phase3 입력
echo "[10x_chain] phase3 start" | tee -a "$LOG"
python3 phase3_ensemble.py \
  --phase2-summary phase2_10x/survivors.csv \
  --out-dir phase3_10x \
  --top-n 5 --pool-per-metric 5 \
  --processes 24 2>&1 | tee -a "$LOG"

# Phase-4
echo "[10x_chain] phase4 start" | tee -a "$LOG"
python3 phase4_3asset.py \
  --spot-top phase3_10x/spot_top.csv \
  --fut-top phase3_10x/fut_top.csv \
  --out-dir phase4_10x 2>&1 | tee -a "$LOG"

echo "[10x_chain] DONE $(date +%T)" | tee -a "$LOG"

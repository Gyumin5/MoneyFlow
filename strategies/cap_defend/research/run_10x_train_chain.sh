#!/usr/bin/env bash
# Holdout-train chain: Phase-2 → Phase-3 → Phase-4 on phase1_10x_train results.
# All phases run with PHASE_END=2023-12-31 (train-only).

set -euo pipefail
cd /home/gmoh/mon/251229/strategies/cap_defend/research

LOG=phase_chain_logs/10x_train_chain_$(date +%H%M).log
mkdir -p phase_chain_logs phase2_10x_train phase3_10x_train phase4_10x_train

export PHASE_END="2023-12-31"
echo "[train_chain] start $(date +%T) PHASE_END=$PHASE_END" | tee -a "$LOG"

# Wait for Phase-1 train done
while true; do
  if [[ -f phase1_10x_train/manifest.json ]]; then
    s=$(python3 -c "import json;print(json.load(open('phase1_10x_train/manifest.json'))['status'])" 2>/dev/null || echo "")
    [[ "$s" == "done" ]] && break
  fi
  sleep 120
done
echo "[train_chain] phase1 done $(date +%T)" | tee -a "$LOG"

# Phase-2
echo "[train_chain] phase2 start" | tee -a "$LOG"
PHASE_END="$PHASE_END" python3 phase2_extract.py \
  --phase1-dir phase1_10x_train \
  --out-dir phase2_10x_train \
  --top-k 75 2>&1 | tee -a "$LOG"

# Phase-3
echo "[train_chain] phase3 start" | tee -a "$LOG"
PHASE_END="$PHASE_END" python3 phase3_ensemble.py \
  --phase2-summary phase2_10x_train/survivors.csv \
  --out-dir phase3_10x_train \
  --top-n 5 --pool-per-metric 5 \
  --processes 24 2>&1 | tee -a "$LOG"

# Phase-4
echo "[train_chain] phase4 start" | tee -a "$LOG"
PHASE_END="$PHASE_END" python3 phase4_3asset.py \
  --spot-top phase3_10x_train/spot_top.csv \
  --fut-top phase3_10x_train/fut_top.csv \
  --out-dir phase4_10x_train 2>&1 | tee -a "$LOG"

echo "[train_chain] DONE $(date +%T)" | tee -a "$LOG"

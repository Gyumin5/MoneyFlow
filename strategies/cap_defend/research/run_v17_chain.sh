#!/usr/bin/env bash
# V17 Phase 체인 러너.
# 전제: Phase-1a (v17_snap_iter.py) 가 background 로 실행 중.
# 동작: top_peaks.csv 대기 → Phase-1b → Phase-2 → Phase-3 순차 실행.

set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$HERE/v17_snap_out"
mkdir -p "$OUT"
cd "$HERE"

LOCK="$HERE/.phase_chain.lock"
if [ -e "$LOCK" ]; then
  echo "[chain] lock 이미 존재: $LOCK — 중복 실행 방지. 종료." >&2
  exit 1
fi
trap 'rm -f "$LOCK"' EXIT
: > "$LOCK"

PEAKS="$OUT/top_peaks.csv"

# === 1) Phase-1a 완료 대기 (최대 6시간) ===
echo "[chain] Phase-1a 완료 대기 (top_peaks.csv)..."
MAX_WAIT_SEC=$((6 * 3600))
WAITED=0
POLL=30
while [ ! -f "$PEAKS" ]; do
  if [ "$WAITED" -ge "$MAX_WAIT_SEC" ]; then
    echo "[chain] Phase-1a timeout ($MAX_WAIT_SEC sec). 종료." >&2
    exit 2
  fi
  sleep "$POLL"
  WAITED=$((WAITED + POLL))
done
echo "[chain] phase-1a done, starting 1b"
touch "$OUT/.phase_1a_done"

# === 2) Phase-1b ===
python3 "$HERE/v17_phase1b.py" 2>&1 | tee "$OUT/phase1b.log"
if [ ! -f "$OUT/phase1b_top.csv" ]; then
  echo "[chain] phase1b_top.csv 없음 — 중단." >&2
  exit 3
fi
touch "$OUT/.phase_1b_done"
echo "[chain] phase-1b done, starting 2"

# === 3) Phase-2 ===
python3 "$HERE/v17_phase2_plateau.py" 2>&1 | tee "$OUT/phase2.log"
if [ ! -f "$OUT/phase2_winners.csv" ]; then
  echo "[chain] phase2_winners.csv 없음 — 중단." >&2
  exit 4
fi
touch "$OUT/.phase_2_done"
echo "[chain] phase-2 done, starting 3"

# === 4) Phase-3 ===
python3 "$HERE/v17_phase3_robust.py" 2>&1 | tee "$OUT/phase3.log"
if [ ! -f "$OUT/phase3_final.csv" ]; then
  echo "[chain] phase3_final.csv 없음 — 중단." >&2
  exit 5
fi
touch "$OUT/.phase_3_done"
echo "[chain] phase-3 done, starting 4 (ensemble)"

# === 5) Phase-4 (EW 앙상블 k=1,2,3) ===
python3 "$HERE/v17_phase4_ensemble.py" 2>&1 | tee "$OUT/phase4.log"
touch "$OUT/.phase_4_done"
echo "[chain] 전체 완료. 결과: $OUT/phase4_ensemble.csv"

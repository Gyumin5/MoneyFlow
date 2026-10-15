#!/usr/bin/env bash
# 자산간 리밸런싱 조건 연구 — 본 채점 14칸 (앵커 7 × 비용 2).
# 한 칸이 941,968 후보 × 창 63개. 같은 기계를 다른 세션이 쓰고 있어 일꾼은 4로 묶는다.
set -u
cd /home/gmoh/mon/251229/strategies/cap_defend/research || exit 1
OUT=/home/gmoh/mon/251229/state/20261015-allocrebal
mkdir -p "$OUT/log"

JOBS="${JOBS:-4}"
E="${E:-113}"

for a in 0 1 2 3 4 5 6; do
  for c in base x5; do
    echo "$a $c"
  done
done | xargs -P "$JOBS" -n 2 bash -c '
  a="$0"; c="$1"
  OUT=/home/gmoh/mon/251229/state/20261015-allocrebal
  t0=$(date +%s)
  python3 -u alloc_grid_run.py --anchor "$a" --cost "$c" --e '"$E"' \
      > "$OUT/log/a${a}_${c}.log" 2>&1
  rc=$?
  echo "a${a} ${c} rc=${rc} $(( $(date +%s) - t0 ))s"
'

echo "=== 요약 ==="
for a in 0 1 2 3 4 5 6; do
  for c in base x5; do
    f="$OUT/grid_a${a}_${c}.json"
    if [ -f "$f" ]; then
      python3 -c "
import json,sys
m=json.load(open(sys.argv[1]))
b=m['base']
print('a%d %-4s n=%d  팔당%.2fms  기준 Cal %.3f MDD %.2f%% ht95 %.2fpp 송금%.2f/년'
      % (m['anchor'], 'x5' if m['stress'] else 'base', m['n'], m['팔당ms'],
         b['Cal'], b['MDD']*100, b['ht95']*100, b['per_yr']))" "$f"
    else
      echo "a${a} ${c} 결과 없음"
    fi
  done
done

#!/usr/bin/env bash
# 위약을 가족 하나만 돌린다. alloc_grid_placebo_main.sh 는 shift 를 다 돌고 나서야
# rand 로 넘어가는데, 집계는 두 가족이 각각 100벌이어야 판정이 된다 — 한쪽만 쌓이면
# 벌 수를 채워도 보정 문턱이 안 선다. 그래서 모자란 가족을 따로 띄울 수 있게 뗀다.
#
# 블록을 내림차순으로 내보낸다. 본 드라이버가 같은 가족에 도달하면 오름차순으로 집으므로
# 둘이 같은 블록에서 만나는 시점이 가장 늦어진다. 끝난 씨앗은 건너뛰지만, 지금 도는
# 중인 씨앗은 파일이 없어 못 건너뛴다 — 겹치면 그 한 벌만 두 번 돈다.
#
# 쓰기: KIND=rand JOBS=2 bash alloc_grid_placebo_fam.sh
set -u
cd /home/gmoh/mon/251229/strategies/cap_defend/research || exit 1
OUT=/home/gmoh/mon/251229/state/20261015-allocrebal
mkdir -p "$OUT/placebo" "$OUT/log"

KIND="${KIND:-rand}"
JOBS="${JOBS:-2}"
BLOCK="${BLOCK:-10}"
TOTAL="${TOTAL:-100}"
E="${E:-113}"

s=$((TOTAL - BLOCK))
while [ "$s" -ge 0 ]; do
  echo "$s"
  s=$((s - BLOCK))
done | xargs -P "$JOBS" -n 1 bash -c '
  s0="$0"
  OUT=/home/gmoh/mon/251229/state/20261015-allocrebal
  t0=$(date +%s)
  python3 -u alloc_grid_placebo.py --kind '"$KIND"' --seed0 "$s0" --count '"$BLOCK"' \
      --anchor 0 --cost base --e '"$E"' \
      > "$OUT/log/pl_'"$KIND"'_${s0}.log" 2>&1
  echo "'"$KIND"' s${s0} rc=$? $(( $(date +%s) - t0 ))s"
'

echo "=== $KIND 가족 완료 ==="
ls "$OUT/placebo" | grep -c "pl_${KIND}_" || true

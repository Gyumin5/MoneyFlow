#!/usr/bin/env bash
# alloc_grid_placebo_fam.sh 와 같은 일을 하되 블록을 오름차순으로 내보낸다.
#
# 쓰는 자리: 같은 가족에 이미 내림차순 일꾼이 붙어 있을 때 반대쪽 끝에서 채운다.
# 둘이 가운데서 만나는 시점이 가장 늦어져 겹치는 블록이 최소가 된다. 이미 끝난 씨앗은
# alloc_grid_placebo.py 가 결과 파일을 보고 건너뛴다 — 지금 도는 중인 씨앗만 못 건너뛰고,
# 겹치면 그 한 벌이 두 번 돈다.
#
# 쓰기: KIND=rand JOBS=3 bash alloc_grid_placebo_fam_up.sh
set -u
cd /home/gmoh/mon/251229/strategies/cap_defend/research || exit 1
OUT=/home/gmoh/mon/251229/state/20261015-allocrebal
mkdir -p "$OUT/placebo" "$OUT/log"

KIND="${KIND:-rand}"
JOBS="${JOBS:-3}"
BLOCK="${BLOCK:-10}"
TOTAL="${TOTAL:-100}"
E="${E:-113}"

s=0
while [ "$s" -lt "$TOTAL" ]; do
  echo "$s"
  s=$((s + BLOCK))
done | xargs -P "$JOBS" -n 1 bash -c '
  s0="$0"
  OUT=/home/gmoh/mon/251229/state/20261015-allocrebal
  t0=$(date +%s)
  python3 -u alloc_grid_placebo.py --kind '"$KIND"' --seed0 "$s0" --count '"$BLOCK"' \
      --anchor 0 --cost base --e '"$E"' \
      > "$OUT/log/pl_'"$KIND"'_up_${s0}.log" 2>&1
  echo "'"$KIND"' s${s0} rc=$? $(( $(date +%s) - t0 ))s"
'

echo "=== $KIND 가족 (오름차순) 완료 ==="
ls "$OUT/placebo" | grep -c "pl_${KIND}_" || true

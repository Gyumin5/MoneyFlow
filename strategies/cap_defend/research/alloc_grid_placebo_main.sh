#!/usr/bin/env bash
# 위약 200벌 (shift 100 + rand 100), 앵커 0 · 기본 비용 = 주 측정 칸.
# 한 벌이 941,968 후보. 블록 하나가 씨앗 10개를 한 프로세스에서 돌려 슬리브 적재를 아낀다.
set -u
cd /home/gmoh/mon/251229/strategies/cap_defend/research || exit 1
OUT=/home/gmoh/mon/251229/state/20261015-allocrebal
mkdir -p "$OUT/placebo" "$OUT/log"

JOBS="${JOBS:-6}"
BLOCK="${BLOCK:-10}"
TOTAL="${TOTAL:-100}"
E="${E:-113}"

for kind in shift rand; do
  s=0
  while [ "$s" -lt "$TOTAL" ]; do
    echo "$kind $s"
    s=$((s + BLOCK))
  done
done | xargs -P "$JOBS" -n 2 bash -c '
  kind="$0"; s0="$1"
  OUT=/home/gmoh/mon/251229/state/20261015-allocrebal
  t0=$(date +%s)
  python3 -u alloc_grid_placebo.py --kind "$kind" --seed0 "$s0" --count '"$BLOCK"' \
      --anchor 0 --cost base --e '"$E"' \
      > "$OUT/log/pl_${kind}_${s0}.log" 2>&1
  echo "${kind} s${s0} rc=$? $(( $(date +%s) - t0 ))s"
'

echo "=== 위약 요약 ==="
python3 - <<'PY'
import glob, json, os
import numpy as np
d = "/home/gmoh/mon/251229/state/20261015-allocrebal/placebo"
rows = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(d, "pl_*.json")))]
if not rows:
    print("결과 없음"); raise SystemExit
for kind in ("shift", "rand", "전체"):
    r = [x for x in rows if kind == "전체" or x["kind"] == kind]
    if not r:
        continue
    dc = np.array([x["best"]["dCal"] for x in r])
    dm = np.array([x["best"]["dMDD"] for x in r])
    dh = np.array([x["best"]["dht95"] for x in r])
    rk = np.array([x["best"]["rank"] for x in r])
    npass = np.array([x["npass"] for x in r])
    print(f"{kind:4s} n={len(r):3d}  칼마개선 95분위 {np.percentile(dc,95):+.3f} 최대 {dc.max():+.3f}"
          f"  낙폭개선 95분위 {np.percentile(dm,95)*100:+.2f}pp"
          f"  이탈개선 95분위 {np.percentile(dh,95)*100:+.2f}pp"
          f"  최소순위 하위5% {np.percentile(rk,5):.4f}"
          f"  전기준통과 최대 {int(npass.max())}")
PY

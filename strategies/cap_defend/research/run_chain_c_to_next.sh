#!/bin/bash
# Next strategies 5개 순차 실행 (C와 병렬 허용, 파일 충돌 없음).
#
# 사용:
#   nohup ./run_chain_c_to_next.sh > run_chain.log 2>&1 &
#
# 세션 종료(SSH 끊김) 후에도 계속 실행됨 (nohup).
# 각 validator는 checkpoint + resume 지원 (중단 → 재실행 시 이어서).

set -uo pipefail
cd "$(dirname "$0")/next_strategies"
mkdir -p out

echo "=== Next strategies 5종 시작 $(date) ==="

NEXT_TESTS=(
    "validate_pullback"
    "validate_vbo"
    "validate_short_mom"
    "validate_breakdown_short"
    "validate_range_mr"
)

for t in "${NEXT_TESTS[@]}"; do
    echo ""
    echo "──────────────────────────────────"
    echo "→ ${t}  ($(date +%Y-%m-%d\ %H:%M:%S))"
    echo "──────────────────────────────────"
    python3 -u "${t}.py" 2>&1 | tee "out/${t}.log"
    echo "  완료 $(date +%H:%M:%S), exit=${PIPESTATUS[0]}"
done

echo ""
echo "=== Next strategies 전체 완료 $(date) ==="
ls -la out/

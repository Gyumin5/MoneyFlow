#!/bin/bash
# next_strategies/run_next.sh - Pullback / VBO / Short Momentum 순차 실행.
# 주의: C tests가 완료된 이후 실행 권장 (동시 실행 시 메모리/IO 경합).

set -euo pipefail
cd "$(dirname "$0")"
mkdir -p out

echo "=== Next strategies 일괄 실행 시작 $(date) ==="

TESTS=(
    "validate_pullback"
    "validate_vbo"
    "validate_short_mom"
)

for t in "${TESTS[@]}"; do
    echo ""
    echo "──────────────────────────────────"
    echo "→ ${t}  ($(date +%H:%M:%S))"
    echo "──────────────────────────────────"
    python3 -u "${t}.py" 2>&1 | tee "out/${t}.log"
    echo "  완료 $(date +%H:%M:%S), exit=${PIPESTATUS[0]}"
done

echo ""
echo "=== 전체 완료 $(date) ==="
ls -la out/

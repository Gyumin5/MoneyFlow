#!/bin/bash
# c_tests_v2/run_all.sh - 17개 테스트 순차 실행 (오프라인 장기 실행용).
#
# 사용:
#   nohup ./run_all.sh > run_all.log 2>&1 &
# 확인:
#   out/*.log + out/*.csv + cache/events_*.pkl

set -euo pipefail
cd "$(dirname "$0")"
mkdir -p out cache

echo "=== C tests v2 일괄 실행 시작 $(date) ==="

TESTS=(
    "test0_extract"             # 이벤트 공통 캐시
    "test1_hold_duration"       # 보유시간 분포 (tp/tstop)
    "test2_regime"              # BTC regime 분해
    "test3_shrink"              # V21-C cash 충돌 & fill ratio
    "test4_bootstrap"           # Holdout 블록 부트스트랩
    "test5_fill_delay"          # 체결 지연 + TX 스트레스
    "test6_coin_contribution"   # 종목별 realized 기여도
    "test7_mae"                 # 봉 내부 MAE (3x wipeout 근접도)
    "test8_data_integrity"      # 1h bar / daily / universe 결손
    "test9_plateau"             # 인접 파라미터 plateau
    "test10_walkforward"        # 여러 train/hold split 안정성
    "test11_universe_npick"     # universe/n_pick/swap_edge 민감도
    "test12_standalone"         # C 단독 (V21 없이)
    "test13_event_signature"    # top best/worst event 분석
    "test14_v21c_correlation"   # V21 vs C 독립 알파 확인
    "test15_cross_anchor"       # 다른 시작일 robustness
    "test16_top_removal"        # 희소 수익 의존도
)

total=${#TESTS[@]}
i=0

# 각 test별 대표 output csv (존재 시 skip, 재실행 resume 용)
declare -A MARKER
MARKER[test0_extract]="cache/events_spot.pkl"
MARKER[test1_hold_duration]="out/test1_hold_duration.csv"
MARKER[test2_regime]="out/test2_regime.csv"
MARKER[test3_shrink]="out/test3_shrink.csv"
MARKER[test4_bootstrap]="out/test4_bootstrap.csv"
MARKER[test5_fill_delay]="out/test5_fill_delay.csv"
MARKER[test6_coin_contribution]="out/test6_coin_contribution.csv"
MARKER[test7_mae]="out/test7_mae_summary.csv"
MARKER[test8_data_integrity]="out/test8_universe.csv"
MARKER[test9_plateau]="out/test9_plateau.csv"
MARKER[test10_walkforward]="out/test10_walkforward.csv"
MARKER[test11_universe_npick]="out/test11_universe_npick.csv"
MARKER[test12_standalone]="out/test12_standalone.csv"
MARKER[test13_event_signature]="out/test13_event_signature.csv"
MARKER[test14_v21c_correlation]="out/test14_v21c_correlation.csv"
MARKER[test15_cross_anchor]="out/test15_cross_anchor.csv"
MARKER[test16_top_removal]="out/test16_top_removal.csv"

for t in "${TESTS[@]}"; do
    i=$((i+1))
    echo ""
    echo "──────────────────────────────────"
    echo "→ [$i/$total] ${t}  ($(date +%H:%M:%S))"
    echo "──────────────────────────────────"
    marker="${MARKER[$t]:-}"
    if [ -n "$marker" ] && [ -f "$marker" ]; then
        echo "  SKIP — 이미 완료 (${marker})"
        continue
    fi
    python3 -u "${t}.py" 2>&1 | tee "out/${t}.log"
    echo "  완료 $(date +%H:%M:%S), exit=${PIPESTATUS[0]}"
done

echo ""
echo "=== 전체 완료 $(date) ==="
echo "결과: out/*.csv , out/*.log"
ls -la out/ | head -40

#!/bin/bash
# OOM 방지 위해 배치 병렬 (동시 3개씩). signal_variants는 joblib 24 자체 소화 → 순차.
set -e
HERE=$(dirname "$(realpath "$0")")
cd "$HERE"
mkdir -p logs out

run_batch() {
    echo "=== batch: $* ==="
    for s in "$@"; do
        log="logs/${s%.py}.log"
        echo "  launching $s → $log"
        python3 "$s" > "$log" 2>&1 &
    done
    wait
    echo "  batch done"
}

echo "=== Phase A: post-filter 테스트 (동시 3개씩) ==="
run_batch test_entry_filters.py test_exit_rules.py test_stop_fine.py
run_batch test_regime.py test_sizing.py test_robustness.py

echo "=== Phase B: signal variants (joblib 24 자체병렬) ==="
log="logs/test_signal_variants.log"
echo "  launching test_signal_variants.py → $log"
python3 test_signal_variants.py > "$log" 2>&1

echo "=== ALL DONE ==="
ls out/*.csv 2>&1

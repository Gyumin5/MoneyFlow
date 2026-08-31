#!/usr/bin/env bash
# 이 저장소의 기본 테스트 명령. 코드 변경의 완료·통과 선언은 이 명령 한 번의 전체 출력으로 한다.
#
# 대상은 tests/test_*.py 와 trade/test_*.py 다. tests/smoke_v25_leverage_down.py 는
# 실계좌에 주문을 내므로 이름 규칙(test_ 접두)에서 제외돼 절대 실행되지 않는다.
# 새 테스트는 같은 규칙의 파일명으로 두면 자동으로 포함된다.
#
# 실행: cd /home/gmoh/mon/251229 && bash run_tests.sh
set -u
cd "$(dirname "$0")"

fail=0
total=0
failed_files=()

for f in tests/test_*.py trade/test_*.py; do
  [ -e "$f" ] || continue
  total=$((total + 1))
  echo "=== $f"
  if python3 "$f"; then
    echo "--- PASS $f"
  else
    echo "--- FAIL $f (rc=$?)"
    fail=$((fail + 1))
    failed_files+=("$f")
  fi
done

echo
echo "==== 요약: 파일 $total 개 중 실패 $fail 개"
for f in ${failed_files+"${failed_files[@]}"}; do echo "  FAIL $f"; done
exit $((fail > 0 ? 1 : 0))

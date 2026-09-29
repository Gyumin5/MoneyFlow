#!/usr/bin/env bash
# 집에서 매일 도는 신호 대조 감시의 실행 껍데기.
#
# 두 갈래로 부른다 — 데이터 갱신과 대조를 시간으로 떼어 놓는다. 한 번에 묶으면
# 갱신이 늦어질 때 대조가 밀린 데이터로 돌거나 아예 안 돌게 된다.
#
#   run_signal_parity.sh refresh   # 09:20 KST — 선물 일봉을 최신으로 당긴다
#   run_signal_parity.sh parity    # 10:00 KST — 라이브 선정과 채택 백테스트를 대조한다
#
# 판정이 0 이 아니면(불일치·비교불능·시간초과·예외) 이 세션에 주입한다. 사람에게
# 직접 쏘지 않는 이유는 raw 출력으로는 판단이 안 되기 때문이다 — 세션이 먼저 읽고
# 무엇이 갈렸는지 해석해 운영자에게 전한다.
#
# 침묵도 고장이다. 매 실행이 state/signal-parity/last.json 에 판정과 시각을 남기므로,
# 그 파일이 오늘 날짜로 갱신되지 않은 것 자체가 감시가 안 돌았다는 증거가 된다.
set -uo pipefail

REPO=/home/gmoh/mon/251229
SESSION=251229
SEND=/home/gmoh/dotfiles/bin/claude-userbot-send
REFRESH_TIMEOUT=600
PARITY_TIMEOUT=600

cd "$REPO" || exit 1

MODE="${1:-parity}"
TODAY=$(date '+%Y-%m-%d')
NOW() { date '+%Y-%m-%d %H:%M KST'; }

mkdir -p logs state/signal-parity
LOG="logs/signal_parity_${TODAY}.log"
STAMP=state/signal-parity/last.json

notify() {
  [ -x "$SEND" ] || return 0
  "$SEND" --session "$SESSION" "$1" >/dev/null 2>&1 || true
}

stamp() {
  # 원자적 저장 — 읽는 쪽이 반쯤 쓰인 파일을 보지 않게.
  local mode="$1" rc="$2" note="$3"
  printf '{"mode":"%s","date":"%s","rc":%s,"at":"%s","note":"%s"}\n' \
    "$mode" "$TODAY" "$rc" "$(NOW)" "$note" > "${STAMP}.tmp"
  mv -f "${STAMP}.tmp" "$STAMP"
}

case "$MODE" in
  refresh)
    {
      echo "=== $(NOW) 데이터 갱신 시작"
      timeout "$REFRESH_TIMEOUT" python3 strategies/cap_defend/refresh_backtest_data.py --target futures
      echo "=== 갱신 rc=$?"
    } >> "$LOG" 2>&1
    last_rc=$(sed -n 's/^=== 갱신 rc=\([0-9]*\)$/\1/p' "$LOG" | tail -1)
    last_rc="${last_rc:-9}"
    stamp refresh "$last_rc" "선물 일봉 갱신"
    if [ "$last_rc" != "0" ]; then
      if [ "$last_rc" = "124" ]; then
        notify "선물 신호 대조 — 데이터 갱신이 ${REFRESH_TIMEOUT}초를 넘겨 끊겼다. 로그 ${REPO}/${LOG}. 10:00 대조는 밀린 데이터로 돌면 비교불능(rc 2)이 된다. 원인을 보고 운영자에게 전해라."
      else
        notify "선물 신호 대조 — 데이터 갱신 실패 rc=${last_rc}. 로그 ${REPO}/${LOG}. 원인을 보고 운영자에게 전해라."
      fi
    fi
    exit 0
    ;;

  parity)
    {
      echo "=== $(NOW) 신호 대조 시작"
      timeout "$PARITY_TIMEOUT" python3 trade/ops/signal_parity_daily.py --days 3
      echo "=== 대조 rc=$?"
    } >> "$LOG" 2>&1
    rc=$(sed -n 's/^=== 대조 rc=\([0-9]*\)$/\1/p' "$LOG" | tail -1)
    rc="${rc:-9}"
    # 이번 실행 구간만 본다. 로그는 날짜 단위로 이어 붙으므로 파일 전체에서 긁으면
    # 아침에 통과한 줄과 방금 실패한 줄이 섞여서 기록이 거짓이 된다.
    verdict=$(awk '/신호 대조 시작/{buf=""} /^판정:/{buf=$0} END{print buf}' "$LOG")
    stamp parity "$rc" "$(printf '%s' "$verdict" | tr -d '"\\')"

    case "$rc" in
      0) : ;;   # 일치 — 조용히 넘어간다. 기록은 STAMP 와 로그에 남는다.
      1) notify "선물 신호 대조 불일치(rc 1). 라이브 선정과 채택 백테스트가 갈렸다. 로그 ${REPO}/${LOG} 의 '다름' 줄에 양쪽 목표 비중이 있다. 어느 쪽이 정본인지 판단해 운영자에게 전해라 — 라이브만 고쳤거나 백테스트만 고친 변경이 있는지 먼저 봐라." ;;
      2) notify "선물 신호 대조 비교불능(rc 2). 통과가 아니다. 사유는 ${REPO}/${LOG} 의 '판정: 비교 못 함' 줄이다 — 데이터 지연, 집·서버 판본 차이, 양성 대조 실패, 예외 넷 중 하나다. 원인을 보고 운영자에게 전해라." ;;
      124) notify "선물 신호 대조가 ${PARITY_TIMEOUT}초를 넘겨 끊겼다(rc 124). 로그 ${REPO}/${LOG}. 오늘 대조는 안 됐다." ;;
      *) notify "선물 신호 대조가 예상 밖으로 끝났다(rc=${rc}). 로그 ${REPO}/${LOG}." ;;
    esac
    exit 0
    ;;

  *)
    echo "쓰기: $0 refresh | parity" >&2
    exit 2
    ;;
esac

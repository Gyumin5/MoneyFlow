#!/usr/bin/env bash
# 집에서 매일 도는 신호 대조 감시의 실행 껍데기.
#
# 두 갈래로 부른다 — 데이터 갱신과 대조를 시간으로 떼어 놓는다. 한 번에 묶으면
# 갱신이 늦어질 때 대조가 밀린 데이터로 돌거나 아예 안 돌게 된다.
#
#   run_signal_parity.sh refresh   # 09:20 KST — 선물(코인)·주식 일봉을 최신으로 당긴다
#   run_signal_parity.sh parity    # 10:00 KST — 다섯 층을 본다. 선물은 라이브 선정함수를
#                                  #   다시 돌려, 주식은 서버가 낸 신호를 읽어 채택 백테스트와
#                                  #   대조한다. 거기에 선물·현물 스냅 상태(저장된 스냅이
#                                  #   스스로 모순 없고 체결 목표와 맞는가)와 주식 실행 결과
#                                  #   (잔고가 전날 잔고 + 그날 주문으로 설명되는가)를 더한다
#
# 판정이 0 이 아니면(불일치·비교불능·시간초과·예외) 이 세션에 주입한다. 사람에게
# 직접 쏘지 않는 이유는 raw 출력으로는 판단이 안 되기 때문이다 — 세션이 먼저 읽고
# 무엇이 갈렸는지 해석해 운영자에게 전한다.
#
# 침묵도 고장이다. 그런데 이 집이 꺼지면 감시도 경보 통로도 같이 죽어서, 아무 일도
# 안 일어나는 것이 "이상 없음" 으로 읽힌다. 그래서 매 실행이 영수증을 서버에 올린다.
# 서버의 독립 크론(parity_watch.py)이 유예시각 뒤에 그 영수증을 보고, 없거나 낡았거나
# 판정이 나쁘면 서버 자기 통로로 알린다. 두 실행에 서로 다른 실행 식별자를 준다 —
# 옛 영수증이 다시 올라온 것을 서버가 다음 날 잡기 위해서다.
# (2026-09-30 토론 run-20260930T054956Z)
set -uo pipefail

REPO=/home/gmoh/mon/251229
SESSION=251229
SEND=/home/gmoh/dotfiles/bin/claude-userbot-send
REFRESH_TIMEOUT=600
PARITY_TIMEOUT=600
# 주소는 저장소 밖 server.conf 에 있다 (공개 저장소라 추적하지 않는다)
_CONF="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/server.conf"
SERVER="${MONEYFLOW_SERVER:-$(grep -m1 "^SERVER=" "$_CONF" 2>/dev/null | cut -d= -f2-)}"
if [ -z "${SERVER:-}" ]; then echo "운영 서버 주소를 찾지 못했다 — $_CONF 의 SERVER= 줄이나 환경변수 MONEYFLOW_SERVER 를 둔다" >&2; exit 2; fi
SERVER_KEY=/home/gmoh/.ssh/id_rsa
SERVER_RECEIPT_DIR=/home/ubuntu/parity-receipts

cd "$REPO" || exit 1

MODE="${1:-parity}"
TODAY=$(date '+%Y-%m-%d')
NOW() { date '+%Y-%m-%d %H:%M KST'; }

mkdir -p logs state/signal-parity/receipts
LOG="logs/signal_parity_${TODAY}.log"
STAMP=state/signal-parity/last.json
RECEIPT="state/signal-parity/receipts/${TODAY}.${MODE}.json"
export PARITY_RUN_ID="${MODE}-$(date '+%Y%m%dT%H%M%S')-$$"

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

push_receipt() {
  # 서버에 원자적으로 올린다. 임시 이름으로 받아서 mv 로 바꾼다 — 서버 크론이
  # 반쯤 쓰인 파일을 읽지 않게. 못 올리면 집에만 알린다. 서버는 영수증이 없는 것을
  # 보고 따로 경보하므로, 이 실패가 침묵으로 끝나지 않는다.
  local src="$1" name="$2"
  [ -f "$src" ] || return 1
  if timeout 60 ssh -i "$SERVER_KEY" -o ConnectTimeout=15 -o BatchMode=yes "$SERVER" \
       "mkdir -p ${SERVER_RECEIPT_DIR} && cat > ${SERVER_RECEIPT_DIR}/.${name}.tmp && \
        mv -f ${SERVER_RECEIPT_DIR}/.${name}.tmp ${SERVER_RECEIPT_DIR}/${name}" \
       < "$src" >/dev/null 2>&1; then
    return 0
  fi
  notify "신호 대조 영수증을 서버에 못 올렸다(${name}). 서버의 침묵 감시가 오늘 영수증이 없다고 경보할 것이다 — 진짜 원인은 감시 실패가 아니라 집→서버 ssh 다. 집 로그 ${REPO}/${LOG}."
  return 1
}

case "$MODE" in
  refresh)
    # 선물(코인 일봉)과 주식을 둘 다 당긴다. 한쪽이 실패해도 나머지는 당긴다 —
    # 먼저 실패한 것 때문에 건너뛰면 그 자산군 대조가 조용히 밀린 데이터로 돈다.
    worst=0
    failed=""
    fut_rc=9
    stock_rc=9
    for target in futures stock; do
      {
        echo "=== $(NOW) 데이터 갱신 시작 ($target)"
        timeout "$REFRESH_TIMEOUT" python3 strategies/cap_defend/refresh_backtest_data.py --target "$target"
        echo "=== 갱신 rc=$? ($target)"
      } >> "$LOG" 2>&1
      rc=$(sed -n "s/^=== 갱신 rc=\([0-9]*\) ($target)$/\1/p" "$LOG" | tail -1)
      rc="${rc:-9}"
      [ "$target" = "futures" ] && fut_rc="$rc" || stock_rc="$rc"
      if [ "$rc" != "0" ]; then
        failed="${failed}${failed:+, }${target} rc=${rc}"
        [ "$rc" -gt "$worst" ] 2>/dev/null && worst="$rc"
      fi
    done
    stamp refresh "$worst" "일봉 갱신 (선물+주식)${failed:+ 실패: $failed}"
    printf '{"mode":"refresh","date":"%s","run_id":"%s","checks":{"선물 일봉":%s,"주식 일봉":%s},"worst":%s,"finished_at":"%s","host":"%s"}\n' \
      "$TODAY" "$PARITY_RUN_ID" "$fut_rc" "$stock_rc" "$worst" "$(NOW)" "$(hostname)" \
      > "${RECEIPT}.tmp" && mv -f "${RECEIPT}.tmp" "$RECEIPT"
    push_receipt "$RECEIPT" "${TODAY}.refresh.json"
    if [ -n "$failed" ]; then
      notify "신호 대조 — 데이터 갱신 실패(${failed}). 로그 ${REPO}/${LOG}. 124 는 ${REFRESH_TIMEOUT}초 초과다. 밀린 데이터로 10:00 대조가 돌면 비교불능(rc 2)이 된다. 원인을 보고 운영자에게 전해라."
    fi
    exit 0
    ;;

  parity)
    {
      echo "=== $(NOW) 신호 대조 시작"
      timeout "$PARITY_TIMEOUT" python3 trade/ops/signal_parity_daily.py --days 3 --receipt "$RECEIPT"
      echo "=== 대조 rc=$?"
    } >> "$LOG" 2>&1
    rc=$(sed -n 's/^=== 대조 rc=\([0-9]*\)$/\1/p' "$LOG" | tail -1)
    rc="${rc:-9}"
    # 이번 실행 구간만 본다. 로그는 날짜 단위로 이어 붙으므로 파일 전체에서 긁으면
    # 아침에 통과한 줄과 방금 실패한 줄이 섞여서 기록이 거짓이 된다.
    # 검사가 여럿이라 '전체 판정:' 줄을 먼저 쓴다 — 마지막 '판정:' 만 보면 나중에 돈
    # 검사의 결과가 그날 전체 판정으로 기록된다.
    verdict=$(awk '/신호 대조 시작/{one="";all=""}
                   /^전체 판정:/{all=$0; next}
                   /^판정:/{one=$0}
                   END{print (all!=""? all : one)}' "$LOG")
    stamp parity "$rc" "$(printf '%s' "$verdict" | tr -d '"\\')"

    # 파이썬이 중간에 죽으면 영수증 자체가 없다. 그때는 껍데기가 대신 적는다 —
    # 검사별 판정을 모르므로 비워 두고, 서버는 '검사 판정이 하나도 없다' 로 경보한다.
    # 서버가 '영수증 없음' 과 '판정 못 남김' 을 구분해서 보게 하려는 것이다.
    if [ ! -f "$RECEIPT" ]; then
      printf '{"mode":"parity","date":"%s","run_id":"%s","checks":{},"worst":%s,"finished_at":"%s","host":"%s","note":"껍데기가 적음 — 파이썬이 판정을 남기지 못했다"}\n' \
        "$TODAY" "$PARITY_RUN_ID" "$rc" "$(NOW)" "$(hostname)" \
        > "${RECEIPT}.tmp" && mv -f "${RECEIPT}.tmp" "$RECEIPT"
    fi
    push_receipt "$RECEIPT" "${TODAY}.parity.json"

    case "$rc" in
      0) : ;;   # 일치 — 조용히 넘어간다. 기록은 STAMP 와 로그에 남는다.
      1) notify "신호 대조 불일치(rc 1) — ${verdict}. 라이브 신호와 채택 백테스트가 갈렸다. 로그 ${REPO}/${LOG} 의 '다름' 줄에 양쪽 목표 비중이 있다. 어느 쪽이 정본인지 판단해 운영자에게 전해라 — 라이브만 고쳤거나 백테스트만 고친 변경이 있는지 먼저 봐라." ;;
      2) notify "신호 대조 비교불능(rc 2) — ${verdict}. 통과가 아니다. 사유는 ${REPO}/${LOG} 의 '판정: 비교 못 함' 줄이다 — 데이터 지연, 집·서버 판본 차이, 양성 대조 실패, 예외 넷 중 하나다. 원인을 보고 운영자에게 전해라." ;;
      124) notify "신호 대조가 ${PARITY_TIMEOUT}초를 넘겨 끊겼다(rc 124). 로그 ${REPO}/${LOG}. 오늘 대조는 안 됐다." ;;
      *) notify "신호 대조가 예상 밖으로 끝났다(rc=${rc}). 로그 ${REPO}/${LOG}." ;;
    esac
    exit 0
    ;;

  *)
    echo "쓰기: $0 refresh | parity" >&2
    exit 2
    ;;
esac

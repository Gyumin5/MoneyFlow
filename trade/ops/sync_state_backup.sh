#!/bin/bash
# 서버의 최신 state 백업을 집 머신으로 가져와 비공개 저장소에 커밋한다.
#
# 왜 집에서 당겨오나: 서버에 깃허브 쓰기 자격증명을 두지 않기 위해서다. 서버는 실매매를
# 돌리는 기계라 그쪽이 털렸을 때 백업 저장소까지 건드릴 수 있으면 백업의 의미가 없다.
#
# 무엇을 올리나: 매일 바뀌는 운영 상태 파일 넷뿐이다. 스냅샷 위상(몇 번째 봉인지)과
# 주식 세 스냅의 다음 앵커 예정일이 이 파일들에만 있어서, 잃으면 재구성이 어렵다.
# API 키는 들어 있지 않다(그건 같은 저장소의 config.py 소관).
#
# 어떻게 올리나: 작업트리에는 최신 하루치만 둔다. 시계열은 깃 이력이 들고 있으므로
# 날짜별 디렉토리를 쌓지 않는다.
#
# 실행: 집 머신의 systemd --user 타이머가 매일 09:20 KST 에 부른다(서버 백업은 09:10).

set -euo pipefail

SERVER="${STATE_SYNC_SERVER:-ubuntu@<서버주소>}"
KEY="${STATE_SYNC_KEY:-$HOME/.ssh/id_rsa}"
REPO="${STATE_SYNC_REPO:-$HOME/moneyflow-config}"
DEST="$REPO/state_backups/latest"
FILES=(binance_state.json kis_trade_state.json signal_state.json trade_state.json)

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S KST')] $*"; }

notify() {
    # 조용한 실패가 이 작업의 유일한 실패 방식이라 알림을 붙인다.
    local msg="$1" token chat
    token=$(python3 -c "import sys;sys.path.insert(0,'$REPO');from config import TELEGRAM_BOT_TOKEN;print(TELEGRAM_BOT_TOKEN)" 2>/dev/null) || return 0
    chat=$(python3 -c "import sys;sys.path.insert(0,'$REPO');from config import TELEGRAM_CHAT_ID;print(TELEGRAM_CHAT_ID)" 2>/dev/null) || return 0
    [ -n "$token" ] && [ -n "$chat" ] || return 0
    curl -s -X POST "https://api.telegram.org/bot${token}/sendMessage" \
        -d chat_id="$chat" -d text="$msg" > /dev/null 2>&1 || true
}

fail() {
    log "실패: $1"
    notify "상태파일 백업 동기화 실패: $1"
    exit 1
}

[ -d "$REPO/.git" ] || fail "비공개 저장소 클론이 없다 ($REPO)"

log "서버에서 최신 백업 날짜 조회"
day=$(ssh -i "$KEY" -o ConnectTimeout=20 -o BatchMode=yes "$SERVER" \
      'ls -1 ~/state_backups 2>/dev/null | grep -E "^[0-9]{4}-[0-9]{2}-[0-9]{2}$" | sort | tail -1') \
      || fail "서버 접속 실패"
[ -n "$day" ] || fail "서버에 백업 디렉토리가 없다"
log "최신 백업 = $day"

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

for f in "${FILES[@]}"; do
    scp -q -i "$KEY" -o ConnectTimeout=20 -o BatchMode=yes \
        "$SERVER:~/state_backups/$day/$f" "$tmp/$f" || fail "$f 가져오기 실패"
    # 쓰다 만 파일을 백업으로 올리지 않는다. 크기가 0 이 아닌 것은 완성의 증거가 아니다.
    python3 -c "import json,sys;json.load(open(sys.argv[1]))" "$tmp/$f" \
        || fail "$f 가 올바른 JSON 이 아니다 — 올리지 않는다"
done

cd "$REPO"
git pull --quiet --rebase || fail "저장소 pull 실패"

mkdir -p "$DEST"
for f in "${FILES[@]}"; do cp "$tmp/$f" "$DEST/$f"; done
# 가져온 시각은 여기 적지 않는다 — 매 실행마다 달라져서 내용이 같아도 커밋이 생긴다.
# 언제 가져왔나는 커밋 시각이 이미 들고 있다.
{
    echo "서버 백업 날짜: $day"
    echo "출처: $SERVER:~/state_backups/$day/"
} > "$DEST/MANIFEST"

if git diff --quiet -- state_backups && git diff --quiet --cached -- state_backups \
   && [ -z "$(git ls-files --others --exclude-standard state_backups)" ]; then
    log "변경 없음 — 커밋 생략 (서버 백업 $day 가 이미 반영돼 있다)"
    exit 0
fi

git add state_backups
git commit --quiet -m "state backup $day"
git push --quiet || fail "push 실패"
log "커밋·푸시 완료 (state backup $day)"

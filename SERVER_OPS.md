# Server Operations — Cap Defend V25 (주식 V25 / 코인현물 V24 / 선물 V25)

운영 서버 주소는 저장소 루트의 `server.conf` 에 있다(git 미추적 — 이 저장소는 공개다). 아래 명령의 `$SERVER` 는 그 파일의 `SERVER=` 값이고, 스크립트는 `trade/ops/server_target.py` 로 읽는다. 본 문서는 서버에서 실제로 도는 스크립트, cron, 헬스체크, 배포·복구 절차를 정리한다. 코드 truth 는 항상 서버 본 + 본 디렉토리(`trade/ops/`) 에 있고, 둘은 항상 일치해야 한다.

## 1. 디렉토리 매핑

| 역할 | 로컬 | 서버 |
|------|------|------|
| 정적 HTML 서버 | `trade/ops/serve.py` | `~/serve.py` |
| Flask API 서버 (포트 5000) | `trade/ops/trade_api_server.py` | `~/trade_api_server.py` |
| 워치독 (5분 주기) | `trade/ops/watchdog_serve.sh` | `~/watchdog_serve.sh` |
| executor 래퍼 (flock + 랜덤 지연) | `trade/ops/run_executor.sh` | `~/run_executor.sh` |
| recommend 래퍼 (flock + 재시도) | `trade/ops/run_recommend.sh` | `~/run_recommend.sh` |
| crontab 사본 | `trade/ops/crontab.txt` | `crontab -l` |
| 코인 executor | `trade/executor_coin.py` | `~/executor_coin.py` |
| 코인 live engine | `trade/coin_live_engine.py` | `~/coin_live_engine.py` |
| 주식 executor | `trade/executor_stock.py` | `~/executor_stock.py` |
| 밤 조회 감시 (읽기 전용) | `trade/ops/balance_read_watch.py` | `~/balance_read_watch.py` |
| 집 감시의 침묵 감시 (매매 경로 아님) | `trade/ops/parity_watch.py` | `~/parity_watch.py` |
| 서버 경보 적기 (매매 경로 아님) | `trade/ops/server_alert_log.py` | `~/server_alert_log.py` |
| 선물 자동매매 | `trade/auto_trade_binance.py` | `~/auto_trade_binance.py` |
| 추천 (general) | `strategies/cap_defend/recommend.py` | `~/recommend.py` |
| 추천 (personal) | `strategies/cap_defend/recommend_personal.py` | `~/recommend_personal.py` |
| 운영 매뉴얼 | `OPERATION_MANUAL.md` | (서버 배포 대상 아님 — 저장소에서 본다) |

상태 파일 (서버에만 존재, gitignore)
- `~/trade_state.json` — 코인현물 V24 live state (members, last_target_snapshot, rebalancing_needed, schema_version)
- `~/kis_trade_state.json` — 주식 V25 live state (snapshots: snap0/snap1/snap2, last_rebal_date)
- `~/binance_state.json` — 선물 V25 live state (strat.canary_on 등)
- `~/signal_state.json` — recommend 출력, executor 입력
- `~/state_backups/YYYY-MM-DD/` — `watchdog_serve.sh` 가 하루 한 번 만드는 cohort. 서버 14일 보관,
  집 사본 90일(아래 "서버 밖 사본")

state 백업 (2026-08-25 전면 수정)
- 넷을 한 묶음으로 본다: `signal_state.json` `trade_state.json` `kis_trade_state.json` `binance_state.json`.
- 임시 디렉토리에 모아 JSON 유효성까지 확인한 뒤 디렉토리 이름을 바꿔 한 번에 공개한다.
  이름이 붙은 `state_backups/YYYY-MM-DD/` 만 복원에 쓴다 — 반쯤 만들어진 백업은 이름이 없다.
- 하나라도 실패하면 공개하지 않고 텔레그램으로 알린 뒤 5분마다 재시도한다. 복구되면 복구 알림 1회.
- 그 전에는 서버에 없는 이름 `coin_trade_state.json` 을 복사하고 선물은 대상이 아니었으며,
  `cp` 실패가 삼켜진 채 성공 플래그가 찍혀 코인현물·선물 상태가 한 번도 백업되지 않았다.
- 복원 절차는 `OPERATION_MANUAL.md` 의 "state 복원 런북" 을 따른다. 넷을 같은 날짜로 기계적으로
  되돌리지 않는다.

서버 밖 사본 (2026-11-03 도입 — 토론 run-20261103T060344Z 판단 1)
- 보관처: 집 `/home/gmoh/backups/251229-state/<날짜>/` (저장소 밖, 디렉토리 700 파일 600)
- 받는 것: `trade/ops/pull_state_backups.py`, 집 크론 `30 * * * *`.
  서버의 완성된 날짜 디렉토리만 고르고(`.staging_` 과 심볼릭 링크 제외), 집에 없는 것을 모두
  받는다. 임시 디렉토리에 받아 원격 SHA-256 대조와 파일별 구조 검사를 통과한 뒤에야 날짜
  이름으로 바꾼다 — 서버의 "완성된 것만 이름을 갖는다" 규약을 집에서도 지킨다.
  이미 있는 묶음은 덮어쓰지 않는다. 서버 것과 내용이 다르면 덮어쓰는 대신 경보한다.
  `rsync --delete` 를 쓰지 않는다 — 서버가 14일 뒤 지운 것을 따라 지우면 기간을 늘린 뜻이 없다.
- 보는 것: `trade/ops/check_state_backup_health.py`, 집 크론 `40 * * * *`.
  수신기의 성공 기록을 믿지 않고 실제 디렉토리를 센다. 15:40 을 넘겼는데 오늘 묶음이 없거나,
  최신이 36시간을 넘겼거나, 필수 파일이 빠졌거나, 수신기가 2시간째 안 돌았거나, 지난 회차
  실패가 남았거나, 여유가 1GB 미만이면 집 세션에 주입한다(`claude-userbot-send`). 억제하지 않는다.
- 보관: 묶음 날짜 기준 90일. 최근 7개와 마지막 복원 확인에 쓴 묶음은 기간과 무관하게 남긴다.
  최신 수신이 실패 중이면 정리를 멈춘다. 서버의 14일과 연동하지 않는다.
- 복원 확인: 마지막은 2026-11-04(묶음 2026-11-04, 자동 수신 경로로 받은 첫 묶음). 다음은
  매월 한 번, 그 시점의 최신 묶음으로 한다. 격리 경로에 네 파일을 펼쳐 읽고
  전략 상태 핵심 필드를 확인한다 — 현물 봉카운터·스냅샷 수, 선물 봉카운터·스냅샷 수, 주식
  스냅샷 수와 직전 리스크온, 신호 갱신 시각. 결과는 `receipt.json` 의 `last_restore_check`.
  실매매 실행기를 `--dry-run` 으로 부르는 것은 복원 확인이 아니다 — 라이브 상태파일을 건드린다.
- 되돌리기: 집 크론 두 줄을 지우면 수신이 멈춘다. 받아 둔 묶음은 남고 서버는 변한 게 없다.
  크론 한 줄 삭제가 "전체 원복" 은 아니다 — 보관처와 기록 파일은 손으로 지워야 사라진다.
- 검사: `trade/test_state_backup_pull.py` (31건). 반쯤 받은 묶음·깨진 JSON·빈 객체·해시 불일치·
  파일 누락·중복 실행·보관 정리·점검 판정을 건다.

## 2. cron 일정

```
@reboot                        cd ~ && nohup python3 serve.py > http.log 2>&1 &
@reboot                        nohup ~/start_trade_api.sh > /dev/null 2>&1 &
*/5 * * * *                    ~/watchdog_serve.sh >> ~/watchdog.log
15 9 * * *                     ~/run_recommend.sh general              # 09:15 매일
15 9 * * *                     ~/run_recommend.sh personal             # 09:15 매일
35 23 * * 1-5                  ~/run_executor.sh stock                # 주식 V25, 평일 23:35 1회
5 9 * * *                      ~/run_executor.sh coin                 # 코인현물 V24, 매일 09:05 1회
5 9 * * *                      python3 ~/auto_trade_binance.py --trade # 선물 V25, 매일 09:05 1회
0 20-23,0-6 * * *              python3 ~/balance_read_watch.py        # 밤 계좌 조회 감시, 매시 정각
40 11 * * *                    python3 ~/parity_watch.py              # 집 신호대조 감시의 침묵 점검
```

집(홈) 크론도 이 운영에 함께 든다.

```
20 9 * * *                     ~/mon/251229/trade/ops/run_signal_parity.sh refresh
0 10 * * *                     ~/mon/251229/trade/ops/run_signal_parity.sh parity
*/10 * * * *                   python3 ~/mon/251229/trade/ops/server_alert_relay.py
```

밤 조회 감시(2026-10-24 설치)는 집행기가 쓰는 `get_balance` 를 그대로 불러 조회가 살아 있는지만
본다. 주문·전략 상태 쓰기가 없고, 연속 2회 실패에서 한 번만 알리고 회복 때 해소를 한 번 알린다.
기록은 `~/balance_watch_streak.json`(연속 실패 횟수와 알림 전송 여부), 로그는 `~/balance_watch.log`.
토큰 캐시 `~/.kis_token.json` 은 집행기와 공유한다 — 쓰기는 잠금 + 임시파일 rename 이다.

서버 경보 중계(2026-10-30 설치). 서버의 두 감시는 서버 자기 텔레그램 봇으로 운영자에게 알린다.
그 통로는 집 유저봇을 일부러 안 탄다 — 집이 통째로 죽어도 알림이 나가야 하기 때문이다. 그래서
집 세션은 그 경보를 못 봤다. 이제 보낸 글을 `~/server_alerts.jsonl` 에 한 줄씩 적고(최근 300줄
보관), 집 크론이 10분마다 그 파일을 읽어 아직 안 본 줄만 세션에 주입한다. 서버 통로는 그대로다 —
중계는 하나 더 두는 것이다. 집 표식은 `state/server-alert-relay/seen.json`, 로그는
`logs/server_alert_relay.log`. 24시간보다 낡은 줄과 한 회차 5건 초과는 안 넣는다. ssh 가 안 되면
조용히 로그만 남긴다 — 서버 통로가 이미 운영자에게 닿아 있어서 중계 실패를 또 경보로 만들면
같은 사건이 두 번 시끄러워진다. 손으로 볼 때는 `python3 trade/ops/server_alert_relay.py --dry-run`.

인증 env 는 crontab 이 아니라 `start_trade_api.sh` 가 단일 출처로 읽는다(2026-07-21 TRADE_PIN 폐지,
DASHBOARD_PIN 단일). 비밀값은 문서·채팅·커밋 어디에도 쓰지 않는다.

저장소 `trade/ops/crontab.txt` 가 선언 기준이다. 서버와 다르면 서버가 truth 고 사본을 고친다.
대조: `ssh ... 'crontab -l'` 출력과 그 파일을 비교한다.

2026-08-25 폐기: 월 1회 BT replay(`30 9 1 * * bt_replay_monthly.py`). 서버 crontab 에 등록돼 있지
않았고 `~/bt_replay.log` 도 없어 한 번도 돈 적이 없다. 게다가 `strategies/cap_defend` 를 import 하는데
서버에는 그 디렉토리가 없어 등록됐더라도 ImportError 로 죽는 구조였고, baseline 도 V24 고정 레버리지
3배 시절 상수라 지금 돌리면 무조건 경보다. 스크립트를 지웠다.
신호 수준 자동 감시는 선물과 주식에 있고 현물에는 없다. 집 크론 둘이 매일 돌고
(09:20 `refresh`, 10:00 `parity`) 서버 `parity_watch.py` 가 11:40 에 그 영수증이 왔는지 본다.
선물은 라이브 선정함수를 집에서 다시 돌려 최근 신호일을 대조하고, 주식은 서버가 실제로 낸
`signal_state.json` 을 읽어 와 대조한다. 자동 회차가 보는 것은 이 둘뿐이다
(`run_signal_parity.sh` 의 `for target in futures stock`).

현물은 손으로 볼 때만 본다 — `--sleeve spotsnap` 이 현물 스냅 상태를 보고,
`v24_shadow_today.py` 는 단발 도구라 어느 crontab 에도 없다. 그것을 "현물은 이미 본다" 로
읽으면 안 된다. 라이브 선정함수와 채택 BT 를 같은 가격으로 대조하는 기준선 하니스는 세
슬리브 모두 있다 — `strategies/cap_defend/research/parity_spot.py` · `parity_fut.py` ·
`parity_stock.py`. 손으로 돌리는 것이고 상시 감시가 아니다.

대조를 손으로 다시 돌릴 때는 반드시 `./trade/ops/run_signal_parity.sh parity` 로 부른다.
`signal_parity_daily.py` 를 맨손으로 부르면 그날 로그도, 서버 영수증도, 나쁜 판정의 세션
주입도 안 된다 — 집에서 고쳐 깨끗해져도 서버는 옛 영수증을 읽고 11:40 에 경보를 쏜다.

설계 의도 (V24/V25 — 모든 자산 1D 단일)
- 코인/선물은 D봉 닫힘 직후 09:05 동시 실행 (4h 멤버 제거, 1일 1회). bar-idempotency 로 같은 봉 중복 매매 방지
- 주식은 미국장 마감 후 평일 23:35 1회 (executor_stock.py 가 직접 전략 계산). 옛 "익일 0~4시 retry" 는 폐지됨
- recommend(09:15) 는 HTML 생성 + 자산배분 트리거(T1/T3U) 체크 + 텔레그램
- 워치독은 헬스체크 실패 시만 재시작. 정상 시 무동작 (idempotent)

## 3. 헬스체크

| 대상 | 명령 | 정상 응답 |
|------|------|----------|
| serve.py | `curl -s http://127.0.0.1:8080/strategy.html -o /dev/null -w '%{http_code}'` | `200` |
| trade_api | `curl -s http://127.0.0.1:5000/health` | `{"status":"ok"}` |
| signal 신선도 | `python3 -c "import json; print(json.load(open('signal_state.json'))['meta']['updated_at'])"` | 26시간 이내 |

웹 접근
- http://$HOST:8080/portfolio_result_gmoh.html (개인 포트폴리오)
- http://$HOST:8080/portfolio_result.html (전체 포트폴리오)
- http://$HOST:8080/strategy.html, /strategy_guide.html, /asset_dashboard.html

`/v22_alloc_report.html` 은 2026-11-03 에 공개 목록에서 내렸다 — 2026-04-28 자 V22 세대
보고서라 현행 자산배분(60/20/20)과 다른데 생성 코드가 없어 고칠 길이 없었다. 서버 파일
`/home/ubuntu/v22_alloc_report.html` 은 지우지 않았고 사본을
`/home/gmoh/backups/251229-reports/2026-11-03/` 에 해시와 같이 보존했다. 그 주소는 이제 404 다.

주의: serve.py 의 `do_HEAD` 는 무조건 404 반환 (의도된 보안). curl `-I` 로 점검하면 실제 GET 이 200 이어도 404 로 보임. 항상 GET 으로 점검.

## 4. 배포 절차

표준 절차
1. 로컬 수정 + 문법/단위 테스트
2. `scp -i ~/.ssh/id_rsa <local> $SERVER:/home/ubuntu/<remote>`
3. 영향 영역 헬스체크 (위 표)
4. API 서버 변경 시 재시작 (2026-07-21 인증 env 단일출처화 이후 — env 를 손으로 export 하지 않는다):
   ```
   ssh ... 'pkill -f "python3 trade_api_server.py"; sleep 2'
   ssh ... 'setsid /home/ubuntu/start_trade_api.sh < /dev/null > /dev/null 2>&1 & exit 0'
   ```
   env(DASHBOARD_PIN/ALLOWED_ORIGINS)는 `/home/ubuntu/.trade_env` 단일출처. TRADE_PIN 은 폐지됐으니
   재도입 금지. 기동 실패 사유는 `~/api_server.log` 의 FATAL 줄.
   주의: 재시작 명령을 한 ssh 로 이어붙이면(`... & sleep; curl ...`) 세션 종료와 함께 자식이 죽어
   서버가 내려간 채로 남는다(2026-08-19 실제 발생). 반드시 위처럼 `setsid ... & exit 0` 로 분리.
   깜빡 놓쳐도 watchdog 이 5분 내 복구하지만, 그 사이 대시보드 조회는 실패한다.
5. cron 다음 실행 결과 로그 확인 (`tail -f ~/recommend.log` 등)
6. git commit + push (서버는 git 저장소가 아님 — 로컬이 단일 source of truth)

매매 경로(`executor_*.py`, `auto_trade_binance.py`)는 위 표준 절차에 다음을 더한다.
실행 예약과 겹치지 않는 낮 시간에 하고, 코드와 상태파일을 함께 백업하고
(`~/<파일>.bak_<날짜>_<시각>`), 임시 이름으로 올려 집·서버 md5 가 같은 것을 본 뒤
`mv` 로 원자적으로 바꾼다. 바꾸기 전에 아래 격리 모의 실행을 통과시킨다.

### 격리 모의 실행 (주식 집행기 기준)

집행기가 쓰는 파일은 두 갈래라 격리 방법이 다르다. 한쪽만 막으면 시험 회차가 운영
기록에 섞인다.

- 작업 디렉토리를 따라가는 것 — `kis_trade_state.json`, `signal_state.json`,
  `trade_state.json`, `executor_stock.log`. 임시 디렉토리로 `cd` 하면 격리된다.
  cron 은 `run_executor.sh` 가 `cd /home/ubuntu` 한 뒤 돌린다.
- 사용자 홈을 따라가는 것 — `~/.kis_token.json`(과 그 잠금), 그리고
  `~/.stock_health.json`·`~/.stock_lock`·`~/stock_abort.log`
  (`HealthGuard(name='stock')` 가 base_dir 없이 불려 기본값 `~` 를 쓴다).
  `HOME` 을 바꿔야 격리된다. 안 바꾸면 시험이 중단으로 끝났을 때 연속 중단 횟수가
  올라 다음 실거래 회차가 막힐 수 있다.

알림은 따로 막지 않는다 — `--dry-run` 이 텔레그램을 로그 한 줄로 돌린다.
`config.py`(자격증명)는 격리하지 않는다. 모듈 탐색 경로의 첫 항목이 스크립트가 있는
`/home/ubuntu` 라 `HOME` 을 바꿔도 그대로 찾고, 실제 계좌를 조회해야 맞다.

```
mkdir -p ~/stockdry/home
cp ~/kis_trade_state.json ~/signal_state.json ~/trade_state.json ~/stockdry/
cp ~/.kis_token.json ~/stockdry/home/          # 발급 빈도 제한 때문에 복사한다
md5sum ~/kis_trade_state.json ~/signal_state.json ~/trade_state.json   # 기준값
cd ~/stockdry && HOME=~/stockdry/home python3 ~/executor_stock.py --dry-run
# 사본을 되돌린 뒤 후보를 같은 자리에서 돌린다
cd ~/stockdry && HOME=~/stockdry/home python3 ~/executor_stock_candidate.py --dry-run
```

본다 — 두 회차의 주문 목록·목표·보유·편차가 같나, 원본 상태파일 셋의 md5 가 기준값
그대로인가, `~/.stock_health.json` 의 수정 시각이 안 변했나. 끝나면
`rm -rf ~/stockdry ~/executor_stock_candidate.py`.

배포 빈번한 파일 (수정 자주)
- `recommend*.py`, `executor_*.py`, `coin_live_engine.py`, `auto_trade_binance.py`

배포 드문 파일 (수정 후 재시작 필요)
- `trade_api_server.py` — 위 4번 명령으로 재시작
- `serve.py` — `pkill -f "python3 serve.py"; nohup python3 serve.py > http.log 2>&1 &`
- `watchdog_serve.sh` — 재시작 불필요 (cron 이 5분 후 자동 적용)

## 4-b. 주식 주문의 접수 여부가 불명일 때

주식 집행기는 주문을 보내기 직전에 그 의도를 `kis_trade_state.json` 의
`inflight_order` 에 적는다(`side`·`ticker`·`qty`·`qty_before`·`at`). 응답이
접수·거절·미전송 중 하나면 그 자리에서 지우고, 응답을 못 받았거나 프로세스가 그
사이에 죽으면 남는다. 다음 회차는 계좌를 읽은 뒤 그 표식을 실제 보유와 대조한다.

- 보유가 `qty_before` 그대로 → 접수 안 됨. 표식을 지우고 그대로 진행한다.
  (회차 시작의 미체결 확정이 통과했을 때만 이 판정이 선다.)
- 보유가 기대대로 바뀜(매도면 `qty_before - qty`) → 체결됨. 지우고 진행한다.
- 둘 다 아님(부분 체결 등) → 그 회차의 주문을 전부 멈추고 알린다. 자동으로 안 푼다.

마지막 경우가 사람이 푸는 자리다. 담당은 운영자 한 사람이고, 알림은 텔레그램으로
간다. 다음 평일 23:35 회차 전까지 풀어야 한다 — 안 풀면 그 회차도 같은 자리에서
멈추고 방어 매도가 하루 밀린다.

푸는 순서는 셋을 확인한 뒤 한 줄이다. 시간이 지났다는 이유로 그냥 지우지 않는다.

1. 그 종목의 체결 내역 — 몇 주가 언제 체결됐나.
2. 미체결 주문이 남아 있지 않다는 것.
3. 지금 보유 수량 — 1·2와 앞뒤가 맞나.

```
cd /home/ubuntu && python3 -c "from common.io import load_json,save_json;\
f='kis_trade_state.json';d=load_json(f);print(d.pop('inflight_order',None));save_json(f,d)"
```

표식을 지우면 다음 회차가 그때의 실제 보유로 목표를 다시 계산한다 — 중간에 체결된
수량은 보유에 이미 반영돼 있으므로 따로 손댈 것이 없다. `rebalancing_needed` 도
건드리지 않는다. 그 회차의 드리프트 자격은 주문이 나가는 순간 이미 회수돼 있어서,
미완성 목표 위에 종목 교체가 얹히지 않는다.

`~/.stock_lock`(HealthGuard)과는 다른 장치다. 그쪽은 사람이 `rm` 으로 푸는 전체
정지이고, 이쪽은 한 주문에 대한 표식이다. 둘 다 떠 있으면 lock 을 먼저 푼다.

## 5. 장애 복구

### 5.1 serve.py 좀비 (2026-04-29 발생)
증상
- 8080 포트 LISTEN 상태인데 curl 응답 없음 (timeout)
- watchdog 가 새 인스턴스 띄우려다 "Address already in use" 로 5분마다 실패

원인
- serve.py 프로세스가 hung/zombie 상태 (SIGTERM 무시), 포트는 그대로 유지

조치 (현재 watchdog 자동화됨)
- pkill -f → 2초 대기 → 안 죽으면 SIGKILL → 그래도 포트 잡혀있으면 fuser -k 8080/tcp
- 3회 연속 재시작 실패 시 텔레그램 알림

수동 복구
```
ssh ... 'fuser -k 8080/tcp; sleep 2; cd ~ && nohup python3 serve.py > http.log 2>&1 &'
```

### 5.2 signal_state stale
증상
- watchdog 로그 "⚠️ signal_state ${N}h old"
- 텔레그램 알림 (1일 1회)

원인 후보
- recommend.py 실패 (run_recommend.log 확인)
- KIS / Yahoo / CoinGecko API 연결 실패
- flock 잔존 (`/tmp/recommend_personal.lock`)

조치
1. `tail -50 ~/recommend_personal.log` 로 에러 확인
2. lock 잔존이면 `rm /tmp/recommend_*.lock`
3. 수동 실행: `cd ~ && python3 recommend_personal.py 2>&1 | tail`

### 5.3 executor 미체결 / pending
증상
- 텔레그램에 pending_trades 알림
- recommend HTML 의 보유 자산이 목표와 큰 차이

원인 후보
- KIS 토큰 만료 → executor_stock 실패
- Upbit/Binance 일시적 API 오류
- 최소주문 미달

조치
1. 다음 cron 실행이 자동 복구 (대부분)
2. 토큰 만료면 `~/.kis_token.json` 삭제 후 재시도
3. 강제 실행: `~/run_executor.sh coin --force` (또는 stock)

### 5.4 cron 자체 정지
증상
- watchdog 로그 비어있음, signal_state 업데이트 안 됨

조치
- `systemctl status cron` 확인. `sudo systemctl restart cron` (사용자 직접, sudo 필요)

## 6. 보안

- API 키 (`config.py`): 절대 git 에 커밋 금지. .gitignore 등재 확인
- ALLOWED_ORIGINS: trade_api_server 는 명시적 origin 만 허용
- 인증은 `DASHBOARD_PIN` 하나다. `TRADE_PIN` 은 2026-07-21 에 폐지했다 — 다시 넣지 않는다.
  모든 기동 경로가 `start_trade_api.sh` 를 거치고 그 스크립트가 env 의 단일 출처다.
  하드코딩 재도입 금지, 실제값은 문서·채팅·커밋 어디에도 쓰지 않는다
- serve.py: 화이트리스트 6개 HTML 만 서빙. HEAD 자동 404
- ssh 접근: `~/.ssh/id_rsa` 키 기반. password 로그인 비활성

## 7. 진단 원커맨드

```bash
# 전체 헬스
ssh -i ~/.ssh/id_rsa $SERVER 'echo "==serve==" && curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8080/strategy.html; echo "==api==" && curl -s http://127.0.0.1:5000/health; echo "==signal age==" && python3 -c "import json,datetime as dt; t=json.load(open(\"/home/ubuntu/signal_state.json\"))[\"meta\"][\"updated_at\"]; print(t)"; echo "==watchdog tail==" && tail -3 ~/watchdog.log'
```

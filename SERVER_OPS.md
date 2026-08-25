# Server Operations — Cap Defend V25 (주식 V25 / 코인현물 V24 / 선물 V25)

운영 서버: `152.69.225.8`. 본 문서는 서버에서 실제로 도는 스크립트, cron, 헬스체크, 배포·복구 절차를 정리한다. 코드 truth 는 항상 서버 본 + 본 디렉토리(`trade/ops/`) 에 있고, 둘은 항상 일치해야 한다.

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
| 선물 자동매매 | `trade/auto_trade_binance.py` | `~/auto_trade_binance.py` |
| 추천 (general) | `strategies/cap_defend/recommend.py` | `~/recommend.py` |
| 추천 (personal) | `strategies/cap_defend/recommend_personal.py` | `~/recommend_personal.py` |
| 운영 매뉴얼 | `OPERATION_MANUAL.md` | (서버 배포 대상 아님 — 저장소에서 본다) |

상태 파일 (서버에만 존재, gitignore)
- `~/trade_state.json` — 코인현물 V24 live state (members, last_target_snapshot, rebalancing_needed, schema_version)
- `~/kis_trade_state.json` — 주식 V25 live state (snapshots: snap0/snap1/snap2, last_rebal_date)
- `~/binance_state.json` — 선물 V25 live state (strat.canary_on 등)
- `~/signal_state.json` — recommend 출력, executor 입력
- `~/state_backups/YYYY-MM-DD/` — `watchdog_serve.sh` 가 하루 한 번 만드는 cohort. 14일 보관

state 백업 (2026-08-25 전면 수정)
- 넷을 한 묶음으로 본다: `signal_state.json` `trade_state.json` `kis_trade_state.json` `binance_state.json`.
- 임시 디렉토리에 모아 JSON 유효성까지 확인한 뒤 디렉토리 이름을 바꿔 한 번에 공개한다.
  이름이 붙은 `state_backups/YYYY-MM-DD/` 만 복원에 쓴다 — 반쯤 만들어진 백업은 이름이 없다.
- 하나라도 실패하면 공개하지 않고 텔레그램으로 알린 뒤 5분마다 재시도한다. 복구되면 복구 알림 1회.
- 그 전에는 서버에 없는 이름 `coin_trade_state.json` 을 복사하고 선물은 대상이 아니었으며,
  `cp` 실패가 삼켜진 채 성공 플래그가 찍혀 코인현물·선물 상태가 한 번도 백업되지 않았다.
- 복원 절차는 `OPERATION_MANUAL.md` 의 "state 복원 런북" 을 따른다. 넷을 같은 날짜로 기계적으로
  되돌리지 않는다.
- 미결: 서버 밖 사본이 없다. VM 이 통째로 사라지면 백업도 같이 사라진다(사용자 판단 대기).

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
```

인증 env 는 crontab 이 아니라 `start_trade_api.sh` 가 단일 출처로 읽는다(2026-07-21 TRADE_PIN 폐지,
DASHBOARD_PIN 단일). 비밀값은 문서·채팅·커밋 어디에도 쓰지 않는다.

저장소 `trade/ops/crontab.txt` 가 선언 기준이다. 서버와 다르면 서버가 truth 고 사본을 고친다.
대조: `ssh ... 'crontab -l'` 출력과 그 파일을 비교한다.

2026-08-25 폐기: 월 1회 BT replay(`30 9 1 * * bt_replay_monthly.py`). 서버 crontab 에 등록돼 있지
않았고 `~/bt_replay.log` 도 없어 한 번도 돈 적이 없다. 게다가 `strategies/cap_defend` 를 import 하는데
서버에는 그 디렉토리가 없어 등록됐더라도 ImportError 로 죽는 구조였고, baseline 도 V24 고정 레버리지
3배 시절 상수라 지금 돌리면 무조건 경보다. 스크립트를 지웠다.
남는 공백은 세 슬리브 전부의 신호 수준 자동 감시다. (정정 2026-08-25: `v24_shadow_today.py` 가
현물을 매일 본다고 적었던 건 사실이 아니다 — 그 스크립트는 어느 crontab 에도 없고 파라미터도
V24 시절에서 멈춰 있다. 매일 도는 신호 감시는 어느 슬리브에도 없다.)
대신 라이브 선정함수와 채택 BT 를 같은 가격으로 대조하는 기준선 하니스가 세 슬리브 모두 있다 —
`strategies/cap_defend/research/parity_spot.py` · `parity_fut.py` · `parity_stock.py`(2026-08-25 신설).
손으로 돌리는 것이고 상시 감시가 아니다. 이걸 매일 도는 감시로 올릴지, 입력 데이터 fingerprint 와
버전 고정 회귀 테스트를 붙일지는 구현 비용이 있어 사용자 판단 대기 항목으로 `progress.md` 에 올려 뒀다.

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
- http://152.69.225.8:8080/portfolio_result_gmoh.html (개인 포트폴리오)
- http://152.69.225.8:8080/v22_alloc_report.html (자산배분 분석)
- http://152.69.225.8:8080/strategy.html, /strategy_guide.html, /asset_dashboard.html

주의: serve.py 의 `do_HEAD` 는 무조건 404 반환 (의도된 보안). curl `-I` 로 점검하면 실제 GET 이 200 이어도 404 로 보임. 항상 GET 으로 점검.

## 4. 배포 절차

표준 절차
1. 로컬 수정 + 문법/단위 테스트
2. `scp -i ~/.ssh/id_rsa <local> ubuntu@152.69.225.8:/home/ubuntu/<remote>`
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

배포 빈번한 파일 (수정 자주)
- `recommend*.py`, `executor_*.py`, `coin_live_engine.py`, `auto_trade_binance.py`

배포 드문 파일 (수정 후 재시작 필요)
- `trade_api_server.py` — 위 4번 명령으로 재시작
- `serve.py` — `pkill -f "python3 serve.py"; nohup python3 serve.py > http.log 2>&1 &`
- `watchdog_serve.sh` — 재시작 불필요 (cron 이 5분 후 자동 적용)

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
- TRADE_PIN: 환경변수 `TRADE_PIN` 로 trade_api 보호 (실제값은 서버 crontab 에만, 문서·커밋 금지)
- serve.py: 화이트리스트 6개 HTML 만 서빙. HEAD 자동 404
- ssh 접근: `~/.ssh/id_rsa` 키 기반. password 로그인 비활성

## 7. 진단 원커맨드

```bash
# 전체 헬스
ssh -i ~/.ssh/id_rsa ubuntu@152.69.225.8 'echo "==serve==" && curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8080/strategy.html; echo "==api==" && curl -s http://127.0.0.1:5000/health; echo "==signal age==" && python3 -c "import json,datetime as dt; t=json.load(open(\"/home/ubuntu/signal_state.json\"))[\"meta\"][\"updated_at\"]; print(t)"; echo "==watchdog tail==" && tail -3 ~/watchdog.log'
```

## 8. 변경 이력

- 2026-07-03 — 문서 stale 갱신: 제목/매뉴얼참조 V22→V25, cron 을 현행(코인/선물 09:05 매일·주식 23:35 평일, 4h 및 0~4시 retry 폐지)으로 정정, 상태파일 버전 표기(코인 V24/주식·선물 V25) 정정, TRADE_PIN 평문 제거.
- 2026-05-28 — 선물 V25 도입 (동적 per-coin L + CROSSED 마진). 코인현물 V24 / 주식 V25 유지.
- 2026-04-30 — V24 마이그레이션 (모든 자산 1D 단일 + drift trigger, cron 4h×6 → 1d×1).
- 2026-04-29 — watchdog_serve.sh robust restart 패치 (SIGKILL fallback + fuser -k + 3회 실패 알림). 서버 좀비 케이스 자동 복구.
- 2026-04-28 — V22 단일 표기 일괄 정리. recommend*.py, V17_OPERATION_MANUAL → V22_OPERATION_MANUAL 갱신.
- 2026-04-27 — V22 마이그레이션 (코인/선물/주식 모두 1D+4h 2멤버 EW + 주식 snap-stagger).

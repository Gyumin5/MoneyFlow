# Server Operations — Cap Defend V22

운영 서버: `152.69.225.8`. 본 문서는 서버에서 실제로 도는 스크립트, cron, 헬스체크, 배포·복구 절차를 정리한다. 코드 truth 는 항상 서버 본 + 본 디렉토리(`trade/ops/`) 에 있고, 둘은 항상 일치해야 한다.

## 1. 디렉토리 매핑

| 역할 | 로컬 | 서버 |
|------|------|------|
| 정적 HTML 서버 | `trade/ops/serve.py` | `~/serve.py` |
| Flask API 서버 (포트 5000) | `trade/ops/trade_api_server.py` | `~/trade_api_server.py` |
| 워치독 (5분 주기) | `trade/ops/watchdog_serve.sh` | `~/watchdog_serve.sh` |
| executor 래퍼 (flock + 랜덤 지연) | `trade/ops/run_executor.sh` | `~/run_executor.sh` |
| recommend 래퍼 (flock + 재시도) | `trade/ops/run_recommend.sh` | `~/run_recommend.sh` |
| 일별 history 갱신 | `trade/ops/daily_history.py` | `~/daily_history.py` |
| crontab 사본 | `trade/ops/crontab.txt` | `crontab -l` |
| 코인 executor | `trade/executor_coin.py` | `~/executor_coin.py` |
| 코인 live engine | `trade/coin_live_engine.py` | `~/coin_live_engine.py` |
| 주식 executor | `trade/executor_stock.py` | `~/executor_stock.py` |
| 선물 자동매매 | `trade/auto_trade_binance.py` | `~/auto_trade_binance.py` |
| 추천 (general) | `strategies/cap_defend/recommend.py` | `~/recommend.py` |
| 추천 (personal) | `strategies/cap_defend/recommend_personal.py` | `~/recommend_personal.py` |
| 운영 매뉴얼 | `V22_OPERATION_MANUAL.md` | `~/V22_OPERATION_MANUAL.md` |

상태 파일 (서버에만 존재, gitignore)
- `~/trade_state.json` — 코인 V22 live state (members, last_target_snapshot, rebalancing_needed)
- `~/kis_trade_state.json` — 주식 V22 live state (snapshots: snap0/snap1/snap2)
- `~/signal_state.json` — recommend 출력, executor 입력
- `~/state_backups/` — 매일 1회 자동 백업, 14일 보관

## 2. cron 일정

```
@reboot                        cd ~ && nohup python3 serve.py > http.log 2>&1 &
@reboot                        export TRADE_PIN=0318 && nohup python3 ~/trade_api_server.py > ~/api_server.log 2>&1 &
*/5 * * * *                    ~/watchdog_serve.sh >> ~/watchdog.log
15 9 * * *                     ~/run_recommend.sh general
15 9 * * *                     ~/run_recommend.sh personal
35 23 * * 1-5                  ~/run_executor.sh stock                # 평일 23:35
05,35 0-4 * * 2-6              ~/run_executor.sh stock                # 익일 0~4시 :05/:35
5 9,13,17,21,1,5 * * *         ~/run_executor.sh coin                 # 4h 동기
5 9,13,17,21,1,5 * * *         python3 ~/auto_trade_binance.py --trade
```

설계 의도
- recommend(09:15) → 주식 executor(23:35~익일 새벽) → 다음 09:15 recommend → ... 의 24h 사이클
- 코인/선물은 4h 봉 닫힘 시각(KST 9/13/17/21/1/5) :05 동시 실행. bar-idempotency 로 같은 봉 중복 매매 방지
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
4. API 서버 변경 시 재시작:
   ```
   ssh ... 'pkill -f "python3 trade_api_server.py"; sleep 2; cd ~ && export TRADE_PIN=0318 ALLOWED_ORIGINS=http://152.69.225.8:8080; nohup python3 trade_api_server.py > api_server.log 2>&1 &'
   ```
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
- TRADE_PIN: 환경변수 `TRADE_PIN=0318` 로 trade_api 보호
- serve.py: 화이트리스트 6개 HTML 만 서빙. HEAD 자동 404
- ssh 접근: `~/.ssh/id_rsa` 키 기반. password 로그인 비활성

## 7. 진단 원커맨드

```bash
# 전체 헬스
ssh -i ~/.ssh/id_rsa ubuntu@152.69.225.8 'echo "==serve==" && curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8080/strategy.html; echo "==api==" && curl -s http://127.0.0.1:5000/health; echo "==signal age==" && python3 -c "import json,datetime as dt; t=json.load(open(\"/home/ubuntu/signal_state.json\"))[\"meta\"][\"updated_at\"]; print(t)"; echo "==watchdog tail==" && tail -3 ~/watchdog.log'
```

## 8. 변경 이력

- 2026-04-29 — watchdog_serve.sh robust restart 패치 (SIGKILL fallback + fuser -k + 3회 실패 알림). 서버 좀비 케이스 자동 복구.
- 2026-04-28 — V22 단일 표기 일괄 정리. recommend*.py, V17_OPERATION_MANUAL → V22_OPERATION_MANUAL 갱신.
- 2026-04-27 — V22 마이그레이션 (코인/선물/주식 모두 1D+4h 2멤버 EW + 주식 snap-stagger).

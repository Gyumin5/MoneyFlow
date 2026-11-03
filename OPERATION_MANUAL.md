# 운영 매뉴얼 — 현행 (주식 V25 / 코인현물 V24 / 선물 V25)

updated: 2026-08-25

이 문서는 "매일 무슨 일이 일어나고, 이상하면 무엇을 보고, 어떻게 되돌리나" 만 다룬다.
버전 이름별 매뉴얼을 따로 두지 않는다 — 지난 버전의 절차는 git 이력에 있고, 여기엔 지금 도는 것만 남긴다.

| 문서 | 역할 |
|---|---|
| `CLAUDE.md` | 전략 파라미터 정본 (선정·헬스·카나리·스냅·드리프트·비용). 값이 다르면 이쪽이 맞다 |
| `SERVER_OPS.md` | 서버 파일 매핑, cron, 헬스체크, 복구 절차 |
| `history.md` | 결정 로그 (왜 그렇게 정했나) + 기각된 실험 |
| 이 문서 | 실행 흐름, 가드가 무엇을 막는지, 사고 대응, 롤백 |
| `strategies/cap_defend/STRATEGY_EVOLUTION.md` | V12~V25 진화와 폐기된 아이디어 |

## 매일 무엇이 도나

| 시간 (KST) | 대상 | 실행 |
|---|---|---|
| 09:05 매일 | 코인현물 V24 | `executor_coin.py` — Upbit, D봉 닫힘 직후 |
| 09:05 매일 | 선물 V25 | `auto_trade_binance.py --trade` — Binance USDT-M |
| 09:15 매일 | 리포트 | `recommend.py` + `recommend_personal.py` — HTML·자산배분 트리거·텔레그램 |
| 11:40 매일 | 신호 대조 감시 | `parity_watch.py` — 집이 올린 영수증을 읽고 어긋나면 알린다 |
| 23:35 평일 | 주식 V25 | `executor_stock.py` — 한국투자증권, 직접 전략 계산 |
| 매시 정각 20~06시 | 계좌조회 감시 | `balance_read_watch.py` |
| */5 | 감시 | `watchdog_serve.sh` |

감시 둘은 매매 경로가 아니다 — 읽고 알리기만 한다. 서버 자기 텔레그램 봇으로 보내고,
같은 글을 `~/server_alerts.jsonl` 에 적어 집 크론이 세션으로 가져간다.

세 sleeve 공통 판정: 시그널은 t-1 확정 종가, 체결은 당일. 매일 상태를 점검하되 종목 교체는
앵커일에만 한다(예외: 주식 V25 는 drift 발화일에도 fresh 재선정).

리밸 필요 판정은 공통으로 `need_rebal = is_daily_bar AND (snap_fire OR (canary_on AND ht ≥ threshold))`,
`ht = Σ|tgt_w − cur_w| / 2` (자본금 기준 half-turnover).

---

# 선물 (V25, Binance USDT-M)

## 파라미터

| 항목 | 값 |
|---|---|
| sleeve | D_SMA42 단일, snap_interval=95, n_snapshots=5, drift=0.015 |
| 전환 후 재평가 | `post_flip_delay=5` — 카나리 ON 봉을 0 으로 세어 다섯 봉 뒤 첫 회차에 전 스냅샷 재선정 |
| 선정/헬스 | universe 3종 cap 1/3, mom2vol(mom 18/127, vol 90d ≤ 5%), 카나리 BTC SMA42 ±1.5% |
| 레버리지 | 동적 per-coin `L = min(BTC_cap, K2_per_coin)`, Lmin/Lmid/Lmax = 2/3/4 |
| 마진모드 | CROSSED (V24 ISOLATED 에서 전환) |
| 체결 밴드 | 명목 ±5% (`DELTA_THRESHOLD = 0.05`) — 채택 BT 의 수량 ±5% 와 정합 |
| 스탑·캐시가드 | 없음. 가드 없음 (분산 + 동적 L 이 유일 방어) |
| 거래비용 | 0.04% (BT 0.06% 로 보수 측정) |
| cash buffer | 1% (`CASH_BUFFER`) |

단독 sleeve BT(5.6yr): Cal 8.12 / CAGR 312% / MDD -38.3% / Sharpe 1.90.
자산배분 60/20/20 + T1 25pp + T3U 20%: Cal 4.07 / CAGR 76.4% / MDD -18.8% / Sharpe 1.84
(5.8yr, 송금 지연 5영업일).

## 동적 레버리지 결정 (매 cron)

```python
prev = btc_close[:-1]                  # 진행중 봉 제외 (look-ahead 차단)
btc_ratio = prev[-1] / mean(prev[-42:])
btc_cap = 4 if btc_ratio > 1.05 else 3 if btc_ratio > 1.015 else 2   # 시장 전체 상태

for coin in selected:                  # 개별 코인 단기 강도
    r = coin_prev[-1] / mean(coin_prev[-7:])
    k2 = 4 if r > 1.075 else 3 if r > 1.025 else 2
    final_L[coin] = min(btc_cap, k2)
```

레버리지가 내려가는 코인은 `set_leverage` 전에 명목을 `new_L / cur_L` 비율로 줄이는 사전매도가
먼저 나간다(Binance -4131 회피). 가격이 약분되므로 정확히 `1 − new/cur` 만큼 팔린다.

## 체결 밴드 (`DELTA_THRESHOLD`, 2026-08-25 정합)

- 채택 BT `backtest_futures_v25._execute_rebalance` 의 수량 ±5% 와 같은 값이다
  (같은 가격에서 명목비 = 수량비라 두 정의는 동치).
- 지배 범위: 매도 게이트 / 매수 게이트 / `needs_rebalance`(= 체결 후 "목표 달성" 완료 판정) 세 곳.
- 밴드를 타지 않는 경로: 목표비중 0 전량청산, 미보유 신규진입, 카나리 OFF `liquidation_only`,
  레버리지·마진모드 변경(및 위 L 하향 사전매도).
- 주문 실패(`ORDER FAILED`)가 있으면 밴드와 무관하게 `rebalancing_needed` 를 유지한다 —
  거절 잔차가 넓어진 밴드 안에 묻혀 달성으로 선언되는 것을 막는다.
- 드리프트 문턱(0.015)과 혼동 금지. 드리프트는 상시 발화 상태라(레버리지 4배에서 진입가 대비 ±1.5%)
  실제 회전수를 정하는 값은 이 체결 밴드다.
- 근거: `reports/2026-08-25-fut-rebal-band.html`, ADR `history.md` 2026-08-25.
  1% 로 좁히면 수익 동률·MDD 악화·체결 1.69배(연 264회 대 156회).
- 테스트: `tests/test_fut_rebal_band.py`.

## 매 cron 실행 순서와 가드

1. 상시 무결성 검사 — 계좌를 건드리기 전에, 매매 직전의 보유(`positions_before`)로 돈다.
   위반이 잡히면 이번 실행은 매매를 안 한다. 체결 뒤로 미루면 우리 주문이 위반으로 잡히거나
   (자기잠김) 주문을 낸 실행마다 검사를 걸러야 해서 사실상 꺼진다. 기준 스냅샷 갱신은
   성공 판정과 분리되어 있다(2026-08-22). 포지션 조회가 실패하면 검사를 생략하고 소실로 단정하지 않는다
2. `build_margin_plan` 으로 심볼별 작업계획을 한 번 만든다 — 조회 불명이 하나라도 있으면 ABORT.
   preflight·변경·사후검증이 전부 이 계획 하나를 쓴다
3. 변경이 필요한 것만 preflight. one-way 모드 변경은 계정 단위 규칙이라 계정 전체가 비어야 하고
   (먼지도 불가), 마진모드 변경은 그 심볼만 비면 된다(무관 심볼 먼지는 허용). 둘 중 하나라도
   잔존이 있으면 ABORT
4. `ensure_position_mode_oneway` — hedge 면 one-way 로 바꾸고 다시 검증한다. 보장 실패면 ABORT.
   3번의 계정 전체 zero 를 통과한 뒤라 포지션이 없을 때만 바뀐다
5. 레버리지를 낮추는 코인은 비율만큼 먼저 reduceOnly 매도한다 — 안 그러면 `set_leverage` 가
   증거금 증가로 -4131 거부를 낸다. 올리는 쪽은 그대로 두고 `execute_rebalance` 가 매수로 처리
6. 코인별 마진모드. 계획이 "변경 필요" 라 한 심볼은 `ensure_margin_type` 이 CROSSED 로 설정하고
   다시 검증한다. "변경 불필요" 심볼은 `verify_margin_type` 로 검증만 하고 설정을 시도하지 않는다 —
   보유 중인 심볼에 set 을 걸면 바이낸스가 거부한다. 어느 쪽이든 어긋나면 ABORT
7. 레버리지. 이미 목표면 `set_leverage` 를 생략한다 — 같은 값으로 불러도 성공을 주므로 매번
   "계좌 변경" 으로 표시되고, 그러면 1번 검사가 매매 경로마다 걸러져 외부개입 탐지가 꺼진다.
   생략했든 설정했든 그 뒤 `verify_leverage` 를 짧게 세 번 재조회한다(전파 지연 대비)
8. 어느 코인이든 verify 실패 → 매매 전체 ABORT (다음 cron 재시도). 부분 유니버스 매매는
   백테스트 가정(3 코인 균등) 위반이다
9. `force_cancel_all_orders` → `execute_rebalance` (매도 먼저, 매수 나중). 취소 범위는 목표 심볼과
   현재 보유 심볼의 합집합이고, 취소 실패도 ABORT 다 — 잔존 주문이 신규 주문과 충돌한다

매매 게이트는 `fut_trade_gate()` 단일 판정이다. 빈 `target_lev_map` 을 무조건 차단하지 않는다 —
산출 실패(lev_abort)만 차단하고, 전액 CASH 는 `liquidation_only` 로 통과시킨다(매수 루프 미진입 +
전 주문 reduceOnly 불변식). 이 구분을 없애면 2026-08-15 청산 차단 사고가 재발한다.

## 롤백

- 체결 밴드: `DELTA_THRESHOLD` 한 줄 원복(0.01). 상태파일 스키마 영향 없음.
- 동적 레버리지 → 고정 L3 ISOLATED: `auto_trade_binance.py` LEVERAGE_FLOOR/MID/CEILING=3/3/3,
  `futures_live_config.py` 동일 복원, 전 포지션 close → 마진모드 ISOLATED → set_leverage 3 → cron 재개.
- 드리프트만 끄기: `DRIFT_ENABLED_FUT = False` (target 계산은 유지, snap_fire 만으로 리밸).

---

# 코인 현물 (V24, Upbit)

## 파라미터

| 항목 | 값 |
|---|---|
| sleeve | D_SMA42 단일, snap_interval=217 (=7×31), n_snapshots=7, drift=0.10 |
| 전환 후 재평가 | `post_flip_delay=5` — 카나리 ON 봉을 0 으로 세어 다섯 봉 뒤 첫 회차에 전 스냅샷 재선정 |
| 선정/헬스 | universe 3종 cap 1/3, mom2vol(mom 20/127, vol 90d ≤ 5%), 카나리 BTC SMA42 ±1.5% |
| 유니버스 | CoinGecko Top40 ∩ Binance spot ∩ Upbit KRW ∩ 253일 이상 ∩ 거래대금 조건 |
| drift 발화 시 | refill v2 — 모멘텀 둘 다 음수(ms<0 AND ml<0)인 코인만 fresh healthy 로 교체 |
| 가드 | 없음. Upbit warning/delisting 코인만 유니버스에서 즉시 제외 |
| 거래비용 | 0.4% 편도 (BT — `trade/v24_shadow_today.py` 가 `tx_cost=0.004`), 실매매는 Upbit 실수수료 |
| cash buffer | 1% |

단독 sleeve BT(6.1yr, 2020-10-01~2026-11-01, 앵커 7곳 평균):
Cal 1.69 / CAGR +51.6% / MDD -30.7% / Sharpe 1.54. 앵커 흔들림이 칼마 1.48~1.86 이라
단일 앵커 값을 인용하지 않는다(2026-11-02 실측, 커밋 5772d27).
옛 표기 4.35/+77.6%/-17.8% 는 2026-09-22 유니버스 결함 수정 이전 기준이다 — 심볼표를
고정 40종에서 폴더 조회로 바꾼 것이 하락의 거의 전부다.

refill v2 는 drift 0.10 에서 앵커 전용과 사실상 동일하다(2026-06-06 검증: 5.4yr 종목교체 0일 —
발화일의 보유 코인은 항상 모멘텀 양수였다). 미래의 모멘텀 급락 + drift 동시 발생 대비 방어로만 남겼다.

## 상태·현금 키

- 상태파일 `trade_state.json`: `members / last_target_snapshot / rebalancing_needed / schema_version`.
- 현금 키는 엔진 내부 `CASH`, 실매매·리포트 `Cash`. 정규화는 `run()` 반환 경계 한 곳(refill 후처리 뒤)에서만
  하고, executor 는 상류를 믿지 않고 진입부·buffer·cap·완료판정에서 다시 병합한다.
  정규화를 refill 앞으로 되돌리면 2026-08-20 헛주문 사고가 재발한다.
- 부분체결은 `pending_trades` 에 남고 monitor 가 다음 사이클에 복구한다.

## 롤백

- 드리프트만 끄기: `coin_live_engine.DRIFT_ENABLED = False` (snap-only. BT 상 알파 손실을 감수하는 임시 조치).
- 종목 교체를 앵커 전용으로: drift 발화 시 refill 을 끄면 되지만, 위 검증대로 현행과 결과가 같아
  실효가 없다 — 문제 원인이 refill 이라는 근거가 있을 때만 손댄다.

---

# 주식 (V25, 한국투자증권)

## 파라미터

| 항목 | 값 |
|---|---|
| 유니버스 | R7 = SPY, QQQ, VEA, EEM, GLD, PDBC, VNQ |
| 방어자산 | IEF, BIL, BNDX, GLD, PDBC |
| 카나리 | EEM SMA200 ±0.5% dead-zone |
| 선정 | Z-score(가중Mom + Sharpe126) 랭킹 → 3-mom(30/72/230) 필터 → Top3 cap 1/3 + Cash |
| 가중Mom | 0.5×ret63 + 0.3×ret126 + 0.2×ret252 (함수명이 아니라 이 정의가 정본) |
| 트랜치 | SNAP_PERIOD=69, STAGGER=23, N_SNAPS=3 |
| drift | 0.05. 발화일에는 그날 fresh selection 으로 snap picks 리필 후 merge → 체결 (`STOCK_DRIFT_REFILL=True`) |
| cash buffer | 7% (계좌 내부) |
| 거래비용 | 0.1% 편도 (BT) |

전략 계산은 `stock_strategy_v25.py` 순수 함수 하나를 executor 와 recommend 가 같이 호출한다.
`signal_state.json` 은 fallback 전용이고, V25 스키마 + 24시간 신선도 검증을 통과할 때만 쓴다.

## 가드 (이 sleeve 에만 있다)

- 가격 기준일 hard check — 모든 ticker 의 last_date 가 일치하지 않으면 SKIP
- T-1 확정 종가만 사용 (장중 partial row 차단)
- 야후 지연 시 KIS 일봉으로 1차 보강 → 공통 최신일 정렬(≤3영업일) → 초과하면 SKIP
- Cold-start SKIP: `prev_risk_on` 과 `canary_risk_on` 이 둘 다 None 이고 EEM 결측이면 SKIP
- EEM 결측으로 risk_on 을 자동 변경하지 않는다 (`prev_risk_on` 유지, `or` 대신 None 검사)

---

# 자산배분 (60/20/20, 수동 송금)

- 주식 60% / 업비트 20% / 바이낸스 20%. 정본은 라이브 코드 `STOCK_RATIO/COIN_RATIO/FUTURES_RATIO`.
- per-sleeve cash buffer: 주식 7% / 현물 1% / 선물 1% (총 cash ≈ 5%).
- 자동 송금·자동 cap 은 폐지됐다(alloc_transit phantom buffer 영구 비활성화).
  트리거가 켜지면 텔레그램 알림만 보내고 사용자가 직접 송금한다.
- 트리거 (OR):
  · T1 — `half_turnover ≥ 25pp`
  · T3U_can — `max((tgt − cur)/tgt) ≥ 20%` AND 해당 sleeve 카나리 ON
    (주식=EEM Risk-On, 현물=BTC > SMA42×1.015, 선물=`binance_state.json` `strat.canary_on`)
- 같은 사유 알림은 24시간에 1회.

---

# 사고 대응

| 증상 | 먼저 볼 것 | 처치 |
|---|---|---|
| 선물 ABORT 알림 | 로그의 ABORT 사유 (position mode / margin / leverage / 데이터 누락) | 사유 해소 후 다음 cron 재시도. 마진모드는 포지션이 없을 때만 UI 에서 바뀐다 |
| 부분체결 | 현물 `pending_trades`, 선물 `rebalancing_needed` | 둘 다 다음 사이클에 재시도. 손으로 주문 넣지 않는다 |
| 목표 미달이 반복 | 로그의 `sell_check/buy_check … delta=` | 밴드(5%) 안이면 정상. 밴드 밖인데 주문이 없으면 최소주문·스텝 라운딩·잔고 확인 |
| 상시 무결성 검사 경고 | 거래소 체결 이력과 기준 스냅샷 | 우리 주문이면 기준 재시딩, 외부 개입·청산이면 원인 규명 후 재개 |
| 드리프트 상시 발화 | `ht` 값과 레버리지 | 구조상 정상이다. 레버리지 4배에서 진입가 대비 ±1.5% 면 문턱을 넘는다 |
| 상태파일 손상 | `~/state_backups/YYYY-MM-DD/` cohort | 아래 "state 복원 런북". 삭제는 초기화가 아니라 상태 손실이다 |
| 백업 실패 알림 | 알림 본문의 실패 파일명 | 워치독이 5분마다 재시도한다. 파일 자체가 깨진 것이면 그게 진짜 사고다 |

## state 백업

- `watchdog_serve.sh` 가 09:10 KST 이후 첫 실행에서 하루 한 번 만든다.
- 대상 넷: `signal_state.json` `trade_state.json` `kis_trade_state.json` `binance_state.json`.
- 임시 디렉토리에 모아 JSON 유효성까지 확인한 뒤 이름을 바꿔 한 번에 공개한다.
  복원에 쓸 수 있는 건 이름이 붙은 `state_backups/YYYY-MM-DD/` cohort 뿐이다.
- 하나라도 실패하면 공개하지 않고 텔레그램 알림 1회 + 5분마다 재시도, 복구되면 복구 알림 1회.
- 확인: `ls ~/state_backups/$(date +%F)/` 에 json 4개 + MANIFEST.
- 서버 밖 사본: 집 `/home/gmoh/backups/251229-state/<날짜>/` 에 같은 묶음이 90일 보관된다.
  집 크론이 매시 30분에 받고 40분에 수신이 멈췄는지 본다. 멈추면 집 세션에 알림이 들어온다.
  받는 쪽도 해시와 구조를 다 확인한 뒤에야 날짜 이름을 붙이므로, 이름이 있는 디렉토리는
  서버 cohort 와 바이트가 같다. 절차는 `SERVER_OPS.md` 의 "서버 밖 사본".
- 한계: 묶음 안 네 파일은 순차 복사라 거래 시점이 서로 맞는다는 보장이 없다. 09:10 이후의
  체결·미체결은 다음 묶음까지 빠진다. 그래서 넷을 같은 날짜로 기계적으로 되돌리지 않는다.

## state 복원 런북

상태파일은 캐시가 아니라 전략 상태다(스냅 멤버, 앵커 카운터, `rebalancing_needed`, 카나리 직전값).
잘못 되돌리면 다음 실행이 첫 실행처럼 굴어 전 스냅을 그날 신호로 초기화한다.
넷을 같은 날짜로 기계적으로 되돌리지 않는다 — 손상된 것만 되돌린다.

1. 매매를 멈춘다. 해당 cron 을 주석 처리하거나 다음 실행 전에 끝낸다.
2. 지금 파일을 타임스탬프 사본으로 남긴다. 손상본도 증거다.
3. 거래소에서 사실을 먼저 확정한다 — 잔고, 포지션, 미체결, 최근 체결.
   상태파일이 아니라 이쪽이 truth 다.
4. 코드 버전과 스키마가 맞는지 본다(`schema_version`). 옛 cohort 를 새 코드에 넣지 않는다.
5. 손상된 파일만 cohort 에서 복사한다. 복사 후 JSON 유효성과 핵심 키를 눈으로 확인한다.
   서버 자체가 사라졌으면 집 사본에서 가져온다 — `/home/gmoh/backups/251229-state/<날짜>/`.
   고를 때는 `ls` 로 가장 최신이 아니라 `schema_version` 이 지금 코드와 맞는 것을 고른다.
6. 백업 시점 이후 실제 체결이 있었다면 그 부분을 손으로 맞춘다
   (선물 진입 L, 현물 members, 주식 snapshots, `last_rebal_date`).
7. 주문이 나가지 않는 경로로 한 번 확인한다. 선물 dry-run 은 반드시 격리 사본에 대고 돌린다 —
   라이브 `binance_state.json` 을 대상으로 하면 `rebalancing_needed` 가 오염된다.
8. 재초기화·중복주문 유발 조건이 없는지 본다(빈 members, 앵커 카운터 0, `pending_trades` 잔재).
9. executor 를 하나씩 되살리고 첫 실행 로그를 끝까지 본다. 넷을 한꺼번에 켜지 않는다.

## 배포 절차

1. 로컬 수정 + `bash run_tests.sh` 한 번의 전체 출력 (`tests/test_*.py` 와 `trade/test_*.py` 둘 다 돈다)
2. 결정 가치가 있으면 ai-debate → 사용자 승인
3. 서버에 타임스탬프 백업 생성 → scp → md5 대조
4. API 서버 변경이면 `/home/ubuntu/start_trade_api.sh` 로만 재기동
5. git commit + push, 결정이면 `history.md` ADR
6. 다음 cron 실행 감독

## 체결 밴드 5% 관측 결과 (2026-10-15 종결)

2026-09-29~10-14 열여섯 회차 전수에서 의도대로 갈랐다. 드리프트는 매 회차 발화했지만
주문은 넷에서만 나갔고(열둘은 밴드 안이라 0건) 열여섯 회차 모두 목표 달성으로 끝났다.
경계 바로 아래인 4.6%·4.1% 도 안 넘어갔다. 원복 사유 여섯 중 다섯이 음성이고
레버리지 전환 왕복도 실제로 관측됐다. 밴드를 되돌릴 이유 없음 — history `#47`.

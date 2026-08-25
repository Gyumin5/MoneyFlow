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
| 23:35 평일 | 주식 V25 | `executor_stock.py` — 한국투자증권, 직접 전략 계산 |
| */5 | 감시 | `watchdog_serve.sh` |

세 sleeve 공통 판정: 시그널은 t-1 확정 종가, 체결은 당일. 매일 상태를 점검하되 종목 교체는
앵커일에만 한다(예외: 주식 V25 는 drift 발화일에도 fresh 재선정).

리밸 필요 판정은 공통으로 `need_rebal = is_daily_bar AND (snap_fire OR (canary_on AND ht ≥ threshold))`,
`ht = Σ|tgt_w − cur_w| / 2` (자본금 기준 half-turnover).

---

# 선물 (V25, Binance USDT-M)

## 파라미터

| 항목 | 값 |
|---|---|
| sleeve | D_SMA42 단일, snap_interval=95, n_snapshots=5, drift=0.03 |
| 선정/헬스 | universe 3종 cap 1/3, mom2vol(mom 18/127, vol 90d ≤ 5%), 카나리 BTC SMA42 ±1.5% |
| 레버리지 | 동적 per-coin `L = min(BTC_cap, K2_per_coin)`, Lmin/Lmid/Lmax = 2/3/4 |
| 마진모드 | CROSSED (V24 ISOLATED 에서 전환) |
| 체결 밴드 | 명목 ±5% (`DELTA_THRESHOLD = 0.05`) — 채택 BT 의 수량 ±5% 와 정합 |
| 스탑·캐시가드 | 없음. 가드 없음 (분산 + 동적 L 이 유일 방어) |
| 거래비용 | 0.04% (BT 0.06% 로 보수 측정) |
| cash buffer | 1% (`CASH_BUFFER`) |

단독 sleeve BT(5.6yr): Cal 8.12 / CAGR 312% / MDD -38.3% / Sharpe 1.90.
자산배분 60/25/15 + T1+T3U: Cal 5.72 / CAGR 106% / MDD -18.6%.

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
- 드리프트 문턱(0.03)과 혼동 금지. 드리프트는 상시 발화 상태라(레버리지 4배에서 진입가 대비 ±1.5%)
  실제 회전수를 정하는 값은 이 체결 밴드다.
- 근거: `reports/2026-08-25-fut-rebal-band.html`, ADR `history.md` 2026-08-25.
  1% 로 좁히면 수익 동률·MDD 악화·체결 1.69배(연 264회 대 156회).
- 테스트: `tests/test_fut_rebal_band.py`.

## 매 cron 실행 순서와 가드

1. `verify_position_mode_oneway` — hedge 모드면 ABORT
2. 코인별 `verify_margin_type` — CROSSED 아니면 ABORT (자동 set 하지 않는다. 비정상 상황 알림)
3. 마진 변경이 필요한 심볼만 preflight → `build_margin_plan` 단일 작업계획으로 변경·검증
4. 이미 목표 레버리지면 `set_leverage` 생략, 아니면 set → verify
5. 어느 코인이든 verify 실패 → 매매 전체 ABORT (다음 cron 재시도)
6. 상시 무결성 검사 — 이 실행이 계좌를 건드리기 전(positions_before)에, 그리고 계좌를 건드리지 않은
   실행에서만 돈다. 기준 스냅샷 갱신은 성공 판정과 분리되어 있다(자기잠김 방지, 2026-08-22)
7. `force_cancel_all_orders` → `execute_rebalance` (매도 먼저, 매수 나중)

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
| 선정/헬스 | universe 3종 cap 1/3, mom2vol(mom 20/127, vol 90d ≤ 5%), 카나리 BTC SMA42 ±1.5% |
| 유니버스 | CoinGecko Top40 ∩ Binance spot ∩ Upbit KRW ∩ 253일 이상 ∩ 거래대금 조건 |
| drift 발화 시 | refill v2 — 모멘텀 둘 다 음수(ms<0 AND ml<0)인 코인만 fresh healthy 로 교체 |
| 가드 | 없음. Upbit warning/delisting 코인만 유니버스에서 즉시 제외 |
| 거래비용 | 0.04% (BT), 실매매는 Upbit 실수수료 |
| cash buffer | 1% |

단독 sleeve BT(5.4yr): Cal 4.63 / CAGR +82% / MDD -18%.

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

# 자산배분 (60/25/15, 수동 송금)

- 주식 60% / 업비트 25% / 바이낸스 15%. 정본은 라이브 코드 `STOCK_RATIO/COIN_RATIO/FUTURES_RATIO`.
- per-sleeve cash buffer: 주식 7% / 현물 1% / 선물 1% (총 cash ≈ 5%).
- 자동 송금·자동 cap 은 폐지됐다(alloc_transit phantom buffer 영구 비활성화).
  트리거가 켜지면 텔레그램 알림만 보내고 사용자가 직접 송금한다.
- 트리거 (OR):
  · T1 — `half_turnover ≥ 20pp`
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
| 상태파일 손상 | 백업 타임스탬프 사본 | 서버 백업에서 복원. 삭제는 초기화가 아니라 상태 손실이다 |

## 배포 절차

1. 로컬 수정 + 테스트 (`tests/` 전부)
2. 결정 가치가 있으면 ai-debate → 사용자 승인
3. 서버에 타임스탬프 백업 생성 → scp → md5 대조
4. API 서버 변경이면 `/home/ubuntu/start_trade_api.sh` 로만 재기동
5. git commit + push, 결정이면 `history.md` ADR
6. 다음 cron 실행 감독

## 관측 중 (2026-08-25~)

체결 밴드 5% 전환 후 09:05 cron 10~14회 동안 로그의 `sell_check/buy_check … delta=` 로
"1% 였다면 냈을 주문" 과 실제를 대조한다(추가 라이브 코드 없음). 원복 조건:
청산·카나리 OFF 실패, 주문 오류가 없는데 5% 초과 잔차를 달성으로 선언, 중복 재주문,
잘못된 레버리지, 리스크리밋 위반, 레버리지 전환 잔차 8% 이상 누적.

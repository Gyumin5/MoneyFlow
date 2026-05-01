# V23 Spec — V22 차세대 (2026-04-30)

상태: DRAFT. AI 검토 + 합의 후 단계별 구현.

## 결정 근거 요약

V22 운영 (2026-04-27 시작) 의 후속 개선. BT 검증 (5.4yr, 2020-10 ~ 2026-04, 일별, look-ahead 차단) 으로 도출한 후보 C.

검증 단계
1. 3D 그리드 (snap × n × drift) — drift sensitivity
2. n_prime 제약 rank-sum — alloc 가로지른 범용성
3. 평탄도 검증 (300 BT) — 인접 파라미터 안정성
4. Option B/C verify — 차별화·서로소 강제 (gcd(spot_n, fut_n)=1)
5. 후보 C 직도입 검증 — drift 발화 빈도, tx 민감도, yearly Cal 분포, 자산 상관/동시 DD

## V22 → V23 변경점

| 자산 | V22 (현행) | V23 (신규) |
|---|---|---|
| stock | snap_days=125, n_snap=3, stagger=42 | snap_days=69, n_snap=3, stagger=23 |
| coin spot | D_SMA42 + H4_SMA240 EW (snap 60 동기) | D-only n_snap=7 단일, snap=217, drift=0.10 |
| coin fut | D_SMA42 + H4_SMA240 EW (snap 90 동기, L3) | D-only n_snap=3 단일, snap=57, drift=0.05 (L3) |
| 자산배분 | 60/40/0 (수동) | 60/40/0 유지 (변경 없음) |
| cron | 4h 동기 (9,13,17,21,1,5시 :05) | 1d 동기 (9시 :05) |
| 4h 데이터 | KLINE_LIMITS '4h' 1500 | 제거 |

차별화 / 서로소 충족
- spot n_snap = 7, fut n_snap = 3 (gcd=1)
- stagger: stock=23, spot=31, fut=19 (모두 distinct prime, 2-digit)

## 단독 sleeve 성능 (BT 5.4yr, 365.25/days CAGR)

| sleeve | Cal | CAGR | MDD | ymin | rebal/yr |
|---|---|---|---|---|---|
| stock sd=69 n=3 | 1.16 | +17% | -15% | +0.59 | (snap 기준) |
| spot sn=217 n=7 d=0.10 | 4.63 | +82% | -18% | -0.14 | 27 |
| fut sn=57 n=3 d=0.05 | 10.74 | +450% | -42% | -0.23 | 119 |

## 포트폴리오 성능

| alloc | Cal | CAGR | MDD | ymin |
|---|---|---|---|---|
| 60/40/0 (운영) | 3.60 | +41% | -11% | +0.56 |
| 60/30/10 (참고) | 4.60 | +62% | -13% | +1.17 |

자산 일별 상관: stock-spot 0.08, stock-fut 0.07, spot-fut 0.77
3자산 동시 -5% DD 일수: 29.2% (60/30/10)

## 비용 민감도 (tx 3x stress)

| asset | tx 1x Cal | tx 3x Cal | ymin 변화 |
|---|---|---|---|
| stock | 1.16 | 1.03 | +0.59 → +0.40 |
| spot | 4.63 | 3.27 | -0.14 → -0.65 (fragile) |
| fut | 10.74 | 10.07 | -0.23 → -0.29 |

## drift 트리거 정의

```python
def half_turnover(cur_w, tgt_w):
    keys = set(cur_w) | set(tgt_w)
    return sum(abs(tgt_w.get(k, 0) - cur_w.get(k, 0)) for k in keys) / 2

def need_rebal_drift(cur_w, tgt_w, threshold):
    return half_turnover(cur_w, tgt_w) >= threshold
```

cur_w 정의 (선물 기준)
- weight = (margin + 미실현 PnL) / port_value
- port_value = 현금 + sum(margins) + sum(unrealized_pnl)
- 자본금 기준 비중 (레버리지 곱한 명목 노출 X)

리밸 발화 조건 (변경, AI 검토 반영)
- 기존: snap 기일에만
- V23: `is_daily_bar AND not crash_cooldown AND (snap_fire OR (canary_on AND half_turnover >= threshold))`

## 마이그레이션 영향

trade_state.json 스키마
- members 구조: 2개 → 1개 (D-only)
- tranches 길이: spot 3 → 7, fut 3 → 3 (변동 없음)
- 첫 실행 시 모든 트랜치를 현재 시그널로 초기화 (기존 동작 유지)
- bar-idempotency 키 갱신 (4h → 1d)

서버 cron
- 코인/선물 4h x 6 → 1d x 1
- 주식 23:35 + 0~4시 보조 시각 유지

## 필요 코드 변경 (sync 대상 — CLAUDE.md 규칙)

### 백테스트
- strategies/cap_defend/backtest_spot_barfreq.py — V23 spec 갱신
- strategies/cap_defend/backtest_futures_full.py — V23 spec
- strategies/cap_defend/futures_ensemble_engine.py — 멤버 1개로 단순화
- strategies/cap_defend/futures_live_config.py — 새 파라미터

### 라이브 엔진
- trade/coin_live_engine.py
  - members: D_SMA42 + H4_SMA240 → D_SMA42 (n_snap=7) 단일
  - drift trigger: sleeve-level half_turnover 비교
  - 4h fetch 제거 (KLINE_LIMITS '4h' 삭제)
  - 디버그 로그: snap fire, drift fire, half_turnover 값, target weight
- trade/executor_coin.py — n_snap=7 트랜치 처리
- trade/auto_trade_binance.py (구현 예정 또는 기존 위치) — fut drift trigger
- trade/executor_stock.py — sd=69, n_snap=3, stagger=23

### 권고
- strategies/cap_defend/recommend.py — V22 → V23 표기, 새 파라미터
- strategies/cap_defend/recommend_personal.py — 동일

### 운영
- trade/ops/run_executor.sh — 변경 없음 (executor 호출 그대로)
- trade/ops/crontab.txt — 4h x 6 → 1d x 1
- trade/ops/serve.py / trade_api_server.py — 검토 (V23 표기)
- trade/asset_dashboard.html — V22 문자열 갱신
- binance_state.json / kis_trade_state.json / signal_state.json — schema 변경 점검 + schema_version 필드

### 문서
- V22_OPERATION_MANUAL.md → V23_OPERATION_MANUAL.md (또는 V22 갱신)
- CLAUDE.md — 코인 spot / 선물 / 마이그레이션 섹션 갱신
- MEMORY.md — V23 운영 명시
- progress.md / history.md — 결정 로그

## 디버깅 로그 정의 (신규)

매 D봉 닫힘 시각 (cron 09:05) 실행 시 라이브 엔진 로그
```
[V23 cron 09:05] {asset} bar_close ts={ts}
  bar_id: 2026-04-30T00:00:00Z
  schema_version: V23
  members: [{member_name}]
  current_weights: {coin: weight, ...}     ← 거래소 잔고 기반 (margin/equity)
  member_signals: {member: target_weights}
  combined_target: {coin: weight, ...}
  half_turnover_cur_to_tgt: 0.045
  drift_threshold: 0.10
  snap_fire: False
  drift_fire: False
  canary_on: True
  crash_cooldown_until: None
  pending_order_exists: False
  rebal_decision: SKIP
  reason: ''
  dry_run: False
  next_snap_fire_in: 12 days
```

리밸 발화 시 텔레그램
```
[V23 {asset} REBAL] reason=drift|snap
  half_turnover: 0.123
  threshold: 0.10
  current_w: {...}
  target_w: {...}
  delta: {coin: ±%, ...}
  fee_estimate: $X
```

부분체결 시 텔레그램
```
[V23 PENDING] {asset} {coin}
  planned_qty: {planned}
  filled_qty: {actual}
  pending_qty: {remaining}
  target_px: {target_px}
  fill_px_avg: {fill_avg}
  slippage_bps: {(fill_avg - target_px) / target_px * 10000:.1f}
  min_notional: {if any}
```

텔레그램 idempotency: 동일 (bar_id, asset, reason) 단위에서 1회만 발송 (drift 핑퐁 시 spam 방지).

## 백테스트 동등성 검증 절차

1. 라이브 엔진 dry-run (매매 X, target 산출만)
2. 동일 시각의 BT 결과와 비교
   - target weight 일치
   - drift 발화 시점 일치
   - snap 기일 일치
   - half_turnover 값 일치 (소수 셋째 자리)
3. 1주일 dry-run 후 매매 활성화

## 운영 적용 절차 (Phase F)

1. 라이브 엔진 코드 + 마이그레이션 스크립트 배포 (scp)
2. cron 정지 (기존 4h 비활성)
3. 마이그레이션 실행 (백업 + 새 구조 초기화)
4. 새 cron 활성화 (1d 09:05)
5. 첫 매매 발생 시 텔레그램 즉시 확인
6. 첫 24h 집중 모니터링 (drift 발화 빈도, 부분체결, ymin 추적)
7. 첫 1주 회고

## AI 검토 (2026-04-30) 반영

codex
- drift 부등호 `>=` 통일. 조건 괄호: `is_daily_bar AND not crash_cooldown AND (snap_fire OR (canary_on AND half_turnover >= threshold))`
- 선물 비중은 자본금 기준 (margin/equity), L3 명목 노출은 주문 산출 단계에서만 적용
- 마이그레이션 항목 추가: 미체결 취소, state 백업+검증+rollback, schema_version 명시, cron lock, bar-idempotency 초기화, spot 7-tranche 초기 스케줄
- 첫 실행 큰 주문 dry-run 권고
- 로그 추가 필드: bar_id, state_schema_version, canary_on, crash_cooldown_until, reason, dry_run, pending_order_exists, min_notional/skip_reason
- 텔레그램 idempotency (동일 bar/reason 단위)
- 첫 주 drift 발화 횟수 + 수수료 별도 집계
- ensemble framework 유지 (단일 member 형태)
- 잔고 기반 current_w 와 BT current_w 비교를 dry-run 검증 항목에 추가
- 추가 sync 대상: binance_state.json, kis_trade_state.json, signal_state.json, 대시보드/API V22 문자열, 로그/알림 포맷

gemini
- drift 위치 (CRITICAL): coin_live_engine.py 는 target 만 산출. cur_w (실제 업비트 잔고) 는 executor_coin.py 가 조회
  - 해결: drift 판별을 executor_coin.py 로 이동 또는 잔고 스냅샷을 엔진 입력으로 주입
  - 선물 auto_trade_binance.py 는 일체형이라 엔진 위치 OK
- 슬리피지 추정 (target_px vs fill_px 차이) 텔레그램 로그에 추가
- 7-tranche 첫날 캡 분할매매 안전장치 필수
- auto_trade_binance.py 내부 스케줄러 점검 (cron 외 자체 sleep 루프 있는지)
- ensemble framework 유지: ENSEMBLE_WEIGHTS = {'D_SMA42': 1.0} 형태
- drift 0.10 핑퐁 리스크 첫 주 1순위 모니터

## 결정 (반영)

1. drift 비교 위치
   - spot: executor_coin.py 가 잔고 조회 후 cur_w 산출 → engine 에 주입 → engine 이 half_turnover 비교 후 EngineResult.rebalancing_needed 에 반영. 또는 executor 가 직접 비교 후 결정.
   - fut: auto_trade_binance.py 일체형 (현재 구조 유지)
   - 결정: 잔고 주입 방식. coin_live_engine 의 compute_live_targets() 가 cur_w 인자 받아서 drift 판별. 변경폭 작음.

2. drift 부등호: `>=` 통일

3. ensemble framework 유지: ENSEMBLE_WEIGHTS = {'D_SMA42': 1.0} (spot), {'D_SMA42': 1.0} (fut)

4. spot 7-tranche 마이그레이션: 첫 실행 cap=1/3 (현행) + 분할매매 (executor 기존). 분할 매매 안전한지 dry-run 1회 필수.

5. drift 비중 정의: 자본금 기준 (margin / equity). 명목 노출 X.

6. state schema version 추가: trade_state.json["schema_version"] = "V23"

7. 마이그레이션 절차
   a. 기존 cron 정지
   b. 미체결 주문 일괄 조회 + 취소
   c. trade_state.json 백업 (timestamp 부) + 새 구조 초기화
   d. 새 cron 활성화 (1d, lock 으로 중복 방지)
   e. 첫 실행 dry-run 1회 (target 산출만, 매매 X)
   f. dry-run 결과 OK 시 매매 활성화

8. 추가 sync 대상: binance_state.json, kis_trade_state.json, signal_state.json, asset_dashboard.html (V22 문자열), trade_api_server.py 의 V22 표기

## 미해결 / 결정 필요

- (모두 반영됨)

## 리스크 및 완화

| 리스크 | 영향 | 완화 |
|---|---|---|
| 첫 실행 부분체결 / 중복주문 | 자금 손실 | dry-run 1주 + 모니터링 강화 |
| drift 핑퐁 (whipsaw) | 수수료 누적 | tx 3x BT 흡수 검증, 첫 주 빈도 모니터 |
| spot tranche 재구성 | 한번에 큰 주문 | 분할매매 (executor 기존 기능) |
| 5.4yr 단일 BT regime fit | 약세장 ymin 우연성 | 워크포워드 정책상 금지 — 모니터링 강화로 대응 |
| spot ymin 0.08 우연성 (gemini 우려) | 약세장 미검증 | drift 0.10 보수적 임계값 + sleeve 분산 |

## 변경 이력

- 2026-04-30 DRAFT (Claude + 사용자 + AI 검토)
- 2026-XX-XX V23 도입 (예정)

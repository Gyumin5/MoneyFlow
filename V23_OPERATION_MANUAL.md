# V23 운영 매뉴얼 (2026-04-30 확정)

V22 → V23. 모든 자산 1D 단일 + drift trigger. 4h 멤버 제거, cron 단순화.

## 결정 근거 (요약)

5.4yr BT (2020-10 ~ 2026-04, look-ahead 차단) 검증
- 3D 그리드 (snap × n_snap × drift) → drift sensitivity
- n_prime 제약 rank-sum (57,600 조합) → alloc 가로지른 범용성
- 평탄도 검증 (300 BT) → 인접 파라미터 안정성
- 차별화·서로소 검증 (Option B/C, 2,220 portfolio)
- C 후보 직도입 검증 (drift 발화 빈도, tx 민감도, yearly Cal, 자산 상관)

AI 합의 (gemini + codex)
- drift `>=` 통일, 자본금 기준 비중, ensemble framework 유지, 마이그레이션 절차 보완

## 파라미터

| 자산 | V22 | V23 |
|---|---|---|
| stock | sd=125, n=3, stagger=42 | sd=69, n=3, stagger=23 |
| coin spot | D_SMA42 + H4_SMA240 EW (sn=60+360 동기) | D_SMA42 단일 sn=217×7 drift=0.10 |
| coin fut | D_SMA42 + 4h_SMA240 EW (sn=90+540 동기, L3) | D_SMA42 단일 sn=57×3 drift=0.05 (L3) |
| 자산배분 | 60/40/0 | 60/40/0 (변경 없음) |
| cron | 4h x 6 (9,13,17,21,1,5시) | 1d x 1 (09시) |

차별화·서로소 충족
- spot n_snap=7, fut n_snap=3 (gcd=1)
- stagger: stock=23, spot=31, fut=19 (모두 distinct prime)

## 단독 sleeve 성능 (BT 5.4yr, CAGR 365.25/days)

| sleeve | Cal | CAGR | MDD | ymin | rebal/yr |
|---|---|---|---|---|---|
| stock sd=69 n=3 | 1.16 | +17% | -15% | +0.59 | (snap 기준) |
| spot sn=217 n=7 d=0.10 | 4.63 | +82% | -18% | -0.14 | 27 |
| fut sn=57 n=3 d=0.05 | 10.74 | +450% | -42% | -0.23 | 119 |

## 포트폴리오 성능

| alloc | Cal | CAGR | MDD | ymin |
|---|---|---|---|---|
| 60/40/0 (운영) | 3.60 | +41% | -11% | +0.56 |
| 60/30/10 (참고, 미적용) | 4.60 | +62% | -13% | +1.17 |

자산 일별 상관: stock-spot 0.08, stock-fut 0.07, spot-fut 0.77

## drift 트리거 정의

```python
def half_turnover(cur_w, tgt_w):
    keys = set(cur_w) | set(tgt_w)
    return sum(abs(tgt_w.get(k, 0) - cur_w.get(k, 0)) for k in keys) / 2

# 발화 조건
need_rebal = is_daily_bar AND not crash_cooldown AND (
    snap_fire OR (canary_on AND half_turnover(cur_w, target) >= threshold)
)
```

cur_w 정의 (자본금 기준)
- spot: 코인 평가액 / 총자산 (executor 가 Upbit 잔고에서 산출 후 engine 에 주입)
- fut: 종목 margin / equity (auto_trade_binance 가 positions['real_weight'] 직접 사용)

threshold
- spot: 0.10
- fut: 0.05

## 코드 변경 요약 (V22 → V23)

라이브 엔진
- trade/coin_live_engine.py: MEMBER_H4_SMA240 제거. MEMBER_D_SMA42 = {snap=217, n_snap=7}. ENSEMBLE_WEIGHTS = {'D_SMA42': 1.0}. half_turnover/evaluate_drift_fire utility 추가. EngineResult.drift_fire/drift_half_turnover/drift_threshold 필드 추가. SCHEMA_VERSION='V23'. KLINE_LIMITS '4h' 제거.
- trade/executor_coin.py: cur_w 자본금 비중 산출 (Upbit 잔고 → KRW 비중) → engine 에 주입. result.drift_fire 시 rebalancing_needed=True. V23 debug log.
- trade/auto_trade_binance.py: STRATEGIES 단일 (D_SMA42 sn=57 n=3). ENSEMBLE_WEIGHTS={'D_SMA42':1.0}. fetch_all_data 에서 4h 제거. drift 트리거 로직 (cur_w fut = real_weight). DRIFT_THRESHOLD_FUT=0.05. SCHEMA_VERSION='V23'.
- trade/executor_stock.py: SNAP_PERIOD_DAYS=69, SNAP_STAGGER_DAYS=23.

권고
- strategies/cap_defend/recommend.py: 헤더 V23, 멤버 표기 단일화.
- strategies/cap_defend/recommend_personal.py: STRATEGY_VERSION='V23', VERSION_HISTORY 추가, FUTURES/COIN_MEMBER_META V23 갱신, STOCK_ANCHOR_DAYS = (1, 24, 47).

설정
- strategies/cap_defend/futures_live_config.py: 단일 strategy, snap_interval_bars=57, drift_threshold=0.05.
- trade/ops/crontab.txt: 4h x 6 → 1d x 1 (5 9 * * *).

마이그레이션
- trade/migrate_v22_to_v23.py: state 파일 백업 + 새 schema 초기화 (snapshots 길이 조정, bar_counter 0, last_bar_ts None, schema_version='V23', rebalancing_needed=True).

## 마이그레이션 절차 (codex 검토 반영, 2026-04-30)

순서 핵심: cron 활성화는 수동 dry-run 검증 통과 후 마지막에. 첫 `--trade` 전에 reason/half_turnover/order_notional/per-coin delta 반드시 확인.

1. 기존 cron 정지 (수동)
2. 미체결 주문 일괄 취소
   ```
   # spot: executor 시작 시 cancel_all() 자동 호출
   # fut: force_cancel_all_orders() 호출
   ```
3. 코드 배포 (scp)
4. 마이그레이션 dry-run
   ```bash
   python3 trade/migrate_v22_to_v23.py --dry-run
   ```
5. 마이그레이션 apply
   ```bash
   python3 trade/migrate_v22_to_v23.py --apply
   ```
6. 수동 dry-run 1회 (target 산출만, 매매 없음)
   ```bash
   python3 trade/executor_coin.py --dry-run
   python3 trade/auto_trade_binance.py --dry-run
   ```
   검증 항목 (필수):
   - schema_version='V23' 모든 state 파일 마크
   - reason / snap_fire / drift_fire / half_turnover 일치
   - target_w 합계 = 1.0, current_w 합계 = 1.0
   - 선물 current_w 가 margin/equity 기준인지 (명목 노출 아님)
   - per-coin delta 가 의도한 첫 주문인지 (V22 → V23 mismatch 폭 확인)
   - 동일 bar 중복 실행 0건
7. 검증 통과 시 cron 활성화 (1d 09:05)
8. 매매 활성화 (--trade) — 다음 09:05 cron 트리거 또는 수동 1회

검증 실패 시
- rebalancing_needed=False 로 강제 복귀 (state JSON 수동 편집)
- 또는 migrate --rollback {ts} 로 V22 복귀

## 디버깅 로그

매 cron 실행 시 라이브 엔진 출력
```
[V23 cron 09:05] {asset} bar_close ts={ts}
  schema_version: V23
  members: ['D_SMA42']
  current_weights: {coin: weight, ...}
  combined_target: {coin: weight, ...}
  half_turnover_cur_to_tgt: 0.045
  drift_threshold: 0.10 (spot) / 0.05 (fut)
  snap_fire: False
  drift_fire: False
  canary_on: True
  pending_order_exists: False
  rebal_decision: SKIP
  reason: ''
```

리밸 발화 시 텔레그램
```
⚠ V23 {asset} drift: ht={ht:.3f} ≥ {threshold:.2f} → 리밸
```

## 운영 사고 대응

부분체결
- spot: pending_trades state. monitor 가 다음 사이클에서 재시도
- fut: rebalancing_needed=True 유지 → 다음 cron 재실행

drift whipsaw 의심 시
- 발화 빈도 (실제 vs BT 119/yr) 비교
- 임계값 0.05 → 0.10 으로 보수화 검토
- 단계적: drift OFF (snap 만), 알파 -36% 손실 감수

상태파일 손상
```bash
python3 trade/migrate_v22_to_v23.py --rollback {YYYYMMDDHHMMSS}
```

## 점검 체크리스트 (배포 후 첫 1주)

- [ ] cron 새 시각 (09:05) 정상 실행 확인
- [ ] 4h cron 비활성 (잔존 cron 없음)
- [ ] schema_version='V23' 모든 state 파일 마크 확인
- [ ] 첫 실행 부분체결 0건 확인
- [ ] drift 발화 횟수 정상 범위 (spot ~5/wk, fut ~2/wk)
- [ ] yearly Cal 추적 (BT vs live 비교)
- [ ] tx_cost 누적 (BT 흡수 범위 내)

## 첫 주 tripwire (보수화 트리거, 2026-04-30 추가)

- spot drift fire `> 2회/day` OR `> 8회/week` → threshold 0.10 → 0.12~0.15 검토
- fut drift fire `> 1회/day` OR `> 4회/week` → threshold 0.05 → 0.07~0.10 검토
- 첫날 마이그레이션 mismatch 1회성 발화는 카운트 분리
- half_turnover 실제 분포가 임계값 근처에 반복 접근 시 핑퐁 의심

## fallback (drift OFF, snap only)

런타임 플래그 (코드 수정 없이 끌 수 있도록):
- `coin_live_engine.py`: `DRIFT_ENABLED = True` 모듈 상수
- `auto_trade_binance.py`: `DRIFT_ENABLED_FUT = True` 모듈 상수
- False 로 토글 시: target 계산은 유지, drift_fire 만 강제 False, snap_fire 만으로 리밸 트리거
- 적용: 핑퐁/whipsaw 의심 시 잠시 OFF 후 분석 (BT 알파 -36% 손실 감수)

## 미해결 / 추후 검토

- 60/30/10 alloc 도입 (사용자 결정 필요)
- 약세장 (코인 -30% 이상) 진입 시 ymin 변동 모니터
- drift 0.10 핑퐁 발생 시 보수화 결정
- spot ymin -0.08 우연성 검증 (월별/연도별 ymin 분포)

# V22 운영 매뉴얼 (2026-04-21 확정)

V21 기반 + 현물 C 슬리브 추가. 주식 V17, 선물 V21 동일.

## 변경 개요

V21 → V22:
- 현물: V21 3멤버 앙상블 유지 + Strategy C 슬리브 추가 (dip-buy mean reversion).
- 선물: V21 그대로 유지 (2022 bear MDD 악화 이슈로 C 미적용).
- 주식: V17 유지.
- 자산배분: 60/35/5 유지.

Cron:
- 현물: 5 9 * * * → 5 * * * * (09:05 1회/일 → 매시간 :05)
  C 슬리브가 1h봉 기반이므로 매시간 시그널 체크.
  V21 D봉 로직은 bar-idempotency로 D봉 닫힐 때만 실행.
- 선물: 변경 없음 (5 9,13,17,21,1,5 * * *).

## C 슬리브 스펙 (현물, champion: s_dthr12_tp3 + A2_bounce_w1)

### 신호
- Interval: 1h봉 (Binance kline)
- dip_bars: 24 (24시간 누적 수익률)
- dip_thr: -0.12 (-12% 이하로 떨어지면 dip 판정)
- tp_pct: 0.03 (+3% 익절)
- tstop_hours: 24 (24시간 후 시간컷)
- universe: 시총 Top15 ∩ Binance USDT ∩ Upbit KRW ∩ V21 effective_universe
- n_pick: 1 (동시 1개 포지션)
- swap_edge: 1 (백테 설정, 라이브 MVP는 swap 없음 — 보유 중 다른 dip 시그널 나와도 유지)
- cap_per_slot: 0.15 (실전 초기, 총자산의 15%까지 단일 포지션)
- 스탑로스: 없음

### A2 bounce 가드 (라이브 재현)
백테 filter_bounce_confirm = "시그널 봉 이후 1h 내 양봉 찾기 → 양봉 다음 봉 Open 진입".

라이브 state machine:
- 봉 T 닫힘 (cron T+5분): dip 조건 통과 + 봉 T가 양봉(Close>Open) → pending_entry 저장
- 봉 T+1 닫힘 (cron T+1시+5분): pending 다음 봉 도달 → 봉 T+1 Open 가격으로 진입
- 봉 T+N: 포지션 TP or tstop 체크 → 청산

### State 스키마 (trade_state.json 추가분)
```json
"c_sleeve": {
  "position": {
    "coin": "ETH",
    "entry_ts": "2026-04-21T10:00:00Z",
    "entry_px": 3420.5,
    "tp_px": 3523.1,
    "tstop_ts": "2026-04-22T10:00:00Z",
    "krw_spent": 1500000,
    "dip_ret": -0.14
  },
  "pending_entry": {
    "coin": "ETH",
    "bar_ts": "2026-04-21T09:00:00Z",
    "dip_ret": -0.14
  },
  "last_signal_bar_ts": "2026-04-21T09:00:00Z"
}
```
모두 없을 수 있음 (position / pending_entry 중 하나만 있거나 둘 다 없음).

## 아키텍처

3단계 분리:
1. `cle.compute_c_intent(state, bars_1h, universe, now)` — 주문 X, CIntent 반환
   action: hold / enter / exit / pending_save / pending_expire
2. `cle.apply_c_to_target(v21_target, c_position, c_intent, total_pv)` — merged target 생성
3. `ec.finalize_c_state(state, intent, fill_result)` — 체결 후 state 갱신

실행자 (`ec.handle_c_only`) 는 V21이 체결 없는 skip 경로에서 C만 단독 체결.

## 체결 순서

1. V21 신호 계산 → V21 target
2. C intent 계산 → CIntent
3. V21 target 변화 없음 (target 불변 + rebalancing_needed=False):
   → handle_c_only (C 단독 체결)
4. V21 target 변화 있음:
   → V21 execute_delta (V21 타겟만 체결)
   → handle_c_only (C 별도 체결)
   (V21+C 동시 이벤트는 연 ~월 1회라 MVP는 순차 체결 수용)
5. state 저장

## 기대 성과 (백테 5.5년)

| 지표 | V21 단독 | V21+C | V21+C champion (V22) |
|------|---------|-------|---------------------|
| Holdout Cal | 1.75 | 2.32 | 3.24 |
| Holdout CAGR | 33.9% | 37.4% | 47.0% |
| Holdout MDD | -19.3% | -16.1% | -14.5% |
| Final (5.5년) | ×13.2 | ×15.2 | ×20.5 |

실전 초기 cap 0.15 기준으로는 개선폭 ~1/2 수준 예상. CAGR 33.9% → ~39.7%.

## 실전 투입 절차

1. 로컬 dry-run 검증 (state 더미로 일련 사이클 시뮬).
2. AI 코드 리뷰 (gemini + codex).
3. 서버 cron 변경 (5 9 * * * → 5 * * * *).
4. 서버 배포 (scp coin_live_engine.py, executor_coin.py).
5. 첫 24시간 집중 모니터링 (매시간 로그, C 발동 여부).
6. 1주일 운영 후 cap 상향 검토 (0.15 → 0.20 → 0.25 → 0.333).

## 주요 리스크

1. V21+C 코인 겹침 — 백테와 동일 허용. ETH 33% V21 + 15% C = 48% 합산.
2. C 진입 pending 만료 (next bar 놓침) — 정확히 +1h 경계 체크로 방지.
3. 매수 체결 미확정 (Upbit rate limit / delay) — position 기록 보류, 다음 사이클 재시도.
4. cron 빈도 10배 증가 — 외부 API (CoinGecko/Binance/Upbit) rate limit 여유 확인.

## 롤백

긴급 롤백 시:
1. git revert HEAD (V22 commits).
2. cron 5 9 * * * 로 복원.
3. 서버 재배포.
4. state의 c_sleeve 섹션 삭제 (선택 — V21 동작엔 무관).

## 주요 파라미터 레퍼런스

`trade/coin_live_engine.py`:
```python
C_SLEEVE_CFG = {
    'interval': '1h',
    'dip_bars': 24,
    'dip_thr': -0.12,
    'tp_pct': 0.03,
    'tstop_hours': 24,
    'universe_size': 15,
    'n_pick': 1,
    'cap_per_slot': 0.15,     # 실전 초기
    'bounce_window_hours': 1,
}
```

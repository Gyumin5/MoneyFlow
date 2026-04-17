# V21 운영 매뉴얼 (2026-04-17 확정)

## 전환 개요

- 현물 코인: V20 (D_SMA50 + 4h_SMA240 50:50) → V21 (D봉 3멤버 EW 1/3씩)
- 선물: V20 d005 4전략 (동적 lev 3~5x, stop -15%, cash_guard) → V21 고정 3x 3멤버 EW 1/3씩, 가드 없음
- 주식: V17 고정 (변경 없음)
- 자산배분 초기값: 60% / 40% / 0% (주식 / 현물 / 선물)
- 밴드: sleeve r30 (자산 가중치 × 30% = 밴드폭) — recommend_personal 알림용
- 자산간 리밸런싱: 수동 (사용자가 recommend_personal 보고 직접)

## 현물 V21 (코인)

앙상블: ENS_spot_k3_4b270476 (k=3 EW)

| 멤버 | Interval | SMA | Mom(S) | Mom(L) | vol_cap | Snap bars |
|---|---|---|---|---|---|---|
| D_SMA50 | D | 50 | 20 | 90 | daily 5% | 90 |
| D_SMA150 | D | 150 | 20 | 60 | daily 5% | 90 |
| D_SMA100 | D | 100 | 20 | 120 | daily 5% | 90 |

공통:
- n_snapshots = 3 (90봉 내에서 30봉 간격 stagger)
- canary_hyst = 1.5%
- health = mom2vol (vol_lookback 90d)
- universe_size = 3
- cap = 1/3 (단일 코인 최대 1/3)
- gap_threshold = -0.15 (직전 완성 D봉 -15% 초과 시 제외)
- exclusion_days = 30

앙상블 가중치: D_SMA50=1/3, D_SMA150=1/3, D_SMA100=1/3

Cron: `5 9 * * *` (한국시간 09:05, 1회/일)

유니버스: CoinGecko Top40 ∩ Binance spot TRADING ∩ Upbit KRW normal ∩ 253일 이상 ∩ 30일 평균 10억원 이상 → Top40 (V20과 동일)

## 선물 V21

앙상블: ENS_fut_L3_k3_12652d57 (k=3 EW)

| 멤버 | Interval | SMA | Mom(S) | Mom(L) | vol_cap | Snap bars |
|---|---|---|---|---|---|---|
| 4h_S240_SN120 | 4h | 240 | 20 | 720 | daily 5% | 120 |
| 4h_S240_SN30 | 4h | 240 | 20 | 480 | daily 5% | 30 |
| 4h_S120_SN120 | 4h | 120 | 20 | 720 | daily 5% | 120 |

공통:
- n_snapshots = 3
- canary_hyst = 1.5%
- health = mom2vol
- universe_size = 3
- cap = 1/3
- 레버리지: 고정 3배 (L3)
- 스탑: 없음 (stop_kind=none, STOP_PCT=0)
- 캐시 가드: 없음 (STOP_GATE_CASH_THRESHOLD=0)
- 동적 레버리지: 비활성 (floor=mid=ceiling=3)

앙상블 가중치: 각 1/3

Cron: `5 9,13,17,21,1,5 * * *` (한국시간, 4h봉 닫힘 + 5분)

## 상태 파일 변경

### 코인 (trade_state.json)
변경 전 (V20): `members.D_SMA50` + `members.4h_SMA240`
변경 후 (V21): `members.D_SMA50` + `members.D_SMA150` + `members.D_SMA100`
다른 키는 유지 (excluded_coins, last_combined, last_target_snapshot 등)

### 선물 (binance_state.json)
변경 전 (V20): `strategies.4h_d005` + `2h_S240` + `2h_S120` + `4h_M20`
변경 후 (V21): `strategies.4h_S240_SN120` + `4h_S240_SN30` + `4h_S120_SN120`

초기 canary 상태: 모두 OFF (첫 실행 시 OFF→ON 전환으로 신규 진입)

## 실전 투입 절차

1. 로컬에서 V21 코드 작성 완료
2. AI 검토 (gemini+codex)
3. 로컬 dry-run (state 더미로 시뮬)
4. 기존 서버 cron 중단
5. 기존 state 파일 백업 (trade_state_v20_backup.json, binance_state_v20_backup.json)
6. 기존 포지션 전량 청산 (Upbit + Binance)
   - Upbit: 모든 코인 → KRW
   - Binance: 모든 포지션 → USDT (계좌 잔액 그대로 유지)
7. V21 초기 state 생성 (canary OFF, excluded_coins={})
8. scp로 V21 파일 배포
9. 서버 cron 교체:
   - 코인: `5 9 * * *`
   - 선물: `5 9,13,17,21,1,5 * * *`
10. 다음 cron 시점에 자동 진입 (canary OFF → ON 전환 시 신규 매수)

## 백테스트

- 현물: `strategies/cap_defend/backtest_spot_barfreq.py` V21 spec 지원 확장
- 선물: `strategies/cap_defend/backtest_futures_v21.py` 신규 생성
- V12~V21 전체 백테스트는 README에서 안내

## 주요 변경 요약

| 항목 | V20 | V21 |
|---|---|---|
| 코인 멤버 수 | 2 (D+4h) | 3 (모두 D) |
| 코인 snap | 30봉(D) / 60봉(4h) | 90봉(D 공통) |
| 코인 cron | 매시간 :05 | 09:05 일 1회 |
| 선물 멤버 수 | 4 (d005/S240/S120/M20) | 3 (S240×2 + S120) |
| 선물 레버리지 | 동적 3~5x | 고정 3x |
| 선물 스탑 | -15% prev_close + cash_guard | 없음 |
| 선물 cron | 매시간 | 4h마다 (6회/일) |

## 복구 / 롤백

긴급 롤백 시:
1. git checkout으로 V20 파일 복원
2. state 백업 파일 복원
3. cron 원래대로 복원
4. 포지션은 새 target 기준으로 리밸되므로 자동 전환

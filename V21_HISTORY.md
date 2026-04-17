# V21 개발 히스토리 (2026-04-15 ~ 2026-04-17)

V20 → V21 전환 과정에서 수행된 실험, 결정, 검증 및 배포 절차 정리.

상세 운영 스펙은 [V21_OPERATION_MANUAL.md](./V21_OPERATION_MANUAL.md), 버전 진화는 [strategies/cap_defend/STRATEGY_EVOLUTION.md](./strategies/cap_defend/STRATEGY_EVOLUTION.md) 참조.

---

## 타임라인 요약

| 날짜 | 주요 작업 |
|---|---|
| 2026-04-15 | iter_refine 5-stage peak refinement 파이프라인 구성 |
| 2026-04-16 | dense grid 결과 Phase-4 robustness 분석, 과적합 의심 |
| 2026-04-17 오전 | 10배수 그리드(10x) 재실행, true blind holdout 검증 |
| 2026-04-17 오후 | Leverage L2/L3/L4 비교, 현물 앙상블 고정, AI 3자 종합 검토 |
| 2026-04-17 저녁 | V21 live engine 구현, 서버 배포, cron 교체 |

---

## Phase-1~4 연구 파이프라인

5,472개 config(10배수 그리드, D+4h봉)을 다단계로 좁혀 최종 앙상블을 산출.

```
Phase-1: brute force (5,472 configs, 10x grid)
  ↓  phase1_10x/raw.csv
Phase-2: plateau filter (axis-neighbor 성과 일치 ≥ 0.85)
  ↓  phase2_10x/survivors.csv (461/938, 49% 통과)
Phase-3: ensemble (k=1/2/3, pool corr gate)
  ↓  phase3_10x/{spot_top,fut_top}.csv
Phase-4: 3자산 allocation grid (stock 60% 고정 + spot + fut)
  ↓  phase4_10x/raw.csv (25,830 rows)
Sub-period rank-sum: 10 semi-annual windows (H1 2021~H2 2025)
True blind holdout: train 2020.10~2023.12 / holdout 2024.01~2026.04
```

핵심 스크립트: `strategies/cap_defend/research/phase{1,2,3,4}_*.py`, `iter_refine.py`, `bridge_iter_to_phase1.py`, `plateau_check.py`, `run_subperiod_ranksum.py`.

---

## 10배수 그리드 재설계 (2026-04-17)

**배경:** dense grid(SMA 39/44/127 같은 연속값)에서 과적합 의심. 엄격 10배수로 재실행.

**10x 그리드 스펙:**
- D: sma [20,30,50,100,150] / ms [10,20,30,60] / ml [60,90,120,240] / snap [20,30,60,90] / vol 0.03/0.05/0.07
- 4h: sma [120,240,480,720] / ms [20,60,120] / ml [120,240,480,720] / snap [30,60,120,180] / vol d0.05/b0.5/b0.7
- 2h 제거 (노이즈 + 동일 universe/canary로 직교성 약함)
- 총 5,472 configs (dense 16,416 대비 33%)

**3-anchor OOS Cal_mean 비교:**

| Allocation | 10x | dense | 개선 |
|---|---|---|---|
| 60/30/10 abs10 | 3.00 | 2.61 | +15% |
| 60/35/5 abs15 | 2.91 | 2.50 | +16% |
| 60/25/15 abs | 2.67 | 2.69 | 동률 |
| 60/35/5 L4 abs15 (운영) | - | 2.32 | (기준) |

**발견:** coarse grid = regularization. 좁은 noise island 회피. 10x top은 전원 k=3 EW, dense top은 k=1 (dense near-dup filter가 멤버 다양성 죽임).

---

## True Blind Holdout (2026-04-17)

train 2020-10 ~ 2023-12 only로 Phase-1~4 전체 재실행 → holdout 2024-01 ~ 2026-04 검증.

| label | train_Cal | hold_Cal | hold_CAGR | hold_MDD |
|---|---|---|---|---|
| 60/25/15 sleeve | 6.66 | 1.66 | 30.2% | -18.2% |
| 60/30/10 sleeve | 6.44 | 1.68 | 29.0% | -17.2% |
| 60/25/15 abs | 7.51 | 1.76 | 27.2% | -15.5% |
| 60/35/5 sleeve | 5.84 | 1.52 | 23.9% | -15.7% |
| 60/35/5 abs | 6.68 | 1.55 | 22.2% | -14.3% |
| 60/30/10 abs | 6.93 | 1.34 | 23.5% | -17.6% |
| 60/40/0 abs | 5.82 | 1.43 | 17.8% | -12.5% |
| 60/40/0 sleeve | 5.42 | 1.39 | 17.0% | -12.2% |

**결론:**
- IS Cal의 ~75%가 선택편향 (train 5~7 → holdout 1.3~1.8)
- 전 후보 holdout Cal > 1.0 (BTC buy&hold 0.53 대비 2.5~3.3배) → 전략 알파 실존
- sleeve > abs (holdout에서 sleeve 안정)
- 선물 10~15% + sleeve 공통 상위
- **경고:** holdout 2024~2026이 상승/회복장 위주 → 진짜 bear OOS 부재

---

## Leverage L2/L3/L4 비교 (Sub-period Rank-sum)

Phase-4 결과에서 배분별 top-1(CalxCAGR)을 뽑아 10 semi-annual windows로 ranksum.

| label | ranksum | fp_Cal | fp_CAGR | fp_MDD |
|---|---|---|---|---|
| **60/35/5 L3** | **61** | 3.23 | 38.5% | -11.9% |
| **60/30/10 L3** | **68** | 3.41 | 43.2% | -12.7% |
| **60/25/15 L3** | **77** | 3.53 | 47.9% | -13.6% |
| 60/35/5 L4 | 84 | 3.19 | 41.5% | -13.0% |
| 60/35/5 L2 | 88 | 3.10 | 36.0% | -11.6% |
| 60/30/10 L4 | 99 | 3.13 | 47.0% | -15.0% |
| 60/30/10 L2 | 100 | 3.17 | 36.7% | -11.6% |
| 60/20/20 L3 | 101 | 3.56 | 53.4% | -15.0% |
| 60/25/15 L4 | 103 | 3.36 | 54.8% | -16.3% |
| 60/25/15 L2 | 105 | 3.12 | 40.2% | -12.9% |
| 60/20/20 L4 | 113 | 3.46 | 60.1% | -17.4% |
| 60/20/20 L2 | 121 | 3.20 | 41.6% | -13.0% |

**L3가 robustness 1~3위 독점.** L4는 절대 CAGR 최고지만 기복 심함 + 청산 위험. L2는 MDD 최저지만 의외로 robustness 열세.

**선물 앙상블 단독 청산 횟수 (선물 100%, 2020.10~):**
- L2 70fb40c0(D봉 k=3): 청산 0회, MDD -41%
- L2 b48b3541(4h k=3): 청산 1회, MDD -34.5%
- L3 12652d57(4h k=3): 청산 2회, MDD -49.4%
- L3 fd2dfed2(4h k=3): 청산 2회, MDD -50.8%

3자산 배분에서 선물 5~20%만 차지하므로 포트폴리오 레벨 청산 위험은 훨씬 낮음.

---

## 현물 앙상블 고정 (AI 3자 만장일치)

후보 3개:

| Tag | k | 멤버 | Cal | CAGR |
|---|---|---|---|---|
| **ENS_spot_k3_4b270476** | 3 | SMA50+SMA150+SMA100 | 2.40 | 63.0% |
| ENS_spot_k2_3c2b46f7 | 2 | SMA50+SMA150 | 2.42 | 63.5% |
| ENS_spot_k2_f8f52424 | 2 | SMA50+SMA100 | 2.39 | 62.7% |

공통 핵심 멤버: `spot_1D_S50_M20_90_d0.05_SN90` (SMA50, Mom20/90, daily vol 5%, snap 90).

**선정: ENS_spot_k3_4b270476 (k=3 EW)** 근거:
1. L2/L3 12개 배분 조건 중 10개에서 1위 (범용 robustness)
2. k=3 분산효과
3. k=2 대비 성과 차이 노이즈 수준 (Cal -0.02, CAGR -0.5%p)
4. 3멤버 전부 D봉/vol5%/snap90 동일 구조 → 실매매 복잡도 증가 미미

---

## AI 3자 종합 검토

**Claude Code Review (Agent):**
- Liq 카운트는 실제 강제청산 맞음 (futures_ensemble_engine.py L118 hit_liq)
- combine_targets 시점 정합성 확인 (look-ahead 명시적 증거 없음)
- mix_eq sleeve vs abs 비용 일관성 의심 (확실하지 않음)

**Gemini:**
- IS에만 2022 bear 포함, Holdout은 상승장 위주 → 진짜 bear OOS 부재
- 변경안 B(60/30/10 L3 sleeve) 지지
- 시스템 서킷브레이커(포트폴리오 -20% 등) 도입 권장

**Codex:**
- Holdout 보고 최종 1개 고르면 data reuse (blindness 훼손)
- B 또는 C 권장 (D 60/20/20은 2022 취약 기록)
- Sharpe 누락된 비교표 보완 필요
- crash week/exchange outage/partial fill 스트레스 테스트 필요

**3자 공통 우려:**
1. Holdout이 bear 부재
2. Data reuse 위험
3. 가드 없음 tail risk
4. 스트레스 테스트 부재

**3자 합의 추천:**
**B안 60/30/10 L3 sleeve** (Cal 3.41 / CAGR 43.2% / MDD -12.7%)
- 현물: k3_4b270476
- 선물: L3 12652d57 (4h봉 k=3)
- 밴드: sleeve r30

---

## Critical Bug Fix (Codex 지적)

**문제:** 초기 auto_trade_binance.py V21에서 `STOP_PCT=0.0 + STOP_GATE_CASH_THRESHOLD=0.0`으로 가드를 비활성화한다고 설정했으나, 실제로는:

- L1251: `if cash_w < 0.0:` → cash_w는 항상 0 이상이므로 거의 항상 False → `sync_stop_orders()`가 항상 실행
- L1268-1270: STOP_PCT=0이면 `stop_price = max(prev_close * 1.0, entry_price * 1.0)` → 현재가 근처 STOP_MARKET 발주
→ 포지션 진입하자마자 스탑 트리거되어 청산될 위험

**수정:**
```python
# auto_trade_binance.py:sync_stop_orders()
if STOP_PCT <= 0.0 or STOP_GATE_CASH_THRESHOLD <= 0.0:
    log.info("STOP OFF (V21: 가드 비활성, STOP_PCT=0)")
    return
```
early return으로 완전 비활성화 보장. 배포 후 "STOP OFF" 로그로 확인됨.

---

## V21 배포 절차 (2026-04-17 저녁)

> 서버 IP/SSH 정보는 개인 운영 매뉴얼 참조. 공개 저장소에 포함하지 않음.

1. 서버 cron 중단 (코인/선물 라인 주석 처리)
2. state 파일 백업:
   - `trade_state_v20_backup_20260417_125334.json`
   - `binance_state_v20_backup_20260417_125334.json`
   - `trade_state_v20_final_20260417_125515.json`
   - `binance_state_v20_final_20260417_125515.json`
   - crontab 백업: `/tmp/crontab_v20_backup_*.txt`
3. 포지션 전량 청산:
   - Upbit: BTC(0.114) / ETH(3.63) / TRX(131,229) 시장가 매도 → KRW 2.31억
   - Binance: BTC / TRX / TON / BNB / CC 전량 MARKET close → $1069 USD
4. V21 파일 scp 배포:
   - `trade/coin_live_engine.py`
   - `trade/auto_trade_binance.py`
   - `trade/executor_coin.py`
   - `strategies/cap_defend/recommend_personal.py`
   - `V21_OPERATION_MANUAL.md`
5. V20 state 파일 move (V21 첫 실행에서 자동 재생성)
6. V21 executor 수동 첫 실행:
   - 코인: D_SMA50 canary ON (BTC > SMA50 + 1.5%), D_SMA150/100 OFF (더 엄격한 SMA) → TRX 11.1% 매수 (2515만 KRW)
   - 선물: 3멤버 모두 canary ON → TRX 33.3% × 3x 매수 (3222 코인 $1048)
   - "STOP OFF (V21: 가드 비활성)" 로그 확인
7. cron V21로 교체:
   - 코인: `5 9 * * *` (일 1회 09:05 KST, D봉 하루 1번)
   - 선물: `5 9,13,17,21,1,5 * * *` (4h마다 6회, 봉 닫힘 + 5분)
8. recommend_personal 실행 → HTML + 텔레그램 정상 확인
9. 13:05 cron 자동 첫 실행: `매매 스킵: rebalancing_needed=false`, `STOP OFF` 확인

---

## V21 최종 스펙 요약

### 현물 (코인) V21: ENS_spot_k3_4b270476

| 멤버 | interval | SMA | Mom(S) | Mom(L) | vol_cap | snap |
|---|---|---|---|---|---|---|
| D_SMA50 | D | 50 | 20 | 90 | daily 5% | 90 |
| D_SMA150 | D | 150 | 20 | 60 | daily 5% | 90 |
| D_SMA100 | D | 100 | 20 | 120 | daily 5% | 90 |

- 1/3씩 EW, canary_hyst 1.5%, Top3, cap 1/3, gap -15% / excl 30일
- Cron: `5 9 * * *`

### 선물 V21: ENS_fut_L3_k3_12652d57

| 멤버 | interval | SMA | Mom(S) | Mom(L) | vol_cap | snap |
|---|---|---|---|---|---|---|
| 4h_S240_SN120 | 4h | 240 | 20 | 720 | daily 5% | 120 |
| 4h_S240_SN30 | 4h | 240 | 20 | 480 | daily 5% | 30 |
| 4h_S120_SN120 | 4h | 120 | 20 | 720 | daily 5% | 120 |

- 1/3씩 EW, 고정 3x 레버리지, 가드 없음 (stop_kind=none)
- Cron: `5 9,13,17,21,1,5 * * *`

### 자산배분 V21

- 주식 V17 유지 (60%)
- 현물 V21 (40%)
- 선물 V21 (0%, 수동 이동 대기)
- 밴드: sleeve r30 (weight × 30%, 최소 2%p)
- 리밸런싱: 수동 (recommend_personal에서 알림만)

---

## 미해결 / 다음 단계

1. **진짜 bear OOS 검증:** 실전 투입 후 첫 bear가 시험대
2. **포트폴리오 시스템 서킷브레이커:** -20% 등 catastrophic guard 설계 검토
3. **스트레스 테스트:** crash week / exchange outage / partial fill / slippage 확대
4. **Shadow paper 병행:** B와 C 2~3개월 병행 추적 후 확정
5. **선물 비중 수동 이동:** 현재 바이낸스 $1069 → 업비트로 이동 (목표 0% vs 실제 4.7%)

---

## 관련 산출물

```
strategies/cap_defend/research/
├── phase1_10x/raw.csv                    (5,472 rows)
├── phase1_10x_train/raw.csv
├── phase2_10x/survivors.csv              (461)
├── phase3_10x/{spot_top,fut_top}.csv
├── phase3_10x_train/{spot_top,fut_top}.csv
├── phase4_10x/raw.csv                    (25,830 rows)
├── phase4_10x_train/raw.csv              (33,840 rows)
├── phase4_10x_robustness/
│   ├── subperiod_ranksum.csv             (8 candidates × 10 windows)
│   ├── subperiod_detail.csv
│   ├── leverage_comparison.csv           (L2/L3/L4 비교)
│   ├── leverage_detail.csv
│   ├── full_sleeve_ranksum.csv           (30 candidates)
│   └── block_bootstrap.csv
├── holdout_candidates.csv
├── holdout_results.csv
├── phase_status.html                     (종합 대시보드)
└── run_subperiod_ranksum.py, run_block_bootstrap.py
```

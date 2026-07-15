# 빈 스냅 트랜치 동적 재진입 — 백테스트 변형 레지스트리 (ai-debate 20260712T033542Z 발굴)

목적: 최선 1개 수렴이 아니라, F0 baseline + 21개 변형을 전부 백테스트해서 그룹별 tradeoff 표로 결과 비교.

## 공통 규칙 (모든 변형 고정)
- 기본 V24 spot: D_SMA42, n_snapshots=7, offsets=[0,31,62,93,124,155,186], snap_interval=217, universe_size=3, cap=1/3, health=mom2vol(ms20>0 & ml127>0 & vol90d<=0.05), canary=BTC/SMA42 hyst 1.5%.
- 모든 재진입 판정은 t-1 종가 신호로 계산, t 체결 (look-ahead 금지).
- empty tranche 정의: target risky weight <= 1e-6 AND 현재 risky value + pending buy notional <= max(1e-6*NAV, min_order_krw).
- K 제한 우선순위: 성과점수 아님. empty_age 내림차순, 동률이면 phase offset [0,31,62,93,124,155,186] 순.
- 별도 명시 없으면 재진입 후 target은 다음 자기 217봉 anchor까지 고정(freeze).
- 모든 primary 변형에서 canary OFF는 기존처럼 즉시 청산(청산 지연/무시 변형은 제외됨).
- 비용: BT 0.04% + Upbit 실수수료 반영. 검증: window×stride rank-sum n≈72 + 10-anchor 평균·분산.

## 변형 목록 (F0 + 21)
| ID | 정의 | 파라미터 | 그룹 |
|----|------|----------|------|
| F0 | 현행 baseline | anchor-only, drift=0.10, refill v2 유지, 동적 empty refill 없음 | G0 |
| A1 | 빈 트랜치 daily 무제한 재진입 | 대상=empty only, 주기=daily, K=∞, 조건=canary_on & healthy>=1, sizing=full, after=freeze | G1,G4 |
| A2 | 빈 트랜치 daily K=1 | A1 + 하루 최대 1개, priority=empty_age then phase | G1,G4 (기본 제한형 대표) |
| A3 | 빈 트랜치 daily K=2 | A1 + 하루 최대 2개 | G1,G4 |
| A4 | 빈 트랜치 5봉 cadence | empty_age%5==0, K=1, healthy>=1, full, freeze | G1 |
| A5 | 빈 트랜치 10봉 cadence | empty_age%10==0, K=1 | G1 |
| A6 | 빈 트랜치 21봉 cadence | empty_age%21==0, K=1 | G1 |
| A7 | 재진입 후 daily 재평가 허용 | A2로 재진입한 tranche는 다음 anchor 전까지도 daily compute_weights 재호출 | G2 |
| D1 | 빈 트랜치 전용 31봉 mini-anchor | empty tranche는 217 대신 31봉마다 재평가, K=1, healthy>=1, full, freeze | G1 |
| C1 | cash 트랜치 잠재 target drift 재진입 | empty에 potential healthy target vs Cash drift ht 계산, ht>=0.10이면 eligible, daily, K=1, freeze (A2 등가성 검증) | G1 |
| S1 | healthy>=2 강화 | A2 + healthy_count>=2일 때만 재진입 | G3 |
| S2 | healthy 3봉 연속 안정성 | A2 + t-1까지 healthy>=1이 3봉 연속 | G3 |
| S3 | canary ON 3봉 연속 안정성 | A2 + t-1까지 canary_on이 3봉 연속 | G3 |
| L1 | 재진입 후 글로벌 쿨다운 | A1 기반, 한 tranche 재진입 후 5봉 동안 다른 empty tranche 진입 금지 | G4 |
| P1 | 50% step entry | A2 + 첫 재진입 50%, 다음 eligible 평가서 조건 유지 시 100% | G5 |
| P2 | healthy 개수 비례 sizing | A2 + risky allocation = h/3 of full target, 나머지 cash | G5 |
| R1 | 부분 현금 트랜치까지 대상 확대 | cash_weight>=50% tranche도 daily 재평가, K=1, healthy>=1, cap 1/3 | G6 |
| W1 | sleeve 전량 현금일 때만 | 7 tranche 모두 empty일 때만 A2 트리거 | G6 |
| H1 | 카나리 ON 한정 ml-only 완화 | A2 trigger, canary_on이면 health=ml>0 & vol<=5%만(ms 제거) | G7 |
| H2 | 카나리 ON 한정 vol_cap 7% | A2 trigger, mom2 유지, canary_on이면 vol_cap=7% | G7 |
| H3 | 카나리 ON 한정 vol_cap 10% | A2 trigger, mom2 유지, canary_on이면 vol_cap=10% | G7 |
| H4 | 카나리 ON 한정 ms lookback 10봉 | A2 trigger, canary_on이면 ms lookback=10, ml>0, vol<=5% | G7 |

## 그룹
- G0 baseline: {F0}
- G1 직접 refill 속도: {A1,A2,A3,A4,A5,A6,D1,C1}
- G2 재진입 후 처리: {A7} (+ F0,A2 대조)
- G3 안정성 필터: {S1,S2,S3} (+ A2 대조)
- G4 동시진입 억제: {A1,A2,A3,L1}
- G5 sizing: {P1,P2} (+ A2 대조)
- G6 대상 범위: {R1,W1} (+ A2 대조)
- G7 health 완화: {H1,H2,H3,H4} (refill 계열과 분리 해석)

## 공통 산출 지표 (변형마다 전부)
CAGR, MDD, Calmar, Sharpe, 비용 차감 전후 CAGR, turnover, 거래횟수, 현금체류일, empty tranche 지속일 p50/p90/max, 재진입 지연 p50/p90, 동시진입수 분포, 재진입 후 5/10/21/31봉 forward return, 재진입 후 5/10/21봉 내 canary OFF율, 즉시 왕복거래율, 종목교체일수, anchor별 평균·분산, 10-anchor 평균·분산, window×stride rank-sum n≈72.

## 과적합 통제 (엄수)
- 변형군·기각선 사전등록(이 파일이 사전등록). 특정 anchor/window 에서만 이긴 변형 기각.
- 재진입 이벤트 n<20이면 재진입 특화 지표는 탐색 결과로만 표기(채택 근거 금지).
- H 계열은 refill 계열과 섞어 사후 승자 선택 금지.
- plateau는 대표값(N=5/10/21/31, vol=7/10)에서만. 촘촘한 스윕 금지.
- 결과 해석은 "승자 1개"가 아니라 그룹별 tradeoff 표. A1=상한선 대조군, A2=기본 제한형 대표.

## 제외된 아이디어 (구현 안 함)
- 미래 n봉 수익률로 재진입 종목 선택(look-ahead).
- 사후 최적 anchor/window 선택.
- canary OFF 청산 지연/최소보유기간(손실방어 훼손).
- 수동 regime flag, crash 구간 전용 rule(라이브 대칭성 약함).
- vol_cap 5~15% 촘촘 스윕(과적합).

## 계측 하니스 사전 결과 (measure_canary_empty.py, 참고)
- "카나리 ON & healthy==0": 5.4yr 2009일 중 136일(6.77%, ON대비 12.9%). 에피소드 29개 median 1일 max 31일.
- 원인 ml<=0 81.7% + vol>5% 71.1% (실제 약세). 2022년 51% 집중, 2023·2026 0.
- 맹목 top3 매수 forward: 20봉 median +3.3%, 하방 p10 -11~-33% (비대칭 위험).
- 예측: 대부분 변형은 2022 구간에서만 F0와 차이. 재진입 이벤트 희소 가능 → 검정력 주의.

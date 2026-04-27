# V21 선물 로버스트니스 플랜 (2026-04-24)

## 배경

- 현재 V21 선물 3멤버 SN=[120, 30, 120] → SN30 은 SN120 의 약수. gcd=30, lcm=120.
  SN120 의 모든 tranche refresh 점이 SN30 에 포함 → 실제로 "동시 refresh" 관측됨 (04-24 01:05).
- 집단 동시 실행 (전세계 동일 전략 사용자) 시장 충격 + 체결 슬리피지 위험 존재.
- 목표: snap 비동기화 + 각 변수 jitter 내성 검증 + 집단 실행 영향 정량화.

## 베이스라인

- 전략: ENS_fut_L3_k3_12652d57
- 멤버 3개, 4h봉, EW 1/3, 고정 3x 레버리지, 가드 없음
- 기간: 2020-10-01 ~ 2026-03-28
- tx 0.04%, 유지증거금 0.4%

---

## Phase 0 — SN 교체 백테 (우선, ~1시간)

목적: SN=[33, 123, 129] 으로 바꿔도 성과 plateau 인지 확인.

실험:
- A. baseline SN=[120, 30, 120]
- B. SN=[123, 33, 123]   (최소 이동)
- C. SN=[33, 123, 129]   (상대소수 최대, 권고)
- D. SN=[39, 111, 141]   (넓은 분산, p=[13, 37, 47])

지표: CAGR, MDD, Sharpe, Cal, turnover, 총 rebal 횟수.
통과: C 가 A 대비 Cal/MDD 악화 5% 이내.

---

## Phase 1 — Jitter sensitivity table (~2~3시간)

목적: 각 변수에 ±5/±10/±20% 노이즈 주입 시 성과 변화. plateau 있는 변수만 live jitter 허용.

축:
- SMA period (240, 240, 120)
- mom_long (720, 480, 720)
- canary_hyst (0.015 공통)
- snap_interval (Phase0 C 확정값)
- vol_threshold (0.05 공통)

실험: 각 변수만 독립으로 base ± [5, 10, 20]% (반올림 정수). 4개 값 × 5축 = 20개 백테 + baseline.

지표: Cal, MDD, Sharpe 표.
통과: ±10% 에서 Cal 변동 <5%, MDD 변동 <5pp 이면 "plateau 변수".

결과: live jitter 가능한 변수 목록 + 권장 jitter 범위.

---

## Phase 2 — Slippage stress test (~1시간)

목적: 집단 동시 실행 시 체결 슬리피지 악화 시뮬레이션.

실험: tx 0.04% 를 [1x, 2x, 5x, 10x, 20x] 로 변경.

지표: Cal, CAGR, MDD.
통과: tx 10x (0.4%) 에서도 Cal > 1.0, MDD < -30% 유지.

해석: 전세계 집단 실행으로 체결가 10배 악화되는 최악 시나리오에서도 전략이 수익성 있는지.

---

## Phase 3 — Monte Carlo execution jitter (~1일)

목적: 매 rebal 마다 params perturbation 시 성과 분포 확인. "live 에서 랜덤 흔들기" 의 통계적 안전 마진.

실험:
- Phase 1 에서 plateau 판정 받은 변수만 대상
- rebal 시점마다 params 를 base × (1 + ε), ε ~ Uniform(-r, r)
- r ∈ {0.02, 0.05, 0.10}
- N=100 seed × 각 r

지표: seed 별 Cal 분포 (mean, std, p5, p95), worst MDD.
통과:
  mean Cal ≥ base × 0.95
  std Cal / mean Cal < 0.1
  worst MDD ≤ base × 1.1 (10% 악화 허용)

결과: 실제 live 에 적용 가능한 jitter 규칙 (변수별 r 값).

---

(Phase 4 cron delay 실측 분석은 2026-04-24 사용자 지시로 제외)

---

## 실행 순서 & 의사결정

1. Phase 0 C 통과 → SN=[33, 123, 129] 서버 반영 (state 스키마 호환 확인 後)
2. Phase 1 통과 변수 리스트 확정
3. Phase 2 통과 시 strategy 내성 확인 (결론만, live 변경 無)
4. Phase 3 통과 시 live jitter 범위 확정 → auto_trade_binance.py 패치

각 phase 결과는 futures_jitter_report.md 에 기록.

## 주의

- V17 주식 phase1a_v2 (36k configs) 백그라운드 실행 중. 선물 백테 병렬 실행 시 CPU 경쟁 가능.
  Phase 0 은 config 4개 × 30초 내외 = 2분, 경쟁 무시 가능.
  Phase 3 은 10k+ run 이라 phase1a_v2 끝난 후 돌려야 함.
- 백테 기간/세팅 bias: 2020~2025 는 상승장 편향. out-of-sample 확보 어려움 → 본 플랜은 "robustness" 만 검증, 수익성 주장 아님.
- live 적용 전 Codex/Gemini 크로스체크 필수 (V21 가드 없는 상태 + jitter 추가는 tail risk 중첩 우려 가능).

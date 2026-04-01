# 바이낸스 선물 자동매매 — 최종 전략 반영 + 라이브 운영 중

## 현재 운영 전략

**`1h_09 + 4h_01 + 4h_09` 앙상블 + `capmom 5/4/3x` + `prev_close 15% + cash_guard(34%)`**

- 서버 크론 매시 `:05` 실행, 매일 `09:00` 리포트
- 신호 전략:
  - `1h_09`: `SMA 168`, `Mom 36/720`, `mom2vol`, `bar vol 0.80`, `snap 27`
  - `4h_01`: `SMA 240`, `Mom 10/30`, `mom1vol`, `daily vol 0.05`, `snap 120`
  - `4h_09`: `SMA 120`, `Mom 20/120`, `mom2vol`, `bar vol 0.60`, `snap 21`
- 실행층:
  - 종목별 동적 레버리지 `cap_mom_blend_543_cash`
  - 종목별 `5x / 4x / 3x`
  - `prev_close 15%`
  - `cash_guard(34%)`
- 라이브 코드 반영:
  - 포지션 조회 버그 수정
  - OHLCV 페이지네이션 수정
  - 주문 재시도 추가
  - 매 실거래 실행마다 stop 재동기화
  - 텔레그램 메시지 간소화

## 연구 완료

### 스탑로스 / 조건부 스탑

- always-on 스탑은 전반적으로 성과 개선 실패
- `prev_close 15%`가 가장 덜 나쁜 단순 스탑
- 재진입 쿨다운, 긴 롤링 룩백, ATR 스탑은 개선 실패
- 조건부 스탑은 유효:
  - `3x prev_close 15% + cash_guard(34%)`: Cal `3.62`, Liq `1`
  - `4x prev_close 15% + cash_guard(34%)`: Cal `3.72`, Liq `11`
  - `5x prev_close 15% + cash_guard(34%)`: Cal `4.01`, Liq `24`

### 동적 레버리지

- 계정 단위 5/4/3보다 종목 단위 5/4/3이 우수
- 최고 성능:
  - `coin_capmom_543_cash + prev_close15 + cash_guard(34%)`
  - Cal `4.30`, CAGR `+250.3%`, MDD `-58.2%`, Liq `7`
- 근접 2위:
  - `coin_mom_543_cash + prev_close15 + cash_guard(34%)`
  - Cal `4.29`, CAGR `+249.9%`, MDD `-58.2%`, Liq `6`

### 신호 전략 재탐색

- 기존 `4h1 + 4h2 + 1h1` 대신 `1h/4h` 전략군을 다시 탐색
- 1h/4h 후보를 따로 스크리닝 후 `2개/3개/4개` 조합까지 검증
- 상위 조합:
  - `1h_09 + 4h_01 + 4h_09`
  - `1h_09 + 1h_05 + 4h_01 + 4h_08`
  - `4h_01 + 4h_09`

### 트랜치 간격 미세조정

- `1h_09`는 `24`보다 `21/27/33`이 우수
- `4h_01`은 `120` 유지가 최선
- `4h_09`는 `18`이 최고였으나, 시간대 정렬 완화를 위해 `21` 채택

최종 선택:
- `1h_09 snap=27`
- `4h_01 snap=120`
- `4h_09 snap=21`

### 기존 전략 대비 최종 비교

동일 실행층 `capmom 543 + prev_close15 + cash_guard(34%)` 기준:

- 기존 `4h1 + 4h2 + 1h1`
  - Cal `4.30`
  - CAGR `+250.3%`
  - MDD `-58.2%`
  - Liq `7`
- 최종 `1h_09 + 4h_01 + 4h_09`
  - Cal `4.98`
  - CAGR `+221.1%`
  - MDD `-44.4%`
  - Liq `4`

해석:
- CAGR은 다소 낮아졌지만,
- MDD와 청산이 줄고 Calmar가 크게 개선되어 실거래 운영 기준으로 교체 가치가 충분하다고 판단

## 관련 문서

- 최종 전략 설명:
  - [strategies/cap_defend/futures_strategy_final.md](/home/gmoh/mon/251229/strategies/cap_defend/futures_strategy_final.md)
- 스탑 연구 요약:
  - [strategies/cap_defend/stoploss_summary.md](/home/gmoh/mon/251229/strategies/cap_defend/stoploss_summary.md)

## 다음 작업

- [ ] 다음 크론 로그/텔레그램 확인
- [ ] moneyflow-config에 바이낸스 키 반영
- [ ] 문서/실험 러너 정리 후 git push
- [ ] Kill-switch 봉 기반 개선

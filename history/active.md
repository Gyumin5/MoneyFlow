# history/active.md — 현재 유효한 결정 compact view

갱신: 2026-08-20 KST. 이 파일은 append-only 가 아니다. 폐기·대체된 결정은 지우고 유효한 것만 남긴다.
원문·근거는 `../history.md` 의 같은 날짜 항목. 실험 시작 전 이 파일부터 읽고, 아래 "재시도 금지" 에
있는 건 다시 돌리지 않는다. 의도적 재현이면 왜 다시 도는지를 progress.md 에 남긴다.

## 1. 운영 전략 (in force)

- 자산배분 60(주식)/25(업비트 현물)/15(바이낸스 선물). 정본은 라이브 코드 recommend.py.
  리밸 트리거 T1(half_turnover ≥ 20pp) OR T3U_can(상대미달 ≥ 20% AND 해당 sleeve 카나리 ON).
  자동 송금 없음 — 알림만 보내고 사용자가 직접 송금. [2026-05-22, 2026-05-26]
- 코인 현물 V24: D_SMA42 단일, snap 217bar/7슬롯, drift 0.10, cap 1/3, 현금버퍼 1%.
  종목 교체는 스냅 리프레시 또는 refill v2(mom short<0 AND long<0) 둘 중 하나로만. [2026-04-30]
- 선물 V25: D_SMA42, snap 95bar/5슬롯, drift 0.03, per-coin 동적 L2~L4 = min(BTC cap, K2), CROSS.
  드리프트 기준통화·비중은 진입마진+PnL / equity (BT 정의와 정합). [2026-05-28, 2026-07-17]
- 주식 V25: R7 유니버스, EEM 카나리 SMA200 ±0.5%, Z-score 랭킹 + 3-mom 필터, cap 1/3+현금,
  snap 69일/3슬롯/stagger 23, drift 0.05. 주식만 drift 발화일에도 fresh selection 으로 교체. [2026-05-29]
- 헬스체크 공통: short-mom > 0 AND long-mom(127일) > 0 AND 90일 vol ≤ 5%.
  lookback 만 현물 20일 / 선물 18일. 카나리는 SMA42 ±1.5% 히스테리시스. [2026-07-21]
- 가드 없음이 전략의 일부다. 스톱·크래시·DD 가드는 2026-04-21 에 본체째 제거했고 분산이 유일 방어.
  되살리려면 시스템 서킷브레이커를 별도 layer 로 설계할 것. [2026-04-21]
- 자산배분 자동 rebal(alloc_transit phantom buffer)은 영구 비활성. 트리거는 알림만 낸다. [2026-05-26]

## 2. 코드 불변식 (깨면 사고가 재발한다)

- 현금 키: 엔진 내부는 대문자 CASH, 정규화는 run() 반환 경계 한 곳(refill 후처리 뒤)에서만.
  executor 는 상류를 신뢰하지 않고 진입부·buffer·cap·완료판정에서 다시 병합하고, 주문 후보
  생성부에서 표기 무관 현금 키를 하드 차단한다. 정규화를 refill 앞으로 되돌리면 08-20 헛주문 재발.
  회귀 테스트 tests/test_cash_key_normalization.py. [2026-08-20]
- 선물 매매 게이트는 fut_trade_gate() 단일 판정. 빈 target_lev_map 을 무조건 차단하지 않는다 —
  산출 실패(lev_abort)만 차단하고, 전액 CASH 는 liquidation_only 로 통과시킨다(매수 루프 미진입
  + 전 주문 reduceOnly 불변식). 이 구분을 없애면 08-15 청산 차단이 재발한다.
  회귀 테스트 tests/test_fut_liquidation_only.py. [2026-08-15]
- 선물 preflight 게이트를 완화하지 않는다. 레버리지 대조는 허용오차 방식 유지(먼지 게이트는 분리).
  [2026-07-15]
- 한투 총자산·현금은 CTRP6548R(투자계좌자산현황) tot_asst_amt 기준. 해외주식 inquire-balance /
  present-balance 단독 사용 금지 — 외화RP 스윕분이 누락된다. [2026-06-03]
- KIS API 연결 실패를 잔고 0원으로 해석하지 않는다(가짜 리밸 알림·히스토리 오기록 원인). [2026-07-04]
- BT SSoT(unified_backtest.py, backtest_futures_v25.py)에 실험 코드를 남기지 않는다.
  변형은 env 토글로 넣고 결론 나면 즉시 revert. [2026-07-23]
- 승인 없이 실거래·인증 안전장치를 완화하거나 강제진행 코드를 추가하지 않는다.
  ai-debate 검토 없이 실매매 코드 배포 금지.

## 3. 운영·보안

- 서버 인증은 DASHBOARD_PIN 단일. TRADE_PIN 폐지(재도입 금지). 기동은 start_trade_api.sh 경유만.
  env 단일출처는 /home/ubuntu/.trade_env. [2026-07-21]
- 포트 5000 잔고조회 GET 6종은 require_auth() 게이트. localhost 통과, 원격은 X-Auth-PIN.
  CORS Access-Control-Allow-Headers 에 X-Auth-PIN 필수(빠지면 대시보드 전체 브레이크). [2026-07-03]
- API 재시작은 ssh 두 번으로 분리한다. pkill 과 기동을 한 ssh 에 이어붙이면 세션 종료와 함께
  자식이 죽어 서버가 내려간 채 남는다(08-19 실제 4분 다운). SERVER_OPS.md 4번. [2026-08-19]
- 대시보드는 조회 전용. 쓰기 엔드포인트(cash_buffer)는 사용자 미사용.
- 기록 분리: 계획은 progress.md, 보고서는 reports/ 한 파일, 중간로그·원시출력은 logs/ 또는
  state/<job-id>/, 결론은 history.md 한 항목 + 보고서 경로. 본문을 history 로 옮기지 않는다.

## 4. 재시도 금지 (다시 돌리면 시간낭비)

- 카나리 OFF→ON 재진입의 분할진입(스냅별 stagger). 현물·선물 전 지표 열위(현물 Cal 4.55→3.29,
  선물 7.58→4.01, MDD 도 악화). 재진입 타이밍은 리스크가 아니라 기회비용이다. [2026-07-23]
- 코인 현물 빈 스냅 재진입 변형 추가 발굴·스윕. 설계공간 22변형으로 소진, 추가는 다중검정 과적합.
  P2(비례사이징) 무승부, H3(vol_cap 10%) 레짐의존+MDD 악화로 기각. F0(현행) 유지. [2026-07-15]
- vol_th(헬스 vol_cap) 양방향 재스윕. 0.06+ 완화는 SAND 류 고변동 알트 편입으로 Cal 붕괴,
  0.03~0.04 타이트화는 유니버스 축소로 현금화·CAGR 붕괴. 0.05 는 과적합 최적점이 아니라
  아웃라이어 필터로 작동 중(현물·선물이 독립적으로 0.05 피크). [2026-07-15, 2026-07-21]
- K2 per-coin 레버리지 상방 L5/L6, 하방 L1/L3 floor. 전부 기각. 특히 L5 임계 1.075 는
  L4 를 상시 L5 로 치환하는 것이라 재검토 시 금지, 보수안 1.10/1.08 만. L6 는 재론 가치 없음.
  [2026-06-02]
- 자산간 T3O(과대-트림) 트리거. 라이브 full 유니버스에선 BNB·SOL 상방을 구조적으로 잘라
  CAGR 희생이 채택바 미달. 임계를 올려도 개선이 평탄하고 40%부터 보호가 사라진다.
  (1차 토론의 조건부 채택은 BNB·SOL 제외 유니버스 기준이었고 2차에서 번복됨.) [2026-06-16]
- 선물 유지증거금 tier-aware 구현. 동적 L4 + CROSS 구조에서 0.4~0.65% 차이는 청산위험 dead zone.
  레버리지 상한을 올리거나 ISOLATED 로 되돌릴 때만 재검증. [2026-07-03]
- 선물 BT 용 simulate() 별도 작성. 청산·crash·DD·BL forced exit 누락으로 가짜 Cal 7.22 가 나온다.
  backtest_futures_full.py 의 external_target_schedule 모드를 쓸 것. [2026-05-14]
- dry-run 을 라이브 binance_state.json 대상으로 실행. 상태가 무조건 저장돼 rebalancing_needed 가
  오염되고 다음 cron 이 오발화한다. [2026-07-15]
- 전략 기대성능을 헤드라인 단일값(전체 CAGR 72.5%)으로 제시. 3층(상방/계획기준 30~45%/no-freak 29%)
  으로만 말한다. 사후 최대승자 제외 수치는 예측이 아니라 취약성 진단용. [2026-06-16]

## 5. 방법론

- 홀드아웃·walk-forward·yearly rank 금지. 윈도우 기반 unified rank-sum 만 쓴다.
- grid 축은 라운드 넘버. 라운드 5배수 grid 에서 geom-mid 는 작동하지 않는다(비율 ≥1.2 stride 필요).
- 라이브 파라미터와 BT 기본값 혼동이 결론을 뒤집는다(선물 ms=18/ml=127 vs BT ms=30/ml=90).
  실험 전 라이브 정합성부터 확인. [2026-05-14]
- 함수명을 믿지 말고 구현을 본다. 주식 calc_weighted_mom 이 순수252 였던 탓에 종목선정이
  채택 BT 와 16.8% 어긋나 있었다. [2026-06-06]
- 결과 보고는 Calmar/Sharpe/CAGR/MDD 4개를 항상 함께. 개선이 턴오버·비용 증가로 설명되면 미채택.

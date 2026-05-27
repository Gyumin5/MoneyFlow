## [2026-05-23] alloc_transit phantom buffer (옵션 D) 도입
tags: V24, alloc_transit, phantom_buffer, drift, ai-debate
- 결정: 자산배분 트리거 fire 시 3자산 동시 cap_ratio 적용 (60/25/15). cap_ratio = min(1.0, target_sleeve_krw / current_sleeve_krw) 매일 갱신. 각 executor 가 effective_pv = actual × cap_ratio. clear: ht ≤ 0.05 AND 최소 1일 유지. floor 0.10 (드물게 자산 90% 매도 한도).
- 근거: T+2 settle 지연 + 사용자 수동 송금 환경에서 stock→coin transit 시 cash idle 문제. ai-debate 3라운드 (codex+gemini+claude). 옵션 A (5상태 머신, 1주) → over-engineering. 옵션 C (사용자 수동) → 1000만원 단위 실수 위험. 옵션 D pure (단순 cap) → 무한루프 자연 차단 (cap 도달 = effective_pv 한계 = 추가 매도 X). Critical 1-6 안전장치 (cap_ratio 검증/floor/schema fallback/stale guard/풍부한 로그/fault injection test 14 케이스 PASS) 후 채택.
- 코드: strategies/cap_defend/recommend_personal.py (flag set/clear/update + cooldown 1일 + 첫 발동 알림) + trade/{executor_stock,executor_coin,auto_trade_binance}.py (cap_ratio 적용 + validation + stale guard) + trade/test_alloc_transit_validation.py + crontab.txt (코인/선물 09:05→09:20)
- 되돌릴 조건: cap_ratio invalid/stale 빈발 또는 자산 10억 초과 시 옵션 A (5상태 머신) 로 승격. drift_whipsaw 시 DRIFT_ENABLED=False 토글 가능
- Prediction: alloc_transit 무한 매도 루프 / flag flapping / state race / cash 회계 오류 사고가 6개월 동안 telemetry/history 상 재발하지 않으면 성공, 재발하면 재검토.

## [2026-05-23] stock V24 drift trigger 추가 (drift_threshold=0.10)
tags: V24, 주식, drift, alloc_rebal
- 결정: stock_engine_snap.py + trade/executor_stock.py 에 drift trigger 추가. cur_w (cash 포함) vs combined target half_turnover ≥ 10pp 면 즉시 rebal.
- 근거: 자산간 alloc rebal 후 stock 계좌에 cash inflow 시 다음 anchor (최대 23일) 까지 idle 문제. BT Cal +2.8% (1.09 → 1.12, MDD 동일). alloc_transit phantom buffer 와 협업 (cap 으로 매도 → drift 로 잔여 deploy 차단).
- 되돌릴 조건: drift whipsaw 빈발 시 threshold 0.15 또는 0.20 으로 상향, 또는 비활성.

## [2026-05-22] 자산배분 M 안 채택 (60/25/15, T1(13)|T3U_can(20))
tags: V24, alloc, 트리거, marginal_stability, ai-debate
- 결정: 자산배분 70/15/15 → 60/25/15. 리밸 트리거 T1(ht≥13pp) OR T3U_can(rel-under≥20% & sleeve canary ON).
- 근거: 363 후보 × n=72 windows unified rank-sum 결과 broad plateau center. 5.4yr BT Cal 3.97 / CAGR 70.74% / MDD -17.81% / Sharpe 2.09. marginal stability 분석: T1(13) std 0.194, T3U_can(20) std 0.216, 양쪽 평탄.
- 되돌릴 조건: yearly Cal rank drop 또는 새 후보 unified rank champ 등장.

## [2026-05-14] 선물 sleeve 재최적화 — V24 유지 + 펀딩 BT 버그 fix
tags: V24, 선물, 펀딩, BT, plateau
- 결정: V24 그대로 유지 (변경 없음). C1/EW 후보 모두 채택하지 않음.
- 근거: 펀딩 BT 버그 fix 후 V24 정식 (sma=42 ms=18 ml=127 live 파라미터) Cal 4.05 확인. C1 (sma=38 ms=20 ml=122) Cal 5.32 우위지만 SMA narrow peak (sma -10% Cal -69%) plateau 결격. EW 50/50 Cal 3.96 으로 V24 보다 낮음 (이전 잘못된 V24 Cal 3.32 기준일 때 EW +12% 매력 있었으나, live 정식 V24 기준일 때 매력 사라짐).
- 코드 변경: backtest_futures_full.py 의 펀딩 매칭 fix (prev_date < t ≤ date 윈도우 sum, D 봉 3회 funding 정확 반영) + external_target_schedule 파라미터 추가 (앙상블 BT 가능, V24/C1 replay parity 0%). 모든 결과 strategies/cap_defend/research/fut_reopt_2026_05/.
- 실험/실패: (1) simulate() 별도 작성 → 청산/crash/DD/BL forced exit 누락으로 EW Cal 7.22 가짜 결과 (재시도 금지, backtest_futures_full.py 의 external_target_schedule 모드 사용). (2) phase_b1 의 라운드 5배수 grid → geom-mid 작동 안함 (비율 ≥1.2 stride 필요). (3) live V24 파라미터 (ms=18 ml=127) 와 BT 기본값 (ms=30 ml=90) 혼동 → 결론 뒤집힘. live 정합성 먼저 확인.
- 되돌릴 조건: 향후 펀딩 데이터/시장 regime 크게 변하면 재실행 (현재 grid + iter_refine 파이프라인 활용).

## [2026-04-30] V24 도입 (모든 자산 1D 단일 + drift trigger)
tags: V24, 마이그레이션, 코인, 선물, 주식, drift
- 결정: 모든 자산 V24 통일. spot D_SMA42 sn=217 n=7 drift=0.10, fut D_SMA42 sn=57 n=3 drift=0.05 L3, stock SNAP_PERIOD=69 STAGGER=23 N_SNAPS=3. 4h 멤버 제거, cron 4h x 6 → 1d x 1 (5 9 * * *)
- 근거: 5.4yr BT (rank-sum across alloc, 평탄도 300 BT, 차별화·서로소 2,220 portfolio, C 후보 직도입 검증) 통과 + AI 합의 (gemini+codex). spot/fut n_snap 서로소 (7/3 gcd=1), stagger 모두 distinct prime (23/31/19). drift trigger: half_turnover(cur_w, tgt_w) >= threshold, cur_w 자본금 기준
- 코드: trade/{coin_live_engine,executor_coin,executor_stock,auto_trade_binance,migrate_v22_to_v24}.py + strategies/cap_defend/{futures_live_config,recommend,recommend_personal}.py + trade/ops/crontab.txt + V24_OPERATION_MANUAL.md
- codex 검토 반영 (조건부 NO-GO → GO): (1) 매뉴얼 마이그레이션 절차 보정 (cron 활성화는 수동 dry-run 검증 통과 후 마지막), (2) DRIFT_ENABLED / DRIFT_ENABLED_FUT 런타임 플래그 추가 (코드 수정 없이 snap-only fallback), (3) 첫 주 tripwire 추가 (spot >2/day or 8/wk → 0.12~0.15 / fut >1/day or 4/wk → 0.07~0.10)
- 되돌릴 조건: drift whipsaw 시 DRIFT_ENABLED=False 토글 → snap-only 운영. yearly Cal 급락 시 migrate --rollback 으로 V22 복귀
- 서버 배포: 사용자 승인 후 Phase F 진행 예정

## [2026-04-28] V22 일괄 적용 정합성 정리
tags: V22, 운영, 문서정리
- 결정: 모든 자산 (주식/코인/선물) V22 단일 표기. 이전 버전 표기 제거 (progress/history active 영역, MEMORY, recommend docstring 등)
- 근거: 사용자 명시 — 운영은 모두 V22, 옛 버전 표기는 혼선만 야기
- 되돌릴 조건: 새로운 버전(V24) 적용 시점에 동일 절차로 갱신
- archive: 03/04-early 월별로 history/ 하위 이동 (보관)

## [2026-04-27] V22 마이그레이션 (코인/선물/주식 동기)
tags: V22, 마이그레이션, 코인, 선물, 주식
- 결정: 코인·선물·주식 모두 V22 통일. 코인/선물 = 1D + 4h 2멤버 EW (snap 60일/90일 동기), 주식 = snap-based 3-tranche stagger (SNAP_PERIOD=126, STAGGER=42, N_SNAPS=3)
- 근거: AI 합의 (Gemini+Codex) — interval 다양성, plateau-center cfg 채택, 가드 분산방어 일원화. 자산배분 60/40/0 수동 유지 (선물 0%)
- 되돌릴 조건: 다음 라운드 백테스트에서 단일 interval 회귀가 robust 하다고 입증되면

## [2026-04-22] V22 코인 C 슬리브 비활성
tags: V22, 코인, C슬리브
- 결정: V22 C 슬리브(champion mean-reversion) cap=0 으로 비활성. 코드/문서 유지, 단독 sleeve 운용 안 함
- 근거: Upbit 1h 등가 백테스트 탈락 — 단독 Cal 3.10(가드없음 D ensemble) 대비 C+D 등가 Cal 1.46, MDD -19% → -44%. champion 이 Binance 1h 기반이라 Upbit 등가 검증 불가
- 되돌릴 조건: Upbit-aware champion 재탐색 결과 등가 sleeve 가 robust 하면 cap_per_slot 복원

## [2026-04-21] V22 가드 전면 제거
tags: V22, 가드, 분산방어
- 결정: stop/cash_guard 본체 코드 삭제, 앙상블 분산이 유일 방어. rebalancing_needed 플래그로 통일
- 근거: 가드 ON/OFF ablation 에서 ON 시 OOS Cal 손실 일관 관찰. 분산방어가 더 robust
- 되돌릴 조건: 시스템 서킷브레이커(-20%) 도입 검토 시 별도 layer 로 추가

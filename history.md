## [2026-07-03] 보안: TRADE_PIN 노출 대응(히스토리 정리+엔드포인트 제거+rotate) + 문서 stale 전면 갱신
tags: 보안, secrets-scan, git-history, trade_api, 문서정합, 자산배분
- 결정: (1) git 히스토리 전체에서 TRADE_PIN 값 filter-repo 치환 제거 후 force-push. (2) 강제거래 엔드포인트 /api/trade/upbit + 헬퍼 run_trade_async 코드 삭제(웹 노출 폐기), recommend_personal.py orphan forceTrade() JS 정리, 미배포 사본 trade/api_server.py 동일. (3) TRADE_PIN 40자리 랜덤 rotate(crontab+실행프로세스). (4) 문서 stale 전면 갱신: 버전표기 주식V25/코인현물V24/선물V25, README·SERVER_OPS 재작성, CLAUDE.md 카나리 SMA300·2%→SMA200·0.5% dead-zone 정정, 자산배분 표기 60/25/15 통일(라이브 코드 recommend.py 정본).
- 근거: TRADE_PIN(4자리)이 crontab/watchdog/구 SERVER_OPS 등 히스토리에 평문 노출 + 포트 5000 이 0.0.0.0 바인딩+iptables 0.0.0.0/0 로 외부 개방 → PIN 만으로 임의 강제매수(/api/trade/upbit) 트리거 가능한 실질 노출. UI 버튼 제거는 클라이언트만 없앤 것으로 서버 엔드포인트는 생존. rotate 만으론 4자리 브루트포스 여지+엔드포인트 생존 → 엔드포인트 제거 우선. config.py 등 실키는 gitignore 로 히스토리 무오염(확정 하드코딩 시크릿=TRADE_PIN 하나).
- 검증: 서버 실측 POST /api/trade/upbit=404, /health=ok, /api/status=200. 옛 PIN 으로 /api/cash_buffer=403 거부. 히스토리 잔존 숫자 PIN 대입 0건. 백업 bundle+crontab.bak(옛 PIN 포함) 삭제.
- 핵심교훈: UI 기능 제거 ≠ 서버 엔드포인트 비활성 — 프로세스·포트·방화벽 직접 확인. 히스토리 정리는 un-leak 아님(GitHub 옛커밋 캐시/외부 clone 잔존) → 노출 시크릿은 반드시 rotate. "push 누락"과 "commit 누락" 구분(GitHub 최신커밋 옛날=로컬 미커밋부터 확인). 자산배분 수치는 라이브 코드 정본으로 통일.
- 되돌릴 조건: 강제거래 웹 트리거 재도입 필요 시 인증강화(길고 랜덤 PIN + 포트 5000 내부화 + rate-limit) 선행 후에만. 잔여 선택(미결): 포트 5000 리버스프록시 잠금(대시보드 유지), IP scrub.
- 커밋: 0356283(문서 stale) / 2bdf1ed(엔드포인트 제거) / e6a940e(자산배분 표기). filter-repo force-push 8cad3f9→0356283.
- Prediction: secrets-scan 클래스(하드코딩 시크릿·평문 PIN 노출·UI제거만으로 방치된 위험 엔드포인트)가 향후 6개월 repo 스캔/telemetry 상 재발하지 않으면 성공, 재발하면 재검토.

## [2026-06-16] 전략 성능 아웃라이어 의존도 + 일반화 기대성능 보정
tags: 일반화, survivorship, 아웃라이어, 유니버스, 기대성능, ai-debate
- 결정: 전략 기대성능을 헤드라인 단일값(전체포트 CAGR 72.5%)으로 제시 금지. 3층 보고 — 상방케이스 72.5%(freak 포함, 기대값 아님) / 계획기준 CAGR 30~45%·중앙 35~40%·Cal~2~3·Sharpe 1.5~1.7 / no-freak 스트레스 29%(비용·수동송금지연·레짐악화 얹으면 20%대 이하). 유니버스는 사후수익 아닌 구조적 사전필터(거래소토큰/밈/유동성/상장기간/규제/선물청산)로 정의, full 유지+BNB/DOGE 관찰플래그. 매매코드 무변경.
- 근거(bt_v25_deoutlier.py, 전체포트 3 sleeve): base CAGR full 72.5% → BNB만 제외 42.9%(-29.6pp) → 점프코인 전부 제외(U5) 29.2%. 헤드라인 절반 가까이가 BNB freak 점프 의존. de-outlier 해도 MDD 안 줄고 악화(-16%) = 정직한 survivorship 보정. 코인 첨도 DOGE 787·BNB 43·XRP 28·ADA 21(점프) vs SOL 8.6·BTC 3.7(정상).
- 핵심교훈: 사후 최대승자 제외는 hindsight — 실시간 식별 불가, 모멘텀 전략은 다음 승자 타게 설계. de-outlier 수치는 예측 아닌 취약성 진단/스트레스로만 사용. ai-debate(run-20260616T013453Z): full 유지 기본, T3O full기준 기각 유지(de-outlier 승률 72~87% 회복은 우측꼬리 제거 순환효과 가능).
- 후속(권장): net backtest(슬리피지·수수료·세금·funding·수동송금지연), leave-one-out 기여도, 레짐별 분해.

## [2026-06-16] 자산간 리밸 T3O(과대-트림) 트리거 — 라이브 기준 기각 (현행 base 유지)
tags: 자산배분, 트리거, T3O, T3U, robustness, ai-debate, 유니버스
- 결정: 현행 T1 20pp/T3U 20% 유지, T3O 미도입. 매매코드 무변경. (1차 토론의 "조건부 채택"은 BNB·SOL 제외 유니버스 기준이었고, 라이브(full, BNB·SOL 포함) 기준 2차 토론에서 번복됨.)
- 근거: T3O 는 4유니버스 조합 전부에서 Cal·Sharpe·MDD 부호 개선이나, 라이브 full 유니버스에선 BNB·SOL 상방을 구조적으로 잘라 CAGR 희생 과다. 임계 grid(full): 20%→35% 올려도 CAGR 희생 8.4→6.3pp(채택바 3-4pp 미달), MDD 개선 -12.4%서 평탄, 40%부턴 보호 소멸. window 승률 50~60% 내내 = 매구간 우위 아닌 꼬리보험(소수 드로다운 의존). ai-debate 2회(run-20260616T011050Z 조건부GO → run-20260616T012423Z 라이브기준 기각).
- 유니버스별 ΔCal(신−현행): full +0.41(CAGR-8.4pp,win50%) / BNB만 +0.51(-6.4pp,50%) / SOL만 +0.07(-3.1pp,77%) / 둘다제외 +0.40(-0.5pp,85%). T3O가 명확히 이로운 곳은 둘다제외뿐인데 그 유니버스 자체가 base CAGR 72.5→34.8 로 큰 수익 희생.
- 코인 아웃라이어 통계(2020-11~): BNB kurt 43.3·왜도 2.61·하루 +71% = 통계적 아웃라이어(제외 명분 데이터 지지). SOL kurt 8.6·왜도 0.51 = 점프아웃라이어 아님(고베타·-96% MDD). Sharpe 상 BTC(0.78) 뚜렷 상회는 BNB 1.08·SOL 1.17 둘뿐.
- 핵심교훈: T3O 는 T3U 의 대칭(과대+카나리OFF) 아님. 비대 sleeve=랠리sleeve→카나리 거의 ON → OFF게이트 too late(+0.15), 무게이트 차익실현 트림이 정합. 카나리 ON 게이트=none 과 동일.
- 되돌릴 조건(사용자 결정): (a) BNB 를 아웃라이어로 유니버스 제외 + (b) 목적함수를 위험조정(Cal/Sharpe/MDD) 우선으로 명시 + 연 ~6pp CAGR 보험료 수용 시, 해당 유니버스서 T3O(35% 근방) 재평가. 그 전엔 base 유지.
- 스크립트: research/bt_v25_t1_t3u_t3o.py, bt_v25_t3o_robust.py, bt_v25_excl_compare.py, bt_v25_t3o_thresh_full.py. 정리 research/T3O_TRIGGER_FINDINGS.md.

## [2026-06-03] KIS 잔고 외화RP 누락 → CTRP6548R 채택 + 자동RP 해지
tags: KIS, 잔고, RP, 표시정합, executor
- 결정: 한투 총자산/현금은 CTRP6548R(투자계좌자산현황) output2.tot_asst_amt 기준. 해외주식 inquire-balance/present-balance 단독 사용 금지(외화RP 자동매매 스윕분 누락). 사용자가 KIS앱에서 자동RP 해지.
- 근거: HTML 총액 419.24M vs 실계좌 440.18M, 차이 ≈ 외화RP(USD)자동매매 20.13M. 해외주식 API 계열은 RP(금융상품)를 구조적으로 못 봄. CTRP6548R 만 RP 포함 총자산을 줌(앱과 100% 일치). 자동RP 해지 시 유휴 USD 가 외화예수금으로 환원돼 present-balance/주문가능액에 전액 포함 → executor PV·funding 도 코드 수정 없이 자동 정확.
- 코드: trade/auto_trade_kis.py(get_account_asset 신규, CTRP6548R), trade/ops/trade_api_server.py(_get_stock_balance_data RP 포함), strategies/cap_defend/recommend_personal.py(line127 보유=total−cash). 서버 배포·검증 완료(present-balance 419.98M→440.11M, 주문가능 18,345→31,592 USD).
- 실험/실패: present-balance tot_asst_amt(419.98M)도 RP 못 잡음 — 처음엔 executor 가 이걸 쓰니 맞을 줄 알았으나 둘 다 누락이었음. RP 는 오직 CTRP6548R 에만.
- 되돌릴 조건: 자동RP 재활성화 시 executor PV/funding 가 다시 RP 누락 → CTRP6548R 기반 패치 필요. 현재는 RP off 라 불필요.
- 한계: 옛 일별 스냅샷(assets.db)은 RP 활성 기간 동안 total 을 ~RP(~20M, 총자산의 ~3%)만큼 과소 기록. 일별 RP 미저장이라 정밀 backfill 불가. 2026-06-03 부터 정확.

## [2026-06-02] K2 per-coin L 상방 L5/L6 + 하방 L1/L3 floor 전부 기각
tags: V25, 선물, leverage, K2, ai-debate, 과적합방지
- 결정: per-coin 동적 L 현행 L2~4 유지. 하방 L1·floor L3, 상방 L5·L6 모두 미채택. 라이브 변경 없음.
- 근거: 하방 변형은 양방향 Calmar 악화 (L1 floor: CAGR↓·MDD 불변 → alloc Cal 2.98→2.63 / L3 floor: CAGR↑이나 MDD↑↑ → 2.98→2.78). L6 만장일치 NO-GO (alloc 이득 0=plateau 평평 3.14, bootstrap 1건 음수, 꼬리위험만). L5(1.10/1.08)는 5게이트 robust 통과 (제외 4/4 +0.11~0.35, bootstrap 12/12, cost 5x edge 유지, window rank dominant) 이고 BT 가 일중 Low+CROSS worst-case 청산 검사하는데도 5.6yr 청산 0건 — 그러나 이득 작음(BNB+SOL +0.16 Cal), Sharpe 미세하락(1.66→1.63 = 이득이 alpha 아닌 leverage-beta), plateau 단조(내부 peak 없음 = bull 샘플 레버리지-베타 시그니처, V25 과적합 규칙 위반), bull-heavy 표본. "robust 통과 ≠ 채택" — 사용자 NO-GO.
- 실험/실패: L5 임계 plateau 가 1.075 에서 최고(Cal 3.18)지만 이는 L4 를 사실상 L5 로 치환=상시 고레버리지화. 재검토 시 1.075 공격안 금지, 보수 1.10/1.08 만. L6 는 재론 가치 없음.
- 되돌릴 조건: 장기 bear/횡보 포함 표본으로 재검증 시 + intraday(<1h) 청산 시뮬 통과 시 L5 보수안 재론 가능. 그 전엔 재론 금지.
- 스크립트: research/bt_k2_l5l6_full.py(5게이트 통합), bt_k2_upside_l5.py, bt_k2_floor_l3.py, bt_k2_l1_downside.py. 토론: ~/.claude/state/ai-debate/run-20260602T060453Z/.

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

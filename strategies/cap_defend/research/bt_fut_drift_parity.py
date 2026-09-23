"""선물 드리프트 비교 좌표계가 라이브와 어긋나 있나 — 발화 수로 잰다 (2026-09-23).

지적(ai-debate run-20260923T004639Z, gemini·claude-fable 공통). 라이브는 목표 비중에
현금버퍼를 반영해 비교한다(auto_trade_binance.py 3098~3118: 위험자산에 (1-buf) 를 곱하고
CASH=buf 를 넣은 tgt_w_norm 과 cur_w_fut 를 비교). 백테스트는 combined 를 합 1 로 정규화한
채로 비교한다(backtest_futures_v25.py 976). 보유 비중은 pv*w*(1-CASH_BUFFER) 로 만들어지므로
재조정 직후에도 백테스트 쪽에는 half-turnover 가 버퍼 크기만큼 남는 구조적 바닥이 생긴다.

맞다면 백테스트의 낮은 문턱 팔들은 "매일 발화" 로 포화돼 서로 구분이 안 되고, 라이브 값
0.01 은 백테스트 0.02 근처에 대응한다. 즉 고원의 좌표가 통째로 밀린다.

재는 법. 엔진에 env 토글 FUT_DRIFT_PARITY 를 붙여 드리프트 비교에만 라이브 변환을 넣고,
같은 문턱 격자에서 발화 수를 양쪽으로 센다. 실행(체결·배분) 경로는 안 건드린다.
바닥이 실재하면 낮은 문턱일수록 발화가 크게 줄고, 줄어든 발화 수가 한 칸 느슨한 문턱의
현행 값과 비슷해진다.

실행: cd /home/gmoh/mon/251229/strategies/cap_defend/research && python3 -u bt_fut_drift_parity.py
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, '/home/gmoh/mon/251229/strategies/cap_defend')

START, END = '2020-10-01', '2026-09-18'
TX_FUT = 0.0006
THRESHOLDS = [0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.05]
FUT_BASE = dict(sma_days=42, mom_short_days=18, mom_long_days=127,
                vol_threshold=0.05, n_snapshots=5, snap_interval_bars=95,
                canary_hyst=0.015, universe_size=3, cap=1 / 3)

_bars = None
_funding = None
_k2 = None


def run_one(thr: float, parity: bool, phase: int = 0) -> dict:
    os.environ['FUT_DRIFT_PARITY'] = 'on' if parity else 'off'
    os.environ['DRIFT_HEALTH_MODE'] = 'refill'
    import importlib
    import backtest_futures_v25 as F
    importlib.reload(F)          # 모듈 상수로 읽는 토글이라 재적재가 필요하다
    kw = dict(FUT_BASE)
    kw['drift_threshold'] = thr
    m = F.run(_bars, _funding, interval='D', leverage=_k2,
              tx_cost=TX_FUT, maint_rate=0.004,
              vol_days=90, vol_mode='daily', post_flip_delay=5,
              phase_offset_bars=phase, health_mode='mom2vol',
              start_date=START, end_date=END, **kw)
    eq = m.get('_equity')
    return dict(rebal=int(m.get('Rebal', 0)), trades=int(m.get('Trades', 0)),
                cagr=float(m.get('CAGR', 0.0)), mdd=float(m.get('MDD', 0.0)),
                final=float(eq.iloc[-1]) if eq is not None else float('nan'))


def main() -> None:
    global _bars, _funding, _k2
    t0 = time.time()
    from unified_backtest import load_data
    from backtest_futures_v25 import build_K2_signal
    print('코인 일봉 로드 중...')
    _bars, _funding = load_data('D')
    _k2 = build_K2_signal(_bars, btc_cap_sma_period=42, btc_cap_thr_mid=1.015,
                          btc_cap_thr_max=1.05, k2_sma_period=7, k2_hyst=0.025,
                          l_min=2.0, l_mid=3.0, l_max=4.0)
    print(f'  심볼 {len(_bars)}종 ({time.time()-t0:.0f}초)\n')

    print('드리프트 비교 좌표계 — 발화 수 (앵커 위상 0, 기본 비용, 5.97년)')
    print(f"  {'문턱':>7} {'현행 발화':>9} {'라이브정합 발화':>15} {'차이':>7} "
          f"{'현행 체결':>9} {'정합 체결':>9}")
    rows = {}
    for thr in THRESHOLDS:
        a = run_one(thr, False)
        b = run_one(thr, True)
        rows[thr] = (a, b)
        print(f"  {thr:>7.4f} {a['rebal']:>9d} {b['rebal']:>15d} "
              f"{b['rebal']-a['rebal']:>+7d} {a['trades']:>9d} {b['trades']:>9d}")

    print('\n  라이브정합 좌표에서의 성적')
    print(f"  {'문턱':>7} {'CAGR':>8} {'MDD':>8} {'최종배수':>9}")
    for thr in THRESHOLDS:
        b = rows[thr][1]
        print(f"  {thr:>7.4f} {b['cagr']:>8.1f} {b['mdd']:>8.2f} {b['final']:>9.2f}")

    print(f'\n총 소요: {time.time()-t0:.0f}초')
    os.environ['FUT_DRIFT_PARITY'] = 'off'


if __name__ == '__main__':
    main()

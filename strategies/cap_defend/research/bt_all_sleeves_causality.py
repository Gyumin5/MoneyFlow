"""세 슬리브 전부에 주문 인과성 검사를 돌린다 — 양성·음성 대조 한 벌 (2026-10-15).

bt_spot_order_causality.py 는 현물 하나만 쟀다. 그것만으로는 검사 자체를 못 믿는다 —
아무거나 실패로 찍는 검사일 수도 있기 때문이다. 깨끗할 것으로 보이는 슬리브가 실제로
통과해야 양성 결과가 뜻을 갖는다.

## 세 슬리브가 서로 다른 질문을 받는다

구조가 다르므로 같은 질문을 던지면 안 된다.

- 현물·선물: 수량 기반이다. 체결가는 그날 시가, 신호는 전날이다. 그래서 물을 것은
  "주문 수량이 그날 종가에 딸려 움직이나" 다. 종가는 시가 체결 뒤에야 알 수 있다.
- 주식: 비중 기반이고 그날 종가로 신호를 만들어 그날 종가로 체결한다. 체결가 자체가
  신호의 입력이라 위 질문이 성립하지 않는다. 그래서 주식에는 다른 것을 묻는다 —
  "라이브처럼 전 거래일 신호로 바꾸면 결과가 달라지나". 크기는
  bt_stock_sig_lag_impact.py 가 따로 잰다. 여기서는 어긋남의 존재만 확정한다.

## 판정 (결과 보기 전에 못박는다)

수량 기반 둘(현물·선물):
  같은 날을 두 번 재생하되 보유·목표 코인의 그날 종가만 저가＋폭×0.25 와 저가＋폭×0.75
  로 바꾼다. 시가·고가·저가와 다른 날은 안 건드린다. 주문 목표 수량과 체결 후 보유
  수량을 비교한다.
  - 실패(시간 누수): 대상·방향이 바뀌거나 수량 차이가 max(1e-10, 1e-10×기준값) 초과.
  - 통과: 허용오차 안.
  - 측정불가: 보유가 있는 리밸런싱 날을 못 찾음.

검사가 유효하려면 둘이 갈려야 한다. 셋 다 통과하거나 셋 다 실패하면 검사가 슬리브를
구분 못 하는 것이므로 그 자체를 결함으로 기록하고 결론을 안 낸다.

주식:
  같은 앵커·같은 구간에서 현행(당일 신호)과 전 거래일 신호 두 팔의 자산곡선을 비교한다.
  - 어긋남 있음: 두 곡선 최대 차이 > 1e-9.
  - 어긋남 없음: 그 이하.

## 선행조건

두 정본 엔진에 ORDER_PROBE 계측과 SPOT_PV_DATE 토글, bt_stock_mom3 에 STOCK_SIG_LAG
토글이 있어야 한다. 측정이 끝나면 전부 되돌리고 재현 패치만 research/ 에 남긴다.

실행: cd /home/gmoh/mon/251229/strategies/cap_defend/research
      python3 bt_all_sleeves_causality.py
"""
from __future__ import annotations

import copy
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import unified_backtest as ub  # noqa: E402
import backtest_futures_v25 as fbt  # noqa: E402
import bt_coin_param_resweep as R  # noqa: E402

TOL_REL = 1e-10


def perturb(bars, date, coins, frac):
    """date 의 Close 만 저가＋폭×frac 으로. 다른 봉·다른 열은 불변."""
    out = dict(bars)
    for c in coins:
        df = bars.get(c)
        if df is None or date not in df.index:
            continue
        d2 = df.copy()
        lo = float(d2.loc[date, 'Low'])
        hi = float(d2.loc[date, 'High'])
        d2.loc[date, 'Close'] = lo + (hi - lo) * frac
        out[c] = d2
    return out


def compare(a, b):
    if not a or not b:
        return [('probe_missing', len(a), len(b))]
    pa, pb = a[0], b[0]
    diffs = []
    for key in ('target_qty', 'holdings_after'):
        ka, kb = pa[key], pb[key]
        for coin in sorted(set(ka) | set(kb)):
            va, vb = ka.get(coin, 0.0), kb.get(coin, 0.0)
            if (coin in ka) != (coin in kb):
                diffs.append((f'{key}:{coin}:membership', va, vb))
                continue
            if abs(va - vb) > max(1e-10, TOL_REL * abs(va)):
                diffs.append((f'{key}:{coin}', va, vb))
    return diffs


def qty_sleeve(name, mod, runner, bars, funding):
    """수량 기반 슬리브 한 곳을 잰다. 통과/실패/측정불가 중 하나를 반환."""
    print(f"\n{'=' * 60}\n[{name}] 주문 수량이 그날 종가에 딸려 움직이나\n{'=' * 60}")

    trace = []
    runner(bars, funding, trace=trace)
    days = [t['date'] for t in trace if t.get('rebal')]
    print(f"  리밸런싱 날 {len(days)} 개")

    chosen = None
    for d in days:
        mod.ORDER_PROBE.clear()
        os.environ['ORDER_PROBE_DATE'] = str(d.date())
        try:
            runner(bars, funding)
        finally:
            os.environ.pop('ORDER_PROBE_DATE', None)
        if mod.ORDER_PROBE:
            p = mod.ORDER_PROBE[0]
            if p['holdings_before'] and p['target_qty']:
                chosen = (d, p)
                break
    if chosen is None:
        print("  측정불가 — 보유가 있는 리밸런싱 날을 못 찾음")
        return '측정불가'

    d, p = chosen
    coins = sorted(set(p['holdings_before']) | set(p['target_qty']))
    print(f"  고른 날 {d.date()} · 직전 보유 {sorted(p['holdings_before'])} · 목표 {sorted(p['target_qty'])}")

    res = {}
    for frac in (0.25, 0.75):
        b2 = perturb(bars, d, coins, frac)
        mod.ORDER_PROBE.clear()
        os.environ['ORDER_PROBE_DATE'] = str(d.date())
        try:
            runner(b2, funding)
        finally:
            os.environ.pop('ORDER_PROBE_DATE', None)
        res[frac] = list(mod.ORDER_PROBE)
        if res[frac]:
            print(f"    종가 {frac:.2f} 자리 → 평가액 {res[frac][0]['pv_used']:.6f}")

    diffs = compare(res[0.25], res[0.75])
    if not diffs:
        print("  통과 — 종가를 바꿔도 주문 수량이 안 바뀐다")
        return '통과'
    print(f"  실패 — 차이 {len(diffs)} 건 (시간 누수)")
    for nm, va, vb in diffs[:6]:
        print(f"    {nm}: {va!r} vs {vb!r}")
    return '실패'


def spot_runner(bars, funding, trace=None):
    os.environ['DRIFT_HEALTH_MODE'] = 'refill'
    ub.run(bars, funding, interval='D', asset_type='spot', leverage=1.0, tx_cost=0.004,
           start_date=R.START, end_date=R.END,
           sma_bars=42, mom_short_bars=20, mom_long_bars=127,
           vol_threshold=0.05, vol_mode='daily',
           n_snapshots=7, snap_interval_bars=217,
           canary_hyst=0.015, drift_threshold=0.10, post_flip_delay=5,
           universe_size=3, cap=1 / 3, selection='greedy',
           stop_kind='none', stop_pct=0.0,
           dd_lookback=60, dd_threshold=-99.0,
           bl_drop=-99.0, bl_days=7, crash_threshold=-99.0,
           health_mode='mom2vol', _trace=trace)


def fut_runner(bars, funding, trace=None):
    os.environ['DRIFT_HEALTH_MODE'] = 'refill'
    kw = dict(R.FUT_BASE)
    fbt.run(bars, funding, interval='D', leverage=R._k2,
            tx_cost=R.TX_FUT, maint_rate=0.004,
            vol_days=90, vol_mode='daily', post_flip_delay=5,
            phase_offset_bars=0, health_mode='mom2vol',
            start_date=R.START, end_date=R.END, _trace=trace, **kw)


def stock_check():
    print(f"\n{'=' * 60}\n[주식] 신호 시점이 라이브와 어긋나 있나\n{'=' * 60}")
    print("  질문이 다르다 — 주식은 체결가 자체가 신호 입력이라 수량 검사가 성립하지 않는다.")
    curves = {}
    for lag in ('0', '1'):
        os.environ['STOCK_SIG_LAG'] = lag
        try:
            curves[lag] = R.run_stock(R.ANCHORS[0], False)
        finally:
            os.environ['STOCK_SIG_LAG'] = '0'
    common = curves['0'].index.intersection(curves['1'].index)
    gap = float((curves['0'].loc[common] - curves['1'].loc[common]).abs().max())
    print(f"  현행(당일 신호) 대 전 거래일 신호 두 곡선 최대 차이 {gap:.6g}")
    if gap > 1e-9:
        print("  어긋남 있음 — 채택 백테스트가 라이브보다 한 봉 앞선 정보로 체결한다")
        return '어긋남'
    print("  어긋남 없음")
    return '없음'


def main() -> int:
    from unified_backtest import load_data
    from backtest_futures_v25 import build_K2_signal
    print("코인 일봉 로드 중...", flush=True)
    bars, funding = load_data('D')
    R._bars, R._funding = bars, funding
    R._k2 = build_K2_signal(bars, btc_cap_sma_period=42, btc_cap_thr_mid=1.015,
                            btc_cap_thr_max=1.05, k2_sma_period=7, k2_hyst=0.025,
                            l_min=2.0, l_mid=3.0, l_max=4.0)
    print(f"  심볼 {len(bars)}종", flush=True)

    out = {}
    out['코인 현물 (25%)'] = qty_sleeve('코인 현물 (25%)', ub, spot_runner, bars, funding)
    out['코인 선물 (15%)'] = qty_sleeve('코인 선물 (15%)', fbt, fut_runner, bars, funding)
    out['미국 주식 (60%)'] = stock_check()

    print(f"\n{'=' * 60}\n정리\n{'=' * 60}")
    for k, v in out.items():
        print(f"  {k:16s} {v}")

    qty = [out['코인 현물 (25%)'], out['코인 선물 (15%)']]
    if qty[0] == qty[1]:
        print("\n  경고 — 수량 기반 두 슬리브가 같은 판정이다. 검사가 둘을 구분 못 하면")
        print("  양성 결과를 그대로 믿을 수 없다. 검사 자체를 먼저 의심한다.")
        return 1
    print("\n  검사 유효 — 양성(현물)과 음성(선물)이 갈렸다. 아무거나 실패로 찍는 검사가 아니다.")
    return 0


if __name__ == '__main__':
    sys.exit(main())

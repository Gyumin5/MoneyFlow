#!/usr/bin/env python3
"""snap_interval 진단 테스트.

4가지 테스트:
1. tx=0 vs tx=0.4%: 비용 제거 후 snap 효과 분리
2. n_snap=1 vs 3: snapshot 합성 효과 검증
3. flip_refresh='all' vs 'oldest': 플립 시 갱신 방식
4. cash_reentry: 현금 재진입 디커플링

베스트 조합 2개 x snap 3개(21, 60, 120)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backtest_spot_barfreq import load_binance_prices, run_backtest
from coin_engine import Params

START = '2020-10-01'
END = '2026-03-31'


def make_params(sma, ms, ml, np_):
    return Params(
        canary='K8', vote_smas=(sma,), vote_moms=(), vote_threshold=1,
        canary_band=1.5, health='HK', health_sma=0,
        health_mom_short=ms, health_mom_long=ml, health_vol_window=ml,
        vol_cap=0.05, sma_period=sma, selection='baseline', n_picks=np_,
        weighting='baseline', top_n=40, risk='baseline', tx_cost=0.004,
        start_date=START, end_date=END,
        dd_exit_lookback=0, dd_exit_threshold=0,
        bl_threshold=0, bl_days=0, drift_threshold=0,
    )


def run_one(prices, fm, funding, params, snap, **kwargs):
    """단일 테스트 실행. kwargs = n_snap, flip_refresh, cash_reentry, tx_cost"""
    tx = kwargs.pop('tx_cost', 0.004)
    r = run_backtest(prices, fm, params,
                     bars_per_day=1, snap_interval_bars=snap,
                     mode='spot', leverage=1.0, tx_cost=tx,
                     funding_data=None, **kwargs)
    m = r['metrics']
    cal = m['CAGR'] / abs(m['MDD']) if m['MDD'] != 0 else 0
    return m['Sharpe'], m['CAGR'], m['MDD'], cal, r['rebal_count']


def print_row(label, snap, sh, cagr, mdd, cal, rbl):
    print('%20s snap=%3s | Sh %5.2f  CAGR %+7.1f%%  MDD %+6.1f%%  Cal %5.2f  Rbl %4d' % (
        label, snap, sh, cagr*100, mdd*100, cal, rbl))


def main():
    prices, fm, funding = load_binance_prices('D', top_n=40)

    configs = [
        ('Ms20/Ml90/N3', 50, 20, 90, 3),
        ('Ms80/Ml240/N3', 50, 80, 240, 3),
    ]
    snaps = [21, 60, 120]

    for cname, sma, ms, ml, np_ in configs:
        params = make_params(sma, ms, ml, np_)
        print('\n' + '='*80)
        print('SMA%d/%s' % (sma, cname))
        print('='*80)

        # ─── Test 1: tx=0 vs tx=0.4% ───
        print('\n--- Test 1: tx=0 vs tx=0.4% (비용 분해) ---')
        for snap in snaps:
            sh0, c0, m0, cal0, r0 = run_one(prices, fm, funding, params, snap, tx_cost=0)
            sh4, c4, m4, cal4, r4 = run_one(prices, fm, funding, params, snap, tx_cost=0.004)
            print_row('tx=0', snap, sh0, c0, m0, cal0, r0)
            print_row('tx=0.4%', snap, sh4, c4, m4, cal4, r4)
            print('%20s snap=%3s | Sh diff %+.2f  CAGR diff %+.1f%%' % (
                '', snap, sh4-sh0, (c4-c0)*100))
            print()

        # ─── Test 2: n_snap=1 vs 3 ───
        print('--- Test 2: n_snap=1 vs n_snap=3 ---')
        for snap in snaps:
            sh1, c1, m1, cal1, r1 = run_one(prices, fm, funding, params, snap, n_snap=1)
            sh3, c3, m3, cal3, r3 = run_one(prices, fm, funding, params, snap, n_snap=3)
            print_row('n_snap=1', snap, sh1, c1, m1, cal1, r1)
            print_row('n_snap=3', snap, sh3, c3, m3, cal3, r3)
            print()

        # ─── Test 3: flip_refresh all vs oldest ───
        print('--- Test 3: flip_refresh=all vs oldest ---')
        for snap in snaps:
            sha, ca, ma, cala, ra = run_one(prices, fm, funding, params, snap, flip_refresh='all')
            sho, co, mo, calo, ro = run_one(prices, fm, funding, params, snap, flip_refresh='oldest')
            print_row('flip=all', snap, sha, ca, ma, cala, ra)
            print_row('flip=oldest', snap, sho, co, mo, calo, ro)
            print()

        # ─── Test 4: cash_reentry ───
        print('--- Test 4: cash_reentry (디커플링) ---')
        for snap in snaps:
            shn, cn, mn, caln, rn = run_one(prices, fm, funding, params, snap, cash_reentry=False)
            shy, cy, my, caly, ry = run_one(prices, fm, funding, params, snap, cash_reentry=True)
            print_row('reentry=off', snap, shn, cn, mn, caln, rn)
            print_row('reentry=on', snap, shy, cy, my, caly, ry)
            print()

        # ─── Test 4+: cash_reentry + snap=NONE ───
        print('--- Test 4+: cash_reentry + snap=NONE ---')
        shn, cn, mn, caln, rn = run_one(prices, fm, funding, params, 99999, cash_reentry=False)
        shy, cy, my, caly, ry = run_one(prices, fm, funding, params, 99999, cash_reentry=True)
        print_row('NONE+reentry=off', 'INF', shn, cn, mn, caln, rn)
        print_row('NONE+reentry=on', 'INF', shy, cy, my, caly, ry)

        # ─── Combo: cash_reentry + flip=oldest ───
        print('\n--- Combo: cash_reentry + flip=oldest ---')
        for snap in snaps:
            sh, c, m, cal, r = run_one(prices, fm, funding, params, snap,
                                       cash_reentry=True, flip_refresh='oldest')
            print_row('reentry+oldest', snap, sh, c, m, cal, r)

    print('\nDone.')


if __name__ == '__main__':
    main()
